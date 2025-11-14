from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..config import ViCoConfig
from ..execution.executor import PlanExecutor
from ..language.guidance import GuidanceController, GuidanceResult, ReasonedPlan
from ..language.reasoner import PolicyReasoner
from ..memory_bridge import MemoryBridgeController, TeamMemoryHub


class ViCoPolicy:
    """High-level orchestrator combining perception, memory, reasoning, and execution."""

    def __init__(
        self,
        cfg: ViCoConfig,
        agent_id: int,
        logger,
        agent_memory,
        env_api: Optional[Dict[str, Any]] = None,
        memory_hub: Optional[TeamMemoryHub] = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.logger = logger
        self.device = device
        self.env_api = env_api or {}
        self.agent_memory = agent_memory
        self.team_hub = memory_hub
        self.memory_bridge = MemoryBridgeController(cfg, agent_id=agent_id, device=device, hub=memory_hub)
        self.reasoner = PolicyReasoner(cfg)
        self.guidance = GuidanceController(cfg, self.reasoner)
        self.executor = PlanExecutor(cfg, agent_id, agent_memory, env_api=self.env_api, logger=logger)
        self.agent_state: Dict[str, Any] = {
            "role": "explore",
            "subgoal": "search",
            "holding_ids": [],
            "skip_targets": {"names": set(), "coords": set()},
            "current_room": None,
            "heuristic_streak": 0,
        }
        self.last_plan: Optional[GuidanceResult] = None
        self.last_reasoner_frame: int = -self.cfg.reasoner_min_interval
        self.active_target: Optional[Tuple[float, float, float]] = None
        self.target_acquire_frame: int = -999
        self._common_sense = self._load_common_sense()
        self._name_map = self._load_name_map()
        self._task_target_lookup: Set[str] = set()
        self._task_container_lookup: Set[str] = set()
        self._grabbable_lookup: Set[str] = set()
        self.agent_state.update(
            {
                "task_type": None,
                "task_targets": [],
                "task_containers": [],
                "grabbable_names": [],
            }
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.memory_bridge.reset()
        self.executor.tick()
        self.agent_state.update({
            "role": "explore",
            "subgoal": "search",
            "holding_ids": [],
            "skip_targets": {"names": set(), "coords": set()},
            "current_room": None,
            "heuristic_streak": 0,
            "visible_summary": [],
        })
        self.last_plan = None
        self.last_reasoner_frame = -self.cfg.reasoner_min_interval
        self.active_target = None

    # ------------------------------------------------------------------
    def set_goal_context(
        self,
        goal_objects: Optional[Dict[str, Any]] = None,
        rooms_name: Optional[List[str]] = None,
        task_type: Optional[str] = None,
    ) -> None:
        goal_objects = goal_objects or {}
        goal_names = {str(name).lower() for name in goal_objects.keys()}
        selected_type = task_type
        knowledge = self._common_sense or {}
        candidate_types = [task_type] if task_type else ["food", "stuff"]
        candidate_types = [t for t in candidate_types if t]
        for candidate in candidate_types:
            candidate_targets = {
                str(n).lower() for n in knowledge.get(candidate, {}).get("target", [])
            }
            if goal_names and candidate_targets and goal_names.issubset(candidate_targets):
                selected_type = candidate
                break
        if selected_type is None:
            best_type = None
            best_overlap = -1
            for candidate in ["food", "stuff"]:
                candidate_targets = {
                    str(n).lower() for n in knowledge.get(candidate, {}).get("target", [])
                }
                overlap = len(goal_names & candidate_targets)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_type = candidate
            selected_type = best_type or "food"
        task_payload = knowledge.get(selected_type, {})
        target_raw = {str(n).lower() for n in task_payload.get("target", [])}
        container_raw = {str(n).lower() for n in task_payload.get("container", [])}
        grabbable_raw = {str(n).lower() for n in knowledge.get("floor_objects", [])}
        if not grabbable_raw:
            grabbable_raw = target_raw | container_raw
        def _expand_with_mapping(names: Set[str]) -> Set[str]:
            expanded = set()
            for raw_name in names:
                expanded.add(raw_name)
                mapped = self._name_map.get(raw_name, raw_name).lower()
                if mapped != raw_name:
                    expanded.add(mapped)
            return expanded
        target_lookup = _expand_with_mapping(target_raw)
        container_lookup = _expand_with_mapping(container_raw)
        grabbable_lookup = _expand_with_mapping(grabbable_raw)
        self._task_target_lookup = target_lookup
        self._task_container_lookup = container_lookup
        self._grabbable_lookup = grabbable_lookup
        self.agent_state["task_type"] = selected_type
        self.agent_state["task_targets"] = sorted(target_lookup)
        self.agent_state["task_containers"] = sorted(container_lookup)
        self.agent_state["grabbable_names"] = sorted(grabbable_lookup)
        self.agent_state["goal_objects_raw"] = goal_objects
        if rooms_name is not None:
            self.agent_state["rooms_name"] = rooms_name
        if self.logger:
            self.logger.info(
                "[Policy] goal context set task=%s goal_objects=%s",
                selected_type,
                goal_objects or list(target_lookup)[:3],
            )

    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        frame = int(obs.get("current_frames", 0))
        # Reset logged failed objects cache for this frame
        self._logged_cal_failed_objects = set()
        prev_status = obs.get("status")
        last_command = self.agent_state.get("last_command")
        if isinstance(prev_status, int) and last_command == "pick":
            pick_id = self.agent_state.get("last_pick_id")
            if prev_status == 1:
                if self.logger:
                    self.logger.info("[Policy] pick success id=%s frame=%s", pick_id, frame)
                # Add successfully picked object to holding_ids immediately
                # This is necessary because held_objects may not be updated in the same frame
                if pick_id is not None:
                    current_holding_ids = self.agent_state.get("holding_ids", [])
                    if pick_id not in current_holding_ids:
                        current_holding_ids.append(pick_id)
                        self.agent_state["holding_ids"] = current_holding_ids
                        if self.logger:
                            self.logger.info(
                                "[Policy] pick success: added to holding_ids id=%s holding_ids=%s",
                                pick_id,
                                current_holding_ids,
                            )
                    # Get object name from visible_objects or object_info
                    obj_name = None
                    if self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
                        obj_info = self.agent_memory.object_info.get(pick_id)
                        if obj_info:
                            obj_name = obj_info.get("name")
                    if obj_name:
                        self._register_skip_target(name=obj_name)
            elif prev_status == 2:
                if self.logger:
                    self.logger.warning("[Policy] pick failed id=%s frame=%s", pick_id, frame)
                last_pick_pos = self.agent_state.get("last_pick_pos")
                if last_pick_pos is not None:
                    self._register_skip_target(position=last_pick_pos)
            self.agent_state.pop("last_command", None)
            self.agent_state.pop("last_pick_id", None)
            self.agent_state.pop("last_pick_pos", None)
        elif last_command == "pick" and self.logger:
            self.logger.debug(
                "[Policy] pick command pending: status=%s (type=%s) frame=%s",
                prev_status,
                type(prev_status).__name__,
                frame,
            )
        self.executor.tick()
        self._update_agent_state(obs)
        self.agent_state["frame"] = frame
        perception_input = self._prepare_perception_inputs(obs)
        bridge_output = self.memory_bridge.process(perception_input)
        snapshot = bridge_output["snapshot"]
        visible_infos = perception_input.get("object_infos", [])
        self.agent_state["last_visible_infos"] = visible_infos
        if self.logger and getattr(self.logger, "debug", None):
            grabbable_count = sum(1 for info in visible_infos if info.get("is_grabbable"))
            task_target_count = sum(1 for info in visible_infos if info.get("is_task_target"))
            self.logger.debug(
                "[Policy] frame=%s visible_infos: total=%s grabbable=%s task_targets=%s",
                frame,
                len(visible_infos),
                grabbable_count,
                task_target_count,
            )
        guard_summary = self.executor.guard_summary()
        guard_keys = set()
        for key in guard_summary.keys():
            if isinstance(key, tuple) and len(key) >= 2:
                x = float(key[0]) / 10.0 if isinstance(key[0], (int, float)) else float(key[0])
                z = float(key[1]) / 10.0 if isinstance(key[1], (int, float)) else float(key[1])
                guard_keys.add((self._quantize_coord(x), self._quantize_coord(z)))
        self.agent_state["nav_guard_coords"] = guard_keys
        if self.team_hub is not None:
            self.team_hub.update_nav_guard(self.agent_id, guard_summary)
        # Get scene_bounds from agent_memory if available
        scene_bounds = None
        if self.agent_memory is not None and hasattr(self.agent_memory, "_scene_bounds"):
            scene_bounds = getattr(self.agent_memory, "_scene_bounds", None)
        
        context_extra = {
            "nav_guard_info": guard_summary,
            "guard_cooldown": 0,
            "task_type": self.agent_state.get("task_type"),
            "task_targets": self.agent_state.get("task_targets", []),
            "task_containers": self.agent_state.get("task_containers", []),
            "grabbable_names": self.agent_state.get("grabbable_names", []),
            "visible_objects": self.agent_state.get("visible_summary", []),
            "goal_objects": self.agent_state.get("goal_objects_raw", {}),
        }
        if scene_bounds:
            context_extra["scene_bounds"] = scene_bounds
            self.agent_state["scene_bounds"] = scene_bounds
        if self.logger and getattr(self.logger, "debug", None):
            team_count = len(snapshot.team_symbolic) if getattr(snapshot, "team_symbolic", None) else 0
            partner_count = 0
            if getattr(snapshot, "per_agent_symbolic", None):
                for participant, entries in snapshot.per_agent_symbolic.items():
                    if participant == self.agent_id:
                        continue
                    partner_count += len(entries)
            self.logger.debug(
                "[Policy] frame=%s shared_context team_symbolic=%s partner_entries=%s",
                frame,
                team_count,
                partner_count,
            )
        force_llm = bool(self.agent_state.pop("force_llm", False))
        force_heuristics = frame - self.last_reasoner_frame < self.cfg.reasoner_min_interval
        if force_llm:
            force_heuristics = False
            context_extra["force_llm"] = True
        guidance_context = self.guidance.build_context(
            agent_id=self.agent_id,
            frame=frame,
            memory=self.memory_bridge,
            snapshot=snapshot,
            agent_state=self.agent_state,
            extra=context_extra,
        )
        guidance = self.guidance.decide(guidance_context, force_heuristics=force_heuristics)
        self.agent_state["last_reasoner_debug"] = guidance.debug
        if self.logger and guidance.debug:
            dbg = guidance.debug
            self.logger.debug(
                "[Policy] frame=%s src=%s force_heuristics=%s candidates=%s llm_error=%s",
                frame,
                guidance.source,
                dbg.get("force_heuristics"),
                dbg.get("candidate_count"),
                dbg.get("llm_error"),
            )
        streak = int(self.agent_state.get("heuristic_streak", 0))
        if guidance.source == "llm":
            streak = 0
        else:
            streak += 1
            threshold = max(4, self.cfg.reasoner_plan_horizon // 3 or 1)
            if streak >= threshold:
                self.agent_state["force_llm"] = True
                self.last_reasoner_frame = frame - self.cfg.reasoner_min_interval - 1
                if self.logger:
                    self.logger.debug(
                        "[Policy] heuristic streak %s reached threshold %s -> forcing LLM refresh",
                        streak,
                        threshold,
                    )
                streak = 0
        self.agent_state["heuristic_streak"] = streak
        override_used = False
        plan = guidance.plan
        holding_ids = self.agent_state.get("holding_ids", [])
        if holding_ids:
            if self.logger:
                self.logger.debug(
                    "[Policy] frame=%s holding_ids=%s checking for deliver",
                    frame,
                    holding_ids,
                )
            if plan is None or plan.action_type != "deliver":
                deliver_plan = self._maybe_force_deliver(visible_infos)
                if deliver_plan is not None:
                    plan = deliver_plan
                    override_used = True
                    if self.logger:
                        self.logger.info(
                            "[Policy] frame=%s overriding with deliver id=%s name=%s dist=%.2f",
                            frame,
                            deliver_plan.target_id,
                            deliver_plan.meta.get("target_name", "?"),
                            deliver_plan.meta.get("distance", -1.0),
                        )
                elif self.logger:
                    self.logger.debug(
                        "[Policy] frame=%s _maybe_force_deliver returned None",
                        frame,
                    )
        if plan is None or plan.action_type != "pick":
            if self.logger and visible_infos:
                grabbable_count = sum(1 for info in visible_infos if info.get("is_grabbable"))
                close_count = sum(1 for info in visible_infos if info.get("is_grabbable") and info.get("distance") is not None and float(info.get("distance", 999)) <= (4.5 if info.get("is_task_target") else 3.0))
                grabbable_names = [info.get("name", "?") for info in visible_infos if info.get("is_grabbable")][:5]
                if grabbable_count == 0 and len(visible_infos) > 0:
                    sample_names = [info.get("name", "?") for info in visible_infos[:5]]
                    sample_mapped = [info.get("name_mapped", "?") for info in visible_infos[:5]]
                    self.logger.debug(
                        "[Policy] frame=%s force_pick_check visible=%s grabbable=0 sample_names=%s sample_mapped=%s grabbable_lookup_size=%s",
                        frame,
                        len(visible_infos),
                        sample_names,
                        sample_mapped,
                        len(self._grabbable_lookup),
                    )
                else:
                    self.logger.debug(
                        "[Policy] frame=%s force_pick_check visible=%s grabbable=%s close=%s holding=%s names=%s",
                        frame,
                        len(visible_infos),
                        grabbable_count,
                        close_count,
                        len(holding_ids),
                        grabbable_names,
                    )
            if not override_used:
                override_plan = self._maybe_force_pick(visible_infos)
                if override_plan is not None:
                    plan = override_plan
                    override_used = True
                elif self.logger and visible_infos:
                    self.logger.debug("[Policy] frame=%s force_pick returned None", frame)
        if plan is None:
            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "no_plan"})
        plan = self._ensure_plan_target(plan)
        # Clamp LLM-generated coordinates to valid room positions before checking
        if plan.target_position is not None and guidance.source == "llm":
            # Use executor's clamp method to ensure coordinates are within valid room
            clamped_pos = self.executor._clamp_target(plan.target_position)
            if clamped_pos != plan.target_position:
                if self.logger:
                    self.logger.debug(
                        "[Policy] clamped LLM target from %s to %s",
                        plan.target_position,
                        clamped_pos,
                    )
                # Update plan with clamped position
                meta = dict(plan.meta)
                meta["original_target"] = plan.target_position
                meta["clamped"] = True
                plan = ReasonedPlan(plan.action_type, plan.target_id, clamped_pos, plan.confidence, meta)
        # Reject plans with targets outside map boundaries and add to skip_targets
        # Exception: if is_goal_position=True, clamp the position instead of rejecting
        is_goal_position = plan.meta.get("is_goal_position", False)
        if plan.target_position is not None and self.agent_memory is not None:
            target_pos = plan.target_position
            rejected = False
            if hasattr(self.agent_memory, "_scene_bounds"):
                bounds = getattr(self.agent_memory, "_scene_bounds", None)
                if bounds:
                    x, y, z = target_pos
                    x_min = bounds.get("x_min")
                    x_max = bounds.get("x_max")
                    z_min = bounds.get("z_min")
                    z_max = bounds.get("z_max")
                    if (x_min is not None and x_max is not None and (x < x_min or x > x_max)) or \
                       (z_min is not None and z_max is not None and (z < z_min or z > z_max)):
                        if is_goal_position:
                            # For goal_position, clamp the position instead of rejecting
                            clamped_x = min(max(x, x_min + 0.5), x_max - 0.5) if x_min is not None and x_max is not None else x
                            clamped_z = min(max(z, z_min + 0.5), z_max - 0.5) if z_min is not None and z_max is not None else z
                            clamped_pos = (clamped_x, y, clamped_z)
                            if self.logger:
                                self.logger.info(
                                    "[Policy] goal_position target outside map bounds, clamping from %s to %s",
                                    target_pos,
                                    clamped_pos,
                                )
                            meta = dict(plan.meta)
                            meta["original_target"] = target_pos
                            meta["clamped"] = True
                            plan = ReasonedPlan(plan.action_type, plan.target_id, clamped_pos, plan.confidence, meta)
                        else:
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] rejecting plan with target outside map bounds: %s (bounds: x=[%s, %s], z=[%s, %s])",
                                    target_pos,
                                    x_min, x_max, z_min, z_max,
                                )
                            # Add to skip_targets to prevent LLM from suggesting it again
                            self._register_skip_target(position=target_pos)
                            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "target_outside_map_bounds"})
                            rejected = True
            # Also check with check_pos_in_room if available
            if not rejected and plan.target_position is not None and self.env_api and "check_pos_in_room" in self.env_api:
                if not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                    if is_goal_position:
                        # For goal_position, try to use executor's clamp_target to find a valid position
                        clamped_pos = self.executor._clamp_target(target_pos)
                        if clamped_pos != target_pos:
                            if self.logger:
                                self.logger.info(
                                    "[Policy] goal_position target outside room, clamped from %s to %s",
                                    target_pos,
                                    clamped_pos,
                                )
                            meta = dict(plan.meta)
                            meta["original_target"] = target_pos
                            meta["clamped"] = True
                            plan = ReasonedPlan(plan.action_type, plan.target_id, clamped_pos, plan.confidence, meta)
                        else:
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] goal_position target outside room and cannot clamp, rejecting"
                                )
                            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "target_outside_room"})
                            rejected = True
                    else:
                        if self.logger:
                            self.logger.warning(
                                "[Policy] rejecting plan with target outside room: %s (adding to skip_targets)",
                                target_pos,
                            )
                        # Add to skip_targets to prevent LLM from suggesting it again
                        self._register_skip_target(position=target_pos)
                        plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "target_outside_room"})
        if plan.action_type != "idle" and guidance.source == "llm":
            self.last_reasoner_frame = frame
            if self.logger:
                self.logger.debug(
                    "[Policy] frame=%s accepted LLM plan action=%s target=%s conf=%.2f",
                    frame,
                    plan.action_type,
                    plan.target_position,
                    plan.confidence,
                )
        if override_used:
            guidance.source = "policy_override"
            plan.meta["override"] = "near_grabbable"
        self.last_plan = guidance
        command, exec_meta = self.executor.execute(plan, snapshot, self.agent_state)
        self.agent_state.setdefault("recent_meta", []).append(exec_meta)
        command_type = command.get("type") if isinstance(command, dict) else None
        if command_type == 3:
            self.agent_state["last_command"] = "pick"
            self.agent_state["last_pick_id"] = command.get("object", plan.target_id)
            self.agent_state["last_pick_pos"] = plan.target_position
        elif command_type == 4:
            self.agent_state["last_command"] = "deliver"
            self.agent_state["last_deliver_id"] = plan.target_id
            self.agent_state["last_deliver_pos"] = plan.target_position
        elif command_type not in (None, "ongoing"):
            self.agent_state["last_command"] = str(command_type)
        if exec_meta.get("forced_failure"):
            target = plan.target_position
            name = plan.meta.get("target_name") if isinstance(plan.meta, dict) else None
            self._register_skip_target(name=name, position=target)
            self.active_target = None
            self.agent_state["force_llm"] = True
            self.last_reasoner_frame = frame - self.cfg.reasoner_min_interval - 1
            if self.logger:
                self.logger.warning(
                    "[Policy] guard failure triggered LLM refresh (target=%s frame=%s)",
                    target,
                    frame,
                )
        return command

    # ------------------------------------------------------------------
    def observe_feedback(self, feedback: Dict[str, Any]) -> None:
        if feedback.get("success") is False:
            reason = feedback.get("reason")
            target = feedback.get("target")
            if reason == "invalid" and target is not None:
                self._register_skip_target(position=target)

    # ------------------------------------------------------------------
    def _prepare_perception_inputs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        if isinstance(rgb, torch.Tensor):
            rgb_tensor = rgb.float() / 255.0 if rgb.dtype != torch.float32 else rgb
        else:
            rgb_array = np.asarray(rgb)
            if rgb_array.dtype != np.float32:
                rgb_array = rgb_array.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb_array)
        visible = obs.get("visible_objects", [])
        if self.logger and getattr(self.logger, "debug", None):
            if not hasattr(self, '_logged_visible_empty'):
                self._logged_visible_empty = False
            if not visible and not self._logged_visible_empty:
                self.logger.debug(
                    "[Policy] _prepare_perception_inputs: obs['visible_objects'] is empty (len=%d)",
                    len(visible) if visible else 0,
                )
                self._logged_visible_empty = True
        object_names = [item.get("name", "") for item in visible]
        agent_pos = None
        if "agent" in obs:
            try:
                agent_pos = np.asarray(obs["agent"], dtype=np.float32)
            except Exception:
                agent_pos = None
        object_infos = []
        for item in visible:
            obj_id = item.get("id")
            obj_name = item.get("name")
            obj_type = item.get("type")
            if obj_id is None:
                continue
            # Filter out agent objects (type=3 or name="agent" or id matches agent_id)
            if obj_type == 3 or (isinstance(obj_name, str) and obj_name.lower() == "agent") or obj_id == self.agent_id:
                continue
            if obj_name is None or (isinstance(obj_name, str) and obj_name.strip() == ""):
                continue
            info: Dict[str, Any] = {
                "id": obj_id,
                "name": obj_name,
                "category": item.get("category"),
                "visible": item.get("visible", True),
                "seg_color": item.get("seg_color"),  # Add seg_color for distance estimation
            }
            position = item.get("position", None)
            location = item.get("location", None)
            # NOTE: visible_objects from tdw_gym.py only contains id, type, seg_color, name (no position field)
            # So position will always be None here, and we need to calculate it
            # Try to get position from multiple sources
            if position is None and location is None:
                # First try agent_memory.get_object_position (from stored object_info)
                if self.agent_memory is not None and hasattr(self.agent_memory, "get_object_position"):
                    try:
                        mem_pos = self.agent_memory.get_object_position(obj_id)
                        if mem_pos is not None:
                            position = list(mem_pos) if isinstance(mem_pos, (list, tuple, np.ndarray)) else mem_pos
                    except (KeyError, AttributeError, Exception):
                        pass
                # Then try agent_memory.object_info
                if position is None and self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
                    try:
                        if obj_id in self.agent_memory.object_info:
                            obj_info = self.agent_memory.object_info[obj_id]
                            mem_pos = obj_info.get("position")
                            if mem_pos is not None:
                                position = list(mem_pos) if isinstance(mem_pos, (list, tuple, np.ndarray)) else mem_pos
                    except (KeyError, AttributeError, Exception):
                        pass
                # If still None, try to calculate position using cal_object_position (like COELA does)
                # Note: agent_memory.update() is already called before policy.act(), so agent_memory.obs is up-to-date
                # CRITICAL: visible_objects from tdw_gym.py does NOT include position, so we MUST calculate it here
                if position is None:
                    has_seg_color = "seg_color" in item
                    has_obs = hasattr(self.agent_memory, "obs") and self.agent_memory.obs is not None
                    has_cal_method = self.agent_memory is not None and hasattr(self.agent_memory, "cal_object_position")
                    
                    # Always log to understand why cal_object_position is not being called
                    if self.logger:
                        self.logger.info(
                            "[Policy] _prepare_perception_inputs: position=None for id=%s name=%s has_seg_color=%s has_obs=%s has_cal_method=%s",
                            obj_id,
                            obj_name,
                            has_seg_color,
                            has_obs,
                            has_cal_method,
                        )
                    
                    if has_cal_method and has_seg_color and has_obs:
                        try:
                            calc_pos, _ = self.agent_memory.cal_object_position(item)
                            if calc_pos is not None:
                                position = list(calc_pos) if isinstance(calc_pos, (list, tuple, np.ndarray)) else calc_pos
                                if self.logger:
                                    self.logger.info(
                                        "[Policy] _prepare_perception_inputs: cal_object_position SUCCEEDED for id=%s name=%s position=%s",
                                        obj_id,
                                        obj_name,
                                        position,
                                    )
                                # Also store it in object_info for future use (agent_memory.get_object_list() might have skipped it)
                                if hasattr(self.agent_memory, "object_info"):
                                    if obj_id not in self.agent_memory.object_info:
                                        self.agent_memory.object_info[obj_id] = {}
                                    self.agent_memory.object_info[obj_id]["position"] = position
                            else:
                                # Log only once per object per frame to reduce log spam
                                if self.logger:
                                    if not hasattr(self, '_logged_cal_failed_objects'):
                                        self._logged_cal_failed_objects = set()
                                    if obj_id not in self._logged_cal_failed_objects:
                                        self.logger.debug(
                                            "[Policy] _prepare_perception_inputs: cal_object_position returned None for id=%s name=%s",
                                            obj_id,
                                            obj_name,
                                        )
                                        self._logged_cal_failed_objects.add(obj_id)
                        except (KeyError, AttributeError, Exception) as exc:
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] _prepare_perception_inputs: cal_object_position FAILED for id=%s name=%s: %s",
                                    obj_id,
                                    obj_name,
                                    exc,
                                )
                    elif self.logger:
                        self.logger.warning(
                            "[Policy] _prepare_perception_inputs: cal_object_position SKIPPED for id=%s name=%s (has_seg_color=%s has_obs=%s has_cal_method=%s)",
                            obj_id,
                            obj_name,
                            has_seg_color,
                            has_obs,
                            has_cal_method,
                        )
                # Also check if position is in the item dict with different keys
                if position is None:
                    for key in ["pos", "world_position", "world_pos", "xyz", "coord", "coordinates"]:
                        if key in item:
                            try:
                                pos_val = item[key]
                                if pos_val is not None:
                                    position = list(pos_val) if isinstance(pos_val, (list, tuple, np.ndarray)) else pos_val
                                    break
                            except Exception:
                                pass
            if position is None:
                position = location
            obj_pos: Optional[np.ndarray] = None
            if position is not None:
                try:
                    obj_pos = np.asarray(position, dtype=np.float32)
                except Exception:
                    obj_pos = None
                if isinstance(position, np.ndarray):
                    info["position"] = position.tolist()
                elif isinstance(position, (list, tuple)):
                    info["position"] = list(position)
                else:
                    info["position"] = position
            if location is not None and "position" not in info:
                if isinstance(location, np.ndarray):
                    info["location"] = location.tolist()
                elif isinstance(location, (list, tuple)):
                    info["location"] = list(location)
                else:
                    info["location"] = location
            if agent_pos is not None and obj_pos is not None:
                try:
                    if agent_pos.shape[0] >= 3 and obj_pos.shape[0] >= 3:
                        info["distance"] = float(np.linalg.norm(agent_pos[[0, 2]] - obj_pos[[0, 2]]))
                    else:
                        info["distance"] = float(np.linalg.norm(agent_pos - obj_pos))
                except Exception:
                    pass
            if "distance" not in info and item.get("distance") is not None:
                try:
                    info["distance"] = float(item.get("distance"))
                except Exception:
                    pass
            raw_name = str(info.get("name", "")).lower()
            if raw_name in ("none", "null", ""):
                continue
            mapped_name = self._name_map.get(raw_name, raw_name).lower()
            name_lc = mapped_name
            info["name_mapped"] = mapped_name
            info["is_task_target"] = name_lc in self._task_target_lookup
            info["is_task_container"] = name_lc in self._task_container_lookup
            info["is_grabbable"] = name_lc in self._grabbable_lookup or info["is_task_target"]
            distance_val = info.get("distance")
            priority = 0.0
            if info["is_task_target"]:
                priority += 4.0
            elif info["is_grabbable"]:
                priority += 2.5
            if isinstance(distance_val, (int, float)) and np.isfinite(float(distance_val)):
                priority += max(0.0, 2.5 - float(distance_val))
            info["priority"] = priority
            object_infos.append(info)
        self.agent_state["visible_summary"] = self._summarize_visible_objects(object_infos)
        return {
            "rgb": rgb_tensor,
            "depth": depth,
            "instruction": obs.get("instruction"),
            "object_names": object_names,
            "object_infos": object_infos,
        }

    def _update_agent_state(self, obs: Dict[str, Any]) -> None:
        holding_ids = []
        holding_slots: Dict[str, Optional[int]] = {}
        held_objects = obs.get("held_objects", []) or []
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug(
                "[Policy] _update_agent_state: held_objects=%s (type=%s)",
                held_objects,
                type(held_objects).__name__,
            )
        valid_held_objects = False
        for entry in held_objects:
            if not entry:
                continue
            obj_id = entry.get("id")
            arm_name = entry.get("arm")
            if obj_id is not None:
                holding_ids.append(obj_id)
                valid_held_objects = True
            if isinstance(arm_name, str):
                holding_slots[arm_name.lower()] = obj_id
        
        # Check if put_in was successful (status == 1 and last_command was deliver)
        prev_status = obs.get("status")
        last_command = self.agent_state.get("last_command")
        if isinstance(prev_status, int) and last_command == "deliver" and prev_status == 1:
            # put_in was successful, clear holding_ids
            if self.logger:
                self.logger.info(
                    "[Policy] deliver success: clearing holding_ids (was %s)",
                    self.agent_state.get("holding_ids", []),
                )
            holding_ids = []
            holding_slots = {}
            valid_held_objects = False  # Don't preserve existing holding_ids
        
        # If held_objects is empty or all None, preserve existing holding_ids
        # This is important because pick success may have just added an object,
        # but obs["held_objects"] may not be updated yet
        # However, don't preserve if put_in was successful (handled above)
        if not valid_held_objects and not (isinstance(prev_status, int) and last_command == "deliver" and prev_status == 1):
            existing_holding_ids = self.agent_state.get("holding_ids", [])
            if existing_holding_ids:
                holding_ids = existing_holding_ids
                if self.logger:
                    self.logger.debug(
                        "[Policy] _update_agent_state: held_objects empty, preserving existing holding_ids=%s",
                        holding_ids,
                    )
        self.agent_state["holding_ids"] = holding_ids
        self.agent_state["holding_slots"] = holding_slots
        if self.logger and holding_ids:
            self.logger.debug(
                "[Policy] _update_agent_state: updated holding_ids=%s holding_slots=%s",
                holding_ids,
                holding_slots,
            )
        if self.env_api and "belongs_to_which_room" in self.env_api:
            position = obs.get("agent")
            if isinstance(position, np.ndarray) or isinstance(position, list):
                room = self.env_api["belongs_to_which_room"](np.array(position))
                self.agent_state["current_room"] = room

    def _ensure_plan_target(self, plan: ReasonedPlan) -> ReasonedPlan:
        current_pos = getattr(self.agent_memory, "position", None)

        skip_coords = self._skip_coords()
        guard_coords = set(self.agent_state.get("nav_guard_coords", set()) or [])
        blocked_coords = skip_coords | guard_coords

        def _to_array(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return value
            if isinstance(value, (list, tuple)):
                return np.array(value, dtype=np.float32)
            return None
        current_arr = _to_array(current_pos)
        target_arr = _to_array(self.active_target)
        if plan.target_position is not None:
            if not self._is_blocked_position(plan.target_position, blocked_coords):
                self.active_target = plan.target_position
                self.target_acquire_frame = self.agent_state.get("frame", -999)
                return plan
            # Don't redirect if this is a goal_position (we need to reach it)
            if plan.meta.get("is_goal_position", False):
                if self.logger:
                    self.logger.debug(
                        "[Policy] target blocked but is_goal_position=True, keeping original plan"
                    )
                # Keep the original plan but with lower confidence
                plan = ReasonedPlan(plan.action_type, plan.target_id, plan.target_position, max(plan.confidence * 0.7, 0.3), plan.meta)
            else:
                meta = dict(plan.meta)
                meta["fallback"] = "explore"
                meta["reason"] = f"redirect_from_blocked:{meta.get('target_name')}"
                plan = ReasonedPlan("search", None, None, max(plan.confidence * 0.5, 0.2), meta)
            self.active_target = None

        if self.active_target is not None:
            if self._is_blocked_position(self.active_target, blocked_coords):
                self.active_target = None
                target_arr = None
        if self.active_target is not None and current_arr is not None and target_arr is not None:
            if current_arr.shape[0] >= 3:
                current_2d = current_arr[[0, 2]]
            else:
                current_2d = current_arr
            if target_arr.shape[0] >= 3:
                target_2d = target_arr[[0, 2]]
            else:
                target_2d = target_arr
            dist = float(np.linalg.norm(current_2d - target_2d))
            if dist > 0.75:
                meta = dict(plan.meta)
                meta.setdefault("fallback", "persist")
                meta["target_position"] = self.active_target
                return ReasonedPlan(
                    action_type="move",
                    target_id=None,
                    target_position=self.active_target,
                    confidence=max(plan.confidence, 0.4),
                    meta=meta,
                )
            else:
                self.active_target = None

        if plan.action_type not in {"search", "move", "explore"}:
            return plan
        if self.agent_memory is None:
            return plan
        try:
            attempts = 0
            while True:
                explore_x, explore_z = self.agent_memory.explore()
                candidate_pos = (float(explore_x), 0.0, float(explore_z))
                if not self._is_blocked_position(candidate_pos, blocked_coords) or attempts >= 10:
                    break
                attempts += 1
        except Exception as exc:  # pragma: no cover
            if self.logger:
                self.logger.warning("[Policy] explore fallback failed: %s", exc)
            return plan
        pos = getattr(self.agent_memory, "position", np.array([explore_x, 0.0, explore_z]))
        pos_arr = _to_array(pos)
        height = float(pos_arr[1]) if pos_arr is not None and pos_arr.shape[0] >= 2 else 0.0
        target_pos: Tuple[float, float, float] = (float(explore_x), height, float(explore_z))
        self.active_target = target_pos
        meta = dict(plan.meta)
        meta["fallback"] = "explore"
        meta["target_position"] = target_pos
        return ReasonedPlan(action_type="move", target_id=None, target_position=target_pos, confidence=max(plan.confidence, 0.3), meta=meta)

    # ------------------------------------------------------------------
    def _register_skip_target(self, name: Optional[str] = None, position: Optional[Tuple[float, float, float]] = None) -> None:
        store = self.agent_state.setdefault("skip_targets", {"names": set(), "coords": set()})
        names_set = store.setdefault("names", set())
        coords_set = store.setdefault("coords", set())
        if name:
            names_set.add(str(name).lower())
        if position is not None:
            coords_set.add(self._coord_key(position))
        store["names"] = names_set
        store["coords"] = coords_set
        self.agent_state["skip_targets"] = store
        if self.team_hub is not None:
            self.team_hub.update_skip_targets(self.agent_id, names_set, coords_set)

    def _skip_coords(self) -> set:
        store = self.agent_state.get("skip_targets", {})
        return set(store.get("coords", set())) if isinstance(store, dict) else set()

    @staticmethod
    def _quantize_coord(value: float, step: float = 0.5) -> float:
        if step <= 0:
            return round(value, 2)
        return round(round(value / step) * step, 2)

    def _is_blocked_position(self, position: Tuple[float, float, float], blocked_coords: set, radius: float = 0.6) -> bool:
        if position is None:
            return False
        try:
            x = float(position[0])
            z = float(position[2])
        except Exception:
            return False
        qx = self._quantize_coord(x)
        qz = self._quantize_coord(z)
        if (qx, qz) in blocked_coords:
            return True
        for bx, bz in blocked_coords:
            if abs(bx - qx) <= radius and abs(bz - qz) <= radius:
                return True
        return False

    @staticmethod
    def _coord_key(position: Tuple[float, float, float]) -> Tuple[float, float]:
        if isinstance(position, (list, tuple)):
            if len(position) >= 3:
                x = position[0]
                z = position[2]
            elif len(position) == 2:
                x = position[0]
                z = position[1]
            else:
                raise ValueError("Position sequence must have length >= 2")
        else:  # type: ignore[unreachable]
            raise ValueError("Position must be list or tuple")
        return (
            ViCoPolicy._quantize_coord(float(x)),
            ViCoPolicy._quantize_coord(float(z)),
        )

    # ------------------------------------------------------------------
    def _load_name_map(self) -> Dict[str, str]:
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "name_map.json"),
            os.path.join(os.getcwd(), "CoELA", "tdw_mat", "dataset", "name_map.json"),
            os.path.join(os.getcwd(), "dataset", "name_map.json"),
        ]
        for path in candidate_paths:
            norm = os.path.abspath(path)
            if os.path.exists(norm):
                try:
                    with open(norm, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return {str(k).lower(): str(v).lower() for k, v in data.items()}
                except Exception as exc:
                    if self.logger:
                        self.logger.warning("[Policy] failed to load name_map from %s (%s)", norm, exc)
        return {}

    def _load_common_sense(self) -> Dict[str, Any]:
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "list.json"),
            os.path.join(os.getcwd(), "CoELA", "tdw_mat", "dataset", "list.json"),
        ]
        for path in candidate_paths:
            norm = os.path.abspath(path)
            try:
                with open(norm, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                continue
            except Exception as exc:
                if self.logger:
                    self.logger.warning("[Policy] failed to load commonsense list from %s (%s)", norm, exc)
        return {}

    def _maybe_force_pick(self, object_infos: List[Dict[str, Any]]) -> Optional[ReasonedPlan]:
        if not object_infos:
            if self.logger:
                self.logger.debug("[Policy] _maybe_force_pick: no object_infos")
            return None
        if self.agent_state.get("holding_ids"):
            if self.logger:
                self.logger.debug("[Policy] _maybe_force_pick: already holding objects")
            return None
        agent_pos = None
        if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos = np.asarray(self.agent_memory.position, dtype=np.float32)
            except Exception:
                pass
        blocked_coords = self._skip_coords().union(self.agent_state.get("nav_guard_coords", set()))
        skip_names = {
            str(name).lower()
            for name in self.agent_state.get("skip_targets", {}).get("names", set())
        }
        # Get holding_ids to exclude already held objects
        holding_ids = set(self.agent_state.get("holding_ids", []))
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        filtered_reasons: Dict[str, int] = {}
        for info in object_infos:
            if not info.get("is_grabbable"):
                filtered_reasons["not_grabbable"] = filtered_reasons.get("not_grabbable", 0) + 1
                continue
            obj_id = info.get("id")
            if obj_id is None:
                filtered_reasons["no_id"] = filtered_reasons.get("no_id", 0) + 1
                continue
            # Skip objects that are already being held
            if obj_id in holding_ids:
                filtered_reasons["already_held"] = filtered_reasons.get("already_held", 0) + 1
                continue
            position = self._normalise_position(info.get("position"))
            if position is None:
                position = self._normalise_position(info.get("location"))
            if position is None:
                # Try to get position from agent_memory
                obj_id = info.get("id")
                if obj_id is not None and self.agent_memory is not None:
                    if hasattr(self.agent_memory, "get_object_position"):
                        try:
                            mem_pos = self.agent_memory.get_object_position(obj_id)
                            if mem_pos is not None:
                                position = self._normalise_position(mem_pos)
                                # Update info with found position
                                if position is not None:
                                    info["position"] = position
                        except Exception:
                            pass
                    if position is None and hasattr(self.agent_memory, "object_info") and obj_id in getattr(self.agent_memory, "object_info", {}):
                        try:
                            obj_info = self.agent_memory.object_info[obj_id]
                            mem_pos = obj_info.get("position")
                            if mem_pos is not None:
                                position = self._normalise_position(mem_pos)
                                # Update info with found position
                                if position is not None:
                                    info["position"] = position
                        except Exception:
                            pass
            # Calculate distance if missing
            if info.get("distance") is None:
                if position is not None and agent_pos is not None:
                    try:
                        pos_arr = np.asarray(position, dtype=np.float32)
                        if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                            info["distance"] = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                        else:
                            info["distance"] = float(np.linalg.norm(agent_pos - pos_arr))
                    except Exception:
                        pass
                # If position is None but we have seg_color, try to estimate distance from depth
                elif position is None and info.get("seg_color") is not None and self.agent_memory is not None:
                    try:
                        seg_color = info.get("seg_color")
                        if seg_color is not None and hasattr(self.agent_memory, "obs") and self.agent_memory.obs is not None:
                            depth = self.agent_memory.obs.get("depth")
                            seg_mask = self.agent_memory.obs.get("seg_mask")
                            if depth is not None and seg_mask is not None:
                                # Find matched pixels
                                if isinstance(seg_color, (tuple, list)):
                                    seg_color = np.array(seg_color, dtype=seg_mask.dtype)
                                if seg_mask.ndim == 3:
                                    color_match = np.all(seg_mask == seg_color, axis=-1)
                                    matched_depth = depth[color_match]
                                    # Filter valid depths
                                    valid_matched_depth = matched_depth[(matched_depth > 0) & (matched_depth < 100)]
                                    if len(valid_matched_depth) > 0:
                                        # Use median depth as distance estimate
                                        estimated_depth = float(np.median(valid_matched_depth))
                                        info["distance"] = estimated_depth
                                        if self.logger:
                                            self.logger.debug(
                                                "[Policy] _maybe_force_pick: estimated distance from depth: id=%s name=%s dist=%.2f",
                                                obj_id,
                                                info.get("name"),
                                                estimated_depth,
                                            )
                    except Exception as exc:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_pick: distance estimation failed: id=%s name=%s error=%s",
                                obj_id,
                                info.get("name"),
                                exc,
                            )
            # For task targets, be more lenient - allow pick even without position if we have distance
            if position is None:
                # If it's a task target and we have distance, we can still try to pick (will use object_id only)
                if info.get("is_task_target") and info.get("distance") is not None:
                    # Allow it, but we'll need to handle this in executor
                    pass
                # If we have distance estimate (even if not task target), allow for grabbable objects
                elif info.get("distance") is not None and info.get("is_grabbable"):
                    # Allow it - we have distance estimate from depth
                    pass
                else:
                    filtered_reasons["no_position"] = filtered_reasons.get("no_position", 0) + 1
                    continue
            if info.get("distance") is None:
                filtered_reasons["no_distance"] = filtered_reasons.get("no_distance", 0) + 1
                continue
            distance = float(info["distance"])
            # Increase max_distance to allow navigation to objects
            # For task targets: allow up to 10m (executor will navigate)
            # For other grabbables: allow up to 8m (executor will navigate)
            max_distance = 10.0 if info.get("is_task_target") else 8.0
            if not np.isfinite(distance) or distance > max_distance:
                filtered_reasons["too_far"] = filtered_reasons.get("too_far", 0) + 1
                continue
            if self._is_blocked_position(position, blocked_coords, radius=0.4):
                filtered_reasons["blocked"] = filtered_reasons.get("blocked", 0) + 1
                continue
            name_lc = str(info.get("name", "")).lower()
            if name_lc in skip_names:
                filtered_reasons["skip_name"] = filtered_reasons.get("skip_name", 0) + 1
                continue
            priority = info.get("priority", 0.0)
            if info.get("is_task_target"):
                priority += 4.0
            score = priority - distance
            candidates.append((score, info))
        if not candidates:
            if self.logger and filtered_reasons:
                self.logger.debug(
                    "[Policy] _maybe_force_pick: no candidates filtered_reasons=%s",
                    filtered_reasons,
                )
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_info = candidates[0][1]
        target_id = best_info.get("id")
        target_pos = self._normalise_position(best_info.get("position"))
        if target_id is None:
            return None
        # If position is None but we have distance, try to estimate position or allow pick without position
        if target_pos is None:
            distance = best_info.get("distance")
            is_task_target = best_info.get("is_task_target", False)
            if distance is not None and agent_pos is not None:
                # Try to estimate position first (preferred for task targets)
                if is_task_target:
                    try:
                        # Get forward vector from agent_memory
                        agent_forward = None
                        if self.agent_memory is not None and hasattr(self.agent_memory, "forward"):
                            agent_forward = self.agent_memory.forward
                        if agent_forward is not None:
                            forward_arr = np.asarray(agent_forward, dtype=np.float32)
                            if len(forward_arr) >= 3:
                                # Normalize forward vector
                                forward_norm = np.linalg.norm(forward_arr)
                                if forward_norm > 1e-6:
                                    forward_arr = forward_arr / forward_norm
                                    # Estimate target position in XZ plane (use X and Z components)
                                    estimated_x = float(agent_pos[0]) + float(distance) * forward_arr[0]
                                    estimated_z = float(agent_pos[2]) + float(distance) * forward_arr[2]
                                    estimated_y = float(agent_pos[1])  # Keep same height
                                    target_pos = (estimated_x, estimated_y, estimated_z)
                                    if self.logger:
                                        self.logger.info(
                                            "[Policy] _maybe_force_pick: estimated position for task target id=%s name=%s dist=%.2f pos=%s",
                                            target_id,
                                            best_info.get("name"),
                                            distance,
                                            target_pos,
                                        )
                    except Exception as exc:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_pick: position estimation failed: id=%s name=%s error=%s",
                                target_id,
                                best_info.get("name"),
                                exc,
                            )
                
                # If position is still None but distance is close enough, allow pick without position
                # Executor will handle this case
                if target_pos is None:
                    if distance <= 2.5:  # Allow pick if distance is close enough
                        if self.logger:
                            self.logger.info(
                                "[Policy] _maybe_force_pick: allowing pick without position (id=%s name=%s dist=%.2f)",
                                target_id,
                                best_info.get("name"),
                                distance,
                            )
                        # Keep target_pos as None, executor will handle it
                    else:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_pick: skipping without position (too far): id=%s name=%s distance=%.2f",
                                target_id,
                                best_info.get("name"),
                                distance,
                            )
                        return None
            else:
                if self.logger:
                    self.logger.debug(
                        "[Policy] _maybe_force_pick: skipping without position: id=%s name=%s is_task_target=%s distance=%s",
                        target_id,
                        best_info.get("name"),
                        is_task_target,
                        distance,
                    )
                return None
        # Reject invalid positions (0,0,0 is likely a default/placeholder)
        # But allow None if distance is close enough (handled above)
        if target_pos is not None:
            if target_pos == (0.0, 0.0, 0.0) or (abs(target_pos[0]) < 0.01 and abs(target_pos[2]) < 0.01):
                if self.logger:
                    self.logger.debug(
                        "[Policy] _maybe_force_pick: skipping invalid position (0,0,0): id=%s name=%s",
                        target_id,
                        best_info.get("name"),
                    )
                return None
        meta = {
            "reason": "policy_override_grabbable",
            "target_name": best_info.get("name"),
            "distance": best_info.get("distance"),
        }
        confidence = 0.95 if best_info.get("is_task_target") else 0.85
        if self.logger:
            self.logger.info(
                "[Policy] overriding with pick id=%s name=%s dist=%.2f",
                target_id,
                best_info.get("name"),
                best_info.get("distance", -1.0),
            )
        return ReasonedPlan("pick", target_id, target_pos, confidence, meta)

    def _maybe_force_deliver(self, object_infos: List[Dict[str, Any]]) -> Optional[ReasonedPlan]:
        """Force a deliver action if holding objects and a nearby container is visible."""
        holding_ids = self.agent_state.get("holding_ids", [])
        if not holding_ids:
            if self.logger:
                self.logger.debug("[Policy] _maybe_force_deliver: no holding_ids")
            return None
        if self.logger:
            self.logger.debug(
                "[Policy] _maybe_force_deliver: holding_ids=%s visible_infos=%d",
                holding_ids,
                len(object_infos),
            )
        agent_pos = None
        if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos = np.asarray(self.agent_memory.position, dtype=np.float32)
            except Exception:
                pass
        if agent_pos is None:
            return None
        # If no containers in visible_infos, try to find containers in agent_memory.object_info
        if not any(info.get("is_task_container", False) for info in object_infos):
            # Try to find containers in agent_memory.object_info
            if self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
                container_candidates = []
                task_container_names = {str(n).lower() for n in self.agent_state.get("task_containers", [])}
                
                # Always check for goal_position_id first (this is the actual deliver target)
                goal_position_id = None
                # env_api can be a dict or an object, so check both ways
                if self.env_api is not None:
                    # Try dict access first (most common case)
                    if isinstance(self.env_api, dict):
                        goal_position_id = self.env_api.get("goal_position_id")
                    # Try attribute access (if it's an object)
                    elif hasattr(self.env_api, "goal_position_id"):
                        goal_position_id = self.env_api.goal_position_id
                    
                    if goal_position_id is not None:
                        if self.logger:
                            self.logger.info(
                                "[Policy] _maybe_force_deliver: found goal_position_id=%s (type=%s, env_api_type=%s)",
                                goal_position_id,
                                type(goal_position_id).__name__,
                                type(self.env_api).__name__,
                            )
                    elif self.logger:
                        self.logger.debug(
                            "[Policy] _maybe_force_deliver: goal_position_id not found in env_api (type=%s, keys=%s)",
                            type(self.env_api).__name__,
                            list(self.env_api.keys()) if isinstance(self.env_api, dict) else "N/A",
                        )
                
                # If task_containers is empty, try to get container_ids from env_api
                if not task_container_names and self.env_api is not None:
                    # Try to get container_ids from env_api
                    container_ids = None
                    if isinstance(self.env_api, dict):
                        container_ids = self.env_api.get("container_ids")
                    elif hasattr(self.env_api, "container_ids"):
                        container_ids = self.env_api.container_ids
                    
                    if container_ids:
                        # Use container_ids as valid container IDs
                        valid_container_ids = set(container_ids)
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_deliver: task_containers empty, using env_api.container_ids=%s",
                                list(valid_container_ids),
                            )
                    elif goal_position_id is not None:
                        # Use goal_position_id as the container
                        valid_container_ids = {goal_position_id}
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_deliver: task_containers empty, using goal_position_id=%s",
                                goal_position_id,
                            )
                    else:
                        valid_container_ids = None
                else:
                    valid_container_ids = None
                
                if self.logger:
                    self.logger.debug(
                        "[Policy] _maybe_force_deliver: searching memory for containers task_containers=%s object_info_count=%d valid_container_ids=%s",
                        list(task_container_names),
                        len(getattr(self.agent_memory, "object_info", {})),
                        list(valid_container_ids) if valid_container_ids else None,
                    )
                # Collect all object names for debugging
                all_obj_names = []
                for obj_id, obj_info in getattr(self.agent_memory, "object_info", {}).items():
                    if obj_id is None:
                        continue
                    obj_name = str(obj_info.get("name", "")).lower() if obj_info.get("name") else ""
                    obj_type = obj_info.get("type")
                    obj_category = str(obj_info.get("category", "")).lower() if obj_info.get("category") else ""
                    all_obj_names.append((obj_id, obj_name, obj_type, obj_category))
                    
                    # Check if this object is a container
                    is_container = False
                    # PRIORITY 1: Always treat goal_position_id as a container (highest priority)
                    if goal_position_id is not None and obj_id == goal_position_id:
                        is_container = True
                    elif task_container_names:
                        # Exact match
                        if obj_name in task_container_names:
                            is_container = True
                        # Partial match: check if any task_container name is contained in obj_name or vice versa
                        elif any(tc_name in obj_name or obj_name in tc_name for tc_name in task_container_names if tc_name and obj_name):
                            is_container = True
                    if not is_container and valid_container_ids is not None and obj_id in valid_container_ids:
                        is_container = True
                    # Type-based: Type 1 is container, but also check type 2 (beds, tables, etc. can be containers)
                    if not is_container and (obj_type == 1 or obj_type == 2):
                        # For type 2, only consider if it's a known container category
                        if obj_type == 1:
                            is_container = True
                        elif obj_type == 2:
                            # Type 2 can be beds, tables, etc. - check name/category
                            if any(keyword in obj_name for keyword in ["bed", "table", "counter", "cabinet", "fridge", "stove", "dishwasher", "microwave", "plate", "bowl", "tray"]):
                                is_container = True
                    if not is_container and ("container" in obj_category or obj_category in ["bed", "table", "counter", "cabinet", "fridge", "stove", "dishwasher", "microwave"]):
                        is_container = True
                    # Also check if name contains container-related keywords
                    if not is_container:
                        container_keywords = ["plate", "bowl", "tray", "dish", "container", "cabinet", "fridge", "stove", "microwave", "dishwasher"]
                        if any(keyword in obj_name for keyword in container_keywords):
                            is_container = True
                    
                    if is_container:
                        obj_pos = obj_info.get("position")
                        if obj_pos is None:
                            if self.logger:
                                self.logger.debug(
                                    "[Policy] _maybe_force_deliver: container id=%s name=%s type=%s has no position",
                                    obj_id,
                                    obj_name,
                                    obj_type,
                                )
                        else:
                            obj_pos = self._normalise_position(obj_pos)
                            if obj_pos is None:
                                if self.logger:
                                    self.logger.debug(
                                        "[Policy] _maybe_force_deliver: container id=%s name=%s type=%s position normalized to None",
                                        obj_id,
                                        obj_name,
                                        obj_type,
                                    )
                            else:
                                try:
                                    pos_arr = np.asarray(obj_pos, dtype=np.float32)
                                    if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                                        distance = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                                    else:
                                        distance = float(np.linalg.norm(agent_pos - pos_arr))
                                except Exception:
                                    distance = 999.0
                                # Distance limits:
                                # - If this is goal_position_id, no distance limit (it's the actual target)
                                # - For type 2 (beds, tables), allow larger distance (20m) since they're fixed furniture
                                # - For other containers, use standard 10m limit
                                if goal_position_id is not None and obj_id == goal_position_id:
                                    max_distance = float('inf')  # No limit for goal_position_id
                                elif obj_type == 2:
                                    max_distance = 20.0  # Limit for beds/tables
                                else:
                                    max_distance = 10.0
                                if np.isfinite(distance) and distance <= max_distance:
                                    container_candidates.append((distance, obj_id, obj_pos, obj_name))
                                    if self.logger:
                                        self.logger.debug(
                                            "[Policy] _maybe_force_deliver: found container candidate id=%s name=%s type=%s dist=%.2f",
                                            obj_id,
                                            obj_name,
                                            obj_type,
                                            distance,
                                        )
                                elif self.logger:
                                    self.logger.debug(
                                        "[Policy] _maybe_force_deliver: container id=%s name=%s type=%s too far (dist=%.2f > %.2f)",
                                        obj_id,
                                        obj_name,
                                        obj_type,
                                        distance,
                                        max_distance,
                                    )
                if container_candidates:
                    # Sort by distance and pick the closest one
                    container_candidates.sort(key=lambda x: x[0])
                    best_distance, best_id, best_pos, best_name = container_candidates[0]
                    
                    # If this is goal_position_id, try to get actual position from env_api if available
                    if goal_position_id is not None and best_id == goal_position_id:
                        # Try to get actual position from env_api (if it has access to object_manager)
                        # For now, use the position from memory but log it
                        if self.logger:
                            self.logger.info(
                                "[Policy] overriding with deliver id=%s name=%s (goal_position_id) dist=%.2f pos=%s",
                                best_id,
                                best_name,
                                best_distance,
                                best_pos,
                            )
                    else:
                        if self.logger:
                            self.logger.info(
                                "[Policy] overriding with deliver id=%s name=%s (from memory) dist=%.2f",
                                best_id,
                                best_name,
                                best_distance,
                            )
                    meta = {
                        "reason": "policy_override_deliver_memory",
                        "target_name": best_name,
                        "distance": best_distance,
                        "is_goal_position": (goal_position_id is not None and best_id == goal_position_id),
                    }
                    return ReasonedPlan("deliver", best_id, best_pos, 0.9, meta)
                # If no candidates found but goal_position_id exists and is too far, return move plan
                elif goal_position_id is not None:
                    # Find goal_position_id in memory even if it was filtered out
                    goal_obj_info = None
                    for obj_id, obj_info in getattr(self.agent_memory, "object_info", {}).items():
                        if obj_id == goal_position_id:
                            goal_obj_info = obj_info
                            break
                    if goal_obj_info is not None:
                        goal_pos = goal_obj_info.get("position")
                        if goal_pos is not None:
                            goal_pos = self._normalise_position(goal_pos)
                            if goal_pos is not None:
                                goal_name = str(goal_obj_info.get("name", "goal_position")).lower()
                                try:
                                    pos_arr = np.asarray(goal_pos, dtype=np.float32)
                                    if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                                        goal_distance = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                                    else:
                                        goal_distance = float(np.linalg.norm(agent_pos - pos_arr))
                                except Exception:
                                    goal_distance = 999.0
                                # If goal_position_id is too far, return move plan to navigate to it
                                if self.logger:
                                    self.logger.info(
                                        "[Policy] goal_position_id=%s (name=%s) too far (dist=%.2f), returning move plan",
                                        goal_position_id,
                                        goal_name,
                                        goal_distance,
                                    )
                                meta = {
                                    "reason": "policy_override_move_to_goal",
                                    "target_name": goal_name,
                                    "distance": goal_distance,
                                    "is_goal_position": True,
                                }
                                return ReasonedPlan("move", goal_position_id, goal_pos, 0.9, meta)
                if self.logger:
                    # Log sample of object names for debugging
                    sample_names = [(id, name, type) for id, name, type, _ in all_obj_names[:10]]
                    self.logger.debug(
                        "[Policy] _maybe_force_deliver: no container candidates found in memory (task_containers=%s, sample_obj_names=%s)",
                        list(task_container_names),
                        sample_names,
                    )
        skip_targets = self.agent_state.get("skip_targets", {"names": set(), "coords": set()})
        skip_names = {str(n).lower() for n in skip_targets.get("names", [])}
        blocked_coords = set()
        for coord in skip_targets.get("coords", []):
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                # coord can be [x, z] (2 elements) or [x, y, z] (3 elements)
                if len(coord) >= 3:
                    blocked_coords.add((round(float(coord[0]), 2), round(float(coord[2]), 2)))
                else:
                    blocked_coords.add((round(float(coord[0]), 2), round(float(coord[1]), 2)))
        candidates = []
        filtered_reasons = {}
        for info in object_infos:
            if not info.get("is_task_container", False):
                filtered_reasons["not_container"] = filtered_reasons.get("not_container", 0) + 1
                continue
            container_id = info.get("id")
            if container_id is None:
                filtered_reasons["no_id"] = filtered_reasons.get("no_id", 0) + 1
                continue
            position = self._normalise_position(info.get("position"))
            if position is None:
                position = self._normalise_position(info.get("location"))
            if position is None:
                # Try to get position from agent_memory
                if container_id is not None and self.agent_memory is not None:
                    if hasattr(self.agent_memory, "get_object_position"):
                        try:
                            mem_pos = self.agent_memory.get_object_position(container_id)
                            if mem_pos is not None:
                                position = self._normalise_position(mem_pos)
                        except Exception:
                            pass
                    if position is None and hasattr(self.agent_memory, "object_info") and container_id in getattr(self.agent_memory, "object_info", {}):
                        try:
                            obj_info = self.agent_memory.object_info[container_id]
                            mem_pos = obj_info.get("position")
                            if mem_pos is not None:
                                position = self._normalise_position(mem_pos)
                        except Exception:
                            pass
            if position is None:
                filtered_reasons["no_position"] = filtered_reasons.get("no_position", 0) + 1
                continue
            if info.get("distance") is None:
                if position is not None and agent_pos is not None:
                    try:
                        pos_arr = np.asarray(position, dtype=np.float32)
                        if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                            info["distance"] = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                        else:
                            info["distance"] = float(np.linalg.norm(agent_pos - pos_arr))
                    except Exception:
                        filtered_reasons["no_distance"] = filtered_reasons.get("no_distance", 0) + 1
                        continue
                else:
                    filtered_reasons["no_distance"] = filtered_reasons.get("no_distance", 0) + 1
                    continue
            distance = float(info["distance"])
            # Increase max_distance to allow navigation to containers
            # Executor will handle navigation, so we can allow further containers
            max_distance = 10.0
            if not np.isfinite(distance) or distance > max_distance:
                filtered_reasons["too_far"] = filtered_reasons.get("too_far", 0) + 1
                continue
            if self._is_blocked_position(position, blocked_coords, radius=0.4):
                filtered_reasons["blocked"] = filtered_reasons.get("blocked", 0) + 1
                continue
            name_lc = str(info.get("name", "")).lower()
            if name_lc in skip_names:
                filtered_reasons["skip_name"] = filtered_reasons.get("skip_name", 0) + 1
                continue
            score = 5.0 - distance
            candidates.append((score, info))
        if not candidates:
            if self.logger and filtered_reasons:
                self.logger.debug(
                    "[Policy] _maybe_force_deliver: no candidates filtered_reasons=%s",
                    filtered_reasons,
                )
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_info = candidates[0][1]
        target_id = best_info.get("id")
        target_pos = self._normalise_position(best_info.get("position"))
        if target_id is None or target_pos is None:
            return None
        meta = {
            "reason": "policy_override_deliver",
            "target_name": best_info.get("name"),
            "distance": best_info.get("distance"),
        }
        confidence = 0.9
        if self.logger:
            self.logger.info(
                "[Policy] overriding with deliver id=%s name=%s dist=%.2f",
                target_id,
                best_info.get("name"),
                best_info.get("distance", -1.0),
            )
        return ReasonedPlan("deliver", target_id, target_pos, confidence, meta)

    @staticmethod
    def _normalise_position(position: Any) -> Optional[Tuple[float, float, float]]:
        if position is None:
            return None
        if isinstance(position, np.ndarray):
            if position.shape[0] < 3:
                return None
            return (float(position[0]), float(position[1]), float(position[2]))
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            return (float(position[0]), float(position[1]), float(position[2]))
        return None

    def _summarize_visible_objects(self, object_infos: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]) -> Tuple[float, float]:
            priority = float(item.get("priority", 0.0))
            distance = float(item.get("distance", np.inf))
            return (-priority, distance)

        summary: List[Dict[str, Any]] = []
        for info in sorted(object_infos, key=sort_key)[:limit]:
            raw_distance = info.get("distance")
            if isinstance(raw_distance, (int, float)) and np.isfinite(float(raw_distance)):
                distance_val: Optional[float] = float(raw_distance)
            else:
                distance_val = None
            entry = {
                "name": info.get("name"),
                "distance": distance_val,
                "target": bool(info.get("is_task_target")),
                "container": bool(info.get("is_task_container")),
                "grabbable": bool(info.get("is_grabbable")),
            }
            summary.append(entry)
        return summary
