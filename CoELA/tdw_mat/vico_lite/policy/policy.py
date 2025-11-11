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
        target_lookup = {
            str(n).lower() for n in task_payload.get("target", [])
        }
        container_lookup = {
            str(n).lower() for n in task_payload.get("container", [])
        }
        grabbable_lookup = {
            str(n).lower() for n in knowledge.get("floor_objects", [])
        }
        if not grabbable_lookup:
            grabbable_lookup = target_lookup | container_lookup
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
        prev_status = obs.get("status")
        last_command = self.agent_state.get("last_command")
        if isinstance(prev_status, int) and last_command == "pick":
            pick_id = self.agent_state.get("last_pick_id")
            if prev_status == 1:
                if self.logger:
                    self.logger.info("[Policy] pick success id=%s frame=%s", pick_id, frame)
            elif prev_status == 2:
                if self.logger:
                    self.logger.warning("[Policy] pick failed id=%s frame=%s", pick_id, frame)
                last_pick_pos = self.agent_state.get("last_pick_pos")
                if last_pick_pos is not None:
                    self._register_skip_target(position=last_pick_pos)
            self.agent_state.pop("last_command", None)
            self.agent_state.pop("last_pick_id", None)
            self.agent_state.pop("last_pick_pos", None)
        self.executor.tick()
        self._update_agent_state(obs)
        self.agent_state["frame"] = frame
        perception_input = self._prepare_perception_inputs(obs)
        bridge_output = self.memory_bridge.process(perception_input)
        snapshot = bridge_output["snapshot"]
        visible_infos = perception_input.get("object_infos", [])
        self.agent_state["last_visible_infos"] = visible_infos
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
        if plan is None or plan.action_type != "pick":
            override_plan = self._maybe_force_pick(visible_infos)
            if override_plan is not None:
                plan = override_plan
                override_used = True
        if plan is None:
            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "no_plan"})
        plan = self._ensure_plan_target(plan)
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
        object_names = [item.get("name", "") for item in visible]
        agent_pos = None
        if "agent" in obs:
            try:
                agent_pos = np.asarray(obs["agent"], dtype=np.float32)
            except Exception:
                agent_pos = None
        object_infos = []
        for item in visible:
            info: Dict[str, Any] = {
                "id": item.get("id"),
                "name": item.get("name", ""),
                "category": item.get("category"),
                "visible": item.get("visible", True),
            }
            position = item.get("position", None)
            if position is None:
                position = item.get("location")
            if position is not None:
                obj_pos: Optional[np.ndarray] = None
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
            name_lc = str(info.get("name", "")).lower()
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
        for entry in obs.get("held_objects", []) or []:
            if not entry:
                continue
            obj_id = entry.get("id")
            arm_name = entry.get("arm")
            if obj_id is not None:
                holding_ids.append(obj_id)
            if isinstance(arm_name, str):
                holding_slots[arm_name.lower()] = obj_id
        self.agent_state["holding_ids"] = holding_ids
        self.agent_state["holding_slots"] = holding_slots
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
            return None
        if self.agent_state.get("holding_ids"):
            return None
        blocked_coords = self._skip_coords().union(self.agent_state.get("nav_guard_coords", set()))
        skip_names = {
            str(name).lower()
            for name in self.agent_state.get("skip_targets", {}).get("names", set())
        }
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for info in object_infos:
            if not info.get("is_grabbable"):
                continue
            if info.get("id") is None:
                continue
            position = self._normalise_position(info.get("position"))
            if position is None:
                continue
            if info.get("distance") is None:
                continue
            distance = float(info["distance"])
            if not np.isfinite(distance) or distance > 1.6:
                continue
            if self._is_blocked_position(position, blocked_coords, radius=0.4):
                continue
            name_lc = str(info.get("name", "")).lower()
            if name_lc in skip_names:
                continue
            priority = info.get("priority", 0.0)
            if info.get("is_task_target"):
                priority += 4.0
            score = priority - distance
            candidates.append((score, info))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_info = candidates[0][1]
        target_id = best_info.get("id")
        target_pos = self._normalise_position(best_info.get("position"))
        if target_id is None or target_pos is None:
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
