from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
        })
        self.last_plan = None
        self.last_reasoner_frame = -self.cfg.reasoner_min_interval
        self.active_target = None

    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        frame = int(obs.get("current_frames", 0))
        self.executor.tick()
        self._update_agent_state(obs)
        self.agent_state["frame"] = frame
        perception_input = self._prepare_perception_inputs(obs)
        bridge_output = self.memory_bridge.process(perception_input)
        snapshot = bridge_output["snapshot"]
        context_extra = {
            "nav_guard_info": self.executor.guard_summary(),
            "guard_cooldown": 0,
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
        plan = guidance.plan or ReasonedPlan("idle", None, None, 0.0, {"reason": "no_plan"})
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
        self.last_plan = guidance
        command, exec_meta = self.executor.execute(plan, snapshot, self.agent_state)
        self.agent_state.setdefault("recent_meta", []).append(exec_meta)
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
            position = item.get("position") or item.get("location")
            if position is not None:
                info["position"] = position
                if agent_pos is not None:
                    try:
                        obj_pos = np.asarray(position, dtype=np.float32)
                        info["distance"] = float(np.linalg.norm(agent_pos[[0, 2]] - obj_pos[[0, 2]])) if agent_pos.shape[0] >= 3 and obj_pos.shape[0] >= 3 else float(np.linalg.norm(agent_pos - obj_pos))
                    except Exception:
                        pass
            if "distance" not in info and item.get("distance") is not None:
                try:
                    info["distance"] = float(item.get("distance"))
                except Exception:
                    pass
            object_infos.append(info)
        return {
            "rgb": rgb_tensor,
            "depth": depth,
            "instruction": obs.get("instruction"),
            "object_names": object_names,
            "object_infos": object_infos,
        }

    def _update_agent_state(self, obs: Dict[str, Any]) -> None:
        self.agent_state["holding_ids"] = [entry.get("id") for entry in obs.get("held_objects", []) if entry]
        if self.env_api and "belongs_to_which_room" in self.env_api:
            position = obs.get("agent")
            if isinstance(position, np.ndarray) or isinstance(position, list):
                room = self.env_api["belongs_to_which_room"](np.array(position))
                self.agent_state["current_room"] = room

    def _ensure_plan_target(self, plan: ReasonedPlan) -> ReasonedPlan:
        current_pos = getattr(self.agent_memory, "position", None)
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
            self.active_target = plan.target_position
            self.target_acquire_frame = self.agent_state.get("frame", -999)
            return plan

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
            skip_coords = self._skip_coords()
            attempts = 0
            while True:
                explore_x, explore_z = self.agent_memory.explore()
                coord_key = self._coord_key((explore_x, explore_z))
                if coord_key not in skip_coords or attempts >= 5:
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

    def _skip_coords(self) -> set:
        store = self.agent_state.get("skip_targets", {})
        return set(store.get("coords", set())) if isinstance(store, dict) else set()

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
        return (round(float(x), 2), round(float(z), 2))
