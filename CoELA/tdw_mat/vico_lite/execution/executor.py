from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..config import ViCoConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    plan: Dict[str, Any]
    navigation: Dict[str, Any]
    forced_failure: bool = False


class PlanExecutor:
    def __init__(
        self,
        cfg: ViCoConfig,
        agent_id: int,
        agent_memory,
        env_api=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.agent_memory = agent_memory
        self.env_api = env_api
        self.logger = logger or LOGGER
        self.guard_skip_decay = cfg.guard_skip_decay
        self._guard_skip: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # ---------------------------------------------------------------------
    def execute(self, plan, snapshot, agent_state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        action_type = plan.action_type
        meta: Dict[str, Any] = {"plan": plan.meta}
        command: Dict[str, Any]
        if action_type in {"move", "search", "assist", "deliver"}:
            command, nav_meta = self._navigate_to(plan.target_position, agent_state, follow=False)
            meta.update(nav_meta)
        elif action_type == "pick":
            command, nav_meta = self._approach_and_pick(plan, agent_state)
            meta.update(nav_meta)
        elif action_type in {"idle", "wait"}:
            command = {"type": "ongoing"}
        else:
            command = {"type": "ongoing"}
            meta["unknown_action"] = action_type
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug(
                "[Executor] grounded plan=%s -> command=%s meta=%s",
                plan,
                command,
                meta,
            )
        return command, meta

    # ---------------------------------------------------------------------
    def _navigate_to(self, target_pos, agent_state, follow: bool = False):
        meta: Dict[str, Any] = {"navigation": "idle"}
        if target_pos is None:
            return {"type": "ongoing"}, meta
        if self.agent_memory is None:
            meta["reason"] = "no_agent_memory"
            return {"type": "ongoing"}, meta
        target_pos = self._clamp_target(target_pos)
        if hasattr(self.agent_memory, "belongs_to_which_room"):
            room = self.agent_memory.belongs_to_which_room(target_pos)
            if room is None and hasattr(self.agent_memory, "center_of_room"):
                current_room = agent_state.get("current_room")
                if current_room is not None:
                    room_center = self.agent_memory.center_of_room(current_room)
                    if room_center is not None:
                        target_pos = (room_center[0], target_pos[1], room_center[2])
        action, path_len = self.agent_memory.move_to_pos(target_pos, follow=follow)
        meta.update({"navigation": "move_to_pos", "distance": path_len})
        guard_threshold = self._guard_threshold()
        if path_len is not None and guard_threshold is not None and path_len > guard_threshold:
            key = tuple(int(x * 10) for x in target_pos[:2])
            self._guard_skip[key] = {"frames": self.guard_skip_decay, "target": target_pos}
            meta.update({"navigation": "guard_turn", "distance": path_len, "reason": "path_too_long", "forced_failure": True})
            if self.logger:
                self.logger.warning(
                    "[Executor] Guard triggered for agent %s (len=%.2f thr=%.2f) target=%s",
                    self.agent_id,
                    path_len,
                    guard_threshold,
                    target_pos,
                )
            return {"type": "ongoing"}, meta
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug(
                "[Executor] navigate target=%s action=%s path_len=%s",
                target_pos,
                action,
                path_len,
            )
        return action or {"type": "ongoing"}, meta

    def _approach_and_pick(self, plan, agent_state):
        target_pos = plan.target_position
        meta = {"navigation": "pick"}
        is_force_pick = plan.meta.get("override") == "near_grabbable" or plan.meta.get("reason") == "policy_override_grabbable"
        force_pick_distance = plan.meta.get("distance")
        if target_pos is not None and not is_force_pick:
            command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True)
            meta.update(nav_meta)
            if nav_meta.get("forced_failure"):
                if self.logger and getattr(self.logger, "debug", None):
                    self.logger.debug(
                        "[Executor] pick aborted due to guard meta=%s",
                        nav_meta,
                    )
                return command, meta
        elif target_pos is not None and is_force_pick:
            if force_pick_distance is not None and force_pick_distance <= 2.0:
                if self.logger:
                    self.logger.info(
                        "[Executor] force_pick: skipping navigation (dist=%.2f <= 2.0m), attempting direct pick",
                        force_pick_distance,
                    )
                meta["navigation"] = "skip_direct_pick"
                meta["distance"] = force_pick_distance
            else:
                command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True)
                meta.update(nav_meta)
                if nav_meta.get("forced_failure"):
                    if self.logger:
                        self.logger.info(
                            "[Executor] force_pick: guard triggered but proceeding anyway (dist=%.2f)",
                            nav_meta.get("distance", -1),
                        )
                    meta["guard_ignored"] = True
        target_id = plan.target_id
        if target_id is None:
            meta.update({"reason": "missing_target_id"})
            return {"type": "ongoing"}, meta
        holding_slots = agent_state.get("holding_slots", {}) if isinstance(agent_state, dict) else {}
        left_busy = holding_slots.get("left") is not None
        right_busy = holding_slots.get("right") is not None
        if left_busy and right_busy:
            meta.update({"reason": "hands_full"})
            return {"type": "ongoing"}, meta
        arm = "right" if left_busy and not right_busy else "left"
        command = {"type": 3, "object": target_id, "arm": arm}
        meta.update({"result": "attempt_pick", "arm": arm})
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug("[Executor] issuing pick command=%s meta=%s", command, meta)
        return command, meta

    def _guard_threshold(self) -> Optional[float]:
        if self.agent_memory is None or not hasattr(self.agent_memory, "map_size"):
            return None
        map_size = getattr(self.agent_memory, "map_size")
        if not map_size:
            return None
        return float(min(map_size)) * self.cfg.navigation_guard_ratio

    def tick(self):
        removal = []
        for key, value in self._guard_skip.items():
            value["frames"] -= 1
            if value["frames"] <= 0:
                removal.append(key)
        for key in removal:
            self._guard_skip.pop(key, None)

    def guard_summary(self):
        return {key: value["frames"] for key, value in self._guard_skip.items()}

    def _clamp_target(self, target_pos):
        if self.agent_memory is None:
            return target_pos
        x, y, z = target_pos
        clamped_x, clamped_z = x, z
        if hasattr(self.agent_memory, "_scene_bounds"):
            bounds = getattr(self.agent_memory, "_scene_bounds", None)
            if bounds:
                clamped_x = min(max(x, bounds.get("x_min", x)), bounds.get("x_max", x))
                clamped_z = min(max(z, bounds.get("z_min", z)), bounds.get("z_max", z))
        if self.env_api and "check_pos_in_room" in self.env_api:
            if not self.env_api["check_pos_in_room"]((clamped_x, clamped_z)):
                if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
                    agent_pos = self.agent_memory.position
                    clamped_x = agent_pos[0] if abs(clamped_x - agent_pos[0]) > 10 else clamped_x
                    clamped_z = agent_pos[2] if abs(clamped_z - agent_pos[2]) > 10 else clamped_z
                else:
                    clamped_x = 0.0
                    clamped_z = 0.0
        if clamped_x != x or clamped_z != z:
            if self.logger and getattr(self.logger, "debug", None):
                self.logger.debug(
                    "[Executor] clamp target from %s to (%s, %s, %s)",
                    target_pos,
                    clamped_x,
                    y,
                    clamped_z,
                )
        return (clamped_x, y, clamped_z)
