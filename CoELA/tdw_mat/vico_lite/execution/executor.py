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
        if target_pos is not None:
            command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True)
            meta.update(nav_meta)
            if nav_meta.get("forced_failure"):
                if self.logger and getattr(self.logger, "debug", None):
                    self.logger.debug(
                        "[Executor] pick aborted due to guard meta=%s",
                        nav_meta,
                    )
                return command, meta
        command = {"type": 6, "id": plan.target_id}
        meta["result"] = "attempt_pick"
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug("[Executor] issuing pick command=%s meta=%s", command, meta)
        return command, meta

    def _guard_threshold(self) -> Optional[float]:
        if self.agent_memory is None or not hasattr(self.agent_memory, "map_size"):
            return None
        map_size = getattr(self.agent_memory, "map_size")
        if not map_size:
            return None
        return float(max(map_size)) * self.cfg.navigation_guard_ratio

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
