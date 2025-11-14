from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

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
        if action_type == "deliver":
            # Deliver: navigate to target, then put_in
            command, nav_meta = self._navigate_and_deliver(plan, agent_state)
            meta.update(nav_meta)
        elif action_type in {"move", "search", "assist"}:
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
    def _navigate_and_deliver(self, plan, agent_state):
        """Navigate to target container and execute put_in action."""
        target_pos = plan.target_position
        target_id = plan.target_id
        meta = {"navigation": "deliver"}
        
        if target_pos is None:
            if self.logger:
                self.logger.warning("[Executor] deliver: target_pos is None")
            meta.update({"reason": "no_target_position"})
            return {"type": "ongoing"}, meta
        
        # Navigate to target first
        command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True)
        meta.update(nav_meta)
        
        # Calculate actual distance from agent to target AFTER navigation (not path_len)
        actual_distance = float("inf")
        if self.agent_memory is not None and hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos = self.agent_memory.position
                if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                    agent_arr = np.array([agent_pos[0], agent_pos[2]])
                    target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                    actual_distance = float(np.linalg.norm(agent_arr - target_arr))
            except Exception as exc:
                if self.logger:
                    self.logger.debug("[Executor] deliver: failed to calculate actual distance: %s", exc)
        
        # Also check distance from plan.meta if available (this is the distance calculated in policy)
        plan_distance = plan.meta.get("distance")
        if plan_distance is not None and isinstance(plan_distance, (int, float)):
            # Use the smaller of actual_distance and plan_distance
            if actual_distance == float("inf") or plan_distance < actual_distance:
                actual_distance = plan_distance
        
        # If we're close enough (actual distance, not path_len), execute put_in
        if actual_distance is not None and actual_distance <= 3.0:  # Increased threshold to 3.0m
            # Close enough to put_in
            if self.logger:
                self.logger.info(
                    "[Executor] deliver: close enough (actual_dist=%.2f), executing put_in",
                    actual_distance,
                )
            # Execute put_in action (type 4)
            command = {"type": 4}
            meta.update({"result": "put_in", "distance": actual_distance, "actual_distance": actual_distance})
        elif nav_meta.get("forced_failure"):
            # Navigation failed, but still try put_in if we have target_id and we're reasonably close
            if target_id is not None and actual_distance <= 5.0:
                if self.logger:
                    self.logger.info(
                        "[Executor] deliver: navigation failed but close enough (dist=%.2f), attempting put_in",
                        actual_distance,
                    )
                command = {"type": 4}
                meta.update({"result": "put_in_after_nav_failure", "distance": actual_distance})
        
        return command, meta

    # ---------------------------------------------------------------------
    def _navigate_to(self, target_pos, agent_state, follow: bool = False):
        meta: Dict[str, Any] = {"navigation": "idle"}
        if target_pos is None:
            return {"type": "ongoing"}, meta
        if self.agent_memory is None:
            meta["reason"] = "no_agent_memory"
            return {"type": "ongoing"}, meta
        
        # Clamp target to map boundaries (this already handles check_pos_in_room)
        original_target = target_pos
        target_pos = self._clamp_target(target_pos)
        
        # Final validation: if target is still outside room after clamping, reject it
        if self.env_api and "check_pos_in_room" in self.env_api:
            if not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                # Target is still outside room, use agent's current position
                if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
                    try:
                        agent_pos = self.agent_memory.position
                        if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                            target_pos = (float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2]))
                            meta["reason"] = "target_outside_room_using_agent_pos"
                            # Verify agent position is valid
                            if not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                                meta["reason"] = "target_outside_room_agent_pos_invalid"
                                return {"type": "ongoing"}, meta
                        else:
                            meta["reason"] = "target_outside_room_invalid"
                            return {"type": "ongoing"}, meta
                    except Exception:
                        meta["reason"] = "target_outside_room_error"
                        return {"type": "ongoing"}, meta
                else:
                    meta["reason"] = "target_outside_room_no_agent_pos"
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
        is_force_pick = plan.meta.get("override") == "near_grabbable" or plan.meta.get("reason") == "policy_override_grabbable"
        force_pick_distance = plan.meta.get("distance")
        
        # Reject invalid positions (0,0,0 is likely a default/placeholder)
        if target_pos is not None and (target_pos == (0.0, 0.0, 0.0) or (abs(target_pos[0]) < 0.01 and abs(target_pos[2]) < 0.01)):
            if self.logger:
                self.logger.warning(
                    "[Executor] rejecting pick with invalid target_pos (0,0,0): target_id=%s",
                    plan.target_id,
                )
            meta.update({"reason": "invalid_target_position"})
            return {"type": "ongoing"}, meta
        
        # Navigate if target_pos is available
        if target_pos is not None:
            if is_force_pick and force_pick_distance is not None and force_pick_distance <= 2.0:
                if self.logger:
                    self.logger.info(
                        "[Executor] force_pick: skipping navigation (dist=%.2f <= 2.0m), attempting direct pick",
                        force_pick_distance,
                    )
                meta["navigation"] = "skip_direct_pick"
                meta["distance"] = force_pick_distance
            elif not is_force_pick:
                command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True)
                meta.update(nav_meta)
                if nav_meta.get("forced_failure"):
                    if self.logger and getattr(self.logger, "debug", None):
                        self.logger.debug(
                            "[Executor] pick aborted due to guard meta=%s",
                            nav_meta,
                        )
                    return command, meta
            else:  # is_force_pick and distance > 2.0
                command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True)
                meta.update(nav_meta)
                if nav_meta.get("forced_failure"):
                    if self.logger:
                        self.logger.info(
                            "[Executor] force_pick: guard triggered but proceeding anyway (dist=%.2f)",
                            nav_meta.get("distance", -1),
                        )
                    meta["guard_ignored"] = True
        else:
            # target_pos is None: allow pick attempt if distance is close enough
            if is_force_pick and force_pick_distance is not None and force_pick_distance <= 2.5:
                if self.logger:
                    self.logger.info(
                        "[Executor] force_pick: no position but distance=%.2f <= 2.5m, attempting direct pick",
                        force_pick_distance,
                    )
                meta["navigation"] = "skip_no_position"
                meta["distance"] = force_pick_distance
                meta["note"] = "pick_without_position"
            else:
                # Position is None and distance is too far or unknown
                if self.logger:
                    self.logger.debug(
                        "[Executor] pick: target_pos is None and distance=%.2f, skipping pick",
                        force_pick_distance if force_pick_distance is not None else -1.0,
                    )
                meta.update({"reason": "no_position_and_too_far"})
                return {"type": "ongoing"}, meta
        
        # Now attempt pick
        target_id = plan.target_id
        if target_id is None:
            meta.update({"reason": "missing_target_id"})
            return {"type": "ongoing"}, meta
        
        # Convert to int and check if it's valid (0 is not a valid object_id)
        try:
            target_id_int = int(target_id)
            if target_id_int == 0:
                if self.logger:
                    self.logger.warning(
                        "[Executor] target_id=%s is 0 (invalid), skipping pick (may cause KeyError)",
                        target_id,
                    )
                meta.update({"reason": "invalid_target_id_zero"})
                return {"type": "ongoing"}, meta
        except (ValueError, TypeError) as exc:
            if self.logger:
                self.logger.warning(
                    "[Executor] target_id=%s cannot be converted to int: %s",
                    target_id,
                    exc,
                )
            meta.update({"reason": "invalid_target_id_type"})
            return {"type": "ongoing"}, meta
        
        # Check if object_id exists in agent_memory.object_info (to prevent KeyError in tdw_gym)
        # This is a best-effort check since we can't directly access object_manager.transforms
        if self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
            try:
                if target_id_int not in self.agent_memory.object_info:
                    if self.logger:
                        self.logger.warning(
                            "[Executor] object_id=%s not in agent_memory.object_info, skipping pick (may cause KeyError)",
                            target_id_int,
                        )
                    meta.update({"reason": "object_not_in_memory"})
                    return {"type": "ongoing"}, meta
            except Exception as exc:
                if self.logger:
                    self.logger.debug(
                        "[Executor] failed to check agent_memory.object_info for id=%s: %s",
                        target_id_int,
                        exc,
                    )
        
        holding_slots = agent_state.get("holding_slots", {}) if isinstance(agent_state, dict) else {}
        left_busy = holding_slots.get("left") is not None
        right_busy = holding_slots.get("right") is not None
        if left_busy and right_busy:
            meta.update({"reason": "hands_full"})
            return {"type": "ongoing"}, meta
        arm = "right" if left_busy and not right_busy else "left"
        command = {"type": 3, "object": target_id_int, "arm": arm}
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
        
        # Get agent's current position for fallback
        agent_pos = None
        agent_pos_full = None
        if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos_full = self.agent_memory.position
                if isinstance(agent_pos_full, (list, tuple, np.ndarray)) and len(agent_pos_full) >= 3:
                    agent_pos = (float(agent_pos_full[0]), float(agent_pos_full[2]))
                else:
                    agent_pos_full = None
            except Exception:
                agent_pos_full = None
        
        # First, try to clamp using scene_bounds with safety margin
        clamped_x, clamped_z = x, z
        if hasattr(self.agent_memory, "_scene_bounds"):
            bounds = getattr(self.agent_memory, "_scene_bounds", None)
            if bounds:
                x_min = bounds.get("x_min")
                x_max = bounds.get("x_max")
                z_min = bounds.get("z_min")
                z_max = bounds.get("z_max")
                # Add safety margin (0.5m) to prevent going outside
                margin = 0.5
                if x_min is not None and x_max is not None:
                    clamped_x = min(max(x, x_min + margin), x_max - margin)
                if z_min is not None and z_max is not None:
                    clamped_z = min(max(z, z_min + margin), z_max - margin)
        
        # Then, check if clamped position is in room
        if self.env_api and "check_pos_in_room" in self.env_api:
            if not self.env_api["check_pos_in_room"]((clamped_x, clamped_z)):
                # If not in room, use agent's current position
                if agent_pos is not None:
                    clamped_x, clamped_z = agent_pos
                elif agent_pos_full is not None:
                    try:
                        if isinstance(agent_pos_full, (list, tuple, np.ndarray)) and len(agent_pos_full) >= 3:
                            clamped_x = float(agent_pos_full[0])
                            clamped_z = float(agent_pos_full[2])
                        else:
                            clamped_x = 0.0
                            clamped_z = 0.0
                    except Exception:
                        clamped_x = 0.0
                        clamped_z = 0.0
                else:
                    clamped_x = 0.0
                    clamped_z = 0.0
                # Verify the fallback position is valid
                if self.env_api and "check_pos_in_room" in self.env_api:
                    if not self.env_api["check_pos_in_room"]((clamped_x, clamped_z)):
                        # Last resort: use (0, 0) if agent position is also invalid
                        clamped_x = 0.0
                        clamped_z = 0.0
                        if self.logger:
                            self.logger.warning(
                                "[Executor] target %s outside room, resetting to (0, 0)",
                                target_pos,
                            )
        
        # Final safety check: ensure clamped position is within scene_bounds
        if hasattr(self.agent_memory, "_scene_bounds"):
            bounds = getattr(self.agent_memory, "_scene_bounds", None)
            if bounds:
                x_min = bounds.get("x_min")
                x_max = bounds.get("x_max")
                z_min = bounds.get("z_min")
                z_max = bounds.get("z_max")
                if x_min is not None and x_max is not None:
                    clamped_x = min(max(clamped_x, x_min), x_max)
                if z_min is not None and z_max is not None:
                    clamped_z = min(max(clamped_z, z_min), z_max)
        
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
