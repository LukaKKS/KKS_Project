from __future__ import annotations

import logging
import math
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
    def _navigate_to(self, target_pos, agent_state, follow: bool = False, skip_clamp: bool = False):
        meta: Dict[str, Any] = {"navigation": "idle"}
        if target_pos is None:
            return {"type": "ongoing"}, meta
        if self.agent_memory is None:
            meta["reason"] = "no_agent_memory"
            return {"type": "ongoing"}, meta
        
        # Clamp target to map boundaries (this already handles check_pos_in_room)
        # BUT: Skip clamping for pick actions - they need the actual object position
        original_target = target_pos
        if not skip_clamp:
            target_pos = self._clamp_target(target_pos)
        
        # Final validation: if target is still outside room after clamping, reject it
        # BUT: Skip validation for pick actions (skip_clamp=True) - they need actual object position
        if not skip_clamp and self.env_api and "check_pos_in_room" in self.env_api:
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
        
        # ViCo-Lite 개선: Continuous movement for long distances
        # Calculate distance to target
        distance_to_target = float("inf")
        if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos = self.agent_memory.position
                if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                    agent_arr = np.array([agent_pos[0], agent_pos[2]])
                    target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                    distance_to_target = float(np.linalg.norm(agent_arr - target_arr))
            except Exception:
                pass
        
        # Use continuous movement for long distances if enabled
        if self.cfg.use_continuous_navigation and distance_to_target > self.cfg.continuous_navigation_threshold:
            # Use continuous movement (type 9: move_to_position)
            if self.logger:
                self.logger.debug(
                    "[Executor] using continuous navigation for long distance: %.2fm > %.2fm, target=%s",
                    distance_to_target,
                    self.cfg.continuous_navigation_threshold,
                    target_pos,
                )
            # Return continuous movement action (type 9)
            action = {
                "type": 9,  # New action type for continuous movement
                "target_position": target_pos,
            }
            meta.update({
                "navigation": "continuous_move_to_position",
                "distance": distance_to_target,
                "method": "continuous",
            })
            return action, meta
        else:
            # Use discrete step-by-step movement (COELA 방식)
            action, path_len = self.agent_memory.move_to_pos(target_pos, follow=follow)
            meta.update({
                "navigation": "move_to_pos",
                "distance": path_len,
                "method": "discrete",
            })
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
        
        # 핵심 수정: object_id 체크 전에 navigation 먼저 시작
        # 10m 제한으로 인해 object_info에 없어도 navigation은 시도해야 함
        navigation_started = False
        navigation_command = None
        
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
                navigation_command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True)
                meta.update(nav_meta)
                navigation_started = True
                if nav_meta.get("forced_failure"):
                    if self.logger and getattr(self.logger, "debug", None):
                        self.logger.debug(
                            "[Executor] pick aborted due to guard meta=%s",
                            nav_meta,
                        )
                    return navigation_command, meta
            else:  # is_force_pick and distance > 2.0
                navigation_command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True)
                meta.update(nav_meta)
                navigation_started = True
                if nav_meta.get("forced_failure"):
                    if self.logger:
                        self.logger.info(
                            "[Executor] force_pick: guard triggered but proceeding anyway (dist=%.2f)",
                            nav_meta.get("distance", -1),
                        )
                    meta["guard_ignored"] = True
        else:
            # 핵심 수정: target_pos is None이지만 distance가 있으면 navigation 시도 (task target인 경우)
            # 10m 제한으로 인해 위치 계산이 실패했지만 depth로 거리는 알 수 있는 경우
            is_task_target = plan.meta.get("is_task_target", False)
            if is_force_pick and force_pick_distance is not None:
                if force_pick_distance <= 2.5:
                    # Close enough for direct pick
                    if self.logger:
                        self.logger.info(
                            "[Executor] force_pick: no position but distance=%.2f <= 2.5m, attempting direct pick",
                            force_pick_distance,
                        )
                    meta["navigation"] = "skip_no_position"
                    meta["distance"] = force_pick_distance
                    meta["note"] = "pick_without_position"
                elif is_task_target:
                    # Task target but too far - try to navigate using object_id
                    # TDW's move_to_position can work with object_id directly
                    if self.logger:
                        self.logger.info(
                            "[Executor] pick: no position but task target distance=%.2f, will attempt pick with object_id only",
                            force_pick_distance,
                        )
                    meta["navigation"] = "pick_with_object_id"
                    meta["distance"] = force_pick_distance
                    meta["note"] = "no_position_using_object_id"
                else:
                    # Not task target and too far
                    if self.logger:
                        self.logger.debug(
                            "[Executor] pick: target_pos is None and distance=%.2f > 2.5m (not task target), skipping pick",
                            force_pick_distance,
                        )
                    meta.update({"reason": "no_position_and_too_far"})
                    return {"type": "ongoing"}, meta
            else:
                # Position is None and distance is unknown
                if self.logger:
                    self.logger.debug(
                        "[Executor] pick: target_pos is None and distance unknown, skipping pick",
                    )
                meta.update({"reason": "no_position_and_no_distance"})
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
        
        # 핵심 수정: object_id 체크를 navigation 이후로 이동
        # Navigation이 이미 시작되었으면 navigation을 계속 진행하고, pick은 나중에 시도
        # 10m 제한으로 인해 object_info에 없어도 navigation은 허용해야 함
        object_in_memory = True
        if self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
            try:
                if target_id_int not in self.agent_memory.object_info:
                    object_in_memory = False
                    is_task_target = plan.meta.get("is_task_target", False)
                    # Navigation이 이미 시작되었으면 navigation을 계속 진행
                    if navigation_started and navigation_command is not None:
                        if self.logger:
                            self.logger.info(
                                "[Executor] object_id=%s not in agent_memory.object_info but navigation already started, continuing navigation (task_target=%s)",
                                target_id_int,
                                is_task_target,
                            )
                        meta.update({"reason": "object_not_in_memory_but_navigating", "will_retry_pick_after_nav": True})
                        return navigation_command, meta
                    # Task target이고 target_pos가 있으면 navigation 허용
                    elif is_task_target and target_pos is not None:
                        if self.logger:
                            self.logger.info(
                                "[Executor] object_id=%s not in agent_memory.object_info but is task target, allowing navigation",
                                target_id_int,
                            )
                        # Navigation은 이미 시작되었거나 시작할 예정이므로 계속 진행
                        if navigation_command is not None:
                            meta.update({"reason": "object_not_in_memory_but_navigating", "will_retry_pick_after_nav": True})
                            return navigation_command, meta
                        # Navigation이 아직 시작되지 않았으면 시작
                        else:
                            navigation_command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True)
                            meta.update(nav_meta)
                            meta.update({"reason": "object_not_in_memory_but_navigating", "will_retry_pick_after_nav": True})
                            return navigation_command, meta
                    else:
                        # Navigation도 없고 task target도 아니면 스킵
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
        
        # Check actual distance before issuing pick command
        # TDW reach_threshold is 2m, so we need to be within 2m to successfully pick
        actual_distance = float("inf")
        if self.agent_memory is not None and hasattr(self.agent_memory, "position") and self.agent_memory.position is not None and target_pos is not None:
            try:
                agent_pos = self.agent_memory.position
                if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                    agent_arr = np.array([agent_pos[0], agent_pos[2]])
                    target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                    actual_distance = float(np.linalg.norm(agent_arr - target_arr))
            except Exception as exc:
                if self.logger:
                    self.logger.debug("[Executor] pick: failed to calculate actual distance: %s", exc)
        
        # Use plan.meta distance if actual_distance calculation failed
        if actual_distance == float("inf") and plan.meta.get("distance") is not None:
            actual_distance = float(plan.meta.get("distance"))
        
        # 방안 4: Pick 거리 임계값 조정 - 2.0m에서 2.5m로 완화
        reach_threshold = 2.5  # 2.0에서 2.5로 증가
        if actual_distance > reach_threshold:
            if self.logger:
                self.logger.debug(
                    "[Executor] pick: too far from target (dist=%.2f > %.2f), continuing navigation instead of pick",
                    actual_distance,
                    reach_threshold,
                )
            # Continue navigation instead of attempting pick
            if target_pos is not None:
                command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True)
                meta.update(nav_meta)
                meta.update({"reason": "too_far_for_pick", "distance": actual_distance, "threshold": reach_threshold})
                return command, meta
            else:
                meta.update({"reason": "too_far_for_pick_no_position", "distance": actual_distance, "threshold": reach_threshold})
                return {"type": "ongoing"}, meta
        
        holding_slots = agent_state.get("holding_slots", {}) if isinstance(agent_state, dict) else {}
        left_busy = holding_slots.get("left") is not None
        right_busy = holding_slots.get("right") is not None
        if left_busy and right_busy:
            meta.update({"reason": "hands_full"})
            return {"type": "ongoing"}, meta
        arm = "right" if left_busy and not right_busy else "left"
        command = {"type": 3, "object": target_id_int, "arm": arm}
        meta.update({"result": "attempt_pick", "arm": arm, "distance": actual_distance})
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug("[Executor] issuing pick command=%s meta=%s (dist=%.2f <= %.2f)", command, meta, actual_distance, reach_threshold)
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
                        # Try to find a valid position: first try scene_bounds center, then agent position
                        valid_pos = None
                        # Try scene_bounds center
                        if hasattr(self.agent_memory, "_scene_bounds"):
                            bounds = getattr(self.agent_memory, "_scene_bounds", None)
                            if bounds:
                                x_min = bounds.get("x_min")
                                x_max = bounds.get("x_max")
                                z_min = bounds.get("z_min")
                                z_max = bounds.get("z_max")
                                if x_min is not None and x_max is not None and z_min is not None and z_max is not None:
                                    center_x = (x_min + x_max) / 2.0
                                    center_z = (z_min + z_max) / 2.0
                                    if self.env_api["check_pos_in_room"]((center_x, center_z)):
                                        valid_pos = (center_x, center_z)
                        # If scene_bounds center is not valid, try agent's current room center
                        if valid_pos is None and agent_pos is not None:
                            if hasattr(self.agent_memory, "belongs_to_which_room") and hasattr(self.agent_memory, "center_of_room"):
                                try:
                                    agent_room = self.agent_memory.belongs_to_which_room(agent_pos_full if agent_pos_full is not None else (agent_pos[0], 0, agent_pos[1]))
                                    if agent_room is not None:
                                        room_center = self.agent_memory.center_of_room(agent_room)
                                        if room_center is not None and self.env_api["check_pos_in_room"]((room_center[0], room_center[2])):
                                            valid_pos = (room_center[0], room_center[2])
                                except Exception:
                                    pass
                        # If still no valid position, try to find a valid position near agent's current position
                        if valid_pos is None and agent_pos is not None:
                            # Try positions around agent's current position (spiral search)
                            search_radius = 1.0
                            max_radius = 5.0
                            found_valid = False
                            while search_radius <= max_radius and not found_valid:
                                # Try 8 directions around agent position
                                for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                                    rad = math.radians(angle)
                                    test_x = agent_pos[0] + search_radius * math.cos(rad)
                                    test_z = agent_pos[1] + search_radius * math.sin(rad)
                                    # Check map bounds first
                                    if hasattr(self.agent_memory, "_scene_bounds"):
                                        bounds = getattr(self.agent_memory, "_scene_bounds", None)
                                        if bounds:
                                            x_min = bounds.get("x_min")
                                            x_max = bounds.get("x_max")
                                            z_min = bounds.get("z_min")
                                            z_max = bounds.get("z_max")
                                            if (x_min is not None and x_max is not None and (test_x < x_min or test_x > x_max)) or \
                                               (z_min is not None and z_max is not None and (test_z < z_min or test_z > z_max)):
                                                continue
                                    # Check if in room
                                    if self.env_api and "check_pos_in_room" in self.env_api:
                                        try:
                                            if self.env_api["check_pos_in_room"]((test_x, test_z)):
                                                valid_pos = (test_x, test_z)
                                                found_valid = True
                                                break
                                        except Exception:
                                            pass
                                if found_valid:
                                    break
                                search_radius += 1.0
                        
                        if valid_pos is not None:
                            clamped_x, clamped_z = valid_pos
                            if self.logger:
                                self.logger.debug(
                                    "[Executor] target %s outside room, using valid fallback position (%s, %s)",
                                    target_pos,
                                    clamped_x,
                                    clamped_z,
                                )
                        else:
                            # Last resort: use agent's current position if available, otherwise (0, 0)
                            if agent_pos is not None:
                                clamped_x, clamped_z = agent_pos
                                if self.logger:
                                    self.logger.warning(
                                        "[Executor] target %s outside room, using agent's current position as fallback (%s, %s)",
                                        target_pos,
                                        clamped_x,
                                        clamped_z,
                                    )
                            else:
                                clamped_x = 0.0
                                clamped_z = 0.0
                                if self.logger:
                                    self.logger.warning(
                                        "[Executor] target %s outside room, resetting to (0, 0) (no agent position available)",
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
        
        # Final check: if clamped position is (0,0) or still outside room, try to find a valid position
        if self.env_api and "check_pos_in_room" in self.env_api:
            if not self.env_api["check_pos_in_room"]((clamped_x, clamped_z)):
                # Try to find a valid position: use scene_bounds center or agent's current room
                final_valid_pos = None
                # Try scene_bounds center first
                if hasattr(self.agent_memory, "_scene_bounds"):
                    bounds = getattr(self.agent_memory, "_scene_bounds", None)
                    if bounds:
                        x_min = bounds.get("x_min")
                        x_max = bounds.get("x_max")
                        z_min = bounds.get("z_min")
                        z_max = bounds.get("z_max")
                        if x_min is not None and x_max is not None and z_min is not None and z_max is not None:
                            center_x = (x_min + x_max) / 2.0
                            center_z = (z_min + z_max) / 2.0
                            if self.env_api["check_pos_in_room"]((center_x, center_z)):
                                final_valid_pos = (center_x, center_z)
                # If scene_bounds center is not valid, try agent's current room center
                if final_valid_pos is None and agent_pos is not None:
                    if hasattr(self.agent_memory, "belongs_to_which_room") and hasattr(self.agent_memory, "center_of_room"):
                        try:
                            agent_room = self.agent_memory.belongs_to_which_room(agent_pos_full if agent_pos_full is not None else (agent_pos[0], 0, agent_pos[1]))
                            if agent_room is not None:
                                room_center = self.agent_memory.center_of_room(agent_room)
                                if room_center is not None and self.env_api["check_pos_in_room"]((room_center[0], room_center[2])):
                                    final_valid_pos = (room_center[0], room_center[2])
                        except Exception:
                            pass
                # If we found a valid position, use it
                if final_valid_pos is not None:
                    clamped_x, clamped_z = final_valid_pos
                    if self.logger:
                        self.logger.debug(
                            "[Executor] final clamp: (%s, %s) outside room, using valid position (%s, %s)",
                            clamped_x,
                            clamped_z,
                            final_valid_pos[0],
                            final_valid_pos[1],
                        )
                elif (clamped_x == 0.0 and clamped_z == 0.0):
                    # (0,0) is invalid, but we have no better option - log warning
                    if self.logger:
                        self.logger.warning(
                            "[Executor] clamped position (0, 0) is outside room and no valid fallback found for target %s",
                            target_pos,
                        )
        
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
