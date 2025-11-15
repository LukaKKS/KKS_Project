from __future__ import annotations

import json
import math
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
            "holding_ids_preserve_frame": -1,  # Frame when holding_ids was last preserved
            "pending_pick_verification": {},  # {pick_id: frame} - pick success 후 held_objects 확인 대기
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
        # Store current_frame in agent_state for use by _maybe_force_deliver
        self.agent_state["current_frame"] = frame
        # Reset logged failed objects cache for this frame
        self._logged_cal_failed_objects = set()
        prev_status = obs.get("status")
        last_command = self.agent_state.get("last_command")
        if isinstance(prev_status, int) and last_command == "pick":
            pick_id = self.agent_state.get("last_pick_id")
            if prev_status == 1:
                # CRITICAL: Check if we actually have the object in held_objects
                # status == 1 might mean reach_for succeeded, but grasp might have failed
                held_objects = obs.get("held_objects", []) or []
                actual_held_ids = []
                for entry in held_objects:
                    if entry and entry.get("id") is not None:
                        actual_held_ids.append(entry.get("id"))
                
                # Only consider it a success if the object is actually in held_objects
                # OR if held_objects is empty but we just issued the pick (might be delayed update)
                is_actually_held = pick_id is not None and pick_id in actual_held_ids
                held_objects_empty = not actual_held_ids
                
                if is_actually_held:
                    # Confirmed: object is actually in held_objects
                    if self.logger:
                        self.logger.info("[Policy] pick success CONFIRMED id=%s frame=%s (in held_objects)", pick_id, frame)
                    # Add successfully picked object to holding_ids immediately
                    if pick_id is not None:
                        current_holding_ids = self.agent_state.get("holding_ids", [])
                        if pick_id not in current_holding_ids:
                            current_holding_ids.append(pick_id)
                            self.agent_state["holding_ids"] = current_holding_ids
                            # Reset preserve frame since we just confirmed a pick
                            self.agent_state["holding_ids_preserve_frame"] = frame
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
                    # Clear last_command since pick is confirmed successful
                    self.agent_state.pop("last_command", None)
                    self.agent_state.pop("last_pick_id", None)
                    self.agent_state.pop("last_pick_pos", None)
                elif held_objects_empty:
                    # held_objects is empty - status==1 might mean reach_for succeeded but grasp failed
                    # Don't treat as success yet, wait for verification
                    if self.logger:
                        self.logger.warning(
                            "[Policy] pick status=1 but held_objects empty id=%s frame=%s (reach_for may have succeeded but grasp failed - waiting for verification)",
                            pick_id,
                            frame,
                        )
                    # Mark for verification but DON'T add to holding_ids yet
                    # If verification succeeds (object appears in held_objects within 5 frames), then add to holding_ids
                    if pick_id is not None:
                        pending_verification = self.agent_state.get("pending_pick_verification", {})
                        pending_verification[pick_id] = frame
                        self.agent_state["pending_pick_verification"] = pending_verification
                        if self.logger:
                            self.logger.info(
                                "[Policy] pick pending verification: id=%s (will check held_objects in next frames)",
                                pick_id,
                            )
                    # CRITICAL: Clear last_command to prevent re-processing the same pick in next frame
                    # The pick action is done (status=1), we're just waiting for verification
                    # If we don't clear last_command, the same pick will be processed again in next frame
                    self.agent_state.pop("last_command", None)
                    # Keep last_pick_id for verification, but clear it after verification completes
                else:
                    # held_objects has other objects but not the one we tried to pick - definitely failed
                    if self.logger:
                        self.logger.warning(
                            "[Policy] pick status=1 but object NOT in held_objects id=%s frame=%s (held_objects=%s) - treating as FAILURE",
                            pick_id,
                            frame,
                            actual_held_ids,
                        )
                    # Treat as failure
                    if pick_id is not None:
                        current_holding_ids = self.agent_state.get("holding_ids", [])
                        if pick_id in current_holding_ids:
                            current_holding_ids.remove(pick_id)
                            self.agent_state["holding_ids"] = current_holding_ids
                        self.agent_state["holding_ids_preserve_frame"] = -1
                        last_pick_pos = self.agent_state.get("last_pick_pos")
                        if last_pick_pos is not None:
                            self._register_skip_target(position=last_pick_pos)
                    self.agent_state.pop("last_command", None)
                    self.agent_state.pop("last_pick_id", None)
                    self.agent_state.pop("last_pick_pos", None)
                    # Continue to next part of act() - don't process as success
                    # (fall through to _update_agent_state)
            elif prev_status == 2:
                if self.logger:
                    self.logger.warning("[Policy] pick failed id=%s frame=%s", pick_id, frame)
                # Remove failed pick from holding_ids if it was added
                if pick_id is not None:
                    current_holding_ids = self.agent_state.get("holding_ids", [])
                    if pick_id in current_holding_ids:
                        current_holding_ids.remove(pick_id)
                        self.agent_state["holding_ids"] = current_holding_ids
                        self.agent_state["holding_ids_preserve_frame"] = -1
                        if self.logger:
                            self.logger.info(
                                "[Policy] pick failed: removed from holding_ids id=%s holding_ids=%s",
                                pick_id,
                                current_holding_ids,
                            )
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
        # Check if navigation completed and we should retry pick
        # If last_command was navigation (move/search) and status indicates completion, retry pick if we were trying to pick
        # 방안 1: Navigation 완료 후 Pick Retry 강화
        # Navigation이 완료되면 (prev_status가 1, success) 자동으로 pick retry
        # type: 9 (continuous movement)도 navigation으로 간주
        is_navigation_command = last_command in ("0", "1", "2", "move", "search", "continuous_move", 9, "9")
        if is_navigation_command and isinstance(prev_status, int):
            last_pick_id = self.agent_state.get("last_pick_id")
            last_pick_pos = self.agent_state.get("last_pick_pos")
            # Navigation 완료 감지: prev_status가 1 (success)이면 완료로 간주
            # prev_status가 0 (ongoing)이면 아직 진행 중이므로 retry하지 않음
            # type: 9의 경우 ActionStatus.success가 1로 매핑됨
            if self.logger and last_pick_id is not None:
                self.logger.debug(
                    "[Policy] navigation retry check: last_command=%s prev_status=%s (type=%s) last_pick_id=%s frame=%s",
                    last_command,
                    prev_status,
                    type(prev_status).__name__,
                    last_pick_id,
                    frame,
                )
            # Navigation 완료 또는 실패 시 거리 확인 후 pick retry
            # prev_status == 1: success (완료)
            # prev_status == 2: failure (실패) - last_command를 clear하고 새로운 navigation 시도 가능하도록
            # prev_status == 0: ongoing (진행 중) - 너무 오래 지속되면 (20프레임 이상) 실패로 간주
            should_check_distance = False
            should_clear_navigation = False
            last_nav_frame = self.agent_state.get("last_nav_start_frame", -1)
            # last_nav_start_frame이 -1이면 현재 프레임으로 설정 (타임아웃 체크를 위해)
            if last_nav_frame < 0 and prev_status == 0:
                last_nav_frame = frame
                self.agent_state["last_nav_start_frame"] = frame
                if self.logger:
                    self.logger.debug(
                        "[Policy] last_nav_start_frame was -1, setting to current frame=%s for timeout tracking",
                        frame,
                    )
            
            if last_pick_id is not None:
                if prev_status == 1:  # Success - 완료
                    should_check_distance = True
                elif prev_status == 2:  # Failure - 실패했지만 거리 확인 후 clear
                    should_check_distance = True
                    should_clear_navigation = True  # 실패했으므로 navigation clear
                    if self.logger:
                        self.logger.debug(
                            "[Policy] navigation failed (prev_status=2), checking distance for pick retry id=%s, will clear navigation",
                            last_pick_id,
                        )
                elif prev_status == 0:  # Ongoing - 너무 오래 지속되면 실패로 간주
                    # Continuous movement가 너무 오래 지속되면 (20프레임 이상) 실패로 간주
                    if last_nav_frame >= 0 and frame - last_nav_frame >= 20:
                        should_check_distance = True
                        should_clear_navigation = True  # 너무 오래 지속되면 실패로 간주
                        if self.logger:
                            self.logger.warning(
                                "[Policy] navigation ongoing for %d frames (>=20), treating as failure and clearing navigation id=%s",
                                frame - last_nav_frame,
                                last_pick_id,
                            )
            elif prev_status == 0 and last_nav_frame >= 0:
                # last_pick_id가 없어도 ongoing 상태가 너무 오래 지속되면 clear (일반 navigation 타임아웃)
                if frame - last_nav_frame >= 30:  # pick이 아닌 일반 navigation은 30프레임 타임아웃
                    should_clear_navigation = True
                    if self.logger:
                        self.logger.warning(
                            "[Policy] navigation ongoing for %d frames (>=30) without pick target, clearing navigation state",
                            frame - last_nav_frame,
                        )
            
            if should_check_distance:
                target_pos = None
                distance = None
                obj_name = None
                
                # First try to find object in visible_infos (가장 정확)
                if visible_infos:
                    for info in visible_infos:
                        if info.get("id") == last_pick_id and info.get("is_grabbable"):
                            distance = info.get("distance")
                            # 거리 임계값을 2.5m로 완화
                            if distance is not None and distance <= 2.5:
                                target_pos = self._normalise_position(info.get("position"))
                                if target_pos is None:
                                    target_pos = self._normalise_position(info.get("location"))
                                obj_name = info.get("name")
                                break
                
                # If not found in visible_infos, use stored last_pick_pos
                if target_pos is None and last_pick_pos is not None:
                    if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
                        try:
                            agent_pos = np.asarray(self.agent_memory.position, dtype=np.float32)
                            pick_pos_arr = np.asarray(last_pick_pos, dtype=np.float32)
                            if agent_pos.shape[0] >= 3 and pick_pos_arr.shape[0] >= 3:
                                distance = float(np.linalg.norm(agent_pos[[0, 2]] - pick_pos_arr[[0, 2]]))
                            else:
                                distance = float(np.linalg.norm(agent_pos - pick_pos_arr))
                            
                            # 거리 임계값을 2.5m로 완화
                            if distance is not None and distance <= 2.5:
                                target_pos = last_pick_pos
                                if self.logger:
                                    self.logger.info(
                                        "[Policy] navigation completed/failed, retrying pick id=%s using stored position dist=%.2f",
                                        last_pick_id,
                                        distance,
                                    )
                        except Exception as exc:
                            if self.logger:
                                self.logger.debug(
                                    "[Policy] failed to calculate distance for retry pick: %s",
                                    exc,
                                )
                
                # Retry pick if we have valid target and distance
                if target_pos is not None and distance is not None and distance <= 2.5:
                    if self.logger:
                        self.logger.info(
                            "[Policy] navigation completed/failed, retrying pick id=%s name=%s dist=%.2f (prev_status=%s)",
                            last_pick_id,
                            obj_name or f"object_{last_pick_id}",
                            distance,
                            prev_status,
                        )
                    pick_plan = ReasonedPlan(
                        action_type="pick",
                        target_id=last_pick_id,
                        target_position=target_pos,
                        confidence=0.9,
                        meta={
                            "reason": "retry_after_navigation",
                            "target_name": obj_name or f"object_{last_pick_id}",
                            "distance": distance,
                            "prev_status": prev_status,
                        },
                    )
                    command, exec_meta = self.executor.execute(pick_plan, snapshot, self.agent_state)
                    self.agent_state.setdefault("recent_meta", []).append(exec_meta)
                    command_type = command.get("type") if isinstance(command, dict) else None
                    if command_type == 3:
                        self.agent_state["last_command"] = "pick"
                        self.agent_state["last_pick_id"] = last_pick_id
                        self.agent_state["last_pick_pos"] = target_pos
                    # Clear navigation tracking
                    self.agent_state.pop("last_nav_start_frame", None)
                    return command
                elif distance is not None and distance > 2.5:
                    if self.logger:
                        self.logger.debug(
                            "[Policy] navigation completed/failed but still too far (dist=%.2f > 2.5m), not retrying pick id=%s",
                            distance,
                            last_pick_id,
                        )
                    # 거리가 너무 멀면 navigation clear (새로운 navigation 시도 가능하도록)
                    if should_clear_navigation:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] clearing navigation state due to failure or timeout (dist=%.2f > 2.5m)",
                                distance,
                            )
                        self.agent_state.pop("last_command", None)
                        self.agent_state.pop("last_pick_id", None)
                        self.agent_state.pop("last_pick_pos", None)
                        self.agent_state.pop("last_nav_start_frame", None)
            # Navigation이 실패했거나 타임아웃되었으면 clear
            if should_clear_navigation and not should_check_distance:
                if self.logger:
                    self.logger.debug(
                        "[Policy] clearing navigation state due to failure or timeout (no pick retry)",
                    )
                self.agent_state.pop("last_command", None)
                self.agent_state.pop("last_pick_id", None)
                self.agent_state.pop("last_pick_pos", None)
                self.agent_state.pop("last_nav_start_frame", None)
        # 방안 5: 에이전트가 아무것도 보지 못할 때 처리
        # 연속으로 visible_infos가 비어있으면 위치 변경 필요
        if not visible_infos or len(visible_infos) == 0:
            no_vision_count = self.agent_state.get("no_vision_count", 0) + 1
            self.agent_state["no_vision_count"] = no_vision_count
            # 5회 이상 연속으로 아무것도 보지 못하면 explore로 위치 변경
            if no_vision_count >= 5:
                if self.logger:
                    self.logger.warning(
                        "[Policy] frame=%s no vision for %d consecutive frames, forcing explore",
                        frame,
                        no_vision_count,
                    )
                # Explore로 위치 변경 - 더 적극적으로 다른 위치로 이동
                # blocked_coords를 추가하여 같은 위치로 이동하지 않도록 함
                blocked_coords = self.agent_state.get("blocked_coords", set())
                current_pos = snapshot.get("agent_position")
                if current_pos:
                    blocked_coords.add(tuple(current_pos[:2]))  # (x, z) 좌표만 사용
                    self.agent_state["blocked_coords"] = blocked_coords
                
                explore_plan = ReasonedPlan("search", None, None, 0.5, {"reason": "no_vision_for_too_long", "fallback": "persist"})
                # _ensure_plan_target은 내부에서 blocked_coords를 계산함
                explore_plan = self._ensure_plan_target(explore_plan)
                
                # If still idle, try to move to a random position far from current position
                if explore_plan.action_type == "idle" and current_pos and no_vision_count >= 10:
                    # Move to a position far from current position (e.g., opposite side of room)
                    import random
                    if hasattr(self, 'env_api') and self.env_api:
                        room_bounds = self.env_api.get("room_bounds")
                        if room_bounds:
                            # Move to a random position in the room, far from current position
                            target_x = random.uniform(room_bounds[0][0], room_bounds[1][0])
                            target_z = random.uniform(room_bounds[0][2], room_bounds[1][2])
                            target_pos = (target_x, current_pos[1], target_z)
                            explore_plan = ReasonedPlan("move", None, target_pos, 0.5, {"reason": "no_vision_for_too_long", "fallback": "random_position"})
                            if self.logger:
                                self.logger.info(
                                    "[Policy] frame=%s no vision for %d frames, moving to random position %s",
                                    frame,
                                    no_vision_count,
                                    target_pos,
                                )
                
                if explore_plan.action_type != "idle":
                    command, exec_meta = self.executor.execute(explore_plan, snapshot, self.agent_state)
                    self.agent_state.setdefault("recent_meta", []).append(exec_meta)
                    # Reset no_vision_count after executing explore action
                    self.agent_state["no_vision_count"] = 0
                    return command
        else:
            # Vision이 있으면 카운터 리셋
            self.agent_state["no_vision_count"] = 0
            # Vision이 있으면 blocked_coords도 일부 클리어 (최근 것만 유지)
            blocked_coords = self.agent_state.get("blocked_coords", set())
            if len(blocked_coords) > 10:
                # Keep only the 5 most recent blocked coordinates
                blocked_coords = set(list(blocked_coords)[-5:])
                self.agent_state["blocked_coords"] = blocked_coords
        
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
                # Check actual held_objects before forcing deliver
                held_objects = obs.get("held_objects", []) or []
                actual_holding_ids = []
                for entry in held_objects:
                    if entry and entry.get("id") is not None:
                        actual_holding_ids.append(entry.get("id"))
                # Only force deliver if we actually have objects OR holding_ids is recent (within 5 frames)
                preserve_frame = self.agent_state.get("holding_ids_preserve_frame", -1)
                current_frame = obs.get("current_frames", 0)
                can_deliver = actual_holding_ids or (preserve_frame >= 0 and (current_frame - preserve_frame) < 5)
                if can_deliver:
                    deliver_plan = self._maybe_force_deliver(visible_infos, actual_holding_ids=actual_holding_ids)
                    if deliver_plan is not None:
                        plan = deliver_plan
                        override_used = True
                elif self.logger:
                    self.logger.debug(
                        "[Policy] _maybe_force_deliver: skipping - no actual held_objects (holding_ids=%s, actual=%s, preserve_frame=%d)",
                        holding_ids,
                        actual_holding_ids,
                        preserve_frame,
                    )
        if plan is not None and plan.action_type == "deliver" and self.logger:
            self.logger.info(
                "[Policy] frame=%s overriding with deliver id=%s name=%s dist=%.2f",
                frame,
                plan.target_id,
                plan.meta.get("target_name", "?"),
                plan.meta.get("distance", -1.0),
            )
        elif plan is None and holding_ids and self.logger:
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
                # Distributed coordination: Each agent independently assigns task_targets to the closest agent
                # All agents use the same deterministic algorithm, so they all get the same assignment
                combined_infos = list(visible_infos) if visible_infos else []
                task_target_assignments: Dict[int, int] = {}  # {obj_id: assigned_agent_id}
                
                if snapshot and hasattr(snapshot, "team_symbolic") and snapshot.team_symbolic:
                    # Collect all task_targets from team_symbolic with distances from each agent
                    task_targets = self.agent_state.get("task_targets", [])
                    task_target_names = {str(name).lower() for name in task_targets}
                    
                    # Group task_targets by obj_id, collecting distances from all agents
                    task_target_distances: Dict[int, List[Tuple[int, float]]] = {}  # {obj_id: [(agent_id, distance), ...]}
                    
                    # First, collect all task_targets from team_symbolic with their positions
                    task_target_entries: Dict[int, Dict[str, Any]] = {}  # {obj_id: entry}
                    for entry in snapshot.team_symbolic:
                        if entry.get("is_grabbable") and entry.get("is_task_target"):
                            obj_name = str(entry.get("name", "")).lower()
                            if obj_name in task_target_names:
                                obj_id = entry.get("id")
                                if obj_id is not None:
                                    task_target_entries[obj_id] = entry
                    
                    # Calculate distances from all agents to each task_target
                    # We need to get other agents' positions to calculate distances
                    # For now, we'll use the distance from team_symbolic (other agent's view)
                    # and recalculate for current agent if we have the object position
                    for obj_id, entry in task_target_entries.items():
                        seen_by_agent = entry.get("agent_id")
                        entry_distance = entry.get("distance")
                        entry_position = self._normalise_position(entry.get("position")) or self._normalise_position(entry.get("location"))
                        
                        # Add distance from the agent who saw it
                        if seen_by_agent is not None and entry_distance is not None and isinstance(entry_distance, (int, float)):
                            if obj_id not in task_target_distances:
                                task_target_distances[obj_id] = []
                            task_target_distances[obj_id].append((seen_by_agent, float(entry_distance)))
                        
                        # Calculate distance from current agent if we have position
                        if agent_pos is not None:
                            current_agent_distance = None
                            # Try to get position from agent_memory first (more accurate)
                            if self.agent_memory is not None and hasattr(self.agent_memory, "object_info") and obj_id in getattr(self.agent_memory, "object_info", {}):
                                try:
                                    obj_info = self.agent_memory.object_info[obj_id]
                                    mem_pos = obj_info.get("position")
                                    if mem_pos is not None:
                                        mem_pos_normalized = self._normalise_position(mem_pos)
                                        if mem_pos_normalized is not None:
                                            pos_arr = np.asarray(mem_pos_normalized, dtype=np.float32)
                                            if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                                                current_agent_distance = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                                except Exception:
                                    pass
                            
                            # If not found in agent_memory, try entry position
                            if current_agent_distance is None and entry_position is not None:
                                try:
                                    pos_arr = np.asarray(entry_position, dtype=np.float32)
                                    if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                                        current_agent_distance = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                                except Exception:
                                    pass
                            
                            # If we calculated distance, add it
                            if current_agent_distance is not None:
                                if obj_id not in task_target_distances:
                                    task_target_distances[obj_id] = []
                                # Check if current agent already has a distance entry (from visible_infos)
                                existing_entry = None
                                for i, (aid, dist) in enumerate(task_target_distances[obj_id]):
                                    if aid == self.agent_id:
                                        existing_entry = i
                                        break
                                if existing_entry is not None:
                                    # Update existing entry if calculated distance is more accurate
                                    task_target_distances[obj_id][existing_entry] = (self.agent_id, current_agent_distance)
                                else:
                                    task_target_distances[obj_id].append((self.agent_id, current_agent_distance))
                    
                    # Also add current agent's distances from visible_infos (may be more accurate)
                    if visible_infos:
                        for info in visible_infos:
                            if info.get("is_task_target") and info.get("is_grabbable"):
                                obj_id = info.get("id")
                                if obj_id is not None:
                                    distance = info.get("distance")
                                    if distance is not None and isinstance(distance, (int, float)):
                                        if obj_id not in task_target_distances:
                                            task_target_distances[obj_id] = []
                                        # Check if current agent already has a distance entry
                                        existing_entry = None
                                        for i, (aid, dist) in enumerate(task_target_distances[obj_id]):
                                            if aid == self.agent_id:
                                                existing_entry = i
                                                break
                                        if existing_entry is not None:
                                            # Use visible_infos distance (more accurate, from current view)
                                            task_target_distances[obj_id][existing_entry] = (self.agent_id, float(distance))
                                        else:
                                            task_target_distances[obj_id].append((self.agent_id, float(distance)))
                    
                    # Assign each task_target to the closest agent (deterministic: if tie, use agent_id order)
                    for obj_id, agent_distances in task_target_distances.items():
                        if not agent_distances:
                            continue
                        # Sort by distance, then by agent_id for deterministic tie-breaking
                        agent_distances.sort(key=lambda x: (x[1], x[0]))
                        closest_agent_id, closest_distance = agent_distances[0]
                        task_target_assignments[obj_id] = closest_agent_id
                        if self.logger:
                            all_distances_str = ", ".join([f"agent{aid}:{d:.2f}" for aid, d in agent_distances[:3]])
                            self.logger.debug(
                                "[Policy] task_target assignment: obj_id=%s -> agent%d (dist=%.2f, all: %s)",
                                obj_id,
                                closest_agent_id,
                                closest_distance,
                                all_distances_str,
                            )
                    
                    # Add task_targets to combined_infos only if assigned to this agent
                    for entry in snapshot.team_symbolic:
                        if entry.get("is_grabbable") and entry.get("is_task_target"):
                            obj_name = str(entry.get("name", "")).lower()
                            if obj_name in task_target_names:
                                obj_id = entry.get("id")
                                if obj_id is not None:
                                    # Only include if assigned to this agent
                                    if task_target_assignments.get(obj_id) == self.agent_id:
                                        already_in_visible = any(
                                            info.get("id") == obj_id for info in combined_infos
                                        )
                                        if not already_in_visible:
                                            combined_entry = dict(entry)
                                            combined_entry["from_team_symbolic"] = True
                                            combined_entry["seen_by_agent"] = entry.get("agent_id")
                                            # Clear distance and position - will be recalculated in _maybe_force_pick
                                            # Position from team_symbolic might be from another agent's view and could be outside map bounds
                                            combined_entry["distance"] = None
                                            combined_entry["position"] = None  # Force recalculation from agent_memory
                                            combined_entry["location"] = None  # Also clear location
                                            combined_infos.append(combined_entry)
                    
                    # Also check visible_infos for task_targets assigned to this agent
                    if visible_infos:
                        for info in visible_infos:
                            if info.get("is_task_target") and info.get("is_grabbable"):
                                obj_id = info.get("id")
                                if obj_id is not None and task_target_assignments.get(obj_id) == self.agent_id:
                                    # Already in combined_infos, but ensure it's there
                                    if not any(existing.get("id") == obj_id for existing in combined_infos):
                                        combined_infos.append(info)
                
                override_plan = self._maybe_force_pick(combined_infos, snapshot=snapshot)
                if override_plan is not None:
                    plan = override_plan
                    override_used = True
                elif self.logger and visible_infos:
                    self.logger.debug("[Policy] frame=%s force_pick returned None", frame)
        if plan is None:
            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "no_plan"})
        plan = self._ensure_plan_target(plan)
        # Clamp LLM-generated coordinates to valid room positions before checking
        # BUT: Don't clamp pick actions - they need the actual object position
        if plan.target_position is not None and guidance.source == "llm" and plan.action_type != "pick":
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
                            # Also skip the object ID if it's a pick action with invalid position
                            if plan.action_type == "pick" and plan.target_id is not None:
                                # Get object name from agent_memory to skip by name
                                if self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
                                    obj_info = self.agent_memory.object_info.get(plan.target_id)
                                    if obj_info and obj_info.get("name"):
                                        obj_name = str(obj_info.get("name")).lower()
                                        self._register_skip_target(name=obj_name)
                                        if self.logger:
                                            self.logger.debug(
                                                "[Policy] skipping object id=%s name=%s due to invalid position outside map bounds",
                                                plan.target_id,
                                                obj_name,
                                            )
                            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "target_outside_map_bounds"})
                            rejected = True
            # 방안 2: (0,0,0) 타겟 문제 해결 + LLM이 z=0.0 위치 제안 시 reject
            if not rejected and plan.target_position is not None and self.env_api and "check_pos_in_room" in self.env_api:
                # Check if target is (0,0,0) or similar invalid position (X and Z are both near 0)
                is_invalid_pos = (abs(target_pos[0]) < 0.01 and abs(target_pos[2]) < 0.01) if len(target_pos) >= 3 else False
                # Also check if LLM suggested z=0.0 position (often invalid)
                is_z_zero = len(target_pos) >= 3 and abs(target_pos[2]) < 0.01 and abs(target_pos[0]) > 0.01
                
                if is_invalid_pos:
                    if self.logger:
                        self.logger.warning(
                            "[Policy] rejecting invalid target position (X≈0, Z≈0, near origin): %s (action=%s)",
                            target_pos,
                            plan.action_type,
                        )
                    plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "invalid_target_position"})
                    rejected = True
                elif is_z_zero and guidance.source == "llm":
                    # LLM이 z=0.0 위치를 제안한 경우 reject하고 skip_targets에 추가
                    if self.logger:
                        self.logger.warning(
                            "[Policy] rejecting LLM suggested z=0.0 position: %s (action=%s), adding to skip_targets",
                            target_pos,
                            plan.action_type,
                        )
                    self._register_skip_target(position=target_pos)
                    plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "llm_invalid_z_zero_position"})
                    rejected = True
                elif not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                    # For pick actions, if position is outside room, set target_pos to None and use object_id only
                    # For other actions, try to use executor's clamp_target to find a valid position
                    if plan.action_type == "pick":
                        # Pick actions: if position is outside room, remove it and use object_id only
                        if self.logger:
                            self.logger.warning(
                                "[Policy] pick target outside room, removing position and using object_id only: %s (id=%s)",
                                target_pos,
                                plan.target_id,
                            )
                        # Remove target_position - executor will use object_id only for navigation
                        meta = dict(plan.meta)
                        meta["target_position"] = None
                        plan = ReasonedPlan(
                            plan.action_type,
                            plan.target_id,
                            None,  # Set target_position to None
                            plan.confidence,
                            meta,
                        )
                        # Don't reject - executor will handle navigation using object_id
                        pass
                    else:
                        clamped_pos = self.executor._clamp_target(target_pos)
                        # Check if clamped position is valid (in room) and not (0,0,0)
                        clamped_is_valid = False
                        if clamped_pos is not None and len(clamped_pos) >= 3:
                            clamped_x, clamped_y, clamped_z = clamped_pos[0], clamped_pos[1], clamped_pos[2]
                            # Check if clamped position is (0,0,0) or invalid
                            is_clamped_zero = (abs(clamped_x) < 0.01 and abs(clamped_z) < 0.01)
                            if not is_clamped_zero:
                                # Check if clamped position is in room
                                try:
                                    clamped_is_valid = self.env_api["check_pos_in_room"]((clamped_x, clamped_z))
                                except Exception:
                                    clamped_is_valid = False
                        if clamped_is_valid:
                            if self.logger:
                                self.logger.info(
                                    "[Policy] target outside room, clamped from %s to %s (action=%s, is_goal=%s)",
                                    target_pos,
                                    clamped_pos,
                                    plan.action_type,
                                    is_goal_position,
                                )
                            meta = dict(plan.meta)
                            meta["original_target"] = target_pos
                            meta["clamped"] = True
                            plan = ReasonedPlan(plan.action_type, plan.target_id, clamped_pos, plan.confidence, meta)
                        else:
                            # Clamping failed or clamped position is still invalid (including (0,0,0))
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] target outside room and cannot clamp to valid position: %s -> %s (adding to skip_targets)",
                                    target_pos,
                                    clamped_pos if clamped_pos is not None else "None",
                                )
                            # Add to skip_targets to prevent LLM from suggesting it again
                            self._register_skip_target(position=target_pos)
                            plan = ReasonedPlan("idle", None, None, 0.0, {"reason": "target_outside_room"})
                            rejected = True
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
        
        # 핵심 수정: continuous movement가 진행 중이면 새로운 navigation 액션 생성 방지
        # 이전 move_to_position이 완료되기 전에 새로운 move_to_position이 들어오면 이전 액션이 취소되는 문제 해결
        # 단, prev_status=2 (failure)이거나 너무 오래 지속되면 (20프레임 이상) 새로운 navigation 허용
        last_command = self.agent_state.get("last_command")
        is_navigation_plan = plan.action_type in ("move", "search")
        # Pick 액션이지만 거리가 멀어서 navigation으로 변환될 가능성이 있는 경우도 확인
        is_pick_that_might_navigate = plan.action_type == "pick"
        if (is_navigation_plan or is_pick_that_might_navigate) and last_command == "continuous_move" and isinstance(prev_status, int):
            # prev_status=2 (failure)이거나 너무 오래 지속되면 새로운 navigation 허용
            if prev_status == 2:
                # 실패했으므로 새로운 navigation 허용
                if self.logger:
                    self.logger.debug(
                        "[Policy] navigation failed (prev_status=2), allowing new navigation action frame=%s plan_action=%s",
                        frame,
                        plan.action_type,
                    )
                # Navigation state는 위에서 clear됨
            elif prev_status == 0:
                # Ongoing인 경우, 너무 오래 지속되면 새로운 navigation 허용
                last_nav_frame = self.agent_state.get("last_nav_start_frame", -1)
                if last_nav_frame >= 0 and frame - last_nav_frame >= 20:
                    # 너무 오래 지속되면 새로운 navigation 허용
                    if self.logger:
                        self.logger.warning(
                            "[Policy] navigation ongoing for %d frames (>=20), allowing new navigation action frame=%s plan_action=%s",
                            frame - last_nav_frame,
                            frame,
                            plan.action_type,
                        )
                    # Navigation state는 위에서 clear됨
                else:
                    # Navigation이 진행 중이면 새로운 navigation 액션 생성하지 않음
                    if self.logger:
                        self.logger.debug(
                            "[Policy] navigation already ongoing (prev_status=0, %d frames), returning ongoing instead of new navigation action frame=%s plan_action=%s",
                            frame - last_nav_frame if last_nav_frame >= 0 else 0,
                            frame,
                            plan.action_type,
                        )
                    return {"type": "ongoing"}
        
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
        elif command_type == 9:
            # type: 9 (continuous movement) - navigation으로 간주
            self.agent_state["last_command"] = "continuous_move"
            # Navigation 시작 프레임 기록 (ongoing이 너무 오래 지속되는지 확인하기 위해)
            self.agent_state["last_nav_start_frame"] = frame
            # Pick을 위한 navigation인 경우 last_pick_id 저장
            if plan.action_type == "pick" and plan.target_id is not None:
                self.agent_state["last_pick_id"] = plan.target_id
                self.agent_state["last_pick_pos"] = plan.target_position
        elif command_type not in (None, "ongoing"):
            self.agent_state["last_command"] = str(command_type)
        # If pick action resulted in navigation (too_far_for_pick), store pick_id for retry after navigation
        if plan.action_type == "pick" and exec_meta.get("reason") == "too_far_for_pick":
            if plan.target_id is not None:
                self.agent_state["last_pick_id"] = plan.target_id
                self.agent_state["last_pick_pos"] = plan.target_position
                if self.logger:
                    self.logger.debug(
                        "[Policy] storing last_pick_id=%s for retry after navigation (too_far_for_pick)",
                        plan.target_id,
                    )
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
        actual_held_ids = []
        for entry in held_objects:
            if not entry:
                continue
            obj_id = entry.get("id")
            arm_name = entry.get("arm")
            if obj_id is not None:
                holding_ids.append(obj_id)
                actual_held_ids.append(obj_id)
                valid_held_objects = True
            if isinstance(arm_name, str):
                holding_slots[arm_name.lower()] = obj_id
        
        # Verify pending picks: if pick was successful but held_objects doesn't contain it after 5 frames, remove it
        current_frame = obs.get("current_frames", 0)
        pending_verification = self.agent_state.get("pending_pick_verification", {})
        if pending_verification and self.logger:
            self.logger.debug(
                "[Policy] _update_agent_state: checking pending_verification=%s (current_frame=%d)",
                pending_verification,
                current_frame,
            )
        verified_picks = []
        verification_removed_ids = []  # Track IDs removed by verification
        for pick_id, pick_frame in pending_verification.items():
            frames_since_pick = current_frame - pick_frame
            if pick_id in actual_held_ids:
                # Successfully verified - object is actually in held_objects
                # Now add it to holding_ids if it's not already there
                if pick_id not in holding_ids:
                    holding_ids.append(pick_id)
                    self.agent_state["holding_ids_preserve_frame"] = current_frame
                    if self.logger:
                        self.logger.info(
                            "[Policy] pick verification SUCCESS: id=%s confirmed in held_objects, added to holding_ids (frame=%d, frames_since_pick=%d)",
                            pick_id,
                            current_frame,
                            frames_since_pick,
                        )
                else:
                    if self.logger:
                        self.logger.debug(
                            "[Policy] pick verification: id=%s confirmed in held_objects (frame=%d, frames_since_pick=%d)",
                            pick_id,
                            current_frame,
                            frames_since_pick,
                        )
                verified_picks.append(pick_id)
                # Remove from verification_removed_ids if it was there (shouldn't happen, but just in case)
                verification_removed_set = self.agent_state.get("verification_removed_ids", set())
                if pick_id in verification_removed_set:
                    verification_removed_set.discard(pick_id)
                    self.agent_state["verification_removed_ids"] = verification_removed_set
            elif frames_since_pick >= 5:
                # 5 frames passed but object is not in held_objects - likely false positive (reach_for succeeded but grasp failed)
                if pick_id in holding_ids:
                    holding_ids.remove(pick_id)
                    verification_removed_ids.append(pick_id)
                if self.logger:
                    self.logger.warning(
                        "[Policy] pick verification FAILED: id=%s not in held_objects after %d frames (reach_for succeeded but grasp likely failed) - clearing pick state to allow retry",
                        pick_id,
                        frames_since_pick,
                    )
                # Clear pick state to allow retry with different approach
                # If this is the current pick, clear last_command and last_pick_id
                current_last_pick_id = self.agent_state.get("last_pick_id")
                if current_last_pick_id == pick_id:
                    self.agent_state.pop("last_command", None)
                    self.agent_state.pop("last_pick_id", None)
                    self.agent_state.pop("last_pick_pos", None)
                    if self.logger:
                        self.logger.info(
                            "[Policy] cleared last_command and last_pick_id for failed pick id=%s to allow retry",
                            pick_id,
                        )
                # Store the failed pick info for potential retry
                self.agent_state["last_failed_pick_id"] = pick_id
                self.agent_state["last_failed_pick_frame"] = current_frame
                verified_picks.append(pick_id)
        # Remove verified picks from pending
        for pick_id in verified_picks:
            pending_verification.pop(pick_id, None)
        self.agent_state["pending_pick_verification"] = pending_verification
        # If verification removed any IDs, also remove them from agent_state to prevent preserve logic from restoring them
        if verification_removed_ids:
            current_state_holding_ids = self.agent_state.get("holding_ids", [])
            for removed_id in verification_removed_ids:
                if removed_id in current_state_holding_ids:
                    current_state_holding_ids.remove(removed_id)
            self.agent_state["holding_ids"] = current_state_holding_ids
            # Also reset preserve_frame if we removed all holding_ids
            if not current_state_holding_ids:
                self.agent_state["holding_ids_preserve_frame"] = -1
            # Store removed IDs to prevent preserve logic from restoring them
            verification_removed_set = self.agent_state.get("verification_removed_ids", set())
            verification_removed_set.update(verification_removed_ids)
            self.agent_state["verification_removed_ids"] = verification_removed_set
        
        # Check if put_in was successful
        # Method 1: status == 1 and last_command was deliver
        # Method 2: last_command was deliver and held_objects is empty (object was actually delivered)
        # Method 3: last_command was deliver and previous holding_ids objects are no longer in held_objects
        prev_status = obs.get("status")
        last_command = self.agent_state.get("last_command")
        previous_holding_ids = self.agent_state.get("holding_ids", [])
        put_in_success = False
        
        if last_command == "deliver":
            # Check if put_in was successful by comparing held_objects
            # If previous holding_ids objects are no longer in held_objects, put_in was successful
            if previous_holding_ids:
                # Check if any previous holding_ids objects are still in held_objects
                still_holding = [hid for hid in previous_holding_ids if hid in actual_held_ids]
                if not still_holding:
                    # All previous holding_ids objects are no longer in held_objects - put_in was successful
                    put_in_success = True
                    if self.logger:
                        self.logger.info(
                            "[Policy] deliver success: all previous holding_ids objects no longer in held_objects (was %s, now %s)",
                            previous_holding_ids,
                            actual_held_ids,
                        )
                elif isinstance(prev_status, int) and prev_status == 1:
                    # status == 1 also indicates success
                    put_in_success = True
                    if self.logger:
                        self.logger.info(
                            "[Policy] deliver success: status=1 and last_command=deliver (was %s, now %s)",
                            previous_holding_ids,
                            actual_held_ids,
                        )
            elif isinstance(prev_status, int) and prev_status == 1:
                # status == 1 indicates success even if no previous holding_ids
                put_in_success = True
                if self.logger:
                    self.logger.info(
                        "[Policy] deliver success: status=1 and last_command=deliver (no previous holding_ids)",
                    )
        
        if put_in_success:
            # put_in was successful, remove delivered objects from holding_ids
            # Even if held_objects still contains them (TDW delay), status==1 means delivery succeeded
            # Remove all previous_holding_ids objects from holding_ids
            if previous_holding_ids:
                # Keep only objects that are actually still in held_objects AND were not in previous_holding_ids
                # This removes all delivered objects from holding_ids
                holding_ids = [hid for hid in actual_held_ids if hid not in previous_holding_ids]
                if self.logger:
                    self.logger.info(
                        "[Policy] deliver success: removed delivered objects from holding_ids (was %s, removed %s, now %s)",
                        previous_holding_ids,
                        previous_holding_ids,
                        holding_ids,
                    )
            else:
                # No previous holding_ids, just use actual_held_ids
                holding_ids = actual_held_ids.copy()
            holding_slots = {}
            valid_held_objects = len(holding_ids) > 0  # Set to True if there are still held objects
            # Clear last_command to prevent repeating the same deliver action
            self.agent_state.pop("last_command", None)
            self.agent_state.pop("last_deliver_id", None)
            self.agent_state.pop("last_deliver_pos", None)
        
        # If held_objects is empty or all None, preserve existing holding_ids
        # This is important because pick success may have just added an object,
        # but obs["held_objects"] may not be updated yet
        # However, don't preserve if put_in was successful (handled above)
        # Also limit preservation time to max 10 frames to avoid stale state
        # Also don't preserve IDs that were removed by verification
        current_frame = obs.get("current_frames", 0)
        if not valid_held_objects and not put_in_success:
            existing_holding_ids = self.agent_state.get("holding_ids", [])
            preserve_frame = self.agent_state.get("holding_ids_preserve_frame", -1)
            max_preserve_frames = 10  # Maximum frames to preserve holding_ids without confirmation
            # Filter out IDs that were removed by verification
            verification_removed_ids = self.agent_state.get("verification_removed_ids", set())
            if verification_removed_ids:
                existing_holding_ids = [hid for hid in existing_holding_ids if hid not in verification_removed_ids]
                # If all IDs were removed, clear preserve_frame and don't preserve
                if not existing_holding_ids:
                    preserve_frame = -1
                    self.agent_state["holding_ids_preserve_frame"] = -1
            
            if existing_holding_ids:
                # Check if we should still preserve (within time limit)
                if preserve_frame < 0 or (current_frame - preserve_frame) < max_preserve_frames:
                    holding_ids = existing_holding_ids
                    if preserve_frame < 0:
                        # First time preserving, record the frame
                        self.agent_state["holding_ids_preserve_frame"] = current_frame
                    if self.logger:
                        self.logger.debug(
                            "[Policy] _update_agent_state: held_objects empty, preserving existing holding_ids=%s (preserve_frame=%d, current_frame=%d)",
                            holding_ids,
                            preserve_frame,
                            current_frame,
                        )
                else:
                    # Too many frames without confirmation, clear holding_ids
                    if self.logger:
                        self.logger.warning(
                            "[Policy] _update_agent_state: held_objects empty for too long (%d frames), clearing stale holding_ids=%s",
                            current_frame - preserve_frame,
                            existing_holding_ids,
                        )
                    holding_ids = []
                    self.agent_state["holding_ids_preserve_frame"] = -1
        else:
            # Valid held_objects found, reset preserve frame
            if valid_held_objects:
                self.agent_state["holding_ids_preserve_frame"] = -1
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
        # 방안 2: explore() 재시도 로직 개선 (10회 → 20회)
        try:
            attempts = 0
            max_attempts = 20  # 10회에서 20회로 증가
            while True:
                explore_x, explore_z = self.agent_memory.explore()
                candidate_pos = (float(explore_x), 0.0, float(explore_z))
                # Check if position is (0,0,0) or invalid
                is_zero_pos = (abs(explore_x) < 0.01 and abs(explore_z) < 0.01)
                # Check if position is in room
                is_valid = True
                if self.env_api and "check_pos_in_room" in self.env_api:
                    try:
                        is_valid = self.env_api["check_pos_in_room"]((candidate_pos[0], candidate_pos[2]))
                    except Exception:
                        is_valid = False
                # Also check map bounds
                if is_valid and hasattr(self.agent_memory, "_scene_bounds"):
                    bounds = getattr(self.agent_memory, "_scene_bounds", None)
                    if bounds:
                        x_min = bounds.get("x_min")
                        x_max = bounds.get("x_max")
                        z_min = bounds.get("z_min")
                        z_max = bounds.get("z_max")
                        if (x_min is not None and x_max is not None and (explore_x < x_min or explore_x > x_max)) or \
                           (z_min is not None and z_max is not None and (explore_z < z_min or explore_z > z_max)):
                            is_valid = False
                if (not is_zero_pos and not self._is_blocked_position(candidate_pos, blocked_coords) and is_valid) or attempts >= max_attempts:
                    break
                attempts += 1
        except Exception as exc:  # pragma: no cover
            if self.logger:
                self.logger.warning("[Policy] explore fallback failed: %s", exc)
            return plan
        # Final check: if explore returned (0,0,0), try to find a valid position near agent's current position
        if abs(explore_x) < 0.01 and abs(explore_z) < 0.01:
            # Try to find a valid position near agent's current position
            if current_arr is not None and current_arr.shape[0] >= 3:
                agent_x = float(current_arr[0])
                agent_z = float(current_arr[2])
                # Try positions around agent's current position (spiral search)
                search_radius = 1.0
                max_radius = 5.0
                found_valid = False
                while search_radius <= max_radius and not found_valid:
                    # Try 8 directions around agent position
                    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                        rad = math.radians(angle)
                        test_x = agent_x + search_radius * math.cos(rad)
                        test_z = agent_z + search_radius * math.sin(rad)
                        # Check map bounds first
                        is_valid = True
                        if hasattr(self.agent_memory, "_scene_bounds"):
                            bounds = getattr(self.agent_memory, "_scene_bounds", None)
                            if bounds:
                                x_min = bounds.get("x_min")
                                x_max = bounds.get("x_max")
                                z_min = bounds.get("z_min")
                                z_max = bounds.get("z_max")
                                if (x_min is not None and x_max is not None and (test_x < x_min or test_x > x_max)) or \
                                   (z_min is not None and z_max is not None and (test_z < z_min or test_z > z_max)):
                                    is_valid = False
                        # Check if in room
                        if is_valid and self.env_api and "check_pos_in_room" in self.env_api:
                            try:
                                if self.env_api["check_pos_in_room"]((test_x, test_z)):
                                    explore_x, explore_z = test_x, test_z
                                    found_valid = True
                                    if self.logger:
                                        self.logger.debug(
                                            "[Policy] explore returned (0,0,0), found valid position near agent: (%s, %s)",
                                            explore_x,
                                            explore_z,
                                        )
                                    break
                            except Exception:
                                pass
                    if found_valid:
                        break
                    search_radius += 1.0
                
                # If still not found, use executor's clamp_target
                if not found_valid:
                    if hasattr(self.executor, "_clamp_target"):
                        base_pos = (agent_x, 0.0, agent_z)
                        clamped = self.executor._clamp_target(base_pos)
                        # Check if clamped position is still (0,0,0) or invalid
                        if clamped is not None and len(clamped) >= 3:
                            clamped_x, clamped_y, clamped_z = clamped[0], clamped[1], clamped[2]
                            if abs(clamped_x) < 0.01 and abs(clamped_z) < 0.01:
                                # Clamped to (0,0,0) - this is invalid, return idle instead
                                if self.logger:
                                    self.logger.warning(
                                        "[Policy] explore returned (0,0,0) and clamp_target also returned (0,0,0), returning idle"
                                    )
                                return ReasonedPlan("idle", None, None, 0.0, {"reason": "explore_failed_no_valid_position"})
                            explore_x, explore_z = clamped_x, clamped_z
                        else:
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] explore returned (0,0,0) and clamp_target returned invalid result, returning idle"
                                )
                            return ReasonedPlan("idle", None, None, 0.0, {"reason": "explore_failed_no_valid_position"})
                else:
                    # Use agent's current position as last resort
                    explore_x, explore_z = agent_x, agent_z
                    if self.logger:
                        self.logger.warning(
                            "[Policy] explore returned (0,0,0), using agent's current position as fallback: (%s, %s)",
                            explore_x,
                            explore_z,
                        )
            else:
                # No agent position available, use executor's clamp_target with (0,0,0)
                if hasattr(self.executor, "_clamp_target"):
                    base_pos = (0.0, 0.0, 0.0)
                    clamped = self.executor._clamp_target(base_pos)
                    if clamped is not None and len(clamped) >= 3:
                        clamped_x, clamped_y, clamped_z = clamped[0], clamped[1], clamped[2]
                        if abs(clamped_x) < 0.01 and abs(clamped_z) < 0.01:
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] explore returned (0,0,0) and clamp_target also returned (0,0,0), returning idle"
                                )
                            return ReasonedPlan("idle", None, None, 0.0, {"reason": "explore_failed_no_valid_position"})
                        explore_x, explore_z = clamped_x, clamped_z
                    else:
                        if self.logger:
                            self.logger.warning(
                                "[Policy] explore returned (0,0,0) and clamp_target returned invalid result, returning idle"
                            )
                        return ReasonedPlan("idle", None, None, 0.0, {"reason": "explore_failed_no_valid_position"})
        pos = getattr(self.agent_memory, "position", np.array([explore_x, 0.0, explore_z]))
        pos_arr = _to_array(pos)
        height = float(pos_arr[1]) if pos_arr is not None and pos_arr.shape[0] >= 2 else 0.0
        target_pos: Tuple[float, float, float] = (float(explore_x), height, float(explore_z))
        # Final validation: if target is still outside room, clamp it
        if self.env_api and "check_pos_in_room" in self.env_api:
            try:
                if not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                    # Clamp target to valid room position
                    if hasattr(self.executor, "_clamp_target"):
                        clamped = self.executor._clamp_target(target_pos)
                        if clamped is not None and len(clamped) >= 3:
                            clamped_x, clamped_y, clamped_z = clamped[0], clamped[1], clamped[2]
                            # Check if clamped position is still (0,0,0) or invalid
                            if abs(clamped_x) < 0.01 and abs(clamped_z) < 0.01:
                                # Clamped to (0,0,0) - this is invalid, return idle instead
                                if self.logger:
                                    self.logger.warning(
                                        "[Policy] explore target outside room and clamp_target returned (0,0,0), returning idle"
                                    )
                                return ReasonedPlan("idle", None, None, 0.0, {"reason": "explore_failed_no_valid_position"})
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] explore target outside room, clamped from %s to %s",
                                    target_pos,
                                    clamped,
                                )
                            target_pos = clamped
                        else:
                            # Invalid clamp result, return idle
                            if self.logger:
                                self.logger.warning(
                                    "[Policy] explore target outside room and clamp_target returned invalid result, returning idle"
                                )
                            return ReasonedPlan("idle", None, None, 0.0, {"reason": "explore_failed_no_valid_position"})
            except Exception:
                pass
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

    def _maybe_force_pick(self, object_infos: List[Dict[str, Any]], snapshot: Optional[Any] = None) -> Optional[ReasonedPlan]:
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
        # Note: Task target assignment to closest agent is already done in act() before calling this function
        # So we don't need to check other_agent_distances here - only objects assigned to this agent are passed in
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
            # 방안 3: visible_infos의 위치를 우선 사용 (가장 정확)
            # agent_memory의 위치는 잘못 계산될 수 있으므로 신뢰하지 않음
            position = self._normalise_position(info.get("position"))
            if position is None:
                position = self._normalise_position(info.get("location"))
            # visible_infos에서 가져온 위치가 없을 때만 agent_memory에서 가져옴
            # 하지만 이 위치는 신뢰하지 않고 검증만 함
            if position is None:
                # Try to get position from agent_memory (last resort, but validate carefully)
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
                                # Validate position: check if it's within map bounds
                                if position is not None:
                                    x, y, z = position
                                    # Check map bounds if available
                                    is_valid_pos = True
                                    if hasattr(self.agent_memory, "_scene_bounds"):
                                        bounds = getattr(self.agent_memory, "_scene_bounds", None)
                                        if bounds:
                                            x_min = bounds.get("x_min")
                                            x_max = bounds.get("x_max")
                                            z_min = bounds.get("z_min")
                                            z_max = bounds.get("z_max")
                                            if (x_min is not None and x_max is not None and (x < x_min or x > x_max)) or \
                                               (z_min is not None and z_max is not None and (z < z_min or z > z_max)):
                                                is_valid_pos = False
                                                if self.logger:
                                                    self.logger.debug(
                                                        "[Policy] _maybe_force_pick: object id=%s position %s is outside map bounds, skipping",
                                                        obj_id,
                                                        position,
                                                    )
                                    # Also check if position is in room
                                    if is_valid_pos and self.env_api and "check_pos_in_room" in self.env_api:
                                        try:
                                            if not self.env_api["check_pos_in_room"]((x, z)):
                                                is_valid_pos = False
                                                if self.logger:
                                                    self.logger.debug(
                                                        "[Policy] _maybe_force_pick: object id=%s position %s is outside room, skipping",
                                                        obj_id,
                                                        position,
                                                    )
                                        except Exception:
                                            pass
                                    if is_valid_pos:
                                        info["position"] = position
                                    else:
                                        # Invalid position - skip this object
                                        filtered_reasons["invalid_position"] = filtered_reasons.get("invalid_position", 0) + 1
                                        obj_name = obj_info.get("name")
                                        if obj_name:
                                            skip_names.add(str(obj_name).lower())
                                            # 방안 3: skip_targets에 추가하여 재시도 방지
                                            self._register_skip_target(name=str(obj_name).lower())
                                        continue
                        except Exception:
                            pass
            
            # 방안 3: 최종 위치 검증
            # COELA는 map bounds 밖의 객체를 object_info에 저장하지 않지만,
            # cal_object_position이 잘못된 위치를 계산할 수 있으므로 모든 위치를 검증해야 함
            # visible_infos의 위치도 cal_object_position으로 계산되므로 검증 필요
            position_from_visible = (info.get("position") is not None or info.get("location") is not None) and position is not None
            if position is not None:
                x, y, z = position
                is_valid_pos = True
                # 모든 위치를 map bounds로 검증 (visible_infos도 cal_object_position으로 계산되므로)
                # COELA 방식: map bounds 밖의 객체는 사용하지 않음
                # Check map bounds for ALL positions (visible_infos도 검증 필요)
                if hasattr(self.agent_memory, "_scene_bounds"):
                    bounds = getattr(self.agent_memory, "_scene_bounds", None)
                    if bounds:
                        x_min = bounds.get("x_min")
                        x_max = bounds.get("x_max")
                        z_min = bounds.get("z_min")
                        z_max = bounds.get("z_max")
                        if (x_min is not None and x_max is not None and (x < x_min or x > x_max)) or \
                           (z_min is not None and z_max is not None and (z < z_min or z > z_max)):
                            is_valid_pos = False
                            if self.logger:
                                self.logger.debug(
                                    "[Policy] _maybe_force_pick: object id=%s name=%s position %s is outside map bounds (COELA-style filtering), skipping",
                                    obj_id,
                                    info.get("name"),
                                    position,
                                )
                # Check if position is in room for ALL positions
                if is_valid_pos and self.env_api and "check_pos_in_room" in self.env_api:
                    try:
                        if not self.env_api["check_pos_in_room"]((x, z)):
                            is_valid_pos = False
                            if self.logger:
                                self.logger.debug(
                                    "[Policy] _maybe_force_pick: object id=%s name=%s position %s is outside room (COELA-style filtering), skipping",
                                    obj_id,
                                    info.get("name"),
                                    position,
                                )
                    except Exception:
                        pass
                
                if not is_valid_pos:
                    # Invalid position - skip this object
                    filtered_reasons["invalid_position"] = filtered_reasons.get("invalid_position", 0) + 1
                    obj_name = info.get("name")
                    if obj_name:
                        skip_names.add(str(obj_name).lower())
                        self._register_skip_target(name=str(obj_name).lower())
                    continue
            
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
            # 핵심 수정: position이 None이지만 depth 기반 거리 추정이 있으면 허용
            # 10m 제한으로 인해 위치 계산이 실패하지만, depth로 거리는 알 수 있는 경우
            if position is None:
                # If it's a task target and we have distance, we can still try to pick (will use object_id only)
                if info.get("is_task_target") and info.get("distance") is not None:
                    # Allow it - we have distance estimate from depth, executor will handle navigation
                    # Store a flag to indicate position is estimated
                    info["position_estimated"] = True
                    if self.logger:
                        self.logger.debug(
                            "[Policy] _maybe_force_pick: allowing task target id=%s name=%s without position (distance=%.2f from depth)",
                            obj_id,
                            info.get("name"),
                            info.get("distance"),
                        )
                # If we have distance estimate (even if not task target), allow for grabbable objects
                elif info.get("distance") is not None and info.get("is_grabbable"):
                    # Allow it - we have distance estimate from depth
                    info["position_estimated"] = True
                    if self.logger:
                        self.logger.debug(
                            "[Policy] _maybe_force_pick: allowing grabbable id=%s name=%s without position (distance=%.2f from depth)",
                            obj_id,
                            info.get("name"),
                            info.get("distance"),
                        )
                else:
                    filtered_reasons["no_position"] = filtered_reasons.get("no_position", 0) + 1
                    continue
            if info.get("distance") is None:
                filtered_reasons["no_distance"] = filtered_reasons.get("no_distance", 0) + 1
                continue
            distance = float(info["distance"])
            # Increase max_distance to allow navigation to objects
            # For task targets: allow unlimited distance (executor will navigate) - prioritize task targets heavily
            # For other grabbables: allow up to 8m (executor will navigate)
            is_task_target = info.get("is_task_target", False)
            max_distance = float('inf') if is_task_target else 8.0
            if not np.isfinite(distance) or (not is_task_target and distance > max_distance):
                # For task targets, always allow (distance can be any value)
                if not is_task_target:
                    filtered_reasons["too_far"] = filtered_reasons.get("too_far", 0) + 1
                    continue
                # For task targets, if distance is not finite, still allow (will navigate)
                elif not np.isfinite(distance):
                    # Use a default large distance for task targets without valid distance
                    distance = 50.0
                    info["distance"] = distance
            # Skip blocked check if position is None (estimated from depth)
            if position is not None and self._is_blocked_position(position, blocked_coords, radius=0.4):
                filtered_reasons["blocked"] = filtered_reasons.get("blocked", 0) + 1
                continue
            name_lc = str(info.get("name", "")).lower()
            if name_lc in skip_names:
                filtered_reasons["skip_name"] = filtered_reasons.get("skip_name", 0) + 1
                continue
            # Note: Task target assignment is already done in act() - only objects assigned to this agent are passed in
            priority = info.get("priority", 0.0)
            if info.get("is_task_target"):
                priority += 4.0
            # For task targets, prioritize heavily - use priority * 2 to ensure they're always chosen over non-task targets
            # For non-task targets, use normal scoring
            if info.get("is_task_target"):
                score = priority * 2.0 - distance * 0.5  # Reduce distance penalty for task targets
            else:
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
        # 핵심 수정: task target이 아닌 객체는 pick하지 않음
        # task는 특정 객체만 pick해야 하므로, task target이 아닌 객체는 무시
        if not best_info.get("is_task_target", False):
            if self.logger:
                self.logger.debug(
                    "[Policy] _maybe_force_pick: skipping non-task-target object id=%s name=%s (task targets only)",
                    target_id,
                    best_info.get("name"),
                )
            return None
        # 핵심 수정: position이 None이지만 distance가 있으면 위치 추정 또는 object_id만으로 pick 시도
        if target_pos is None:
            distance = best_info.get("distance")
            is_task_target = best_info.get("is_task_target", False)
            position_estimated = best_info.get("position_estimated", False)
            if distance is not None and agent_pos is not None:
                # Try to estimate position first (preferred for task targets)
                if is_task_target or position_estimated:
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
                                    estimated_pos = (estimated_x, estimated_y, estimated_z)
                                    
                                    # Check if estimated position is in room - if not, don't use it (will use object_id only)
                                    is_in_room = True
                                    if self.env_api and "check_pos_in_room" in self.env_api:
                                        try:
                                            is_in_room = self.env_api["check_pos_in_room"]((estimated_x, estimated_z))
                                        except Exception:
                                            is_in_room = False
                                    
                                    if is_in_room:
                                        target_pos = estimated_pos
                                        if self.logger:
                                            self.logger.info(
                                                "[Policy] _maybe_force_pick: estimated position for task target id=%s name=%s dist=%.2f pos=%s",
                                                target_id,
                                                best_info.get("name"),
                                                distance,
                                                target_pos,
                                            )
                                    else:
                                        # Estimated position is outside room - don't use it, will use object_id only
                                        if self.logger:
                                            self.logger.warning(
                                                "[Policy] _maybe_force_pick: estimated position outside room, using object_id only (id=%s name=%s dist=%.2f pos=%s)",
                                                target_id,
                                                best_info.get("name"),
                                                distance,
                                                estimated_pos,
                                            )
                                        target_pos = None  # Will use object_id only in executor
                    except Exception as exc:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_pick: position estimation failed: id=%s name=%s error=%s",
                                target_id,
                                best_info.get("name"),
                                exc,
                            )
                
                # If position is still None, allow pick without position for task targets (any distance) or close objects
                # Executor will handle this case using reach_for (type 3) for task targets
                if target_pos is None:
                    # Task targets: allow any distance (executor will use reach_for with object_id)
                    # Non-task targets: only allow if distance <= 2.5m
                    if is_task_target or distance <= 2.5:
                        if self.logger:
                            if is_task_target:
                                self.logger.info(
                                    "[Policy] _maybe_force_pick: allowing task target pick without position (id=%s name=%s dist=%.2f, will use reach_for)",
                                    target_id,
                                    best_info.get("name"),
                                    distance,
                                )
                            else:
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
                                "[Policy] _maybe_force_pick: skipping without position (too far, not task target): id=%s name=%s distance=%.2f",
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
            "is_task_target": best_info.get("is_task_target", False),  # Executor에서 사용
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

    def _maybe_force_deliver(self, object_infos: List[Dict[str, Any]], actual_holding_ids: Optional[List[int]] = None) -> Optional[ReasonedPlan]:
        """Force a deliver action if holding objects and a nearby container is visible."""
        holding_ids = self.agent_state.get("holding_ids", [])
        preserve_frame = self.agent_state.get("holding_ids_preserve_frame", -1)
        # Get current_frame from agent_state if available (set by act() before calling this)
        current_frame = self.agent_state.get("current_frame", 0)
        
        # Prefer actual_holding_ids if provided, otherwise use holding_ids from state
        if actual_holding_ids is not None:
            if not actual_holding_ids:
                # If actual_holding_ids is empty but holding_ids exists and was recently updated (within 5 frames),
                # we might still want to try deliver (pick might have just succeeded)
                if holding_ids and preserve_frame >= 0:
                    frames_since_pick = current_frame - preserve_frame if current_frame > 0 else 999
                    if frames_since_pick <= 5:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_deliver: no actual_holding_ids but holding_ids=%s was recently updated (preserve_frame=%d, frames_since=%d), will try deliver",
                                holding_ids,
                                preserve_frame,
                                frames_since_pick,
                            )
                        # Use holding_ids from state (recently picked, might not be in held_objects yet)
                        # Don't return None, continue with holding_ids
                    else:
                        if self.logger:
                            self.logger.debug(
                                "[Policy] _maybe_force_deliver: no actual_holding_ids and holding_ids=%s is stale (preserve_frame=%d, frames_since=%d)",
                                holding_ids,
                                preserve_frame,
                                frames_since_pick,
                            )
                        return None
                else:
                    if self.logger:
                        self.logger.debug("[Policy] _maybe_force_deliver: no actual_holding_ids")
                    return None
            else:
                # Use actual_holding_ids, but verify they match holding_ids (for logging)
                if holding_ids and set(actual_holding_ids) != set(holding_ids):
                    if self.logger:
                        self.logger.warning(
                            "[Policy] _maybe_force_deliver: actual_holding_ids=%s differs from holding_ids=%s, using actual",
                            actual_holding_ids,
                            holding_ids,
                        )
                holding_ids = actual_holding_ids
        elif not holding_ids:
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
                # If no candidates found but goal_position_id exists, try to get position from env_api
                elif goal_position_id is not None:
                    # First try to find goal_position_id in memory
                    goal_obj_info = None
                    goal_pos = None
                    goal_name = "goal_position"
                    for obj_id, obj_info in getattr(self.agent_memory, "object_info", {}).items():
                        if obj_id == goal_position_id:
                            goal_obj_info = obj_info
                            goal_pos = goal_obj_info.get("position")
                            goal_name = str(goal_obj_info.get("name", "goal_position")).lower()
                            break
                    
                    # If not found in memory, try to get from env_api
                    if goal_pos is None and self.env_api is not None:
                        if isinstance(self.env_api, dict) and "get_goal_position" in self.env_api:
                            try:
                                goal_pos = self.env_api["get_goal_position"]()
                                if self.logger:
                                    self.logger.info(
                                        "[Policy] _maybe_force_deliver: goal_position_id=%s not in memory, got position from env_api: %s",
                                        goal_position_id,
                                        goal_pos,
                                    )
                            except Exception as exc:
                                if self.logger:
                                    self.logger.debug(
                                        "[Policy] _maybe_force_deliver: failed to get goal_position from env_api: %s",
                                        exc,
                                    )
                    
                    if goal_pos is not None:
                        goal_pos = self._normalise_position(goal_pos)
                        if goal_pos is not None:
                            try:
                                pos_arr = np.asarray(goal_pos, dtype=np.float32)
                                if pos_arr.shape[0] >= 3 and agent_pos.shape[0] >= 3:
                                    goal_distance = float(np.linalg.norm(agent_pos[[0, 2]] - pos_arr[[0, 2]]))
                                else:
                                    goal_distance = float(np.linalg.norm(agent_pos - pos_arr))
                            except Exception:
                                goal_distance = 999.0
                            
                            # Always return deliver plan for goal_position_id (no distance limit)
                            # The executor will handle navigation if needed
                            if self.logger:
                                self.logger.info(
                                    "[Policy] _maybe_force_deliver: overriding with deliver to goal_position_id=%s (name=%s) dist=%.2f pos=%s",
                                    goal_position_id,
                                    goal_name,
                                    goal_distance,
                                    goal_pos,
                                )
                            meta = {
                                "reason": "policy_override_deliver_goal",
                                "target_name": goal_name,
                                "distance": goal_distance,
                                "is_goal_position": True,
                            }
                            return ReasonedPlan("deliver", goal_position_id, goal_pos, 0.95, meta)
                        elif self.logger:
                            self.logger.warning(
                                "[Policy] _maybe_force_deliver: goal_position_id=%s position normalized to None",
                                goal_position_id,
                            )
                    elif self.logger:
                        self.logger.warning(
                            "[Policy] _maybe_force_deliver: goal_position_id=%s not found in memory and env_api.get_goal_position not available",
                            goal_position_id,
                        )
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
