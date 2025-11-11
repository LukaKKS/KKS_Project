from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import ViCoConfig
from ..memory.coordination import SharedStateSnapshot
from ..memory_bridge.controller import MemoryBridgeController
from .reasoner import PolicyReasoner, ReasonerOutput


@dataclass
class ReasonedPlan:
    action_type: str
    target_id: Optional[int]
    target_position: Optional[Tuple[float, float, float]]
    confidence: float
    meta: Dict[str, Any]


@dataclass
class GuidanceResult:
    plan: Optional[ReasonedPlan]
    role: Optional[str]
    subgoal: Optional[str]
    source: str
    debug: Dict[str, Any] = field(default_factory=dict)


class GuidanceController:
    def __init__(self, cfg: ViCoConfig, reasoner: PolicyReasoner) -> None:
        self.cfg = cfg
        self.reasoner = reasoner

    def build_context(
        self,
        agent_id: int,
        frame: int,
        memory: MemoryBridgeController,
        snapshot: SharedStateSnapshot,
        agent_state: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        skip_targets = agent_state.get("skip_targets", {"names": set(), "coords": set()}) or {"names": set(), "coords": set()}
        names_set = {str(name).lower() for name in skip_targets.get("names", set())}
        coords_set = {tuple(coord) for coord in skip_targets.get("coords", set())}
        snapshot_skip = getattr(snapshot, "skip_targets", None)
        if isinstance(snapshot_skip, dict):
            for entry in snapshot_skip.get("names", []):
                names_set.add(str(entry).lower())
            for coord in snapshot_skip.get("coords", []):
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    coords_set.add((float(coord[0]), float(coord[1])))
        names = sorted(names_set)
        coords_sorted = sorted(coords_set)
        coords = [list(coord) for coord in coords_sorted]
        coord_strings = [f"[{coord[0]:.2f}, {coord[1]:.2f}]" for coord in coords_sorted]

        combined_nav_guard: Dict[Tuple[float, float], int] = {}
        snapshot_nav_guard = getattr(snapshot, "nav_guard_info", None)
        if isinstance(snapshot_nav_guard, dict):
            for key, value in snapshot_nav_guard.items():
                if isinstance(key, (list, tuple)) and len(key) >= 2:
                    coord = (float(key[0]), float(key[1]))
                    combined_nav_guard[coord] = max(int(value), combined_nav_guard.get(coord, 0))
        extra_nav_guard = {}
        if extra:
            extra_nav_guard = extra.get("nav_guard_info", {})
        if isinstance(extra_nav_guard, dict):
            for key, value in extra_nav_guard.items():
                if isinstance(key, (list, tuple)) and len(key) >= 2:
                    coord = (float(key[0]), float(key[1]))
                else:
                    coord = key  # best effort
                if isinstance(coord, tuple) and len(coord) >= 2:
                    coord = (float(coord[0]), float(coord[1]))
                    combined_nav_guard[coord] = max(int(value), combined_nav_guard.get(coord, 0))

        context: Dict[str, Any] = {
            "agent_id": agent_id,
            "frame": frame,
            "role": agent_state.get("role", "explore"),
            "subgoal": agent_state.get("subgoal", "search"),
            "holding_ids": agent_state.get("holding_ids", []),
            "current_room": agent_state.get("current_room"),
            "recent_actions": memory.recent_actions(5),
            "memory_step": snapshot.step,
            "memory_symbolic": snapshot.symbolic,
            "skip_targets": {"names": names, "coords": coords},
            "skip_targets_text": {"names": names, "coords": coord_strings},
        }
        for key in ("task_type", "task_targets", "task_containers", "grabbable_names", "visible_objects", "goal_objects"):
            value = agent_state.get(key)
            if value:
                context[key] = value
        partner_symbolic = {}
        if snapshot.per_agent_symbolic:
            for participant, entries in snapshot.per_agent_symbolic.items():
                if participant == agent_id:
                    continue
                partner_symbolic[str(participant)] = entries
        if snapshot.team_symbolic:
            context["team_symbolic"] = snapshot.team_symbolic
        if partner_symbolic:
            context["partner_symbolic"] = partner_symbolic
        if extra:
            context.update({k: v for k, v in extra.items() if k != "nav_guard_info"})
        context["nav_guard_info"] = combined_nav_guard
        return context

    def decide(self, context: Dict[str, Any], force_heuristics: bool = False) -> GuidanceResult:
        output: ReasonerOutput = self.reasoner.decide(context, force_heuristics=force_heuristics)
        if output.plan is None:
            return GuidanceResult(
                plan=None,
                role=output.role,
                subgoal=output.subgoal,
                source=output.source,
                debug=output.debug,
            )
        plan = ReasonedPlan(
            action_type=output.plan.action_type,
            target_id=output.plan.target_id,
            target_position=output.plan.target_position,
            confidence=output.plan.confidence,
            meta=output.plan.meta,
        )
        return GuidanceResult(
            plan=plan,
            role=output.role,
            subgoal=output.subgoal,
            source=output.source,
            debug=output.debug,
        )
