from __future__ import annotations

from dataclasses import dataclass
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
        context: Dict[str, Any] = {
            "agent_id": agent_id,
            "frame": frame,
            "role": agent_state.get("role", "explore"),
            "subgoal": agent_state.get("subgoal", "search"),
            "holding_ids": agent_state.get("holding_ids", []),
            "room": agent_state.get("current_room"),
            "recent_actions": memory.recent_actions(5),
            "memory_step": snapshot.step,
            "memory_symbolic": snapshot.symbolic,
            "skip_targets": agent_state.get("skip_targets", set()),
        }
        if extra:
            context.update(extra)
        return context

    def decide(self, context: Dict[str, Any], force_heuristics: bool = False) -> GuidanceResult:
        output: ReasonerOutput = self.reasoner.decide(context, force_heuristics=force_heuristics)
        if output.plan is None:
            return GuidanceResult(plan=None, role=output.role, subgoal=output.subgoal, source=output.source)
        plan = ReasonedPlan(
            action_type=output.plan.action_type,
            target_id=output.plan.target_id,
            target_position=output.plan.target_position,
            confidence=output.plan.confidence,
            meta=output.plan.meta,
        )
        return GuidanceResult(plan=plan, role=output.role, subgoal=output.subgoal, source=output.source)
