from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _default_heuristic_weights() -> Dict[str, float]:
    return {
        "distance": 1.5,
        "visibility": 0.6,
        "novelty": 0.8,
        "role_alignment": 0.7,
        "penalty": 1.0,
    }


def _default_reflection_actions() -> List[str]:
    return ["pick", "deliver"]


def _default_plan_horizon_actions() -> List[str]:
    return ["search", "move"]


def _default_action_blacklist() -> List[str]:
    return []


def _default_action_whitelist() -> Optional[List[str]]:
    return None


@dataclass
class ViCoConfig:
    """Configuration container for ViCo-Lite modules."""

    # Environment / control
    max_frames: int = 3000
    target_stall_frames: int = 120
    skip_target_decay: int = 120
    guard_skip_decay: int = 60

    # Visual encoders
    clip_model: str = "ViT-B/32"
    clip_dim: int = 512
    kosmos_model: str = "microsoft/kosmos-2-patch14-224"
    kosmos_dim: int = 0
    depth_fusion_dim: int = 0
    alignment_dim: int = 512
    latent_dim: int = 512 + alignment_dim

    # Symbolic abstraction
    symbolic_vocab_size: int = 512
    symbolic_top_k: int = 5

    # LLM guidance / reasoner
    use_llm_guidance: bool = True
    use_reasoner: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_min_interval: int = 90
    llm_summary_targets: int = 3
    alignment_temperature: float = 0.01

    reasoner_model: str = "gpt-4o-mini"
    reasoner_temperature: float = 0.2
    reasoner_candidate_topk: int = 3
    reasoner_reflection: bool = True
    reasoner_min_interval: int = 30
    reasoner_cache_ttl: int = 120
    reasoner_plan_horizon: int = 15
    reasoner_reflection_actions: List[str] = field(default_factory=_default_reflection_actions)
    reasoner_reflection_interval: int = 5
    reasoner_plan_horizon_actions: List[str] = field(default_factory=_default_plan_horizon_actions)
    reasoner_heuristic_weights: Dict[str, float] = field(default_factory=_default_heuristic_weights)

    # Reasoner candidate filtering
    action_whitelist: Optional[List[str]] = field(default_factory=_default_action_whitelist)
    action_blacklist: List[str] = field(default_factory=_default_action_blacklist)

    # Memory parameters
    memory_ema_decay: float = 0.9
    memory_max_history: int = 50

    # Navigation
    navigation_guard_ratio: float = 1.25

    # Logging
    log_json_pretty: bool = False

    # Misc
    device: str = "cpu"
    force_heuristics: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]
