from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch


@dataclass
class SharedStateSnapshot:
    step: int
    ema_latent: torch.Tensor
    history: List[torch.Tensor]
    symbolic: List[Dict[str, object]]
    per_agent_latents: Dict[int, torch.Tensor] = field(default_factory=dict)
    per_agent_symbolic: Dict[int, List[Dict[str, object]]] = field(default_factory=dict)
    team_symbolic: List[Dict[str, object]] = field(default_factory=list)
    agent_id: Optional[int] = None
    skip_targets: Dict[str, List] = field(default_factory=dict)
    nav_guard_info: Dict[Tuple[float, float], int] = field(default_factory=dict)


class SharedMemory:
    """Maintains an exponential moving average of agent latents."""

    def __init__(self, latent_dim: int, decay: float = 0.9, max_history: int = 50, device: str = "cpu") -> None:
        self.latent_dim = latent_dim
        self.decay = decay
        self.device = device
        self.max_history = max_history
        self._ema: Optional[torch.Tensor] = None
        self._history: Deque[torch.Tensor] = deque(maxlen=max_history)
        self._symbolic: List[Dict[str, object]] = []
        self.step: int = 0

    def update(self, latent: torch.Tensor, symbolic: Optional[List[Dict[str, object]]] = None) -> None:
        latent = latent.to(self.device)
        if latent.ndim > 1:
            latent = latent.mean(dim=0)
        if self._ema is None:
            self._ema = latent
        else:
            self._ema = self.decay * self._ema + (1 - self.decay) * latent
        self._history.append(latent.detach())
        if symbolic is not None:
            self._symbolic = symbolic
        self.step += 1

    def snapshot(self) -> SharedStateSnapshot:
        ema = self._ema if self._ema is not None else torch.zeros(self.latent_dim, device=self.device)
        return SharedStateSnapshot(
            step=self.step,
            ema_latent=ema.detach().clone(),
            history=list(self._history),
            symbolic=self._symbolic,
            per_agent_latents={},
            per_agent_symbolic={},
            team_symbolic=self._symbolic,
            agent_id=None,
        )
