from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import torch

from ..config import ViCoConfig
from ..memory.coordination import SharedStateSnapshot
from ..perception.pipeline import PerceptionOutput


class TeamMemoryHub:
    """Shared latent space for multiple agents."""

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.decay = cfg.memory_ema_decay
        self.max_history = cfg.memory_max_history
        self.latent_dim = cfg.latent_dim
        self._agent_latents: Dict[int, torch.Tensor] = {}
        self._agent_symbolic: Dict[int, List[dict]] = {}
        self._ema: Optional[torch.Tensor] = None
        self._history: Deque[torch.Tensor] = deque(maxlen=self.max_history)
        self._step: int = 0
        self._current_episode: Optional[int] = None

    def begin_episode(self, episode_index: Optional[int]) -> None:
        """Reset hub state when a new episode starts."""
        if episode_index is None:
            episode_index = -1
        if self._current_episode != episode_index:
            self._current_episode = episode_index
            self.reset_episode()

    def reset_episode(self) -> None:
        self._agent_latents.clear()
        self._agent_symbolic.clear()
        self._ema = None
        self._history.clear()
        self._step = 0

    def register_agent(self, agent_id: int) -> None:
        self._agent_symbolic.setdefault(agent_id, [])

    def reset_agent(self, agent_id: int) -> None:
        self._agent_latents.pop(agent_id, None)
        self._agent_symbolic.pop(agent_id, None)

    def update(self, agent_id: int, perception: PerceptionOutput) -> SharedStateSnapshot:
        latent = perception.fused_latent.detach().to(self.device)
        if latent.ndim > 1:
            latent = latent.flatten()
        self._agent_latents[agent_id] = latent
        symbolic = perception.symbolic or []
        self._agent_symbolic[agent_id] = symbolic

        stacked = torch.stack(list(self._agent_latents.values()), dim=0) if self._agent_latents else torch.zeros(
            self.latent_dim, device=self.device
        )
        mean_latent = stacked.mean(dim=0) if stacked.ndim > 1 else stacked
        if self._ema is None:
            self._ema = mean_latent
        else:
            self._ema = self.decay * self._ema + (1 - self.decay) * mean_latent
        if mean_latent.numel() > 0:
            self._history.append(mean_latent.detach())
        self._step += 1

        team_symbolic: List[dict] = []
        for participant, entries in self._agent_symbolic.items():
            for entry in entries:
                enriched = dict(entry)
                enriched.setdefault("agent_id", participant)
                team_symbolic.append(enriched)

        return SharedStateSnapshot(
            step=self._step,
            ema_latent=self._ema.detach().clone() if self._ema is not None else torch.zeros(
                self.latent_dim, device=self.device
            ),
            history=list(self._history),
            symbolic=self._agent_symbolic.get(agent_id, []),
            per_agent_latents={idx: tensor.detach().clone() for idx, tensor in self._agent_latents.items()},
            per_agent_symbolic={idx: entries for idx, entries in self._agent_symbolic.items()},
            team_symbolic=team_symbolic,
            agent_id=agent_id,
        )


