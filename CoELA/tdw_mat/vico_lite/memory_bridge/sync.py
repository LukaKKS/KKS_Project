from __future__ import annotations

from typing import Dict

import torch


class MemorySynchronizer:
    """Blend new latents with previous ones to reduce drift."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self._last: Dict[str, torch.Tensor] = {}

    def sync(self, key: str, latent: torch.Tensor) -> torch.Tensor:
        if key not in self._last:
            self._last[key] = latent
            return latent
        self._last[key] = self.alpha * self._last[key] + (1 - self.alpha) * latent
        return self._last[key]

    def reset(self) -> None:
        self._last.clear()
