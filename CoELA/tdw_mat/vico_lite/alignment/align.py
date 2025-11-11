from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CrossModalAligner(nn.Module):
    """Simple cross-modal alignment head to project modalities into a shared space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if vision.ndim == 1:
            vision = vision.unsqueeze(0)
        if text.ndim == 1:
            text = text.unsqueeze(0)
        fused = torch.cat([vision, text], dim=-1)
        aligned = self.fc(fused)
        aligned = aligned / aligned.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return fused, aligned
