from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class GeometryFuser:
    """Extract simple geometric statistics from depth maps."""

    def __init__(self, output_dim: int = 0, device: str = "cpu") -> None:
        self.output_dim = output_dim
        self.device = device

    def fuse(self, depth: Optional[np.ndarray]) -> torch.Tensor:
        if self.output_dim <= 0 or depth is None:
            return torch.zeros(self.output_dim, device=self.device)
        if isinstance(depth, torch.Tensor):
            depth_tensor = depth.detach().float().to(self.device)
        else:
            depth_tensor = torch.from_numpy(depth).float().to(self.device)
        valid = depth_tensor[(depth_tensor > 0) & torch.isfinite(depth_tensor)]
        if valid.numel() == 0:
            return torch.zeros(self.output_dim, device=self.device)
        stats = torch.stack(
            [
                valid.mean(),
                valid.std(unbiased=False),
                valid.min(),
                valid.max(),
            ]
        )
        if self.output_dim <= stats.numel():
            return stats[: self.output_dim]
        padded = torch.zeros(self.output_dim, device=self.device)
        padded[: stats.numel()] = stats
        return padded
