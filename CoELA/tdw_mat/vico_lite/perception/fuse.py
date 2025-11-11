from __future__ import annotations

from typing import Sequence

import torch


def concatenate(features: Sequence[torch.Tensor]) -> torch.Tensor:
    tensors = [feat for feat in features if feat is not None and feat.numel() > 0]
    if not tensors:
        return torch.empty(0)
    aligned = [feat if feat.ndim == 2 else feat.unsqueeze(0) for feat in tensors]
    base = torch.cat(aligned, dim=-1)
    return base.squeeze(0)
