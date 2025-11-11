from __future__ import annotations

from typing import Optional

import torch

from ..perception.clip_utils import load_clip


class ClipTextEncoder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu") -> None:
        self.device = device
        self.model, _ = load_clip(model_name, device=device)
        self.model.requires_grad_(False)

    def encode(self, text: str) -> torch.Tensor:
        import clip  # type: ignore

        tokens = clip.tokenize([text])
        with torch.no_grad():
            tokens = tokens.to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return features.squeeze(0)
