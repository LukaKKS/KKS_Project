from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..config import ViCoConfig
from .clip_encoder import FrozenClipEncoder
from .kosmos_encoder import Kosmos2Encoder
from .symbolic import build_symbolic_descriptions


@dataclass
class PerceptionOutput:
    vision_latent: torch.Tensor
    text_latent: torch.Tensor
    fused_latent: torch.Tensor
    symbolic: Any


class PerceptionAlignmentPipeline:
    """Combine CLIP, Kosmos-2 and symbolic projection into a shared latent."""

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.clip = FrozenClipEncoder(cfg.clip_model, device=device)
        self.kosmos = (
            Kosmos2Encoder(cfg.kosmos_model, device=device, embedding_dim=cfg.kosmos_dim)
            if cfg.kosmos_dim > 0
            else None
        )
        head = cfg.clip_dim + cfg.alignment_dim + max(0, cfg.kosmos_dim)
        self.output_dim = head

    def process(self, observation: Dict[str, Any], instruction: Optional[str]) -> PerceptionOutput:
        rgb = self._ensure_tensor(observation.get("rgb"))
        text = instruction or ""
        clip_image = self.clip.encode_image(rgb)
        clip_text = self.clip.encode_text(text)
        kosmos_feat = None
        if self.kosmos is not None:
            kosmos_feat = self.kosmos.encode(rgb)
        fused_components = [clip_image.squeeze(0)]
        if kosmos_feat is not None and kosmos_feat.numel() > 0:
            fused_components.append(kosmos_feat)
        fused_components.append(clip_text)
        fused = torch.cat(fused_components, dim=-1)
        similarity = clip_image @ clip_text.unsqueeze(-1)
        symbolic = build_symbolic_descriptions(similarity, observation.get("object_names", []), self.cfg.symbolic_top_k)
        return PerceptionOutput(vision_latent=clip_image.squeeze(0), text_latent=clip_text, fused_latent=fused, symbolic=symbolic)

    def _ensure_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(self.device)
        else:
            raise ValueError("RGB image must be provided as torch.Tensor or np.ndarray")
        if tensor.ndim == 3 and tensor.shape[0] in {3, 4}:
            return tensor.float()
        if tensor.ndim == 4:
            return tensor.float()[0]
        raise ValueError(f"Unexpected RGB tensor shape: {tensor.shape}")
