from __future__ import annotations

import functools
from typing import Tuple

import torch

try:
    import clip
except ImportError:  # pragma: no cover - handled at runtime
    clip = None  # type: ignore


@functools.lru_cache(maxsize=2)
def load_clip(model_name: str = "ViT-B/32", device: str = "cpu") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load a CLIP model and tokenizer with simple caching."""
    if clip is None:
        raise ImportError(
            "The 'clip' package is required. Install via pip install git+https://github.com/openai/CLIP.git"
        )
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model, preprocess
