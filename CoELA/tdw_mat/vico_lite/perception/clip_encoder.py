from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .clip_utils import load_clip

try:
    import clip
except ImportError:  # pragma: no cover - runtime dependency
    clip = None  # type: ignore


class FrozenClipEncoder:
    """Wrapper around OpenAI CLIP for RGB embedding extraction."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu") -> None:
        self.device = device
        self.model, self.preprocess = load_clip(model_name, device=device)
        self.model.requires_grad_(False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an RGB image tensor/array using CLIP's preprocess."""
        array: np.ndarray
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.size(0) == 3:
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        elif isinstance(image, np.ndarray):
            array = image
        else:
            raise ValueError("Unsupported image type for CLIP encoding")
        if array.ndim == 3 and array.shape[2] == 3:
            pass
        elif array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected image shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(array)
        with torch.no_grad():
            processed = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(processed)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return features

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using CLIP tokenizer."""
        if clip is None:
            raise ImportError(
                "The 'clip' package is required. Install via pip install git+https://github.com/openai/CLIP.git"
            )
        tokens = clip.tokenize([text])
        with torch.no_grad():
            tokens = tokens.to(self.device)
            feature = self.model.encode_text(tokens)
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return feature.squeeze(0)

    def preprocess_image(self, image) -> torch.Tensor:
        array = image
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        if isinstance(array, np.ndarray) and array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        pil = Image.fromarray(array)
        return self.preprocess(pil).unsqueeze(0)


__all__ = ["FrozenClipEncoder"]
