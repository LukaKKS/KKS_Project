from __future__ import annotations

import logging
from typing import Optional

import torch

LOGGER = logging.getLogger(__name__)


class Kosmos2Encoder:
    """Lightweight wrapper around Kosmos-2, returning zeros if unavailable."""

    def __init__(self, model_name: str, device: str = "cpu", embedding_dim: int = 0) -> None:
        self.device = device
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.model = None
        self.processor = None
        if embedding_dim <= 0:
            LOGGER.info("Kosmos-2 disabled (embedding_dim<=0)")
            return
        try:
            from transformers import Kosmos2ForConditionalGeneration, Kosmos2Processor  # type: ignore

            self.processor = Kosmos2Processor.from_pretrained(model_name)
            self.model = Kosmos2ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            LOGGER.info("Loaded Kosmos-2 model %s", model_name)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            LOGGER.warning("Failed to load Kosmos-2 (%s). Using zero embeddings.", exc)
            self.model = None
            self.processor = None

    def encode(self, rgb: Optional[torch.Tensor], depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.embedding_dim <= 0 or self.model is None or self.processor is None or rgb is None:
            return torch.zeros(self.embedding_dim, device=self.device)
        # Simple mean pooling fallback if model fails during encode
        try:
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].mean(dim=1)
                hidden = hidden.squeeze(0)
                hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                return hidden
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Kosmos-2 encode failed (%s). Returning zeros.", exc)
            return torch.zeros(self.embedding_dim, device=self.device)
