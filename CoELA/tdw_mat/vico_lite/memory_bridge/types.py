from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ObservationBundle:
    rgb: torch.Tensor
    depth: Optional[torch.Tensor]
    instruction: Optional[str]
    objects: List[Dict[str, Any]]


@dataclass
class MemoryUpdate:
    latent: torch.Tensor
    symbolic: Optional[List[Dict[str, Any]]]
    meta: Dict[str, Any]
