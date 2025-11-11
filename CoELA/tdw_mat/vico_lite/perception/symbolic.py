from __future__ import annotations

from typing import Dict, List

import torch


def build_symbolic_descriptions(similarity: torch.Tensor, object_names: List[str], top_k: int = 5) -> List[Dict[str, object]]:
    """Return a list of symbolic descriptions given similarity scores."""
    if similarity.ndim == 1:
        similarity = similarity.unsqueeze(0)
    results: List[Dict[str, object]] = []
    for scores in similarity:
        values, indices = torch.topk(scores, k=min(top_k, scores.shape[-1]))
        entry = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            name = object_names[idx] if idx < len(object_names) else f"obj_{idx}"
            entry.append({"name": name, "score": float(val)})
        results.append({"candidates": entry})
    return results
