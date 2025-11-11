from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def build_symbolic_descriptions(
    similarity: torch.Tensor,
    object_infos: Sequence[Dict[str, object]],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """Return a list of symbolic descriptions given similarity scores and object metadata."""
    if similarity.ndim == 1:
        similarity = similarity.unsqueeze(0)
    results: List[Dict[str, object]] = []
    infos = list(object_infos)
    for scores in similarity:
        values, indices = torch.topk(scores, k=min(top_k, scores.shape[-1]))
        entry = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            base: Dict[str, object]
            if idx < len(infos):
                base = dict(infos[idx])
            else:
                base = {}
            base.setdefault("name", base.get("name", f"obj_{idx}"))
            base["score"] = float(val)
            entry.append(base)
        results.append({"candidates": entry})
    return results
