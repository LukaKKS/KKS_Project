from __future__ import annotations

from typing import Any, Dict, List


class StateMerger:
    """Merge symbolic entries from perception into a shared list."""

    def merge(self, existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not existing:
            return incoming
        merged = list(existing)
        existing_set = {entry.get("id") for entry in existing if "id" in entry}
        for entry in incoming:
            if entry.get("id") not in existing_set:
                merged.append(entry)
        return merged
