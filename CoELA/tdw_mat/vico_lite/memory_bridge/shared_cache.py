from __future__ import annotations

from typing import Any, Dict


class SharedCache:
    """Simple dict-like cache for cross-module data."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def clear(self) -> None:
        self._store.clear()
