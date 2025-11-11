from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class ActionRecorder:
    def __init__(self, maxlen: int = 200) -> None:
        self._buffer: Deque[Dict[str, object]] = deque(maxlen=maxlen)

    def record(self, action: Dict[str, object]) -> None:
        self._buffer.append(action)

    def last(self, n: int = 5) -> List[Dict[str, object]]:
        if n <= 0:
            return []
        return list(self._buffer)[-n:]

    def clear(self) -> None:
        self._buffer.clear()
