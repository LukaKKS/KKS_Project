from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..config import ViCoConfig
from ..memory.coordination import SharedMemory, SharedStateSnapshot
from ..perception.pipeline import PerceptionAlignmentPipeline, PerceptionOutput
from .action_recorder import ActionRecorder
from .hub import TeamMemoryHub
from .shared_cache import SharedCache
from .state_merger import StateMerger
from .sync import MemorySynchronizer


@dataclass
class MemoryBridgeOutput:
    perception: PerceptionOutput
    snapshot: SharedStateSnapshot


class MemoryBridgeController:
    def __init__(
        self,
        cfg: ViCoConfig,
        agent_id: int,
        device: str = "cpu",
        hub: Optional[TeamMemoryHub] = None,
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.device = device
        self.cache = SharedCache()
        self.action_recorder = ActionRecorder()
        self.state_merger = StateMerger()
        self.synchronizer = MemorySynchronizer(alpha=cfg.memory_ema_decay)
        self.perception = PerceptionAlignmentPipeline(cfg, device=device)
        self.hub = hub
        self.shared_memory: Optional[SharedMemory]
        if self.hub is None:
            self.shared_memory = SharedMemory(
                cfg.latent_dim,
                decay=cfg.memory_ema_decay,
                max_history=cfg.memory_max_history,
                device=device,
            )
        else:
            self.shared_memory = None

    def reset(self) -> None:
        self.cache.clear()
        self.action_recorder.clear()
        self.synchronizer.reset()
        if self.hub is None:
            self.shared_memory = SharedMemory(
                self.cfg.latent_dim,
                decay=self.cfg.memory_ema_decay,
                max_history=self.cfg.memory_max_history,
                device=self.device,
            )
        else:
            self.hub.register_agent(self.agent_id)

    def process(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        rgb = observation.get("rgb")
        instruction = observation.get("instruction")
        perception = self.perception.process(observation, instruction)
        synced = self.synchronizer.sync(f"agent_{self.agent_id}", perception.fused_latent)
        perception.fused_latent = synced

        snapshot: SharedStateSnapshot
        if self.hub is not None:
            snapshot = self.hub.update(self.agent_id, perception)
        else:
            assert self.shared_memory is not None
            self.shared_memory.update(synced, symbolic=perception.symbolic)
            snapshot = self.shared_memory.snapshot()

        return {
            "perception": perception,
            "snapshot": snapshot,
        }

    def record_action(self, data: Dict[str, Any]) -> None:
        self.action_recorder.record(data)

    def recent_actions(self, n: int = 5):
        return self.action_recorder.last(n)
