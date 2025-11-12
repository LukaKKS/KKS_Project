from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np

from agent_memory import AgentMemory
from vico_lite.config import ViCoConfig
from vico_lite.policy import ViCoPolicy


class ViCoAgent:
    def __init__(
        self,
        agent_id,
        logger,
        max_frames,
        args=None,
        output_dir: str = "results",
        device: str = "cpu",
        shared_memory_hub=None,
    ):
        self.agent_type = "vico_agent"
        self.agent_id = agent_id
        self.logger = logger
        self.max_frames = max_frames
        self.args = args
        self.output_dir = output_dir
        self.device = device
        self.cfg = ViCoConfig()
        self.map_size = (240, 120)
        self.scene_bounds = {
            "x_min": -15.0,
            "x_max": 15.0,
            "z_min": -7.5,
            "z_max": 7.5,
        }
        self.save_img = True
        self.gt_mask = True
        self.goal_objects: Dict[str, Any] = {}
        self.rooms_name = None
        self.agent_memory: Optional[AgentMemory] = None
        self.policy: Optional[ViCoPolicy] = None
        self.env_api: Optional[Dict[str, Any]] = None
        self.agent_color = [-1, -1, -1]
        self.shared_hub = shared_memory_hub

    def reset(
        self,
        obs: Dict[str, Any],
        goal_objects: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        env_api: Optional[Any] = None,
        agent_color=None,
        agent_id: int = 0,
        rooms_name=None,
        gt_mask: bool = True,
        save_img: bool = True,
        episode_index: Optional[int] = None,
    ):
        self.output_dir = output_dir or self.output_dir
        self.goal_objects = goal_objects or {}
        self.rooms_name = rooms_name
        self.gt_mask = gt_mask
        self.save_img = save_img
        self.agent_color = agent_color if agent_color is not None else [-1, -1, -1]
        if isinstance(env_api, list):
            self.env_api = env_api[self.agent_id]
        else:
            self.env_api = env_api or {}
        self.agent_memory = AgentMemory(
            agent_id=self.agent_id,
            agent_color=self.agent_color,
            output_dir=self.output_dir,
            gt_mask=self.gt_mask,
            gt_behavior=True,
            env_api=self.env_api,
            constraint_type=None,
            map_size=self.map_size,
            scene_bounds=self.scene_bounds,
        )
        # Pass logger to agent_memory for debugging
        self.agent_memory.logger = self.logger
        if self.shared_hub is not None:
            self.shared_hub.begin_episode(episode_index)
            self.shared_hub.register_agent(self.agent_id)
        self.policy = ViCoPolicy(
            self.cfg,
            agent_id=self.agent_id,
            logger=self.logger,
            agent_memory=self.agent_memory,
            env_api=self.env_api,
            memory_hub=self.shared_hub,
            device=self.device,
        )
        self.policy.reset()
        self.policy.set_goal_context(goal_objects=self.goal_objects, rooms_name=self.rooms_name)
        self.agent_memory.update(obs, save_img=self.save_img)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent_memory is None or self.policy is None:
            raise RuntimeError("Agent not initialised. Call reset() first.")
        self.agent_memory.update(obs, save_img=self.save_img)
        augmented_obs = self._augment_observation(obs)
        return self.policy.act(augmented_obs)

    # ------------------------------------------------------------------
    def _augment_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        augmented = copy.deepcopy(obs)
        if "instruction" not in augmented:
            augmented["instruction"] = self._goal_instruction()
        if "visible_objects" not in augmented:
            augmented["visible_objects"] = []
        return augmented

    def _goal_instruction(self) -> str:
        if not self.goal_objects:
            return "goal: cooperate to transport target objects"
        parts = []
        for name, count in self.goal_objects.items():
            parts.append(f"{count} x {name}")
        return "Collect " + ", ".join(parts)
