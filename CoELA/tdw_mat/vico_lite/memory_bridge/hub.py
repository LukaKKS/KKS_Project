from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import torch

from ..config import ViCoConfig
from ..memory.coordination import SharedStateSnapshot
from ..perception.pipeline import PerceptionOutput


class TeamMemoryHub:
    """Shared latent space for multiple agents."""

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.decay = cfg.memory_ema_decay
        self.max_history = cfg.memory_max_history
        self.latent_dim = cfg.latent_dim
        self._agent_latents: Dict[int, torch.Tensor] = {}
        self._agent_symbolic: Dict[int, List[dict]] = {}
        self._agent_skip_names: Dict[int, Set[str]] = {}
        self._agent_skip_coords: Dict[int, Set[Tuple[float, float]]] = {}
        self._agent_nav_guard: Dict[int, Dict[Tuple[float, float], int]] = {}
        # 장애물 좌표 공유: {agent_id: {(x, z): frame}} - frame은 장애물이 감지된 시점
        self._agent_obstacles: Dict[int, Dict[Tuple[float, float], int]] = {}
        self._ema: Optional[torch.Tensor] = None
        self._history: Deque[torch.Tensor] = deque(maxlen=self.max_history)
        self._step: int = 0
        self._current_episode: Optional[int] = None

    @staticmethod
    def _quantize(value: float, step: float = 0.5) -> float:
        if step <= 0:
            return round(value, 2)
        return round(round(value / step) * step, 2)

    def _quantize_coord_pair(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        return (self._quantize(float(coord[0])), self._quantize(float(coord[1])))

    def begin_episode(self, episode_index: Optional[int]) -> None:
        """Reset hub state when a new episode starts."""
        if episode_index is None:
            episode_index = -1
        if self._current_episode != episode_index:
            self._current_episode = episode_index
            self.reset_episode()

    def reset_episode(self) -> None:
        self._agent_latents.clear()
        self._agent_symbolic.clear()
        self._agent_skip_names.clear()
        self._agent_skip_coords.clear()
        self._agent_nav_guard.clear()
        self._agent_obstacles.clear()
        self._ema = None
        self._history.clear()
        self._step = 0

    def register_agent(self, agent_id: int) -> None:
        self._agent_symbolic.setdefault(agent_id, [])
        self._agent_skip_names.setdefault(agent_id, set())
        self._agent_skip_coords.setdefault(agent_id, set())
        self._agent_nav_guard.setdefault(agent_id, {})
        self._agent_obstacles.setdefault(agent_id, {})

    def reset_agent(self, agent_id: int) -> None:
        self._agent_latents.pop(agent_id, None)
        self._agent_symbolic.pop(agent_id, None)
        self._agent_skip_names.pop(agent_id, None)
        self._agent_skip_coords.pop(agent_id, None)
        self._agent_nav_guard.pop(agent_id, None)
        self._agent_obstacles.pop(agent_id, None)

    def update(self, agent_id: int, perception: PerceptionOutput) -> SharedStateSnapshot:
        latent = perception.fused_latent.detach().to(self.device)
        if latent.ndim > 1:
            latent = latent.flatten()
        self._agent_latents[agent_id] = latent
        symbolic = perception.symbolic or []
        self._agent_symbolic[agent_id] = symbolic

        stacked = torch.stack(list(self._agent_latents.values()), dim=0) if self._agent_latents else torch.zeros(
            self.latent_dim, device=self.device
        )
        mean_latent = stacked.mean(dim=0) if stacked.ndim > 1 else stacked
        if self._ema is None:
            self._ema = mean_latent
        else:
            self._ema = self.decay * self._ema + (1 - self.decay) * mean_latent
        if mean_latent.numel() > 0:
            self._history.append(mean_latent.detach())
        self._step += 1

        team_symbolic: List[dict] = []
        for participant, entries in self._agent_symbolic.items():
            for entry in entries:
                enriched = dict(entry)
                enriched.setdefault("agent_id", participant)
                team_symbolic.append(enriched)

        skip_names: Set[str] = set()
        skip_coords: Set[Tuple[float, float]] = set()
        for values in self._agent_skip_names.values():
            skip_names.update(values)
        for values in self._agent_skip_coords.values():
            skip_coords.update(self._quantize_coord_pair(coord) for coord in values)

        nav_guard_info: Dict[Tuple[float, float], int] = {}
        for guard_map in self._agent_nav_guard.values():
            for key, cooldown in guard_map.items():
                q_key = self._quantize_coord_pair(key)
                if q_key not in nav_guard_info or nav_guard_info[q_key] < cooldown:
                    nav_guard_info[q_key] = int(cooldown)

        # 장애물 좌표 집계: 모든 에이전트의 장애물 좌표를 합침
        # TTL(Time To Live)은 120프레임 (약 4초) - 오래된 장애물은 자동으로 제거
        obstacle_coords: Dict[Tuple[float, float], int] = {}
        current_frame = self._step
        obstacle_ttl = 120  # 120프레임 동안 유지
        for agent_id, obstacle_map in self._agent_obstacles.items():
            for coord, detected_frame in obstacle_map.items():
                # TTL 체크: 오래된 장애물은 제거
                if current_frame - detected_frame < obstacle_ttl:
                    q_coord = self._quantize_coord_pair(coord)
                    # 가장 최근에 감지된 프레임을 저장
                    if q_coord not in obstacle_coords or obstacle_coords[q_coord] < detected_frame:
                        obstacle_coords[q_coord] = detected_frame
        # TTL이 지난 장애물 제거
        for agent_id in list(self._agent_obstacles.keys()):
            self._agent_obstacles[agent_id] = {
                coord: frame for coord, frame in self._agent_obstacles[agent_id].items()
                if current_frame - frame < obstacle_ttl
            }

        return SharedStateSnapshot(
            step=self._step,
            ema_latent=self._ema.detach().clone() if self._ema is not None else torch.zeros(
                self.latent_dim, device=self.device
            ),
            history=list(self._history),
            symbolic=self._agent_symbolic.get(agent_id, []),
            per_agent_latents={idx: tensor.detach().clone() for idx, tensor in self._agent_latents.items()},
            per_agent_symbolic={idx: entries for idx, entries in self._agent_symbolic.items()},
            team_symbolic=team_symbolic,
            agent_id=agent_id,
            skip_targets={
                "names": sorted(skip_names),
                "coords": [list(coord) for coord in sorted(skip_coords)],
            },
            nav_guard_info=nav_guard_info,
            obstacle_coords=obstacle_coords,
        )

    def update_skip_targets(
        self,
        agent_id: int,
        names: Iterable[str],
        coords: Iterable[Tuple[float, float]],
    ) -> None:
        if agent_id not in self._agent_skip_names:
            self.register_agent(agent_id)
        name_set = {str(name).lower() for name in names if isinstance(name, str)}
        coord_set = set()
        for coord in coords:
            try:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    coord_set.add(self._quantize_coord_pair((float(coord[0]), float(coord[1]))))
                elif isinstance(coord, tuple) and len(coord) == 2:
                    coord_set.add(self._quantize_coord_pair(coord))
            except Exception:
                continue
        if name_set:
            self._agent_skip_names[agent_id].update(name_set)
        if coord_set:
            self._agent_skip_coords[agent_id].update(coord_set)

    def update_nav_guard(self, agent_id: int, nav_guard: Dict[Tuple[float, float], int]) -> None:
        if agent_id not in self._agent_nav_guard:
            self.register_agent(agent_id)
        normalised: Dict[Tuple[float, float], int] = {}
        for key, value in nav_guard.items():
            if isinstance(key, tuple):
                coord = tuple(float(v) for v in key[:2])
            elif isinstance(key, list):
                coord = tuple(float(v) for v in key[:2])
            else:
                continue
            normalised[self._quantize_coord_pair(coord)] = int(value)
        self._agent_nav_guard[agent_id] = normalised

    def update_obstacles(self, agent_id: int, obstacle_coords: Iterable[Tuple[float, float]], current_frame: int) -> None:
        """장애물 좌표를 공유 메모리에 저장"""
        if agent_id not in self._agent_obstacles:
            self.register_agent(agent_id)
        normalised: Dict[Tuple[float, float], int] = {}
        for coord in obstacle_coords:
            try:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    q_coord = self._quantize_coord_pair((float(coord[0]), float(coord[1])))
                    # 가장 최근 프레임으로 업데이트 (같은 좌표가 여러 번 감지되면 최신 것으로)
                    if q_coord not in normalised or normalised[q_coord] < current_frame:
                        normalised[q_coord] = current_frame
            except Exception:
                continue
        # 기존 장애물과 병합 (최신 프레임 우선)
        for q_coord, frame in normalised.items():
            if q_coord not in self._agent_obstacles[agent_id] or self._agent_obstacles[agent_id][q_coord] < frame:
                self._agent_obstacles[agent_id][q_coord] = frame


