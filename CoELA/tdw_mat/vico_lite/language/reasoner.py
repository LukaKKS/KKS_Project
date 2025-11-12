from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..config import ViCoConfig

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class ReasonerPlan:
    action_type: str
    target_id: Optional[int]
    target_position: Optional[Tuple[float, float, float]]
    confidence: float
    meta: Dict[str, Any]


@dataclass
class ReasonerOutput:
    plan: Optional[ReasonerPlan]
    role: Optional[str]
    subgoal: Optional[str]
    source: str
    debug: Dict[str, Any]


class PolicyReasoner:
    """LLM-backed planner with heuristic fallback and optional reflection."""

    _JSON_FENCE_RE = re.compile(r"```json\s*([\s\S]+?)\s*```", re.IGNORECASE)

    def __init__(self, cfg: ViCoConfig) -> None:
        self.cfg = cfg
        self.client = self._setup_client()
        self.last_call_ts: float = 0.0
        self._warning_logged: Dict[str, bool] = {}
        self._prompt_template = self._load_prompt_template()
        self._last_signature: Optional[Tuple[str, Optional[Tuple[float, float, float]]]] = None

    def _setup_client(self):  # type: ignore[override]
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            env_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_path):
                try:
                    from dotenv import load_dotenv  # type: ignore

                    load_dotenv(env_path)
                    api_key = os.getenv("OPENAI_API_KEY")
                except Exception:  # pragma: no cover
                    pass
        if api_key is None or OpenAI is None:
            LOGGER.warning("PolicyReasoner running in heuristic-only mode (missing OpenAI client).")
            return None
        try:
            import httpx
            client = OpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(30.0, connect=10.0),  # 30초 타임아웃, 연결 10초
            )
            return client
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to initialise OpenAI client (%s). Falling back to heuristics.", exc)
            return None

    # Public API -----------------------------------------------------------------
    def decide(self, context: Dict[str, Any], force_heuristics: bool = False) -> ReasonerOutput:
        debug: Dict[str, Any] = {"force_heuristics": force_heuristics}
        plan = None
        role = context.get("role")
        subgoal = context.get("subgoal")
        source = "heuristic"
        candidates: List[Dict[str, Any]] = []

        if not force_heuristics and self.cfg.use_reasoner and self.client is not None:
            now = time.time()
            cooldown = self.cfg.reasoner_min_interval
            if now - self.last_call_ts >= cooldown:
                try:
                    candidates = self._query_candidates(context)
                    debug["candidate_count"] = len(candidates)
                    if candidates:
                        plan = self._select_plan(context, candidates)
                        source = "llm"
                        self.last_call_ts = now
                        if self.cfg.reasoner_reflection and plan is not None:
                            reflection = self._reflect(plan, context)
                            if reflection:
                                debug["reflection"] = reflection
                                verdict = reflection.get("verdict", "approve").lower()
                                if "reject" in verdict or "retry" in verdict:
                                    debug["reflection_verdict"] = verdict
                                    plan = None
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("PolicyReasoner candidates failed (%s)", exc)
                    debug["llm_error"] = str(exc)
            else:
                debug["cooldown_remaining"] = cooldown - (now - self.last_call_ts)

        if plan is None:
            plan = self._heuristic_plan(context)
            source = "heuristic"
        return ReasonerOutput(plan=plan, role=role, subgoal=subgoal, source=source, debug=debug)

    # Candidate generation -------------------------------------------------------
    def _query_candidates(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(context)
        response = self._chat_completion(prompt, context)
        if response is None:
            return []
        last_exc: Optional[Exception] = None
        for block in self._iter_json_blocks(response):
            try:
                payload = json.loads(block)
                if isinstance(payload, list):
                    return payload[: self.cfg.reasoner_candidate_topk]
                if isinstance(payload, dict) and "candidates" in payload:
                    items = payload.get("candidates", [])
                    if isinstance(items, list):
                        return items[: self.cfg.reasoner_candidate_topk]
            except Exception as exc:  # pragma: no cover
                last_exc = exc
        if last_exc is not None:
            LOGGER.warning(
                "PolicyReasoner failed to parse candidate JSON (frame=%s err=%s resp=%r)",
                context.get("frame"),
                last_exc,
                response[:200],
            )
        return []

    def _build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        description = context.get("memory_symbolic", [])
        nav_guard = []
        guard_info = context.get("nav_guard_info", {}) or {}
        if isinstance(guard_info, dict):
            for key, value in guard_info.items():
                if isinstance(key, (list, tuple)):
                    guard_key = list(key)
                else:
                    guard_key = key
                nav_guard.append({"target": guard_key, "cooldown": value})
        skip_targets = context.get("skip_targets", {"names": [], "coords": []})
        skip_text = context.get("skip_targets_text", {})
        name_strings = skip_text.get("names") or skip_targets.get("names", [])
        skip_summary = {
            "names": skip_targets.get("names", []),
            "coords": skip_targets.get("coords", []),
            "coords_str": skip_text.get("coords", []),
            "names_str": ", ".join(name_strings) if name_strings else "",
        }
        scene_bounds = context.get("scene_bounds")
        payload = {
            "frame": context.get("frame"),
            "role": context.get("role"),
            "subgoal": context.get("subgoal"),
            "holding_ids": context.get("holding_ids"),
            "current_room": context.get("current_room"),
            "memory_symbolic": description,
            "team_symbolic": context.get("team_symbolic", []),
            "partner_symbolic": context.get("partner_symbolic", {}),
            "skip_targets": skip_summary,
            "nav_guard_info": nav_guard,
            "recent_actions": context.get("recent_actions"),
            "task_type": context.get("task_type"),
            "task_targets": context.get("task_targets", []),
            "task_containers": context.get("task_containers", []),
            "grabbable_names": context.get("grabbable_names", []),
            "visible_objects": context.get("visible_objects", []),
            "goal_objects": context.get("goal_objects", {}),
        }
        if scene_bounds:
            payload["scene_bounds"] = scene_bounds
        return [
            {"role": "system", "content": self._prompt_template},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def _load_prompt_template(self) -> str:
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "LLM", "prompt_vico_lite_reasoner.txt"),
            os.path.join(os.getcwd(), "CoELA", "tdw_mat", "LLM", "prompt_vico_lite_reasoner.txt"),
        ]
        for path in candidate_paths:
            norm_path = os.path.abspath(path)
            if os.path.exists(norm_path):
                try:
                    with open(norm_path, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception as exc:
                    LOGGER.warning("PolicyReasoner failed to load prompt template (%s): %s", norm_path, exc)
        LOGGER.warning("PolicyReasoner using fallback prompt template.")
        return (
            "You are a cooperative household robot assistant. Respond with a JSON object containing a"
            " 'candidates' array. Each candidate must include action, target_id, target_pos, reason,"
            " confidence, distance, visibility, novelty, role_alignment, penalty. Avoid previously"
            " skipped targets and keep positions within bounds."
        )

    def _chat_completion(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.reasoner_model,
                messages=messages,
                temperature=self.cfg.reasoner_temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content is None:
                LOGGER.warning("PolicyReasoner received empty content (frame=%s)", (context or {}).get("frame"))
                return None
            frame = (context or {}).get("frame")
            if not self._warning_logged.get("raw_response"):
                LOGGER.warning("PolicyReasoner raw response (frame=%s snippet=%s)", frame, content[:200])
                self._warning_logged["raw_response"] = True
            return content
        except Exception as exc:  # pragma: no cover
            frame = (context or {}).get("frame", "?")
            exc_type = type(exc).__name__
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                LOGGER.warning("PolicyReasoner chat completion timeout (frame=%s, exc=%s)", frame, exc_type)
            else:
                LOGGER.warning("PolicyReasoner chat completion failed (frame=%s, exc=%s: %s)", frame, exc_type, str(exc)[:200])
            return None

    # Reflection ----------------------------------------------------------------
    def _reflect(self, plan: ReasonerPlan, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if plan.action_type not in self.cfg.reasoner_reflection_actions:
            return None
        if context.get("frame", 0) % self.cfg.reasoner_reflection_interval != 0:
            return None
        messages = [
            {
                "role": "system",
                "content": "Inspect the following plan and respond with {\"verdict\": \"approve/reject\"}.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "plan": plan.meta,
                        "context": {
                            "role": context.get("role"),
                            "subgoal": context.get("subgoal"),
                            "holding": context.get("holding_ids"),
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self._chat_completion(messages, context)
        if response is None:
            return None
        for block in self._iter_json_blocks(response):
            try:
                payload = json.loads(block)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return {"verdict": response}

    # Heuristics -----------------------------------------------------------------
    def _select_plan(self, context: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[ReasonerPlan]:
        weights = self.cfg.reasoner_heuristic_weights
        skip_spec = context.get("skip_targets", {}) or {}
        skip_names = {str(name).lower() for name in skip_spec.get("names", [])}
        skip_coords = {
            (round(float(coord[0]), 2), round(float(coord[1]), 2))
            for coord in skip_spec.get("coords", [])
            if isinstance(coord, (list, tuple)) and len(coord) >= 2
        }
        task_targets = {str(name).lower() for name in context.get("task_targets", [])}
        task_containers = {str(name).lower() for name in context.get("task_containers", [])}
        grabbable_names = {str(name).lower() for name in context.get("grabbable_names", [])}
        guard_spec = context.get("nav_guard_info", {}) or {}
        guard_coords: Set[Tuple[float, float]] = set()
        if isinstance(guard_spec, dict):
            for key in guard_spec.keys():
                if isinstance(key, (list, tuple)) and len(key) >= 2:
                    guard_coords.add((round(float(key[0]), 2), round(float(key[1]), 2)))
                elif isinstance(key, str):
                    try:
                        parts = key.strip("()[]").split(",")
                        if len(parts) >= 2:
                            guard_coords.add((round(float(parts[0]), 2), round(float(parts[1]), 2)))
                    except Exception:
                        continue
        blocked_coords = skip_coords | guard_coords
        block_radius = 0.6

        def _is_blocked(x: float, z: float) -> bool:
            qx = round(x, 2)
            qz = round(z, 2)
            if (qx, qz) in blocked_coords:
                return True
            for bx, bz in blocked_coords:
                if abs(bx - qx) <= block_radius and abs(bz - qz) <= block_radius:
                    return True
            return False
        action_bonus = {
            "pick": 6.0,
            "deliver": 6.0,
            "assist": 3.0,
            "move": 1.0,
            "search": 0.0,
            "wait": -2.0,
            "idle": -5.0,
        }
        scored: List[Tuple[float, Dict[str, Any]]] = []
        last_signature = self._last_signature
        for candidate in candidates:
            raw_action = str(candidate.get("action", "search") or "search").strip().lower()
            action = self._normalise_action(raw_action)
            if not action:
                continue
            if self.cfg.action_blacklist and action in self.cfg.action_blacklist:
                continue
            if self.cfg.action_whitelist and action not in self.cfg.action_whitelist:
                continue
            target_name = candidate.get("target_name")
            if isinstance(target_name, str) and target_name.lower() in skip_names:
                continue
            target_pos = candidate.get("target_pos") or candidate.get("target_position")
            target_tuple: Optional[Tuple[float, float, float]]
            if isinstance(target_pos, (list, tuple)) and len(target_pos) == 3:
                target_tuple = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
            else:
                target_tuple = None
            if action in {"move", "search", "deliver", "assist", "pick"} and target_tuple is None:
                continue
            if target_tuple is not None:
                coord_key = (round(float(target_tuple[0]), 2), round(float(target_tuple[2]), 2))
                # First check skip_coords (most strict)
                if coord_key in skip_coords:
                    if LOGGER and not self._warning_logged.get("skip_coord_filtered"):
                        LOGGER.debug("PolicyReasoner filtering candidate: coord %s in skip_coords", coord_key)
                        self._warning_logged["skip_coord_filtered"] = True
                    continue
                if _is_blocked(float(target_tuple[0]), float(target_tuple[2])):
                    continue
                # Check scene_bounds if available (with safety margin)
                scene_bounds = context.get("scene_bounds")
                if scene_bounds:
                    x, y, z = target_tuple
                    x_min = scene_bounds.get("x_min")
                    x_max = scene_bounds.get("x_max")
                    z_min = scene_bounds.get("z_min")
                    z_max = scene_bounds.get("z_max")
                    # Add safety margin to prevent going near boundaries (increased to 1.0m for better safety)
                    margin = 1.0
                    if (x_min is not None and x_max is not None and (x < x_min + margin or x > x_max - margin)) or \
                       (z_min is not None and z_max is not None and (z < z_min + margin or z > z_max - margin)):
                        # Filter out candidates outside or too close to scene bounds
                        if LOGGER and not self._warning_logged.get("scene_bounds_filtered"):
                            LOGGER.debug(
                                "PolicyReasoner filtering candidate: %s outside scene_bounds (x=[%s, %s], z=[%s, %s])",
                                target_tuple,
                                x_min, x_max, z_min, z_max,
                            )
                            self._warning_logged["scene_bounds_filtered"] = True
                        continue
            name_lc = str(target_name).lower() if isinstance(target_name, str) else ""
            is_task_target = name_lc in task_targets
            is_task_container = name_lc in task_containers
            is_grabbable = name_lc in grabbable_names or is_task_target
            distance = self._safe_float(candidate.get("distance"), 1.0)
            visibility = self._safe_float(candidate.get("visibility"), 0.5)
            novelty = self._safe_float(candidate.get("novelty"), 0.5)
            alignment = self._safe_float(candidate.get("role_alignment"), 0.5)
            penalty = self._safe_float(candidate.get("penalty"), 0.0)
            score = (
                weights.get("distance", 1.0) * (1.0 / (distance + 1e-3))
                + weights.get("visibility", 0.5) * visibility
                + weights.get("novelty", 0.5) * novelty
                + weights.get("role_alignment", 0.5) * alignment
                - weights.get("penalty", 1.0) * penalty
            )
            score += action_bonus.get(action, 0.0)
            if action == "pick":
                if not isinstance(target_name, str):
                    score -= 3.0
                    continue
                if is_task_target:
                    score += 5.0
                elif is_grabbable:
                    score += 2.5
                else:
                    score -= 2.0
                score -= 0.3 * max(distance - 1.2, 0.0)
            elif action == "deliver":
                if is_task_container:
                    score += 3.0
                else:
                    score -= 1.0
            elif action == "move":
                if is_task_target or is_task_container:
                    score += 1.0
                elif name_lc:
                    score -= 0.4
            if action == "search":
                score -= 0.4 * max(distance - 3.0, 0.0)
            if last_signature is not None:
                last_action, last_target = last_signature
                if action == last_action:
                    if action in {"search", "move"} and target_tuple is not None and last_target is not None:
                        lt = (round(last_target[0], 2), round(last_target[2], 2))
                        ct = (round(target_tuple[0], 2), round(target_tuple[2], 2))
                        if lt == ct:
                            score -= 4.0
                    elif action in {"pick", "deliver"}:
                        score += 1.0
                    elif action == "wait":
                        score -= 3.0
            candidate["action"] = action
            candidate["_score"] = score
            candidate["_target_tuple"] = target_tuple
            scored.append((score, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        has_non_wait = any(item[1].get("action") not in {"wait", "idle"} for item in scored)
        chosen: Optional[Dict[str, Any]] = None
        for _, candidate in scored:
            action = str(candidate.get("action", "search"))
            if has_non_wait and action in {"wait", "idle"}:
                continue
            target_tuple = candidate.get("_target_tuple")
            if action in {"move", "search", "deliver", "assist", "pick"} and target_tuple is None:
                continue
            chosen = candidate
            break
        if chosen is None:
            return None
        action = str(chosen.get("action", "search"))
        target_tuple = chosen.get("_target_tuple")
        if "_target_tuple" in chosen:
            chosen = dict(chosen)
            chosen.pop("_target_tuple", None)
        plan = ReasonerPlan(
            action_type=action,
            target_id=chosen.get("target_id"),
            target_position=target_tuple,  # type: ignore[arg-type]
            confidence=float(chosen.get("confidence", 0.6)),
            meta=chosen,
        )
        self._last_signature = (plan.action_type, plan.target_position)
        return plan

    def _heuristic_plan(self, context: Dict[str, Any]) -> ReasonerPlan:
        symbolic = context.get("memory_symbolic", [])
        skip_spec = context.get("skip_targets", {}) or {}
        skip_names = {str(name).lower() for name in skip_spec.get("names", [])}
        task_targets = {str(name).lower() for name in context.get("task_targets", [])}
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for entry in symbolic:
            for candidate in entry.get("candidates", []):
                name = candidate.get("name")
                score = float(candidate.get("score", 0.0))
                if isinstance(name, str) and name.lower() in skip_names:
                    continue
                distance = float(candidate.get("distance", 1.0))
                heur_score = self._score_candidate(distance=distance, similarity=score)
                # prefer task targets
                if isinstance(name, str) and name.lower() in task_targets:
                    heur_score += 2.0
                elif task_targets:
                    heur_score -= 0.5
                candidates.append((heur_score, {"action": "search", "target_name": name, "distance": distance}))
        if not candidates:
            return ReasonerPlan("search", None, None, 0.3, {"reason": "fallback"})
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        meta = {
            "reason": "heuristic",
            "score": candidates[0][0],
            "target_name": best.get("target_name"),
        }
        return ReasonerPlan(action_type=best["action"], target_id=None, target_position=None, confidence=0.5, meta=meta)

    def _score_candidate(self, distance: float, similarity: float) -> float:
        weights = self.cfg.reasoner_heuristic_weights
        distance_penalty = weights.get("distance", 1.0) * (1.0 / (distance + 1e-3))
        novelty = weights.get("novelty", 0.5) * similarity
        visibility = weights.get("visibility", 0.5) * similarity
        return distance_penalty + novelty + visibility

    # JSON helpers ---------------------------------------------------------------
    def _iter_json_blocks(self, response: Optional[str]) -> Iterable[str]:
        if not response:
            return []
        text = response.strip()
        matches = list(self._JSON_FENCE_RE.finditer(text))
        if matches:
            return [match.group(1).strip() for match in matches]
        return [text]

    def _safe_float(self, value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                lowered = value.strip().lower()
                qualitative = {
                    "high": 0.9,
                    "medium": 0.5,
                    "low": 0.2,
                    "near": 0.5,
                    "close": 0.3,
                    "far": 1.5,
                    "visible": 0.8,
                    "clear": 0.7,
                    "unknown": default,
                }
                if lowered in qualitative:
                    return qualitative[lowered]
        return default

    def _normalise_action(self, action: str) -> str:
        if not action:
            return ""
        aliases = {
            "scan": "search",
            "explore": "search",
            "inspect": "search",
            "observe": "search",
            "survey": "search",
            "investigate": "search",
            "look": "search",
            "look_around": "search",
            "move_to": "move",
            "go_to": "move",
            "goto": "move",
            "navigate": "move",
            "walk": "move",
            "approach": "move",
            "stay": "wait",
            "hold": "wait",
        }
        normalized = aliases.get(action, action)
        return normalized


__all__ = ["PolicyReasoner", "ReasonerPlan", "ReasonerOutput"]
