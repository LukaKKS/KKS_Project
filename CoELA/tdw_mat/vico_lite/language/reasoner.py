from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
            client = OpenAI(api_key=api_key)
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
        response = self._chat_completion(prompt)
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
            LOGGER.warning("Failed to parse candidate JSON (%s)", last_exc)
        return []

    def _build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        description = context.get("memory_symbolic", [])
        skip_targets = list(context.get("skip_targets", []))
        prompt = (
            "You are a cooperative household robot assistant. "
            "Plan the next high-level action in JSON with fields action, target_id, target_pos, reason, confidence."
        )
        if skip_targets:
            prompt += f" Avoid targets: {skip_targets}."
        prompt += " Use short reasoning."
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "frame": context.get("frame"),
                        "role": context.get("role"),
                        "subgoal": context.get("subgoal"),
                        "holding_ids": context.get("holding_ids"),
                        "memory": description,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _chat_completion(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.reasoner_model,
                messages=messages,
                temperature=self.cfg.reasoner_temperature,
            )
            content = response.choices[0].message.content
            return content
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("PolicyReasoner chat completion failed (%s)", exc)
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
        response = self._chat_completion(messages)
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
        skip_targets = set(context.get("skip_targets", []))
        best_score = -float("inf")
        best_candidate: Optional[Dict[str, Any]] = None
        for candidate in candidates:
            action = str(candidate.get("action", "search"))
            if self.cfg.action_blacklist and action in self.cfg.action_blacklist:
                continue
            if self.cfg.action_whitelist and action not in self.cfg.action_whitelist:
                continue
            target_name = candidate.get("target_name")
            if target_name in skip_targets:
                continue
            distance = float(candidate.get("distance", 1.0) or 1.0)
            visibility = float(candidate.get("visibility", 0.5) or 0.5)
            novelty = float(candidate.get("novelty", 0.5) or 0.5)
            alignment = float(candidate.get("role_alignment", 0.5) or 0.5)
            penalty = float(candidate.get("penalty", 0.0) or 0.0)
            score = (
                weights.get("distance", 1.0) * (1.0 / (distance + 1e-3))
                + weights.get("visibility", 0.5) * visibility
                + weights.get("novelty", 0.5) * novelty
                + weights.get("role_alignment", 0.5) * alignment
                - weights.get("penalty", 1.0) * penalty
            )
            candidate["_score"] = score
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate is None:
            return None
        target_pos = best_candidate.get("target_pos") or best_candidate.get("target_position")
        if isinstance(target_pos, list) and len(target_pos) == 3:
            target_tuple = tuple(float(x) for x in target_pos)  # type: ignore[assignment]
        elif isinstance(target_pos, tuple) and len(target_pos) == 3:
            target_tuple = tuple(float(x) for x in target_pos)
        else:
            target_tuple = None
        plan = ReasonerPlan(
            action_type=str(best_candidate.get("action", "search")),
            target_id=best_candidate.get("target_id"),
            target_position=target_tuple,  # type: ignore[arg-type]
            confidence=float(best_candidate.get("confidence", 0.6)),
            meta=best_candidate,
        )
        return plan

    def _heuristic_plan(self, context: Dict[str, Any]) -> ReasonerPlan:
        symbolic = context.get("memory_symbolic", [])
        skip_targets = set(context.get("skip_targets", []))
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for entry in symbolic:
            for candidate in entry.get("candidates", []):
                name = candidate.get("name")
                score = float(candidate.get("score", 0.0))
                if name in skip_targets:
                    continue
                distance = float(candidate.get("distance", 1.0))
                heur_score = self._score_candidate(distance=distance, similarity=score)
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


__all__ = ["PolicyReasoner", "ReasonerPlan", "ReasonerOutput"]
