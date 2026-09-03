"""Reward-ranked candidate selection for stage 2 (RFT).

* DPO consumes ``(chosen, rejected)`` pairs and updates the policy with a pairwise
  objective. Removing it removes that objective.
* What stage 2 actually needs is a *filter*: sample the policy several times, score every
  candidate with the composite reward, keep only those that pass the shared criterion, and
  fine-tune on the survivors with ordinary cross-entropy. The rejected candidates are
  simply dropped -- they never enter a loss term at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import EvaluationConfig, TrainingConfig
from ..eval.pass_criteria import PassOutcome, evaluate_sample

LOGGER = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One sampled output with everything needed to judge and trace it."""

    prompt_id: str
    text: str
    product: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    reward_detail: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[PassOutcome] = None
    #: Provenance, carried through to the augmented QA manifest.
    dataset: str = ""
    video: str = ""
    event_index: int = 0
    interval: Tuple[int, int] = (0, 0)

    @property
    def passed(self) -> bool:
        return bool(self.outcome and self.outcome.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id, "reward": round(self.reward, 5),
            "passed": self.passed, "dataset": self.dataset, "video": self.video,
            "event_index": self.event_index, "interval": list(self.interval),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "reward_detail": self.reward_detail,
        }


@dataclass
class FilterReport:
    """What the filter admitted, and what it threw away and why."""

    n_candidates: int = 0
    n_passed: int = 0
    n_accepted: int = 0
    n_prompts: int = 0
    n_prompts_with_accept: int = 0
    rejected_reasons: Dict[str, int] = field(default_factory=dict)
    accepted_reward_mean: float = 0.0
    min_reward: float = 0.6
    top_k: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_candidates": self.n_candidates, "n_passed": self.n_passed,
            "n_accepted": self.n_accepted, "n_prompts": self.n_prompts,
            "n_prompts_with_accept": self.n_prompts_with_accept,
            "accept_rate": round(self.n_accepted / self.n_candidates, 4)
            if self.n_candidates else 0.0,
            "accepted_reward_mean": round(self.accepted_reward_mean, 4),
            "min_reward": self.min_reward, "top_k": self.top_k,
            "rejected_reasons": dict(sorted(
                self.rejected_reasons.items(), key=lambda kv: -kv[1])[:10]),
        }


def score_candidates(
    candidates: Sequence[Candidate],
    truth_by_prompt: Dict[str, Dict[str, Any]],
    evaluation: Optional[EvaluationConfig] = None,
    required_fields: Sequence[str] = (),
) -> List[Candidate]:
    """Attach a :class:`PassOutcome` to every candidate, in place."""
    evaluation = evaluation or EvaluationConfig()
    for candidate in candidates:
        truth = truth_by_prompt.get(candidate.prompt_id, {})
        candidate.outcome = evaluate_sample(
            candidate.product, truth, evaluation, required_fields)
    return list(candidates)


def select(
    candidates: Sequence[Candidate],
    config: Optional[TrainingConfig] = None,
) -> Tuple[List[Candidate], FilterReport]:
    """Rank each prompt's candidates by reward and admit the qualifying ones.

    Returns ``(accepted, report)``. Ordering inside a prompt is by reward descending, with
    the pass flag as the primary key -- a passing candidate always outranks a
    higher-reward failing one, so the cap on ``rft_accept_top_k`` can never spend its
    budget on failures while a pass sits below the line.
    """
    config = config or TrainingConfig()
    report = FilterReport(min_reward=config.rft_min_reward,
                          top_k=config.rft_accept_top_k)

    by_prompt: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        by_prompt.setdefault(candidate.prompt_id, []).append(candidate)

    report.n_candidates = len(candidates)
    report.n_prompts = len(by_prompt)
    accepted: List[Candidate] = []

    for prompt_id, group in by_prompt.items():
        group.sort(key=lambda c: (c.passed, c.reward), reverse=True)
        admitted = 0
        for candidate in group:
            if candidate.passed:
                report.n_passed += 1
            if admitted >= config.rft_accept_top_k:
                report.rejected_reasons["over top-k for this prompt"] = (
                    report.rejected_reasons.get("over top-k for this prompt", 0) + 1)
                continue
            if not candidate.passed:
                reason = (candidate.outcome.reasons[0]
                          if candidate.outcome and candidate.outcome.reasons
                          else "did not pass the criterion")
                report.rejected_reasons[reason] = (
                    report.rejected_reasons.get(reason, 0) + 1)
                continue
            if candidate.reward < config.rft_min_reward:
                key = f"reward below {config.rft_min_reward}"
                report.rejected_reasons[key] = report.rejected_reasons.get(key, 0) + 1
                continue
            accepted.append(candidate)
            admitted += 1
        if admitted:
            report.n_prompts_with_accept += 1

    if accepted:
        report.accepted_reward_mean = float(np.mean([c.reward for c in accepted]))
    report.n_accepted = len(accepted)

    LOGGER.info("candidate filter: %d/%d accepted across %d/%d prompts",
                report.n_accepted, report.n_candidates,
                report.n_prompts_with_accept, report.n_prompts)
    return accepted, report


def group_rewards(candidates: Sequence[Candidate]) -> List[List[float]]:
    """Per-prompt reward groups, in the shape the reward-spread diagnostic wants."""
    by_prompt: Dict[str, List[float]] = {}
    for candidate in candidates:
        by_prompt.setdefault(candidate.prompt_id, []).append(candidate.reward)
    return list(by_prompt.values())


def group_outcomes(candidates: Sequence[Candidate]) -> Dict[str, List[PassOutcome]]:
    """Per-prompt outcomes, in the shape the pass@k diagnostic wants."""
    by_prompt: Dict[str, List[PassOutcome]] = {}
    for candidate in candidates:
        if candidate.outcome is not None:
            by_prompt.setdefault(candidate.prompt_id, []).append(candidate.outcome)
    return by_prompt


__all__ = [
    "Candidate", "FilterReport", "score_candidates", "select",
    "group_rewards", "group_outcomes",
]
