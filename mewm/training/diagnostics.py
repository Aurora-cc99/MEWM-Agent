"""Training diagnostics: loss curves, reward histograms, and fold summaries."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import TrainingConfig
from ..eval.pass_criteria import PassOutcome, stage_rates

LOGGER = logging.getLogger(__name__)


@dataclass
class FormatAlignment:

    rate: float = 0.0
    n: int = 0
    parse_failures: int = 0
    contract_failures: int = 0
    examples: List[str] = field(default_factory=list)
    threshold: float = 0.95

    @property
    def ok(self) -> bool:
        return self.n > 0 and self.rate >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rate": round(self.rate, 4), "n": self.n, "ok": self.ok,
            "threshold": self.threshold, "parse_failures": self.parse_failures,
            "contract_failures": self.contract_failures,
            "examples": self.examples[:3],
        }


def format_alignment(outcomes: Sequence[PassOutcome],
                     threshold: float = 0.95) -> FormatAlignment:
    report = FormatAlignment(n=len(outcomes), threshold=threshold)
    if not outcomes:
        return report
    passed = 0
    for outcome in outcomes:
        if outcome.format_ok:
            passed += 1
            continue
        parse_failed = any("did not parse" in r for r in outcome.reasons)
        if parse_failed:
            report.parse_failures += 1
        else:
            report.contract_failures += 1
        if len(report.examples) < 3 and outcome.reasons:
            report.examples.append(outcome.reasons[0])
    report.rate = passed / len(outcomes)
    return report


@dataclass
class PlateauReport:

    slope: float = 0.0
    cv: float = 0.0
    window: int = 0
    final: float = 0.0
    best: float = 0.0
    max_slope: float = 1e-3
    max_cv: float = 0.05
    still_descending: bool = False

    @property
    def ok(self) -> bool:
        return (self.window >= 4 and abs(self.slope) <= self.max_slope
                and self.cv <= self.max_cv)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "slope": round(self.slope, 8), "cv": round(self.cv, 6),
            "window": self.window, "final_loss": round(self.final, 6),
            "best_loss": round(self.best, 6),
            "still_descending": self.still_descending,
            "max_slope": self.max_slope, "max_cv": self.max_cv,
        }


def loss_plateau(
    losses: Sequence[float], window_ratio: float = 0.2,
    max_slope: float = 1e-3, max_cv: float = 0.05,
) -> PlateauReport:
    values = np.asarray([float(v) for v in losses if np.isfinite(v)], dtype=np.float64)
    report = PlateauReport(max_slope=max_slope, max_cv=max_cv)
    if values.size == 0:
        return report

    report.final = float(values[-1])
    report.best = float(values.min())

    span = max(4, int(round(values.size * float(np.clip(window_ratio, 0.05, 1.0)))))
    span = min(span, values.size)
    tail = values[-span:]
    report.window = int(span)

    if span >= 2:
        steps = np.arange(span, dtype=np.float64)
        slope = float(np.polyfit(steps, tail, 1)[0])
        report.slope = slope
        report.still_descending = slope < -max_slope

    mean = float(np.mean(np.abs(tail)))
    report.cv = float(np.std(tail) / mean) if mean > 1e-12 else 0.0
    return report


def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0 or n <= 0:
        return 0.0
    k = min(k, n)
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return float(np.clip(1.0 - product, 0.0, 1.0))


@dataclass
class HeadroomReport:

    pass_1: float = 0.0
    pass_k: float = 0.0
    k: int = 8
    n_prompts: int = 0
    samples_per_prompt: float = 0.0
    gap_min: float = 0.15
    floor: float = 0.5
    per_prompt: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def gap(self) -> float:
        return max(0.0, self.pass_k - self.pass_1)

    @property
    def ok(self) -> bool:
        return self.n_prompts > 0 and self.gap >= self.gap_min and self.pass_k >= self.floor

    @property
    def verdict(self) -> str:
        if self.n_prompts == 0:
            return "no_data"
        if self.pass_k < self.floor:
            return "undertrained"
        if self.gap < self.gap_min:
            return "saturated"
        return "rl_ready"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_1": round(self.pass_1, 4), "pass_k": round(self.pass_k, 4),
            "k": self.k, "gap": round(self.gap, 4), "gap_min": self.gap_min,
            "floor": self.floor, "verdict": self.verdict, "ok": self.ok,
            "n_prompts": self.n_prompts,
            "samples_per_prompt": round(self.samples_per_prompt, 2),
            "per_prompt": self.per_prompt[:10],
        }


def headroom(
    per_prompt_outcomes: Dict[str, Sequence[PassOutcome]],
    k: int = 8, gap_min: float = 0.15, floor: float = 0.5,
) -> HeadroomReport:
    report = HeadroomReport(k=k, gap_min=gap_min, floor=floor)
    ones: List[float] = []
    ks: List[float] = []
    counts: List[int] = []

    for prompt_id, outcomes in per_prompt_outcomes.items():
        n = len(outcomes)
        if n == 0:
            continue
        c = sum(1 for o in outcomes if o.passed)
        p1 = pass_at_k(n, c, 1)
        pk = pass_at_k(n, c, k)
        ones.append(p1)
        ks.append(pk)
        counts.append(n)
        report.per_prompt.append({
            "prompt_id": prompt_id, "n": n, "c": c,
            "pass_1": round(p1, 4), "pass_k": round(pk, 4),
        })

    if not ones:
        return report
    report.pass_1 = float(np.mean(ones))
    report.pass_k = float(np.mean(ks))
    report.n_prompts = len(ones)
    report.samples_per_prompt = float(np.mean(counts))
    report.per_prompt.sort(key=lambda row: row["pass_1"] - row["pass_k"])
    return report


@dataclass
class RewardDistribution:

    n_groups: int = 0
    mean: float = 0.0
    pooled_std: float = 0.0
    mean_group_std: float = 0.0
    quantiles: Dict[str, float] = field(default_factory=dict)
    n_all_low: int = 0
    n_all_high: int = 0
    n_spread: int = 0
    std_min: float = 0.05
    mean_low: float = 0.20
    mean_high: float = 0.85

    @property
    def verdict(self) -> str:
        if self.n_groups == 0:
            return "no_data"
        if self.n_spread >= max(1, self.n_groups // 2):
            return "rl_ready"
        if self.n_all_low > self.n_all_high:
            return "undertrained"
        return "saturated"

    @property
    def ok(self) -> bool:
        return self.verdict == "rl_ready"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict, "ok": self.ok, "n_groups": self.n_groups,
            "mean": round(self.mean, 4), "pooled_std": round(self.pooled_std, 4),
            "mean_group_std": round(self.mean_group_std, 4),
            "quantiles": self.quantiles,
            "n_all_low": self.n_all_low, "n_all_high": self.n_all_high,
            "n_spread": self.n_spread,
            "std_min": self.std_min,
            "mean_band": [self.mean_low, self.mean_high],
        }


def classify_group(rewards: Sequence[float], std_min: float = 0.05,
                   mean_low: float = 0.20, mean_high: float = 0.85) -> str:
    values = np.asarray([float(r) for r in rewards], dtype=np.float64)
    if values.size == 0:
        return "all_low"
    spread = float(values.std())
    mean = float(values.mean())
    if spread >= std_min and mean_low <= mean <= mean_high:
        return "spread"
    return "all_high" if mean > mean_high else "all_low"


def reward_distribution(
    groups: Sequence[Sequence[float]], std_min: float = 0.05,
    mean_low: float = 0.20, mean_high: float = 0.85,
) -> RewardDistribution:
    report = RewardDistribution(std_min=std_min, mean_low=mean_low, mean_high=mean_high)
    populated = [list(g) for g in groups if len(g) > 0]
    if not populated:
        return report

    report.n_groups = len(populated)
    pooled = np.asarray([r for g in populated for r in g], dtype=np.float64)
    report.mean = float(pooled.mean())
    report.pooled_std = float(pooled.std())
    report.mean_group_std = float(np.mean([np.std(g) for g in populated]))
    report.quantiles = {
        "p10": round(float(np.percentile(pooled, 10)), 4),
        "p50": round(float(np.percentile(pooled, 50)), 4),
        "p90": round(float(np.percentile(pooled, 90)), 4),
    }

    for group in populated:
        kind = classify_group(group, std_min, mean_low, mean_high)
        if kind == "spread":
            report.n_spread += 1
        elif kind == "all_low":
            report.n_all_low += 1
        else:
            report.n_all_high += 1
    return report


@dataclass
class SufficiencyReport:

    fold: str = ""
    round_index: int = 0
    in_sample: bool = True
    measured_on: str = "training_pool"
    format: Optional[FormatAlignment] = None
    plateau: Optional[PlateauReport] = None
    headroom: Optional[HeadroomReport] = None
    rewards: Optional[RewardDistribution] = None
    stage_rates: Dict[str, float] = field(default_factory=dict)

    @property
    def decision(self) -> str:
        if self.format is not None and not self.format.ok:
            return "continue_sft"
        if self.plateau is not None and self.plateau.still_descending:
            return "continue_sft"

        head = self.headroom.verdict if self.headroom else "no_data"
        spread = self.rewards.verdict if self.rewards else "no_data"

        if "undertrained" in (head, spread):
            return "continue_sft"
        if "rl_ready" in (head, spread):
            return "rl_ready"
        if head == "saturated" and spread == "saturated":
            return "saturated"
        return "continue_sft"

    def reasons(self) -> List[str]:
        out: List[str] = []
        if self.format is not None and not self.format.ok:
            out.append(f"format alignment {self.format.rate:.3f} < "
                       f"{self.format.threshold:.2f}")
        if self.plateau is not None and self.plateau.still_descending:
            out.append(f"loss still descending (slope {self.plateau.slope:.2e})")
        if self.headroom is not None:
            out.append(f"pass@1 {self.headroom.pass_1:.3f} -> pass@{self.headroom.k} "
                       f"{self.headroom.pass_k:.3f} (gap {self.headroom.gap:.3f}): "
                       f"{self.headroom.verdict}")
        if self.rewards is not None:
            out.append(f"reward groups: {self.rewards.n_spread} spread / "
                       f"{self.rewards.n_all_low} all-low / "
                       f"{self.rewards.n_all_high} all-high: {self.rewards.verdict}")
        if self.in_sample:
            out.append("measured in-sample on the training pool: a plateau here indicates "
                       "memorisation rather than convergence, and pass@1 is optimistic")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold, "round": self.round_index,
            "decision": self.decision, "reasons": self.reasons(),
            "in_sample": self.in_sample, "measured_on": self.measured_on,
            "format_alignment": self.format.to_dict() if self.format else None,
            "loss_plateau": self.plateau.to_dict() if self.plateau else None,
            "headroom": self.headroom.to_dict() if self.headroom else None,
            "reward_distribution": self.rewards.to_dict() if self.rewards else None,
            "stage_rates": self.stage_rates,
        }


def assess(
    outcomes_by_prompt: Dict[str, Sequence[PassOutcome]],
    reward_groups: Sequence[Sequence[float]],
    losses: Sequence[float],
    config: Optional[TrainingConfig] = None,
    fold: str = "",
    round_index: int = 0,
    in_sample: bool = True,
) -> SufficiencyReport:
    config = config or TrainingConfig()
    flat = [o for outcomes in outcomes_by_prompt.values() for o in outcomes]

    report = SufficiencyReport(
        fold=fold, round_index=round_index, in_sample=in_sample,
        measured_on="training_pool" if in_sample else "held_out_subjects",
        format=format_alignment(flat, config.format_pass_rate),
        plateau=loss_plateau(losses, config.plateau_window,
                             config.plateau_max_slope, config.plateau_max_cv),
        headroom=headroom(outcomes_by_prompt, config.pass_k,
                          config.pass_gap_min, config.pass_at_k_floor),
        rewards=reward_distribution(reward_groups, config.reward_std_min,
                                    config.reward_mean_low, config.reward_mean_high),
        stage_rates=stage_rates(flat),
    )
    LOGGER.info("fold %s round %d: %s (%s)", fold, round_index, report.decision,
                "; ".join(report.reasons()))
    return report


__all__ = [
    "FormatAlignment", "format_alignment", "PlateauReport", "loss_plateau",
    "pass_at_k", "HeadroomReport", "headroom", "RewardDistribution",
    "classify_group", "reward_distribution", "SufficiencyReport", "assess",
]
