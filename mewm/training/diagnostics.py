"""SFT sufficiency diagnostics: is the policy trained enough to hand over to RL?

**1a Format alignment.** Does the policy follow the instruction contract -- reason in the
required order and emit parseable JSON? Measured as the share of sampled outputs that
satisfy :mod:`mewm.eval.pass_criteria.check_format`.

**1b Loss plateau.** Has the curve stopped moving? Judged on the trailing window by two
statistics together: the slope (is it still descending?) and the coefficient of variation
(has the noise settled?). Either one alone is misleading -- a curve can be flat on average
while oscillating, and it can be smooth while still descending steadily.

**2a pass@k headroom.** ``pass@k >= pass@1`` holds by construction for any sampler, so the
inequality itself carries no information. What matters is the *gap*: a large gap means the
policy can already produce a passing answer but does not rank it first, which is precisely
what a policy-gradient update fixes. A small gap at a low rate means the capability is
absent and more SFT is the answer; a small gap at a high rate means saturation.

**2b Reward spread.** A group of samples with no spread carries no preference information
*whether the rewards are all low or all high*, because the group-relative advantage
``(R - mean)/std`` is then identically zero and the update is a no-op. All-low means SFT is
undertrained. All-high means the prompt is exhausted and should leave the RL prompt set.
Only a spread group is trainable.

**On what these numbers are measured over.** When ``TrainingConfig.n_val_subjects`` is 0 --
the configured default -- there is no held-out subject and every statistic here is computed
on the same material the policy was fitted to. Such a report carries ``in_sample=True`` and
every consumer prints it, because an in-sample plateau is evidence of memorisation rather
than convergence and an in-sample pass@1 is optimistic. The arithmetic does not change; the
interpretation does.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import TrainingConfig
from ..eval.pass_criteria import PassOutcome, stage_rates

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1a -- format alignment
# ---------------------------------------------------------------------------


@dataclass
class FormatAlignment:
    """Share of sampled outputs that satisfy the instruction contract."""

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
    """Judgement 1a."""
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


# ---------------------------------------------------------------------------
# 1b -- loss plateau
# ---------------------------------------------------------------------------


@dataclass
class PlateauReport:
    """Whether the loss curve has stopped moving."""

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
        """A plateau needs both a flat trend and settled noise."""
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
    """Judgement 1b.

    The slope is a least-squares fit over the trailing window, normalised per step so the
    threshold does not depend on how long training ran. ``still_descending`` is reported
    separately from ``ok``: a curve that is descending faster than the tolerance has not
    plateaued, and stopping there would leave capability on the table.
    """
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


# ---------------------------------------------------------------------------
# 2a -- pass@k
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` from ``c`` passing samples out of ``n`` draws.

    ``1 - C(n-c, k) / C(n, k)``, evaluated as a product to stay stable for large ``n``.
    The naive alternative -- draw k samples and check -- is a high-variance estimate of
    the same quantity, and at the sample counts available here the variance dominates the
    signal the gate is trying to read.
    """
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
    """pass@1 against pass@k: is there anything for RL to sharpen?"""

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
        """Headroom exists: the policy can pass but does not rank the pass first."""
        return self.n_prompts > 0 and self.gap >= self.gap_min and self.pass_k >= self.floor

    @property
    def verdict(self) -> str:
        if self.n_prompts == 0:
            return "no_data"
        if self.pass_k < self.floor:
            return "undertrained"          # cannot pass even with k tries
        if self.gap < self.gap_min:
            return "saturated"            # already ranks its best answer first
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
    """Judgement 2a, averaged over prompts."""
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
    # Surface the prompts with the widest gap first: those are the ones RL can move.
    report.per_prompt.sort(key=lambda row: row["pass_1"] - row["pass_k"])
    return report


# ---------------------------------------------------------------------------
# 2b -- reward distribution
# ---------------------------------------------------------------------------


@dataclass
class RewardDistribution:
    """Three-way verdict over per-prompt reward groups."""

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
        """``undertrained`` / ``saturated`` / ``rl_ready``.

        Decided on the *majority* of groups rather than the pooled mean: a pool whose
        mean sits in range can still be made entirely of degenerate groups, half of them
        at the floor and half at the ceiling, and that pool trains nothing.
        """
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
    """One group's contribution: ``spread`` / ``all_low`` / ``all_high``.
    """
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
    """Judgement 2b."""
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


# ---------------------------------------------------------------------------
# Composite gate
# ---------------------------------------------------------------------------


@dataclass
class SufficiencyReport:
    """All four judgements plus the decision they imply."""

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
        """``continue_sft`` / ``rl_ready`` / ``saturated``.
        """
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
        """Why the decision came out the way it did."""
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
    """Run all four judgements and compose the gate decision."""
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
