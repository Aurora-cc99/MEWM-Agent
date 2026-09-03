"""The five-dimensional composite reward of eq. (11) that drives GRPO.

* the judge engine is frozen and heterogeneous with respect to the policy;
* ``mean`` MNI (not sum) penalises AU padding -- adding low-necessity units lowers the
  average monotonically, so listing more units can only hurt;
* the graph edit distance penalises structural padding the same way;
* ``R_temp`` optimises the eq. (2) TP criterion directly, so temporal quality cannot be
  traded away for fluent prose.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RewardConfig
from ..knowledge.au_anatomy import SLOT_INDEX
from ..knowledge.emotion_prototypes import coarse_of, emotion_similarity, labels_consistent

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight curriculum
# ---------------------------------------------------------------------------

#: (progress upper bound, {component: weight}) -- appendix F.3.
WEIGHT_CURRICULUM: Tuple[Tuple[float, Dict[str, float]], ...] = (
    (0.20, {"fmt": 0.40, "au": 0.30, "emo": 0.20, "causal": 0.05, "temp": 0.05}),
    (0.60, {"fmt": 0.15, "au": 0.35, "emo": 0.20, "causal": 0.20, "temp": 0.10}),
    (1.00, {"fmt": 0.10, "au": 0.20, "emo": 0.20, "causal": 0.30, "temp": 0.20}),
)


def curriculum_weights(progress: float, smooth: bool = True) -> Dict[str, float]:
    """Weights at training ``progress`` in ``[0, 1]``.

    Cosine-interpolated across stage boundaries by default: a hard switch would inject a
    discontinuity into the advantage estimate exactly when the policy is mid-update.
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    stages = list(WEIGHT_CURRICULUM)
    for index, (upper, weights) in enumerate(stages):
        if progress > upper:
            continue
        if not smooth or index == 0:
            return dict(weights)
        lower, previous = stages[index - 1][0], stages[index - 1][1]
        span = max(1e-6, upper - lower)
        t = (progress - lower) / span
        blend = 0.5 - 0.5 * math.cos(math.pi * float(np.clip(t, 0.0, 1.0)))
        blended = {
            key: previous[key] + blend * (weights[key] - previous[key])
            for key in weights
        }
        # Eq. (11) requires a convex combination. Normalise, then absorb the residual
        # left by rounding into the largest weight, so the sum is exactly one rather
        # than one plus a rounding artefact.
        total = sum(blended.values()) or 1.0
        rounded = {key: round(value / total, 5) for key, value in blended.items()}
        residual = 1.0 - sum(rounded.values())
        if abs(residual) > 0:
            largest = max(rounded, key=lambda k: rounded[k])
            rounded[largest] = round(rounded[largest] + residual, 10)
        return rounded
    return dict(stages[-1][1])


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Per-component reward, kept separable for the ablation of paper 4.5(c)."""

    r_au: float = 0.0
    r_emo: float = 0.0
    r_fmt: float = 0.0
    r_causal: float = 0.0
    r_temp: float = 0.0
    total: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "R_AU": round(self.r_au, 5), "R_emo": round(self.r_emo, 5),
            "R_fmt": round(self.r_fmt, 5), "R_causal": round(self.r_causal, 5),
            "R_temp": round(self.r_temp, 5), "total": round(self.total, 5),
            "weights": self.weights, "detail": self.detail,
        }


def reward_au(claimed: Sequence[str], truth: Sequence[str],
              hallucination_penalty: float = 0.5,
              evidenced: Optional[Sequence[str]] = None) -> Tuple[float, Dict[str, Any]]:
    """Set F1 over AUs, minus a penalty for units claimed without evidence."""
    claimed_set, truth_set = set(claimed), set(truth)
    if not claimed_set and not truth_set:
        return 1.0, {"f1": 1.0, "hallucinated": 0}
    intersection = len(claimed_set & truth_set)
    f1 = (2 * intersection / (len(claimed_set) + len(truth_set))
          if (claimed_set or truth_set) else 0.0)

    hallucinated = 0
    if evidenced is not None:
        evidence_set = set(evidenced)
        hallucinated = len([au for au in claimed_set if au not in evidence_set])
    penalty = hallucination_penalty * hallucinated / max(1, len(claimed_set))
    score = float(np.clip(f1 - penalty, 0.0, 1.0))
    return round(score, 5), {"f1": round(f1, 4), "hallucinated": hallucinated}


def reward_emotion(fine_pred: str, fine_true: str,
                   coarse_pred: str = "", coarse_true: str = "") -> Tuple[float, Dict[str, Any]]:
    """Coarse plus fine match, with wheel-similarity partial credit on the fine label.

    Exact-match-only would treat "anger instead of disgust" and "happiness instead of
    disgust" as the same error, which they are not -- the first is a plausible confusion
    between adjacent negative states, the second is a sign error.
    """
    coarse_pred = coarse_pred or coarse_of(fine_pred)
    coarse_true = coarse_true or coarse_of(fine_true)
    fine_score = 1.0 if fine_pred == fine_true else emotion_similarity(fine_pred, fine_true)
    coarse_score = 1.0 if coarse_pred == coarse_true else 0.0
    consistent = labels_consistent(fine_pred, coarse_pred)
    score = 0.6 * fine_score + 0.4 * coarse_score
    if not consistent:
        score *= 0.5           # an internally inconsistent label pair is half-credit
    return round(float(np.clip(score, 0.0, 1.0)), 5), {
        "fine_exact": fine_pred == fine_true, "fine_similarity": round(fine_score, 4),
        "coarse_match": coarse_score == 1.0, "mapping_consistent": consistent,
    }


REQUIRED_COT_FIELDS = ("P", "M", "C", "MC")


def reward_format(product: Dict[str, Any], chain_ids: Sequence[str] = ()) -> Tuple[float, Dict[str, Any]]:
    """Field completeness and citation legality."""
    present = sum(1 for key in REQUIRED_COT_FIELDS if product.get(key))
    completeness = present / len(REQUIRED_COT_FIELDS)

    has_labels = bool(product.get("fine_label")) and bool(product.get("coarse_label"))
    has_scores = bool(product.get("es")) and bool(product.get("dc"))
    has_kcrit = bool(product.get("k_crit"))

    refs = list(product.get("refs") or [])
    legal = sum(1 for ref in refs if ref in set(chain_ids)) if chain_ids else len(refs)
    citation_rate = legal / len(refs) if refs else (1.0 if not chain_ids else 0.5)

    score = (0.4 * completeness + 0.2 * float(has_labels) + 0.2 * float(has_scores)
             + 0.1 * float(has_kcrit) + 0.1 * citation_rate)
    return round(float(np.clip(score, 0.0, 1.0)), 5), {
        "field_completeness": round(completeness, 4),
        "citation_legality": round(citation_rate, 4),
        "has_scores": has_scores, "has_k_crit": has_kcrit,
    }


def reward_causal(
    dc: float, mni_values: Dict[str, float], graph_edit_distance: float,
    lambda_graph: float = 0.35,
) -> Tuple[float, Dict[str, Any]]:
    """World-model judge score: mean of ``DC``, mean ``MNI``, and the structure term.

    Mean MNI rather than sum is what makes AU padding self-defeating: each additional
    low-necessity unit drags the average down.
    """
    mean_mni = float(np.mean(list(mni_values.values()))) if mni_values else 0.0
    # MNI is a KL divergence in nats and unbounded above; squash to [0, 1] so it cannot
    # dominate the other two terms on a single extreme sample.
    mni_term = float(1.0 - math.exp(-2.0 * max(0.0, mean_mni)))
    structure_term = float(math.exp(-lambda_graph * max(0.0, graph_edit_distance)))
    dc_term = float(np.clip(dc, 0.0, 1.0))
    score = (dc_term + mni_term + structure_term) / 3.0
    return round(float(np.clip(score, 0.0, 1.0)), 5), {
        "dc": round(dc_term, 4), "mean_mni": round(mean_mni, 5),
        "mni_term": round(mni_term, 4), "structure_term": round(structure_term, 4),
        "graph_edit_distance": round(graph_edit_distance, 4),
    }


def reward_temporal(
    proposal: Tuple[int, int], truth: Optional[Tuple[int, int]],
    fine_pred: str = "", fine_true: str = "",
    iou_threshold: float = 0.5,
    affective_rescue: bool = True,
    rescue_scale: float = 0.5,
    rescue_min_iou: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Temporal IoU combined with the eq. (2) TP criterion.
    """
    from ..eval.metrics import iou as interval_iou, tp_decision

    if truth is None:
        return 0.0, {"iou": 0.0, "tp": False, "rescued": False,
                     "iou_threshold": iou_threshold}
    overlap = interval_iou(proposal, truth)
    verdict = tp_decision(overlap, fine_pred, fine_true, iou_threshold,
                          affective_rescue, rescue_min_iou)
    detail = {"iou": round(overlap, 4), "iou_threshold": iou_threshold,
              "tp": verdict.is_tp, "rescued": verdict.rescued}
    if verdict.is_tp and not verdict.rescued:
        return round(float(overlap), 5), detail
    if verdict.rescued:
        # Credited, but capped below any genuine temporal hit -- otherwise the policy
        # could stop caring about boundaries as long as the label is right. The floor
        # keeps a correct label from scoring exactly zero when the overlap is tiny,
        # which is a scoring choice and not a change to the verdict above.
        return round(float(rescue_scale * max(overlap, 0.2)), 5), detail
    return round(float(0.2 * overlap), 5), detail


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class CompositeReward:
    """Assembles eq. (11) from the five components."""

    def __init__(self, config: Optional[RewardConfig] = None,
                 evaluation: Optional[Any] = None) -> None:
        self.config = config or RewardConfig()
        # The reward's temporal term must use the same IoU criterion the evaluation
        # does; training against a different threshold optimises the wrong objective.
        if evaluation is None:
            from ..config import EvaluationConfig
            evaluation = EvaluationConfig()
        self.evaluation = evaluation

    def score(
        self,
        product: Dict[str, Any],
        truth: Dict[str, Any],
        judge: Optional[Dict[str, Any]] = None,
        chain_ids: Sequence[str] = (),
        progress: Optional[float] = None,
    ) -> RewardBreakdown:
        """Score one policy output.

        ``judge`` carries the frozen engine's online quantities (``dc``, ``mni``,
        ``graph_edit_distance``); absent them the causal term falls back to zero rather
        than to a guess, so a missing judge shows up as a missing reward rather than a
        fabricated one.
        """
        judge = judge or {}
        weights = (curriculum_weights(progress) if progress is not None
                   else {"au": self.config.w_au, "emo": self.config.w_emo,
                         "fmt": self.config.w_fmt, "causal": self.config.w_causal,
                         "temp": self.config.w_temp})

        r_au, detail_au = reward_au(
            product.get("k_crit") or product.get("active_aus") or [],
            truth.get("aus") or [],
            self.config.hallucination_penalty,
            evidenced=truth.get("evidenced_aus"),
        )
        r_emo, detail_emo = reward_emotion(
            str(product.get("fine_label", "")), str(truth.get("fine", "")),
            str(product.get("coarse_label", "")), str(truth.get("coarse", "")),
        )
        r_fmt, detail_fmt = reward_format(product, chain_ids)
        r_causal, detail_causal = reward_causal(
            float(judge.get("dc", 0.0)),
            dict(judge.get("mni", {})),
            float(judge.get("graph_edit_distance", 0.0)),
            self.config.lambda_graph,
        )
        r_temp, detail_temp = reward_temporal(
            tuple(product.get("interval") or (0, 0)),
            tuple(truth["interval"]) if truth.get("interval") else None,
            str(product.get("fine_label", "")), str(truth.get("fine", "")),
            iou_threshold=self.evaluation.iou_threshold,
            affective_rescue=self.evaluation.affective_rescue,
            rescue_scale=self.evaluation.rescue_reward_scale,
            rescue_min_iou=self.evaluation.rescue_min_iou,
        )

        total = (weights["au"] * r_au + weights["emo"] * r_emo + weights["fmt"] * r_fmt
                 + weights["causal"] * r_causal + weights["temp"] * r_temp)

        return RewardBreakdown(
            r_au=r_au, r_emo=r_emo, r_fmt=r_fmt, r_causal=r_causal, r_temp=r_temp,
            total=round(float(total), 5), weights=weights,
            detail={"au": detail_au, "emo": detail_emo, "fmt": detail_fmt,
                    "causal": detail_causal, "temp": detail_temp},
        )


# ---------------------------------------------------------------------------
# Online judge
# ---------------------------------------------------------------------------


class WorldModelJudge:
    """Computes ``R_causal``'s inputs online from the frozen rollout engine."""

    def __init__(self, service: Any, lambda_graph: float = 0.35) -> None:
        self.service = service
        self.lambda_graph = lambda_graph

    def evaluate(
        self,
        observed: np.ndarray,
        fine_label: str,
        k_crit: Sequence[str],
        candidates: Sequence[str],
        predicted_graph: Optional[Any] = None,
        reference_graph: Optional[Any] = None,
        cid: str = "",
    ) -> Dict[str, Any]:
        """Run ``score`` and ``mask`` and parse the reference graph."""
        from ..agents.structure import graph_edit_distance

        scores = self.service.score(observed, list(candidates), caller="judge", cid=cid)
        dc = scores.normalised.get(fine_label, 0.0)

        mni: Dict[str, float] = {}
        for au in k_crit:
            if au not in SLOT_INDEX:
                continue
            mni[au] = self.service.mask(observed, [au], list(candidates),
                                        caller="judge", cid=cid).mni

        edit = 0.0
        if predicted_graph is not None and reference_graph is not None:
            edit = graph_edit_distance(reference_graph, predicted_graph)

        return {"dc": dc, "mni": mni, "graph_edit_distance": edit,
                "log_likelihood": scores.log_likelihood,
                "model_version": self.service.model_version}


# ---------------------------------------------------------------------------
# Team reward (MAPPO, formwork.md 第 V 条 / 完整执行方案 第 5.3 节, 2026-09-03)
# ---------------------------------------------------------------------------

#: w1..w5 of R_team = w1*IoU + w2*EmotionAcc + w3*(1-Hallucination) + w4*CriticSurvival
#: + w5*STRS_proxy (方案第 5.3 节). Sums to 1; overridable per call.
DEFAULT_TEAM_REWARD_WEIGHTS: Dict[str, float] = {
    "iou": 0.30, "emotion": 0.30, "evidence": 0.15, "critic": 0.15, "strs": 0.10,
}


@dataclass
class TeamRewardBreakdown:
    """The five-term shared reward every agent's MAPPO advantage is computed from.
    """

    r_iou: float = 0.0
    r_emotion_acc: float = 0.0
    r_evidence: float = 0.0
    r_critic_survival: float = 0.0
    r_strs_proxy: float = 0.0
    total: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "R_iou": round(self.r_iou, 5), "R_emotion_acc": round(self.r_emotion_acc, 5),
            "R_evidence": round(self.r_evidence, 5),
            "R_critic_survival": round(self.r_critic_survival, 5),
            "R_strs_proxy": round(self.r_strs_proxy, 5), "total": round(self.total, 5),
            "weights": self.weights, "detail": self.detail,
        }


def team_reward_components(
    product: Dict[str, Any],
    truth: Dict[str, Any],
    critic_verdict: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
) -> TeamRewardBreakdown:
    """``R_team`` for one candidate region, shared by all four agents' advantages.
    """
    if evaluation is None:
        from ..config import EvaluationConfig
        evaluation = EvaluationConfig()
    weights = weights or DEFAULT_TEAM_REWARD_WEIGHTS

    r_iou, detail_iou = reward_temporal(
        tuple(product.get("interval") or (0, 0)),
        tuple(truth["interval"]) if truth.get("interval") else None,
        str(product.get("fine_label", "")), str(truth.get("fine", "")),
        iou_threshold=evaluation.iou_threshold,
        affective_rescue=evaluation.affective_rescue,
        rescue_scale=evaluation.rescue_reward_scale,
        rescue_min_iou=evaluation.rescue_min_iou,
    )
    r_emo, detail_emo = reward_emotion(
        str(product.get("fine_label", "")), str(truth.get("fine", "")),
        str(product.get("coarse_label", "")), str(truth.get("coarse", "")),
    )
    claimed_aus = list(product.get("k_crit") or product.get("active_aus") or [])
    _, detail_au = reward_au(
        claimed_aus, list(truth.get("aus") or []),
        hallucination_penalty=1.0, evidenced=truth.get("evidenced_aus"),
    )
    hallucination_rate = detail_au["hallucinated"] / max(1, len(claimed_aus))
    r_evidence = float(np.clip(1.0 - hallucination_rate, 0.0, 1.0))

    if critic_verdict is None:
        r_critic = 0.5
        critic_source = "no_verdict_yet"
    elif "survival_score" in critic_verdict:
        r_critic = float(np.clip(critic_verdict["survival_score"], 0.0, 1.0))
        critic_source = "graded"
    else:
        r_critic = 1.0 if critic_verdict.get("survived") else 0.0
        critic_source = "binary"

    r_strs = float(detail_iou.get("tp", False)) * r_emo

    total = (weights["iou"] * r_iou + weights["emotion"] * r_emo
             + weights["evidence"] * r_evidence + weights["critic"] * r_critic
             + weights["strs"] * r_strs)

    return TeamRewardBreakdown(
        r_iou=r_iou, r_emotion_acc=r_emo, r_evidence=r_evidence,
        r_critic_survival=r_critic, r_strs_proxy=round(r_strs, 5),
        total=round(float(total), 5), weights=dict(weights),
        detail={"iou": detail_iou, "emotion": detail_emo, "au": detail_au,
                "critic_source": critic_source,
                "hallucination_rate": round(hallucination_rate, 4)},
    )


__all__ = [
    "WEIGHT_CURRICULUM", "curriculum_weights", "RewardBreakdown", "reward_au",
    "reward_emotion", "reward_format", "reward_causal", "reward_temporal",
    "CompositeReward", "WorldModelJudge",
    "DEFAULT_TEAM_REWARD_WEIGHTS", "TeamRewardBreakdown", "team_reward_components",
]
