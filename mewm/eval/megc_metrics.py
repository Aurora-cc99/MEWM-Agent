"""MEGC protocol metrics: spotting and recognition scores per dataset split."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..knowledge.emotion_prototypes import (
    COARSE_EMOTIONS, FINE_EMOTIONS, canonical_fine_label, coarse_of,
)
from .metrics import (
    DEFAULT_IOU_THRESHOLD, au_set_metrics, bleu, count_errors, iou, rouge_n,
    tp_decision, unweighted_average_recall, unweighted_f1,
)


MEGC_COARSE: Tuple[str, ...] = ("positive", "negative", "surprise")
MEGC_FINE: Tuple[str, ...] = ("happiness", "surprise", "fear", "disgust", "anger",
                              "sadness")
REPO_COARSE: Tuple[str, ...] = tuple(COARSE_EMOTIONS)
REPO_FINE: Tuple[str, ...] = tuple(FINE_EMOTIONS)

EXPRESSION_TYPES: Tuple[str, ...] = ("micro-expression", "macro-expression")

_UNAVAILABLE = "unavailable"
_OK = "ok"


def _unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": _UNAVAILABLE, "reason": reason}
    out.update(extra)
    return out


def strs(f1_spot: float, f1_analysis: float) -> float:
    return round(float(f1_spot) * float(f1_analysis), 4)


def _prf(tp: int, n_pred: int, n_true: int) -> Dict[str, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def _greedy_pairs(proposals: Sequence[Tuple[int, int]],
                  truth: Sequence[Tuple[int, int]],
                  iou_threshold: float) -> List[Tuple[int, int, float]]:
    scored: List[Tuple[float, int, int]] = []
    for i, p in enumerate(proposals):
        for j, t in enumerate(truth):
            overlap = iou(p, t)
            if overlap > 0:
                scored.append((overlap, i, j))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_p: set = set()
    used_t: set = set()
    pairs: List[Tuple[int, int, float]] = []
    for overlap, i, j in scored:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append((i, j, overlap))
    return pairs


def spotting_scores(
    per_video: Sequence[Dict[str, Any]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    rescue_min_iou: float = 0.0,
) -> Dict[str, Any]:
    if not per_video:
        return _unavailable("no videos carried spotting proposals")

    strict = {"tp": 0, "fp": 0, "fn": 0}
    rescued = {"tp": 0, "fp": 0, "fn": 0}
    type_true: List[str] = []
    type_pred: List[str] = []
    n_rescue_only = 0

    for row in per_video:
        proposals = [tuple(x) for x in row.get("proposals", [])]
        truth = [tuple(x) for x in row.get("truth", [])]
        p_types = list(row.get("proposal_types", []))
        t_types = list(row.get("truth_types", []))
        p_labels = list(row.get("proposal_labels", []))
        t_labels = list(row.get("truth_labels", []))

        pairs = _greedy_pairs(proposals, truth, iou_threshold)
        matched_p_strict: set = set()
        matched_p_rescued: set = set()
        matched_t_strict: set = set()
        matched_t_rescued: set = set()

        for i, j, overlap in pairs:
            decision = tp_decision(
                overlap,
                fine_pred=p_labels[i] if i < len(p_labels) else "",
                fine_true=t_labels[j] if j < len(t_labels) else "",
                iou_threshold=iou_threshold,
                affective_rescue=True,
                rescue_min_iou=rescue_min_iou,
            )
            if decision.is_tp_strict:
                matched_p_strict.add(i)
                matched_t_strict.add(j)
                type_true.append(t_types[j] if j < len(t_types) else "")
                type_pred.append(p_types[i] if i < len(p_types) else "")
            if decision.is_tp:
                matched_p_rescued.add(i)
                matched_t_rescued.add(j)
                if not decision.is_tp_strict:
                    n_rescue_only += 1

        strict["tp"] += len(matched_t_strict)
        strict["fp"] += len(proposals) - len(matched_p_strict)
        strict["fn"] += len(truth) - len(matched_t_strict)
        rescued["tp"] += len(matched_t_rescued)
        rescued["fp"] += len(proposals) - len(matched_p_rescued)
        rescued["fn"] += len(truth) - len(matched_t_rescued)

    strict_prf = _prf(strict["tp"], strict["tp"] + strict["fp"],
                      strict["tp"] + strict["fn"])
    rescued_prf = _prf(rescued["tp"], rescued["tp"] + rescued["fp"],
                       rescued["tp"] + rescued["fn"])

    if type_true:
        supported = [t for t in EXPRESSION_TYPES if t in set(type_true)]
        spot_uf1, _ = unweighted_f1(type_true, type_pred, labels=supported)
        spot_uf1_declared, spot_per_class = unweighted_f1(
            type_true, type_pred, labels=list(EXPRESSION_TYPES))
        spot_uar = unweighted_average_recall(type_true, type_pred, labels=supported)
        unweighted: Dict[str, Any] = {
            "status": _OK,
            "spot_uf1": spot_uf1,
            "spot_uar": spot_uar,
            "macro_divisor": len(supported),
            "classes_with_support": supported,
            "spot_uf1_all_declared_classes": spot_uf1_declared,
            "per_class_f1": spot_per_class,
            "n_pairs": len(type_true),
            "classes": list(EXPRESSION_TYPES),
            "basis": "strict-IoU matched proposal/truth pairs only",
        }
    else:
        unweighted = _unavailable(
            "" % iou_threshold,
            classes=list(EXPRESSION_TYPES))

    return {
        "status": _OK,
        "iou_threshold": iou_threshold,
        "n_videos": len(per_video),
        "strict_iou": {
            "criterion": "temporal IoU >= %.2f, no affective rescue"
                         % iou_threshold,
            **strict, **strict_prf,
        },
        "paper_eq2": {
            "criterion": ""
                         % rescue_min_iou,
            "n_rescued_only": n_rescue_only,
            **rescued, **rescued_prf,
        },
        "unweighted_type": unweighted,
        "headline_f1": strict_prf["f1"],
        "headline_note": "",
    }


COUNT_QUANTITIES: Tuple[str, ...] = ("expression", "micro", "macro")


def count_scores(pairs_by_quantity: Dict[str, Sequence[Tuple[int, int]]]
                 ) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": _OK}
    scored_any = False
    for quantity in COUNT_QUANTITIES:
        pairs = list(pairs_by_quantity.get(quantity, []))
        if not pairs:
            out[quantity] = _unavailable(
                "no answer to the %s-count question parsed into an integer" % quantity)
            continue
        scored_any = True
        predicted = [int(p) for p, _ in pairs]
        truth = [int(t) for _, t in pairs]
        errors = count_errors(predicted, truth)
        out[quantity] = {"status": _OK, "n": len(pairs), **errors}
    if not scored_any:
        return _unavailable("no count question produced a parseable integer answer")
    return out


def au_scores(predicted: Sequence[Sequence[str]],
              truth: Sequence[Sequence[str]]) -> Dict[str, Any]:
    predicted = list(predicted)
    truth = list(truth)
    if not predicted or not truth:
        return _unavailable("no AU-inventory answer parsed into a set of action units")
    scores = au_set_metrics(predicted, truth)
    return {
        "status": _OK,
        "n": min(len(predicted), len(truth)),
        "f1_au": scores["au_f1"],
        "jaccard_au": scores["au_jaccard"],
        "definition": "F1_AU = 2|A_hat & A|/(|A_hat|+|A|); "
                      "Jaccard_AU = |A_hat & A|/|A_hat | A|",
    }


def _emotion_pair(y_true: Sequence[str], y_pred: Sequence[str],
                  labels: Sequence[str], vocabulary: str) -> Dict[str, Any]:
    kept_true: List[str] = []
    kept_pred: List[str] = []
    n_dropped = 0
    for t, p in zip(y_true, y_pred):
        if t not in labels:
            n_dropped += 1
            continue
        kept_true.append(t)
        kept_pred.append(p)
    if not kept_true:
        return _unavailable(
            "no sample's ground-truth label falls inside the %s vocabulary" % vocabulary,
            classes=list(labels), n_dropped_out_of_vocabulary=n_dropped)
    supported = [label for label in labels if label in set(kept_true)]
    uf1, per_class = unweighted_f1(kept_true, kept_pred, labels=supported)
    uf1_declared, per_class_declared = unweighted_f1(kept_true, kept_pred,
                                                    labels=list(labels))
    uar = unweighted_average_recall(kept_true, kept_pred, labels=supported)
    return {
        "status": _OK,
        "vocabulary": vocabulary,
        "classes": list(labels),
        "classes_with_support": supported,
        "classes_without_support": [l for l in labels if l not in set(supported)],
        "reg_uf1": uf1,
        "reg_uar": uar,
        "macro_divisor": len(supported),
        "reg_uf1_all_declared_classes": uf1_declared,
        "per_class_f1": per_class_declared,
        "n": len(kept_true),
        "n_dropped_out_of_vocabulary": n_dropped,
    }


def recognition_scores(fine_true: Sequence[str], fine_pred: Sequence[str]
                       ) -> Dict[str, Any]:
    if not fine_true:
        return _unavailable("no event carried both a predicted and a reference label")

    canon_true: List[str] = []
    canon_pred: List[str] = []
    for t, p in zip(fine_true, fine_pred):
        ct, _ = canonical_fine_label(t)
        cp, recognised = canonical_fine_label(p)
        canon_true.append(ct)
        canon_pred.append(cp if recognised else (p or "").strip().lower())

    coarse_true = [coarse_of(t) for t in canon_true]
    coarse_pred = [coarse_of(p) if p in REPO_FINE else (p or "") for p in canon_pred]

    return {
        "status": _OK,
        "n_events": len(canon_true),
        "fine": {
            "megc": _emotion_pair(canon_true, canon_pred, MEGC_FINE, ""),
            "repo": _emotion_pair(canon_true, canon_pred, REPO_FINE, ""),
        },
        "coarse": {
            "megc": _emotion_pair(coarse_true, coarse_pred, MEGC_COARSE,
                                  ""),
            "repo": _emotion_pair(coarse_true, coarse_pred, REPO_COARSE,
                                  ""),
        },
        "headline": "fine.megc.reg_uf1 / coarse.megc.reg_uf1 -- the comparable pair",
    }


def _bleu_unsmoothed(candidate: str, reference: str, max_n: int = 4) -> float:
    cand = candidate.split()
    ref = reference.split()
    if not cand or not ref:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        cand_grams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(cand) - n + 1):
            key = tuple(cand[i:i + n])
            cand_grams[key] = cand_grams.get(key, 0) + 1
        ref_grams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(ref) - n + 1):
            key = tuple(ref[i:i + n])
            ref_grams[key] = ref_grams.get(key, 0) + 1
        total = sum(cand_grams.values())
        if not total:
            return 0.0
        overlap = sum(min(c, ref_grams.get(g, 0)) for g, c in cand_grams.items())
        if overlap == 0:
            return 0.0
        log_p += math.log(overlap / total) / max_n
    brevity = min(1 - len(ref) / len(cand), 0.0)
    return round(math.exp(brevity + log_p), 4)


def text_scores(candidates: Sequence[str], references: Sequence[str],
                max_n: int = 4) -> Dict[str, Any]:
    paired = [(c, r) for c, r in zip(candidates, references)
              if isinstance(c, str) and isinstance(r, str) and c.strip() and r.strip()]
    if not paired:
        return _unavailable("no generation had a non-empty reference answer to score "
                            "against")
    smoothed = [bleu(c, r, max_n) for c, r in paired]
    plain = [_bleu_unsmoothed(c, r, max_n) for c, r in paired]
    rouge1 = [rouge_n(c, r, 1) for c, r in paired]
    n = len(paired)
    return {
        "status": _OK,
        "n": n,
        "bleu": round(sum(plain) / n, 4),
        "bleu_smoothed": round(sum(smoothed) / n, 4),
        "rouge_1": round(sum(rouge1) / n, 4),
        "max_n": max_n,
        "tokenisation": "whitespace",
        "smoothing_note": "",
    }


def megc_report(
    spotting: Dict[str, Any],
    counting: Dict[str, Any],
    recognition_au: Dict[str, Any],
    recognition_emotion: Dict[str, Any],
    recognition_text: Dict[str, Any],
    strs_text: Dict[str, Any],
    f1_analysis_on_tp: Optional[float] = None,
    f1_analysis_on_truth: Optional[float] = None,
) -> Dict[str, Any]:
    f1_spot = spotting.get("headline_f1") if spotting.get("status") == _OK else None

    if f1_spot is None or f1_analysis_on_tp is None:
        strs_block: Dict[str, Any] = _unavailable(
            "" % (
                "" if f1_spot is None
                else ""),
            definition="")
    else:
        strs_block = {
            "status": _OK,
            "strs": strs(f1_spot, f1_analysis_on_tp),
            "f1_spot": f1_spot,
            "f1_analysis": f1_analysis_on_tp,
            "f1_analysis_basis": "",
            "f1_analysis_on_ground_truth_intervals": f1_analysis_on_truth,
            "definition": "a",
        }

    return {
        "spotting": {
            "interval": spotting,
            "counting": counting,
        },
        "recognition": {
            "action_units": recognition_au,
            "emotion": recognition_emotion,
            "text": recognition_text,
        },
        "strs": {
            "score": strs_block,
            "text": strs_text,
        },
        "provenance": {
            "spotting": "",
            "counting": "",
            "recognition": "",
            "strs": "",
        },
    }


__all__ = [
    "MEGC_COARSE", "MEGC_FINE", "REPO_COARSE", "REPO_FINE", "EXPRESSION_TYPES",
    "COUNT_QUANTITIES", "strs", "spotting_scores", "count_scores", "au_scores",
    "recognition_scores", "text_scores", "megc_report",
]
