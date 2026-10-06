"""Localisation diagnostics: per-fold error breakdown and false-alarm analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["localisation_report", "signal_auc", "threshold_reachability",
           "extent_anatomy", "filter_ceiling", "rank_auc"]


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.0


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    raw = np.empty(allv.size, dtype=np.float64)
    raw[order] = np.arange(1, allv.size + 1)
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, raw)
    ranks = (sums / counts)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _match(intervals: Sequence[Tuple[int, int, float]],
           truth: Sequence[Tuple[int, int]],
           iou_threshold: float) -> Tuple[int, int, int]:
    used: set = set()
    tp = 0
    for interval in sorted(intervals, key=lambda x: -x[2]):
        best, best_i = 0.0, -1
        for i, t in enumerate(truth):
            if i in used:
                continue
            overlap = _iou(interval[:2], t)
            if overlap > best:
                best, best_i = overlap, i
        if best >= iou_threshold and best_i >= 0:
            used.add(best_i)
            tp += 1
    return tp, len(intervals) - tp, len(truth) - tp


def signal_auc(
    s_curve: np.ndarray,
    micro_truth: Sequence[Tuple[int, int]],
    all_truth: Sequence[Tuple[int, int]] = (),
    frame_offset: int = 0,
) -> Dict[str, Any]:
    s = np.asarray(s_curve, dtype=np.float64).reshape(-1)
    if s.size == 0 or not micro_truth:
        return {"status": "unavailable",
                "reason": "no curve or no micro-expression ground truth"}

    micro_mask = np.zeros(s.size, dtype=bool)
    for on, off in micro_truth:
        lo, hi = max(0, on - frame_offset), min(s.size - 1, off - frame_offset)
        if hi >= lo:
            micro_mask[lo:hi + 1] = True
    annotated = micro_mask.copy()
    for on, off in all_truth:
        lo, hi = max(0, on - frame_offset), min(s.size - 1, off - frame_offset)
        if hi >= lo:
            annotated[lo:hi + 1] = True

    pos = s[micro_mask]
    neg = s[~annotated]
    if pos.size == 0 or neg.size == 0:
        return {"status": "unavailable",
                "reason": "ground truth does not intersect the curve"}
    return {
        "status": "ok",
        "auc_vs_unannotated": round(rank_auc(pos, neg), 4),
        "auc_vs_all_non_micro": round(rank_auc(pos, s[~micro_mask]), 4),
        "mean_inside": round(float(pos.mean()), 4),
        "mean_outside": round(float(neg.mean()), 4),
        "n_frames_inside": int(pos.size),
        "n_frames_outside": int(neg.size),
        "note": ("0.5 = the curve says nothing about where micro-expressions "
                 "are; below 0.5 = it points away from them"),
    }


def threshold_reachability(
    s_curve: np.ndarray,
    micro_truth: Sequence[Tuple[int, int]],
    tau_hi: float,
    frame_offset: int = 0,
) -> Dict[str, Any]:
    s = np.asarray(s_curve, dtype=np.float64).reshape(-1)
    if s.size == 0 or not micro_truth:
        return {"status": "unavailable", "reason": "no curve or no ground truth"}

    percentiles, peaks, above = [], [], 0
    for on, off in micro_truth:
        lo, hi = max(0, on - frame_offset), min(s.size - 1, off - frame_offset)
        if hi < lo:
            continue
        peak = float(s[lo:hi + 1].max())
        peaks.append(peak)
        percentiles.append(float((s < peak).mean() * 100))
        above += peak >= tau_hi
    if not peaks:
        return {"status": "unavailable", "reason": "ground truth outside the curve"}
    return {
        "status": "ok",
        "n_events": len(peaks),
        "tau_hi": tau_hi,
        "tau_hi_percentile_of_curve": round(float((s < tau_hi).mean() * 100), 2),
        "peak_percentile_mean": round(float(np.mean(percentiles)), 2),
        "peak_percentile_median": round(float(np.median(percentiles)), 2),
        "n_events_reaching_tau_hi": int(above),
        "frac_events_reaching_tau_hi": round(above / len(peaks), 4),
        "note": ("an event whose peak S is below tau_hi cannot be proposed by "
                 "hysteresis at any duration setting -- this is a hard recall cap"),
    }


def extent_anatomy(
    intervals: Sequence[Tuple[int, int, int, float]],
    micro_truth: Sequence[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    if not micro_truth:
        return {"status": "unavailable", "reason": "no micro-expression ground truth"}

    best_ious, ratios, apex_positions = [], [], []
    found = hit = midpoint_covered = 0
    for on, off in micro_truth:
        span = max(1, off - on)
        best, best_iv = 0.0, None
        for iv in intervals:
            overlap = _iou(iv[:2], (on, off))
            if overlap > best:
                best, best_iv = overlap, iv
        best_ious.append(best)
        found += best > 0
        hit += best >= iou_threshold
        if best_iv is not None:
            ratios.append((best_iv[1] - best_iv[0] + 1) / (off - on + 1))
        mid = (on + off) // 2
        midpoint_covered += any(iv[0] <= mid <= iv[1] for iv in intervals)
        for iv in intervals:
            if iv[1] >= on - span and iv[0] <= off + span:
                apex_positions.append((iv[2] - on) / span)

    arr = np.array(best_ious)
    out: Dict[str, Any] = {
        "status": "ok",
        "n_truth": len(micro_truth),
        "n_intervals": len(intervals),
        "n_found_any_overlap": found,
        "n_hit_at_threshold": hit,
        "n_midpoint_covered": midpoint_covered,
        "best_iou_mean": round(float(arr.mean()), 4),
        "best_iou_median": round(float(np.median(arr)), 4),
        "iou_curve": {str(t): int((arr >= t).sum())
                      for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)},
    }
    if ratios:
        out["duration_ratio_median"] = round(float(np.median(ratios)), 4)
        out["duration_ratio_mean"] = round(float(np.mean(ratios)), 4)
        out["duration_note"] = ("an interval r times the truth's length caps IoU "
                                "at r even when perfectly centred (r<1), so a "
                                "median below 0.5 caps the score by itself")
    if apex_positions:
        pos = np.array(apex_positions)
        out["apex_position"] = {
            "n_nearby": int(pos.size),
            "inside_event": round(float(((pos >= 0) & (pos <= 1)).mean()), 4),
            "central_half": round(float(((pos >= 0.25) & (pos <= 0.75)).mean()), 4),
            "histogram": {f"[{lo:.2f},{hi:.2f})": int(((pos >= lo) & (pos < hi)).sum())
                          for lo, hi in ((-1.0, -0.5), (-0.5, -0.25), (-0.25, 0.0),
                                         (0.0, 0.25), (0.25, 0.5), (0.5, 0.75),
                                         (0.75, 1.0), (1.0, 1.25), (1.25, 1.5),
                                         (1.5, 2.0))},
            "note": ("a dip in the central bins with mass on both sides is the "
                     "signature of a velocity-like curve: one event writes two "
                     "bursts, at onset and at offset, with a trough at the apex"),
        }
    return out


def error_taxonomy(
    intervals: Sequence[Tuple[int, int, int, float]],
    micro_truth: Sequence[Tuple[int, int]],
    macro_truth: Sequence[Tuple[int, int]] = (),
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    matched: set = set()
    tp_ids: set = set()
    for idx in sorted(range(len(intervals)), key=lambda i: -intervals[i][3]):
        iv = intervals[idx]
        best, best_i = 0.0, -1
        for i, t in enumerate(micro_truth):
            if i in matched:
                continue
            overlap = _iou(iv[:2], t)
            if overlap > best:
                best, best_i = overlap, i
        if best >= iou_threshold and best_i >= 0:
            matched.add(best_i)
            tp_ids.add(idx)

    fp_extent = fp_macro = fp_spurious = 0
    for idx, iv in enumerate(intervals):
        if idx in tp_ids:
            continue
        if any(_iou(iv[:2], t) > 0 for t in micro_truth):
            fp_extent += 1
        elif any(_iou(iv[:2], t) > 0 for t in macro_truth):
            fp_macro += 1
        else:
            fp_spurious += 1

    fn_geometry = fn_blind = 0
    for i, t in enumerate(micro_truth):
        if i in matched:
            continue
        if any(_iou(iv[:2], t) > 0 for iv in intervals):
            fn_geometry += 1
        else:
            fn_blind += 1

    n_fp = fp_extent + fp_macro + fp_spurious
    n_fn = fn_geometry + fn_blind
    return {
        "tp": len(tp_ids),
        "fp": n_fp,
        "fn": n_fn,
        "fp_extent": fp_extent,
        "fp_macro": fp_macro,
        "fp_spurious": fp_spurious,
        "fn_geometry": fn_geometry,
        "fn_blind": fn_blind,
        "fp_spurious_share": round(fp_spurious / n_fp, 4) if n_fp else 0.0,
        "fn_blind_share": round(fn_blind / n_fn, 4) if n_fn else 0.0,
    }


def filter_ceiling(
    per_video: Sequence[Tuple[Sequence[Tuple[int, int, int, float]],
                             Sequence[Tuple[int, int]]]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    def score(select) -> Dict[str, float]:
        tp = fp = fn = 0
        for intervals, truth in per_video:
            kept = [(iv[0], iv[1], iv[3]) for iv in select(intervals, truth)]
            a, b, c = _match(kept, truth, iou_threshold)
            tp, fp, fn = tp + a, fp + b, fn + c
        p, r, f = _prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
                "recall": round(r, 4), "f1": round(f, 4)}

    as_is = score(lambda iv, t: iv)
    oracle = score(lambda iv, t: [x for x in iv
                                  if any(_iou(x[:2], y) >= iou_threshold for y in t)])
    return {
        "as_is": as_is,
        "oracle_filter": oracle,
        "headroom_from_filtering": round(oracle["f1"] - as_is["f1"], 4),
        "note": ("oracle_filter keeps only the intervals that already reach the "
                 "IoU threshold; it is unattainable and bounds every reranking "
                 "or calibration scheme. Its recall equals as_is recall by "
                 "construction -- filtering cannot find what was never proposed"),
    }


def localisation_report(
    videos: Sequence[Any],
    curves: Dict[str, np.ndarray],
    intervals_by_video: Dict[str, Sequence[Tuple[int, int, int, float]]],
    offsets: Optional[Dict[str, int]] = None,
    tau_hi: float = 3.5,
    iou_threshold: float = 0.5,
    include_per_video: bool = True,
) -> Dict[str, Any]:
    offsets = offsets or {}
    per_video: Dict[str, Any] = {}
    pooled_pos: List[np.ndarray] = []
    pooled_neg: List[np.ndarray] = []
    pooled_pairs: List[Tuple[Any, Any]] = []
    aucs, reach, ratios, best_all = [], [], [], []
    taxonomy_total = {"tp": 0, "fp": 0, "fn": 0, "fp_extent": 0, "fp_macro": 0,
                      "fp_spurious": 0, "fn_geometry": 0, "fn_blind": 0}
    n_truth = n_intervals = 0

    for video in videos:
        key = video.video_key
        curve = curves.get(key)
        if curve is None:
            continue
        micro_truth = [tuple(e.interval) for e in video.micro_events()]
        if not micro_truth:
            continue
        all_truth = [tuple(e.interval) for e in video.events]
        ivs = list(intervals_by_video.get(key, []))
        offset = int(offsets.get(key, 0))
        n_truth += len(micro_truth)
        n_intervals += len(ivs)

        sig = signal_auc(curve, micro_truth, all_truth, offset)
        thr = threshold_reachability(curve, micro_truth, tau_hi, offset)
        ext = extent_anatomy(ivs, micro_truth, iou_threshold)
        macro_truth = [t for t in all_truth if t not in set(micro_truth)]
        tax = error_taxonomy(ivs, micro_truth, macro_truth, iou_threshold)
        for k, v in tax.items():
            if k in taxonomy_total:
                taxonomy_total[k] += v
        if sig.get("status") == "ok":
            aucs.append(sig["auc_vs_unannotated"])
            s = np.asarray(curve, dtype=np.float64).reshape(-1)
            mask = np.zeros(s.size, dtype=bool)
            for on, off in micro_truth:
                lo, hi = max(0, on - offset), min(s.size - 1, off - offset)
                if hi >= lo:
                    mask[lo:hi + 1] = True
            ann = mask.copy()
            for on, off in all_truth:
                lo, hi = max(0, on - offset), min(s.size - 1, off - offset)
                if hi >= lo:
                    ann[lo:hi + 1] = True
            pooled_pos.append(s[mask])
            pooled_neg.append(s[~ann])
        if thr.get("status") == "ok":
            reach.append(thr["frac_events_reaching_tau_hi"])
        if ext.get("status") == "ok":
            best_all.append((ext["n_found_any_overlap"], ext["n_hit_at_threshold"],
                             ext["n_truth"]))
            if "duration_ratio_median" in ext:
                ratios.append(ext["duration_ratio_median"])
        pooled_pairs.append((ivs, micro_truth))
        if include_per_video:
            per_video[key] = {"subject": video.subject, "signal": sig,
                              "threshold": thr, "extent": ext, "errors": tax}

    report: Dict[str, Any] = {
        "scope": "micro-expression events only",
        "iou_threshold": iou_threshold,
        "n_videos_with_micro_truth": len(pooled_pairs),
        "n_micro_truth_events": n_truth,
        "n_intervals": n_intervals,
    }
    if pooled_pos and pooled_neg:
        pos, neg = np.concatenate(pooled_pos), np.concatenate(pooled_neg)
        report["layer_1_signal"] = {
            "pooled_auc": round(rank_auc(pos, neg), 4),
            "per_video_auc_mean": round(float(np.mean(aucs)), 4),
            "per_video_auc_median": round(float(np.median(aucs)), 4),
            "n_videos_auc_above_half": int(sum(1 for a in aucs if a > 0.5)),
            "n_videos": len(aucs),
            "mean_S_inside": round(float(pos.mean()), 4),
            "mean_S_outside": round(float(neg.mean()), 4),
            "verdict": ("informative" if rank_auc(pos, neg) > 0.55 else
                        "uninformative" if rank_auc(pos, neg) >= 0.5 else
                        "anti-informative: S is LOWER inside micro-expressions "
                        "than outside, so no decoder over S can localise them"),
        }
    if reach:
        frac = float(np.mean(reach))
        report["layer_2_threshold"] = {
            "mean_frac_events_reaching_tau_hi": round(frac, 4),
            "tau_hi": tau_hi,
            "verdict": (
                ("tau_hi is reached by %.0f%% of events, so it is not a recall "
                 "cap -- but a trigger this far below the curve's typical value "
                 "fires almost everywhere, which is a precision problem"
                 % (100 * frac)) if frac > 0.95 else
                ("tau_hi is reached by only %.0f%% of events: a hard recall cap "
                 "that no change to duration or merge settings can lift"
                 % (100 * frac))),
        }
    if best_all:
        found = sum(a for a, _, _ in best_all)
        hits = sum(b for _, b, _ in best_all)
        missed_entirely = n_truth - found
        report["layer_3_extent"] = {
            "n_found_any_overlap": found,
            "n_hit_at_threshold": hits,
            "n_truth": n_truth,
            "n_never_overlapped": missed_entirely,
            "recall_any_overlap": round(found / n_truth, 4) if n_truth else 0.0,
            "recall_at_threshold": round(hits / n_truth, 4) if n_truth else 0.0,
            "duration_ratio_median": (round(float(np.median(ratios)), 4)
                                      if ratios else None),
            "verdict": (
                ("%d of %d events are never overlapped by any interval: a "
                 "detection problem. Extent only explains the remaining %d."
                 % (missed_entirely, n_truth, found - hits))
                if missed_entirely > 0.5 * n_truth else
                ("intervals overlap %d of %d events but only %d reach the "
                 "threshold: an extent problem among the events that were "
                 "found, with %d never found at all"
                 % (found, n_truth, hits, missed_entirely))
                if found > hits else
                "intervals mostly miss the events outright"),
        }
    report["layer_4_filtering"] = filter_ceiling(pooled_pairs, iou_threshold)
    if taxonomy_total["fp"] or taxonomy_total["fn"]:
        n_fp = taxonomy_total["fp"] or 1
        n_fn = taxonomy_total["fn"] or 1
        taxonomy_total["fp_spurious_share"] = round(
            taxonomy_total["fp_spurious"] / n_fp, 4)
        taxonomy_total["fn_blind_share"] = round(
            taxonomy_total["fn_blind"] / n_fn, 4)
        spurious = taxonomy_total["fp_spurious"] / n_fp
        blind = taxonomy_total["fn_blind"] / n_fn
        taxonomy_total["verdict"] = (
            ("%.0f%% of false positives land on nothing annotated and %.0f%% of "
             "misses were never overlapped at all: the curve is firing in the "
             "wrong places, which is a signal defect no decoder can repair"
             % (100 * spurious, 100 * blind)) if spurious > 0.5 and blind > 0.5
            else
            ("%.0f%% of misses were overlapped but drawn wrong: a geometry "
             "defect, fixable downstream of the curve" % (100 * (1 - blind)))
            if blind <= 0.5 else
            ("%.0f%% of false positives overlap a real expression: the curve "
             "fires on genuine motion but the extent or the channel routing is "
             "wrong" % (100 * (1 - spurious))))
        report["layer_5_error_taxonomy"] = taxonomy_total
    if include_per_video:
        report["per_video"] = per_video
    return report
