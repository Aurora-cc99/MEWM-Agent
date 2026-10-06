"""Temporal IoU histogram diagnostics for spotting evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

N_BINS = 10


def truth_best_ious(
    proposals: Sequence[Tuple[int, int]],
    truths: Sequence[Tuple[int, int]],
) -> List[float]:
    best = []
    for truth in truths:
        peak = 0.0
        for proposal in proposals:
            lo = max(truth[0], proposal[0])
            hi = min(truth[1], proposal[1])
            inter = max(0, hi - lo + 1)
            union = ((truth[1] - truth[0] + 1) + (proposal[1] - proposal[0] + 1)
                     - inter)
            if union > 0:
                value = inter / union
                if value > peak:
                    peak = value
        best.append(round(peak, 4))
    return best


def bin_index(iou: float, n_bins: int = N_BINS) -> int:
    return min(n_bins - 1, max(0, int(iou * n_bins)))


def distribution(ious: Sequence[float], n_bins: int = N_BINS) -> Dict[str, object]:
    counts = [0] * n_bins
    for value in ious:
        counts[bin_index(value, n_bins)] += 1
    bins = [
        {
            "bin": i,
            "range": f"[{i / n_bins:.1f}, {(i + 1) / n_bins:.1f})",
            "count": counts[i],
            "share": round(counts[i] / len(ious), 4) if ious else 0.0,
        }
        for i in range(n_bins)
    ]
    return {
        "n_truths": len(ious),
        "n_bins": n_bins,
        "bins": bins,
        "n_above_0.5": int(sum(1 for value in ious if value > 0.5)),
        "n_zero": int(sum(1 for value in ious if value <= 0.0)),
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }


def render_histogram(ious: Sequence[float], path: Path,
                     threshold: float = 0.5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = [0] * N_BINS
    for value in ious:
        counts[bin_index(value)] += 1
    labels = [f"{i / N_BINS:.1f}-{(i + 1) / N_BINS:.1f}" for i in range(N_BINS)]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(labels, counts, color="#4c72b0", edgecolor="white")
    threshold_bin = int(threshold * N_BINS) - 0.5
    axis.axvline(x=threshold_bin, color="red", linestyle="--", linewidth=1.4,
                 label=f"TP threshold ({threshold})")
    for bar, count in zip(bars, counts):
        if count:
            axis.text(bar.get_x() + bar.get_width() / 2, count + 0.15, str(count),
                      ha="center", va="bottom", fontsize=9)
    axis.set_xlabel("best per-truth IoU")
    axis.set_ylabel("ground-truth events")
    axis.set_title("IoU distribution over micro-expression ground truth")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_distribution(root: Path, rows: List[Dict[str, object]],
                      ious: List[float], threshold: float = 0.5) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stats = distribution(ious)
    payload = {"statistics": stats, "per_truth": rows}
    (root / "iou_distribution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if rows:
        with open(root / "iou_distribution.csv", "w", newline="",
                  encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    render_histogram(ious, root / "iou_distribution.png", threshold=threshold)

    tp_rows = [row for row in rows if row.get("is_tp")]
    tp_ious = [value for row, value in zip(rows, ious) if row.get("is_tp")]
    if len(tp_ious) != len(tp_rows):
        tp_ious = [float(row["best_iou"]) for row in tp_rows]
    tp_stats = distribution(tp_ious)
    tp_payload = {"statistics": tp_stats, "per_truth": tp_rows,
                 "note": "restricted to true-positive ground-truth events "
                         f"(best_iou > {threshold})"}
    (root / "iou_distribution_tp.json").write_text(
        json.dumps(tp_payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if tp_rows:
        with open(root / "iou_distribution_tp.csv", "w", newline="",
                  encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tp_rows)
        render_histogram(tp_ious, root / "iou_distribution_tp.png",
                         threshold=threshold)


__all__ = ["N_BINS", "truth_best_ious", "distribution", "render_histogram",
           "save_distribution"]
