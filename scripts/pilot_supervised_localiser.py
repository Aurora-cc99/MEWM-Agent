"""Pilot: does a supervised localiser beat the analytic curve on real casme_sq data?

Stage I is by far the most expensive part of this and it does not depend on the fold, so
features are extracted once for every annotated video and cached to a single npz. Every
LOSO fold then trains off the cache. Re-running with the cache present skips extraction
entirely.

Baseline to beat, measured on the same 31 videos / 52 events:
    pooled frame AUC 0.4301, 8/31 videos above 0.5, 0/31 above 0.7
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mewm.config import load_config
from mewm.data.datasets import load_dataset
from mewm.training.localiser_supervised import (
    LocaliserTrainConfig, VideoSample, _frame_auc, build_frame_dataset,
    evaluate_localiser, train_localiser,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("pilot")

CACHE = Path(__file__).resolve().parent.parent / ".runlogs" / "localiser_features.npz"
OUT = Path(__file__).resolve().parent.parent / ".runlogs" / "localiser_pilot.json"


def build_cache(dataset: str, limit: int = 0, stride: int = 1, max_frames: int = 0):
    config = load_config()
    index = load_dataset(dataset)
    videos = [v for v in index.videos if v.micro_events()]
    if limit:
        videos = videos[:limit]
    LOGGER.info("extracting features for %d annotated video(s)", len(videos))

    started = time.time()
    samples = build_frame_dataset(
        videos, config, max_frames=max_frames, stride=stride, require_events=True)
    LOGGER.info("extraction took %.1f s for %d sample(s)",
                time.time() - started, len(samples))

    blob = {}
    for i, s in enumerate(samples):
        blob[f"features_{i}"] = s.features
        blob[f"labels_{i}"] = s.labels
        blob[f"ignore_{i}"] = s.ignore
        blob[f"meta_{i}"] = np.array([s.video_key, s.subject], dtype=object)
    blob["n"] = np.array([len(samples)])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **blob)
    LOGGER.info("cached %d sample(s) to %s", len(samples), CACHE)
    return samples


def load_cache():
    blob = np.load(CACHE, allow_pickle=True)
    n = int(blob["n"][0])
    samples = []
    for i in range(n):
        meta = blob[f"meta_{i}"]
        samples.append(VideoSample(
            video_key=str(meta[0]), subject=str(meta[1]),
            features=blob[f"features_{i}"], labels=blob[f"labels_{i}"],
            ignore=blob[f"ignore_{i}"], frames=[]))
    LOGGER.info("loaded %d cached sample(s) from %s", len(samples), CACHE)
    return samples


def analytic_baseline(samples):
    """Frame AUC of the summed analytic slot error, recomputed from the cache.

    The summed-error column is the last feature block appended by ``frame_features``
    when a slot error is supplied, so the baseline is measured on exactly the same
    frames and the same mask as the trained model -- otherwise the comparison would be
    between two different denominators.
    """
    aucs = []
    for s in samples:
        # activations(K) + velocity(K) + acceleration(K) + err(K) + err_sum(1) + ...
        k = (s.features.shape[1] - 1 - 6 - 6 - 1) // 5
        col = 4 * k
        auc = _frame_auc(s.features[:, col], s.labels, ~s.ignore)
        if np.isfinite(auc):
            aucs.append(auc)
    return {"mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "n_above_0.5": int(sum(1 for a in aucs if a > 0.5)),
            "n_above_0.7": int(sum(1 for a in aucs if a > 0.7)),
            "n": len(aucs)}


def loso(samples, epochs: int = 60):
    """One fold per held-out subject. Trained on the rest, evaluated on the held-out."""
    subjects = sorted({s.subject for s in samples})
    LOGGER.info("LOSO over %d subject(s): %s", len(subjects), subjects)

    folds = []
    for subject in subjects:
        train = [s for s in samples if s.subject != subject]
        test = [s for s in samples if s.subject == subject]
        if not test or sum(s.n_positive for s in test) == 0:
            LOGGER.info("fold %s: no positive frames held out, skipping", subject)
            continue
        if len({s.subject for s in train}) < 2:
            LOGGER.info("fold %s: too few training subjects, skipping", subject)
            continue

        config = LocaliserTrainConfig(epochs=epochs, seed=20260828)
        checkpoint = train_localiser(train, config, fold_name=f"loso_{subject}")
        report = evaluate_localiser(checkpoint, test, device=config.device)
        LOGGER.info("fold %s: held-out AUC %.4f (val %.4f)",
                    subject, report["pooled_mean_auc"], checkpoint.metrics["best_val_auc"])
        folds.append({
            "subject": subject,
            "val_auc": checkpoint.metrics["best_val_auc"],
            "test_auc": report["pooled_mean_auc"],
            "n_test_videos": report["n_videos"],
            "per_video": report["per_video"],
        })
    return folds


def main() -> int:
    rebuild = "--rebuild" in sys.argv
    limit = 0
    for arg in sys.argv:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])

    if rebuild or not CACHE.exists():
        samples = build_cache("casme_sq", limit=limit)
    else:
        samples = load_cache()

    if not samples:
        print("no samples", file=sys.stderr)
        return 1

    baseline = analytic_baseline(samples)
    LOGGER.info("analytic baseline: mean AUC %.4f, %d/%d above 0.5, %d above 0.7",
                baseline["mean_auc"], baseline["n_above_0.5"], baseline["n"],
                baseline["n_above_0.7"])

    folds = loso(samples)
    test_aucs = [f["test_auc"] for f in folds if np.isfinite(f["test_auc"])]
    per_video = [v for f in folds for v in f["per_video"] if np.isfinite(v["auc"])]

    summary = {
        "n_samples": len(samples),
        "n_subjects": len({s.subject for s in samples}),
        "n_positive_frames": int(sum(s.n_positive for s in samples)),
        "analytic_baseline": baseline,
        "supervised_loso": {
            "mean_fold_auc": float(np.mean(test_aucs)) if test_aucs else float("nan"),
            "n_folds": len(folds),
            "per_video_mean_auc": (
                float(np.mean([v["auc"] for v in per_video])) if per_video else float("nan")),
            "n_videos_above_0.5": int(sum(1 for v in per_video if v["auc"] > 0.5)),
            "n_videos_above_0.7": int(sum(1 for v in per_video if v["auc"] > 0.7)),
            "n_videos": len(per_video),
        },
        "folds": folds,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 68)
    print("analytic  : mean AUC %.4f  above0.5 %d/%d  above0.7 %d"
          % (baseline["mean_auc"], baseline["n_above_0.5"], baseline["n"],
             baseline["n_above_0.7"]))
    s = summary["supervised_loso"]
    print("supervised: mean AUC %.4f  above0.5 %d/%d  above0.7 %d"
          % (s["per_video_mean_auc"], s["n_videos_above_0.5"], s["n_videos"],
             s["n_videos_above_0.7"]))
    print("=" * 68)
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
