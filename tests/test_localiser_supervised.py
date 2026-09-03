"""Checks for the supervised localiser.

The module exists because the analytic curve scores a *below-chance* frame AUC (0.4301),
so the tests that matter are the ones that would catch a model which looks trained but
has only learned the prior. Two in particular:

* ``test_learns_separable_signal`` builds a signal that is only visible *conditional on*
  a confound -- the same construction as the real failure, where facial change is
  informative only once head motion is accounted for. A model that ignores the confound
  scores ~0.5 here, so the check is not satisfiable by memorising the base rate.
* ``test_split_is_subject_disjoint`` pins the leakage guard. Splitting by window instead
  of by subject is the single easiest way to make every number in this module look good
  and mean nothing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mewm.training.localiser_supervised import (  # noqa: E402
    LocaliserCheckpoint, LocaliserTrainConfig, SupervisedLocaliser, TrainedLocaliser,
    VideoSample, _frame_auc, _robust_z, _split_subjects, _windows, evaluate_localiser,
    frame_features, frame_labels, train_localiser,
)

CHECKS = 0
FAILURES: List[str] = []


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL: {message}")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    interval: Tuple[int, int]
    is_micro: bool = True


@dataclass
class _Video:
    video_key: str = "s01_v01"
    subject: str = "s01"
    events: List[_Event] = field(default_factory=list)

    def micro_events(self):
        return [e for e in self.events if e.is_micro]

    def macro_events(self):
        return [e for e in self.events if not e.is_micro]

    def ground_truth_intervals(self, micro_only: bool = True):
        events = self.micro_events() if micro_only else self.events
        return [e.interval for e in events]


@dataclass
class _Representation:
    slot_activations: np.ndarray
    head_motion: np.ndarray = None
    coherence_gate: np.ndarray = None
    frames: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation and features
# ---------------------------------------------------------------------------


def test_robust_z() -> None:
    print("robust normalisation")

    constant = np.full((50, 3), 7.0)
    out = _robust_z(constant)
    check(np.all(np.isfinite(out)), "constant column must not produce NaN/inf")
    check(np.allclose(out, 0.0), "constant column must normalise to exactly zero")

    x = np.random.default_rng(0).normal(size=(500, 2)) * 3.0 + 10.0
    z = _robust_z(x)
    check(abs(float(np.median(z))) < 0.1, "median of normalised data should be ~0")
    check(0.7 < float(np.std(z)) < 1.4, "MAD scaling should put std near 1")

    # A macro-expression is the outlier that must NOT set the scale.
    spiked = np.concatenate([np.zeros((100, 1)), np.full((5, 1), 1000.0)])
    zs = _robust_z(spiked)
    check(np.all(np.abs(zs[:100]) < 1e-6),
          "an extreme outlier must not rescale the quiet majority")


def test_frame_features() -> None:
    print("frame features")

    n, k = 120, 8
    rng = np.random.default_rng(1)
    representation = _Representation(
        slot_activations=rng.normal(size=(n, k)),
        head_motion=rng.normal(size=(n, 6)),
        coherence_gate=np.ones((n, k)),
        frames=list(range(n)))

    features = frame_features(representation)
    check(features.shape[0] == n, "feature rows must equal frame count")
    check(features.dtype == np.float32, "features must be float32 for torch")
    check(np.all(np.isfinite(features)), "features must be finite")

    # activations + velocity + acceleration + head(6) + head_v(6) + speed(1) + gate(k)
    expected = 3 * k + 6 + 6 + 1 + k
    check(features.shape[1] == expected,
          f"expected {expected} feature columns, got {features.shape[1]}")

    with_error = frame_features(representation, slot_error=np.abs(rng.normal(size=(n, k))))
    check(with_error.shape[1] == expected + k + 1,
          "slot error should add K columns plus its summed curve")

    # Head pose must actually reach the features -- this is the whole design premise.
    no_head = _Representation(slot_activations=representation.slot_activations,
                              head_motion=None, coherence_gate=None,
                              frames=list(range(n)))
    check(frame_features(no_head).shape[1] == 3 * k,
          "dropping head motion and the gate must drop exactly their columns")

    nasty = _Representation(
        slot_activations=np.full((40, 4), np.nan), frames=list(range(40)))
    check(np.all(np.isfinite(frame_features(nasty))),
          "NaN activations must be sanitised, not propagated")

    try:
        frame_features(_Representation(slot_activations=np.zeros((2, 4)), frames=[0, 1]))
        check(False, "a 2-frame video should raise, not silently produce features")
    except ValueError:
        check(True, "short video raises ValueError")


def test_frame_labels() -> None:
    print("frame labels")

    video = _Video(events=[
        _Event((10, 14), is_micro=True),
        _Event((40, 60), is_micro=False),
    ])
    frames = list(range(100))
    labels, ignore = frame_labels(video, frames)

    check(labels.sum() == 5, "a (10,14) interval is 5 inclusive frames")
    check(np.all(labels[10:15] == 1.0), "micro interval must be labelled positive")
    check(labels[9] == 0.0 and labels[15] == 0.0, "labels must not bleed past the interval")

    check(np.all(ignore[40:61]), "macro frames must be masked out of the loss")
    check(not ignore[10:15].any(), "micro frames must never be masked")
    check(not ignore[0:10].any(), "quiet frames are real negatives, not ignored")
    check(ignore.sum() == 21, "only the 21 macro frames should be ignored")

    # Frames the representation never decoded must not be labelled by index accident.
    sparse = list(range(0, 100, 5))
    sparse_labels, _ = frame_labels(video, sparse)
    check(sparse_labels.sum() == 1,
          "with stride 5 only frame 10 falls inside (10,14)")


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def test_frame_auc() -> None:
    print("frame AUC")

    labels = np.array([0, 0, 1, 1], dtype=float)
    mask = np.ones(4, dtype=bool)

    check(close(_frame_auc(np.array([0.0, 0.1, 0.9, 1.0]), labels, mask), 1.0),
          "perfect separation is AUC 1.0")
    check(close(_frame_auc(np.array([1.0, 0.9, 0.1, 0.0]), labels, mask), 0.0),
          "perfectly inverted is AUC 0.0")
    check(close(_frame_auc(np.full(4, 5.0), labels, mask), 0.5),
          "a constant curve must be exactly 0.5, not 1.0 -- ties count as half")

    check(np.isnan(_frame_auc(np.arange(4.0), np.zeros(4), mask)),
          "no positives means AUC is undefined, not 0")
    check(np.isnan(_frame_auc(np.arange(4.0), np.ones(4), mask)),
          "no negatives means AUC is undefined, not 1")

    # Masked frames must be genuinely excluded, not merely down-weighted.
    scores = np.array([9.0, 0.0, 1.0])
    lab = np.array([0.0, 0.0, 1.0])
    check(close(_frame_auc(scores, lab, np.array([False, True, True])), 1.0),
          "masking the confusing negative must lift AUC to 1.0")
    check(close(_frame_auc(scores, lab, np.array([True, True, True])), 0.5),
          "unmasked, the same data is 0.5 -- proof the mask is load-bearing")


# ---------------------------------------------------------------------------
# Splitting and windowing
# ---------------------------------------------------------------------------


def _sample(subject: str, n: int = 300, positives=((100, 115),)) -> VideoSample:
    labels = np.zeros(n, dtype=np.float32)
    for a, b in positives:
        labels[a:b] = 1.0
    return VideoSample(
        video_key=f"{subject}_v", subject=subject,
        features=np.zeros((n, 4), dtype=np.float32),
        labels=labels, ignore=np.zeros(n, dtype=bool), frames=list(range(n)))


def test_split_is_subject_disjoint() -> None:
    print("subject-disjoint split")

    samples = [_sample(f"s{i:02d}") for i in range(8)]
    config = LocaliserTrainConfig(val_subject_fraction=0.25, seed=3)
    train, val = _split_subjects(samples, config)

    check(not (set(train) & set(val)), "train and val subjects must be disjoint")
    check(len(val) >= 1, "at least one validation subject is required")
    check(set(train) | set(val) == {s.subject for s in samples},
          "every subject must land on exactly one side")

    # A subject with no positive frames cannot validate a ranking.
    mixed = [_sample("a"), _sample("b"), _sample("c", positives=())]
    train2, val2 = _split_subjects(mixed, LocaliserTrainConfig(seed=3))
    check("c" in train2 and "c" not in val2,
          "an event-free subject must go to train, never to val")

    single = [_sample("only")]
    train3, val3 = _split_subjects(single, config)
    check(val3 == [] and train3 == ["only"],
          "one subject cannot be split; validation must be empty, not a copy of train")


def test_windows_cover_events() -> None:
    print("event-centred windowing")

    rng = np.random.default_rng(0)
    sample = _sample("s01", n=2000, positives=((500, 515), (1500, 1512)))
    spans = _windows(sample, window=256, rng=rng)

    positive = np.flatnonzero(sample.labels > 0)
    breaks = np.flatnonzero(np.diff(positive) > 1)
    for group in np.split(positive, breaks + 1):
        covered = any(a <= int(group.mean()) < b for a, b in spans)
        check(covered, f"event near frame {int(group.mean())} must be inside some window")

    check(all(0 <= a and b <= len(sample.labels) for a, b in spans),
          "windows must stay inside the video")
    check(len(spans) > 2, "uniform windows must be sampled alongside the event windows")

    short = _sample("s02", n=100)
    check(_windows(short, window=256, rng=rng) == [(0, 100)],
          "a video shorter than the window yields exactly one full-length span")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_model_shapes() -> None:
    print("model")
    import torch

    config = LocaliserTrainConfig(channels=16, dilations=(1, 2, 4))
    model = SupervisedLocaliser(n_features=11, config=config)

    out = model(torch.zeros(3, 128, 11))
    check(tuple(out.shape) == (3, 128), f"expected (3,128) logits, got {tuple(out.shape)}")
    check(model.receptive_field == 1 + 2 * 2 * (1 + 2 + 4),
          f"receptive field should be 15 frames, got {model.receptive_field}")

    full = SupervisedLocaliser(n_features=11)
    check(full.receptive_field >= 60,
          "the shipped dilations must span ~2s at 30fps to see an event in context")

    # Centred convolutions: a spike must influence outputs on BOTH sides, or the model
    # is causal and cannot use an offset to identify an onset.
    model.eval()
    with torch.no_grad():
        base = model(torch.zeros(1, 64, 11))
        spiked = torch.zeros(1, 64, 11)
        spiked[0, 32, :] = 5.0
        delta = (model(spiked) - base).abs()[0].numpy()
    check(delta[:32].sum() > 1e-6 and delta[33:].sum() > 1e-6,
          "a mid-sequence spike must change outputs before and after it")


def test_learns_separable_signal() -> None:
    print("training on a conditionally-separable signal")

    rng = np.random.default_rng(7)
    samples: List[VideoSample] = []
    for subject in range(6):
        n = 600
        head_speed = np.abs(rng.normal(size=(n, 1)))
        # The confound must DOMINATE the event, or the task is separable without
        # conditioning and the test proves nothing. 6x head speed against a 2.0 event
        # bump puts the marginal near chance while the conditional stays easy.
        facial = rng.normal(size=(n, 2)) * 0.5 + head_speed * 6.0

        labels = np.zeros(n, dtype=np.float32)
        for centre in (150, 300, 450):
            labels[centre:centre + 14] = 1.0
        # The event adds facial motion WITHOUT head motion. Marginally the facial
        # channel is dominated by the head confound, exactly as measured on casme_sq
        # (scene share 0.847); only a model that conditions on head speed can win.
        facial[labels > 0] += 2.0

        features = np.concatenate([facial, head_speed], axis=1).astype(np.float32)
        samples.append(VideoSample(
            video_key=f"s{subject}_v", subject=f"s{subject}",
            features=features, labels=labels,
            ignore=np.zeros(n, dtype=bool), frames=list(range(n))))

    # Confirm the task is actually confound-dominated: the raw facial channel alone
    # must be near chance, or the test proves nothing.
    pooled = np.concatenate([s.features[:, 0] for s in samples])
    pooled_labels = np.concatenate([s.labels for s in samples])
    naive = _frame_auc(pooled, pooled_labels, np.ones(pooled.size, dtype=bool))
    check(naive < 0.75,
          f"the naive marginal signal should be weak by construction, got {naive:.3f}")

    config = LocaliserTrainConfig(
        epochs=40, channels=32, dilations=(1, 2, 4, 8), window=256,
        batch_windows=4, device="cpu", patience=40, seed=11)
    checkpoint = train_localiser(samples, config, fold_name="synthetic")

    best = checkpoint.metrics["best_val_auc"]
    check(best > 0.85,
          f"trained AUC should clear 0.85 on a learnable task, got {best:.4f}")
    check(best > naive + 0.1,
          f"training must beat the naive marginal ({best:.3f} vs {naive:.3f})")
    check(checkpoint.metrics["n_positive_frames"] > 0, "positive frames must be counted")
    check(1.0 <= checkpoint.metrics["pos_weight"] <= config.pos_weight_cap,
          "pos_weight must be capped, not the raw imbalance ratio")


def test_pos_weight_is_capped() -> None:
    print("class-imbalance handling")

    # 3 positive frames in 4000: the raw ratio is >1000, which destabilises training.
    samples = [_sample(f"s{i}", n=4000, positives=((100, 103),)) for i in range(3)]
    for s in samples:
        s.features = np.random.default_rng(0).normal(size=(4000, 3)).astype(np.float32)

    config = LocaliserTrainConfig(epochs=1, channels=8, dilations=(1,),
                                  window=128, batch_windows=2, device="cpu", seed=5)
    checkpoint = train_localiser(samples, config, fold_name="imbalance")
    check(checkpoint.metrics["pos_weight"] == config.pos_weight_cap,
          "an extreme imbalance must saturate the cap")


# ---------------------------------------------------------------------------
# Checkpoint provenance
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_and_leak_guard(tmp: Path) -> None:
    print("checkpoint provenance")

    samples = [_sample(f"s{i:02d}", n=400) for i in range(4)]
    for s in samples:
        s.features = np.random.default_rng(2).normal(size=(400, 3)).astype(np.float32)

    config = LocaliserTrainConfig(epochs=2, channels=8, dilations=(1, 2),
                                  window=128, batch_windows=2, device="cpu", seed=5)
    checkpoint = train_localiser(samples, config, fold_name="fold_s07")

    path = tmp / "localiser.pt"
    checkpoint.save(path)
    check(path.exists(), "checkpoint file must be written")

    loaded = LocaliserCheckpoint.load(path)
    check(loaded.n_features == checkpoint.n_features, "feature width must round-trip")
    check(loaded.fold_name == "fold_s07", "fold name must round-trip")
    check(sorted(loaded.train_subjects) == sorted(checkpoint.train_subjects),
          "train subjects must round-trip -- they are the contamination proof")

    seen = (loaded.train_subjects + loaded.val_subjects)[0]
    try:
        loaded.assert_excludes([seen])
        check(False, "predicting on a trained subject must raise")
    except RuntimeError as exc:
        check(seen in str(exc), "the error must name the leaked subject")

    loaded.assert_excludes(["s99_unseen"])
    check(True, "an unseen subject passes the guard")

    localiser = TrainedLocaliser(loaded, device="cpu")
    representation = _Representation(
        slot_activations=np.random.default_rng(3).normal(size=(200, 4)),
        frames=list(range(200)))
    try:
        localiser.curve(representation)
        check(False, "a feature-width mismatch must raise, not silently mis-predict")
    except ValueError as exc:
        check("does not match" in str(exc), "mismatch error must explain the cause")


def test_evaluate_reports_spread(tmp: Path) -> None:
    print("evaluation reporting")

    samples = [_sample(f"s{i:02d}", n=300) for i in range(3)]
    for s in samples:
        s.features = np.random.default_rng(4).normal(size=(300, 3)).astype(np.float32)

    config = LocaliserTrainConfig(epochs=1, channels=8, dilations=(1,),
                                  window=128, batch_windows=2, device="cpu", seed=5)
    checkpoint = train_localiser(samples, config, fold_name="fold_eval")

    held_out = [_sample("s99", n=300)]
    held_out[0].features = np.random.default_rng(5).normal(size=(300, 3)).astype(np.float32)

    report = evaluate_localiser(checkpoint, held_out, device="cpu")
    check(report["n_videos"] == 1, "every held-out video must appear")
    check("per_video" in report and len(report["per_video"]) == 1,
          "per-video detail is required -- a pooled mean hides which videos improved")
    check("n_above_0.7" in report and "n_above_0.5" in report,
          "the spread counters must be reported alongside the mean")
    check(report["per_video"][0]["subject"] == "s99", "provenance must be carried through")


def main() -> int:
    import tempfile

    print("=" * 68)
    print("supervised localiser checks")
    print("=" * 68)

    test_robust_z()
    test_frame_features()
    test_frame_labels()
    test_frame_auc()
    test_split_is_subject_disjoint()
    test_windows_cover_events()
    test_model_shapes()
    test_learns_separable_signal()
    test_pos_weight_is_capped()
    with tempfile.TemporaryDirectory() as d:
        test_checkpoint_roundtrip_and_leak_guard(Path(d))
        test_evaluate_reports_spread(Path(d))

    print("=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED of {CHECKS} checks")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
