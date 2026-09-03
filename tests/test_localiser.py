"""Tests for the M2b extent decoder and the localisation diagnostics.

Two things are being pinned here, and they are different in kind.

The decoder is a *mechanism*: given a curve with two bursts a known distance
apart, it must recover that distance and place its apex in the trough between
them. That is checkable on synthetic input with an exact expected answer, and it
is checked below at several durations, at two frame rates, and against the
failure mode that motivated the whole design (an interval on one flank).

The diagnostics are an *instrument*: their job is to return the right verdict
about a signal, including the uncomfortable verdict. So the tests feed them a
curve that is deliberately anti-correlated with the ground truth and require
that the AUC comes back below 0.5 -- an instrument that cannot report bad news
is worse than no instrument, because a run of it would read as a clean bill of
health.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mewm.config import SpottingConfig  # noqa: E402
from mewm.engines.m2_localiser import MicroLocaliser, double_burst_kernel  # noqa: E402
from mewm.engines.m2_spotting import CandidateInterval, PhysioEvent  # noqa: E402
from mewm.eval.localisation_diagnostics import (  # noqa: E402
    error_taxonomy,
    extent_anatomy, filter_ceiling, localisation_report, rank_auc, signal_auc,
    threshold_reachability,
)

PASS = FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def iou_of(interval, on: int, off: int) -> float:
    """IoU of a CandidateInterval against a ground-truth span, inclusive."""
    inter = max(0, min(interval.t_off, off) - max(interval.t_on, on) + 1)
    union = ((interval.t_off - interval.t_on + 1) + (off - on + 1) - inter)
    return inter / union if union else 0.0


def synth_curve(length: int, events, baseline=1.0, amplitude=8.0, flank=0.3,
                seed=0) -> np.ndarray:
    """A velocity-like curve: each event writes a burst at onset and at offset.

    This is the empirically observed shape of the real S curve, not an
    idealisation chosen to flatter the decoder -- prediction error is large where
    the face is *moving* and small at the apex, where velocity passes through
    zero. Any decoder that assumes a single central hump gets the apex wrong on
    input like this.
    """
    rng = np.random.default_rng(seed)
    s = baseline + 0.15 * rng.standard_normal(length)
    for on, off in events:
        duration = off - on + 1
        width = max(1, int(round(flank * duration)))
        for lo in (on, off - width + 1):
            hi = min(length, lo + width)
            lo = max(0, lo)
            if hi > lo:
                s[lo:hi] += amplitude
    return s


# ---------------------------------------------------------------------------
print("[kernel]")

for duration in (5, 9, 14, 21, 40):
    k = double_burst_kernel(duration, flank_fraction=0.3, trough_weight=1.0)
    check(f"kernel at duration {duration} has that length", k.size == duration,
          f"got {k.size}")
    check(f"kernel at duration {duration} is zero-mean",
          abs(float(k.sum())) < 1e-9, f"sum={k.sum():.3e}")
    check(f"kernel at duration {duration} is unit-norm",
          abs(float(np.linalg.norm(k)) - 1.0) < 1e-9)
    check(f"kernel at duration {duration} is positive on both flanks",
          k[0] > 0 and k[-1] > 0)
    check(f"kernel at duration {duration} dips in the middle",
          k[duration // 2] < 0, f"centre={k[duration // 2]:.3f}")
    check(f"kernel at duration {duration} is symmetric",
          np.allclose(k, k[::-1], atol=1e-12))

k_tiny = double_burst_kernel(1)
check("a sub-minimum duration is clamped rather than raising", k_tiny.size >= 3)
check("zero trough weight still yields a usable kernel",
      abs(float(np.linalg.norm(double_burst_kernel(9, trough_weight=0.0))) - 1.0)
      < 1e-9)

k_wide = double_burst_kernel(9, flank_fraction=0.9)
check("an over-wide flank fraction cannot erase the trough",
      float(k_wide[4]) < float(k_wide[0]),
      "flanks must stay above the centre for the kernel to mean anything")


# ---------------------------------------------------------------------------
print("\n[duration bank]")

cfg = SpottingConfig()
loc30 = MicroLocaliser(cfg, fps=30.0)
loc200 = MicroLocaliser(cfg, fps=200.0)

# The default config imposes NO length prior (min/max_micro_seconds are 0), so the
# bank is derived from the curve it is asked about. These checks are on the
# curve-aware form; the bare property is only the introspective placeholder.
bank30 = loc30.duration_bank_for(1200)
bank200 = loc200.duration_bank_for(8000)

check("the bank is non-empty at 30 fps", len(bank30) > 0)
check("the bank is sorted and unique", list(bank30) == sorted(set(bank30)))
check("with no configured floor the bank starts at the shortest decodable extent",
      min(bank30) == MicroLocaliser.MIN_DECODABLE_FRAMES, str(bank30))
check("with no configured ceiling the bank scales with the curve, not a prior",
      max(loc30.duration_bank_for(2400)) > max(loc30.duration_bank_for(1200)),
      f"{loc30.duration_bank_for(1200)} vs {loc30.duration_bank_for(2400)}")
check("a longer extent than any preset micro ceiling is searchable",
      max(bank30) > int(round(0.6 * 30)), str(bank30))
check("both frame rates search the same physical duration when the curve is the "
      "same length in seconds",
      abs(max(bank200) / 200.0 - max(bank30) / 30.0) < 0.05,
      f"{max(bank200)}@200fps vs {max(bank30)}@30fps")

# Opting back in to a bounded range must still work -- the bounds are off by
# default, not removed.
bounded = SpottingConfig(min_micro_seconds=0.20, max_micro_seconds=0.60)
loc_bounded = MicroLocaliser(bounded, fps=30.0)
bank_b = loc_bounded.duration_bank_for(1200)
check("a configured floor clamps the bank",
      min(bank_b) >= int(round(bounded.min_micro_seconds * 30)) - 1, str(bank_b))
check("a configured ceiling clamps the bank",
      max(bank_b) <= int(round(bounded.max_micro_seconds * 30)) + 1, str(bank_b))


# ---------------------------------------------------------------------------
print("\n[extent recovery]")

for truth_duration in (9, 14, 17):
    on = 60
    off = on + truth_duration - 1
    curve = synth_curve(240, [(on, off)], seed=truth_duration)
    out = loc30.localise(curve, 0, None, [])
    check(f"a {truth_duration}-frame event yields at least one interval",
          len(out) > 0)
    if not out:
        continue
    best = max(out, key=lambda iv: iou_of(iv, on, off))
    iou = iou_of(best, on, off)
    check(f"a {truth_duration}-frame event is recovered at IoU >= 0.5",
          iou >= 0.5, f"IoU={iou:.3f} claim=({best.t_on},{best.t_off}) "
                      f"truth=({on},{off})")
    ratio = (best.t_off - best.t_on + 1) / truth_duration
    check(f"the recovered extent is within 40% of a {truth_duration}-frame truth",
          0.6 <= ratio <= 1.4, f"ratio={ratio:.3f}")
    centre_error = abs(best.apex - (on + off) / 2) / truth_duration
    check(f"the apex of a {truth_duration}-frame event lands centrally",
          centre_error <= 0.3,
          f"apex={best.apex} expected~{(on + off) / 2:.1f} "
          f"(normalised error {centre_error:.2f})")


# ---------------------------------------------------------------------------
print("\n[the failure mode it was built to fix]")

on, off = 80, 93
curve = synth_curve(240, [(on, off)], seed=7)
argmax = int(np.argmax(curve))
check("the raw curve's argmax sits on a flank, not the apex",
      abs(argmax - (on + off) / 2) > 0.25 * (off - on + 1),
      f"argmax={argmax}, event centre={(on + off) / 2:.1f} -- this is why "
      f"peak-of-S apexing is wrong on a velocity-like curve")
out = loc30.localise(curve, 0, None, [])
if out:
    best = max(out, key=lambda iv: max(0, min(iv.t_off, off) - max(iv.t_on, on) + 1))
    check("the decoder's apex is closer to the truth centre than the argmax is",
          abs(best.apex - (on + off) / 2) < abs(argmax - (on + off) / 2),
          f"decoder apex={best.apex}, argmax={argmax}, truth centre="
          f"{(on + off) / 2:.1f}")

check("a flat curve yields no intervals rather than spurious ones",
      len(loc30.localise(np.full(200, 2.0), 0, None, [])) == 0)
check("an empty curve is handled", len(loc30.localise(np.array([]), 0, None, [])) == 0)
check("a curve shorter than the smallest kernel is handled",
      len(loc30.localise(np.array([1.0, 2.0, 1.0]), 0, None, [])) == 0)

shifted = loc30.localise(synth_curve(240, [(80, 93)], seed=7), 1000, None, [])
check("t_start offsets the emitted frame indices",
      bool(shifted) and all(iv.t_on >= 1000 for iv in shifted),
      "intervals must be reported in the video's own frame numbering")


# ---------------------------------------------------------------------------
print("\n[separation and caps]")

many = synth_curve(600, [(60, 73), (200, 213), (360, 373), (500, 513)], seed=3)
out = loc30.localise(many, 0, None, [])
check("four well-separated events yield several intervals", len(out) >= 2,
      f"n={len(out)}")
check("no more intervals than the configured per-video cap",
      len(out) <= cfg.localiser_max_per_video, f"n={len(out)}")
pairs = [(a, b) for i, a in enumerate(out) for b in out[i + 1:]]
worst = max((max(0, min(a.t_off, b.t_off) - max(a.t_on, b.t_on) + 1)
             / ((a.t_off - a.t_on + 1) + (b.t_off - b.t_on + 1)
                - max(0, min(a.t_off, b.t_off) - max(a.t_on, b.t_on) + 1))
             for a, b in pairs), default=0.0)
check("surviving intervals respect the NMS overlap ceiling",
      worst <= cfg.localiser_nms_iou + 1e-9, f"worst pairwise IoU={worst:.3f}")
check("intervals are emitted in descending score order",
      all(out[i].peak_S >= out[i + 1].peak_S for i in range(len(out) - 1)),
      "downstream top-k selection depends on this ordering")
check("every interval carries a note naming the decoder",
      all("M2b" in (iv.notes or "") for iv in out),
      "provenance has to survive into the evidence block")
check("every apex lies within its own interval",
      all(iv.t_on <= iv.apex <= iv.t_off for iv in out))


# ---------------------------------------------------------------------------
print("\n[diagnostics: rank AUC]")

check("a perfectly separated pair scores 1.0",
      rank_auc(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0)
check("a reversed pair scores 0.0",
      rank_auc(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == 0.0)
check("all-ties score 0.5",
      abs(rank_auc(np.ones(5), np.ones(7)) - 0.5) < 1e-12,
      "ties must not be silently broken in either side's favour")
check("an empty side yields NaN rather than a misleading 0.5",
      np.isnan(rank_auc(np.array([]), np.ones(3))))


# ---------------------------------------------------------------------------
print("\n[diagnostics: the verdicts]")

good = synth_curve(300, [(50, 63), (180, 193)], seed=1)
sig = signal_auc(good, [(50, 63), (180, 193)], [(50, 63), (180, 193)])
check("a curve that fires on the events reports AUC above 0.5",
      sig["status"] == "ok" and sig["auc_vs_unannotated"] > 0.5,
      str(sig.get("auc_vs_unannotated")))

anti = np.full(300, 10.0)
for a, b in ((50, 63), (180, 193)):
    anti[a:b + 1] = 1.0
bad = signal_auc(anti, [(50, 63), (180, 193)], [(50, 63), (180, 193)])
check("a curve that is QUIETER inside the events reports AUC below 0.5",
      bad["auc_vs_unannotated"] < 0.5, str(bad.get("auc_vs_unannotated")))
check("the anti-correlated case reports mean_inside below mean_outside",
      bad["mean_inside"] < bad["mean_outside"])
check("no ground truth is reported as unavailable, not as a score of zero",
      signal_auc(good, [])["status"] == "unavailable")
check("ground truth outside the curve is reported as unavailable",
      signal_auc(np.ones(10), [(500, 520)])["status"] == "unavailable")

macro_only = signal_auc(good, [(50, 63)], [(50, 63), (180, 193)])
check("macro-expression frames are excluded from the negatives",
      macro_only["n_frames_outside"] < 300 - 14,
      "counting macro frames as negatives would credit the curve for firing "
      "on the wrong event class")

thr = threshold_reachability(good, [(50, 63), (180, 193)], tau_hi=3.5)
check("threshold reachability reports a fraction in [0,1]",
      0.0 <= thr["frac_events_reaching_tau_hi"] <= 1.0)
check("events on a strong curve reach a tau of 3.5",
      thr["frac_events_reaching_tau_hi"] == 1.0)
check("a tau above every sample is reported as unreachable",
      threshold_reachability(good, [(50, 63)], tau_hi=1e6)
      ["frac_events_reaching_tau_hi"] == 0.0)
check("the peak percentile is reported on the video's own distribution",
      0.0 <= thr["peak_percentile_median"] <= 100.0)


# ---------------------------------------------------------------------------
print("\n[diagnostics: extent anatomy]")

truth = [(100, 113)]
exact = extent_anatomy([(100, 113, 106, 9.0)], truth, 0.5)
check("an exact match is one hit", exact["n_hit_at_threshold"] == 1)
check("an exact match has IoU 1.0", exact["best_iou_median"] == 1.0)
check("an exact match has duration ratio 1.0", exact["duration_ratio_median"] == 1.0)

half = extent_anatomy([(100, 105, 103, 9.0)], truth, 0.5)
check("a too-short interval overlaps but does not hit",
      half["n_found_any_overlap"] == 1 and half["n_hit_at_threshold"] == 0,
      "this is the arithmetic that caps the real score: for an interval "
      "contained in the truth, IoU is exactly the duration ratio")
check("a too-short interval reports a ratio below half",
      half["duration_ratio_median"] < 0.5)

# The boundary itself, pinned: 7 frames inside a 14-frame truth is IoU exactly
# 0.500, because the union is just the truth. So a median duration ratio of 0.52
# -- what the engine actually produces -- is not by itself disqualifying; it
# leaves almost no margin, and the observed median IoU is far below 0.52 only
# because the intervals are also phase-shifted onto the event's flanks.
boundary = extent_anatomy([(100, 106, 103, 9.0)], truth, 0.5)
check("a contained interval at ratio exactly 0.5 has IoU exactly 0.5",
      boundary["best_iou_median"] == 0.5)
check("IoU exactly at the threshold counts as a hit",
      boundary["n_hit_at_threshold"] == 1,
      "MEGC scores TP at IoU >= 0.5, inclusive")

flanks = extent_anatomy([(96, 102, 99, 9.0), (111, 117, 114, 8.0)], truth, 0.5)
hist = flanks["apex_position"]["histogram"]
check("two flank intervals put no apex in the central half",
      flanks["apex_position"]["central_half"] == 0.0,
      "this is the double-burst signature the diagnostics exist to name")
check("the flank histogram has mass on both sides of the centre",
      hist["[-0.25,0.00)"] + hist["[0.00,0.25)"] > 0
      and hist["[0.75,1.00)"] + hist["[1.00,1.25)"] > 0, str(hist))
check("no intervals is reported without dividing by zero",
      extent_anatomy([], truth, 0.5)["n_hit_at_threshold"] == 0)
check("no ground truth is reported as unavailable",
      extent_anatomy([(1, 2, 1, 1.0)], [], 0.5)["status"] == "unavailable")
check("the IoU curve is monotone non-increasing in the threshold",
      all(exact["iou_curve"][str(a)] >= exact["iou_curve"][str(b)]
          for a, b in zip((0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                          (0.2, 0.3, 0.4, 0.5, 0.6, 0.7))))


# ---------------------------------------------------------------------------
print("\n[diagnostics: the filtering ceiling]")

noisy = [([(100, 113, 106, 9.0)] + [(300 + 20 * i, 306 + 20 * i, 303 + 20 * i, 5.0)
                                    for i in range(50)], [(100, 113)])]
ceil = filter_ceiling(noisy, 0.5)
check("as-is precision is crushed by 50 false positives",
      ceil["as_is"]["precision"] < 0.05, str(ceil["as_is"]))
check("the oracle filter recovers F1 1.0 when a true hit is present",
      ceil["oracle_filter"]["f1"] == 1.0, str(ceil["oracle_filter"]))
check("filtering headroom is reported as positive here",
      ceil["headroom_from_filtering"] > 0.9)

hopeless = [([(200, 206, 203, 9.0)] * 1, [(100, 113)])]
ceil2 = filter_ceiling(hopeless, 0.5)
check("with no true hit to keep, the oracle filter scores zero too",
      ceil2["oracle_filter"]["f1"] == 0.0,
      "this is the finding that rules out reranking: a perfect filter over "
      "proposals that never reach the threshold buys exactly nothing")
check("the oracle filter cannot change recall",
      ceil2["oracle_filter"]["recall"] == ceil2["as_is"]["recall"])


# ---------------------------------------------------------------------------
print("\n[diagnostics: the error taxonomy]")

# One micro truth at (100,113), one macro truth at (300,340).
tax = error_taxonomy(
    intervals=[(100, 113, 106, 9.0),      # exact -> TP
               (104, 110, 107, 8.0),      # overlaps the micro, too short -> extent
               (305, 330, 315, 7.0),      # lands on the macro event -> macro
               (700, 712, 706, 6.0)],     # nothing there -> spurious
    micro_truth=[(100, 113)],
    macro_truth=[(300, 340)],
    iou_threshold=0.5)
check("an exact interval is the TP", tax["tp"] == 1)
check("an overlapping-but-short interval is an extent FP", tax["fp_extent"] == 1)
check("an interval on a macro event is a macro FP", tax["fp_macro"] == 1,
      "the curve fired on real motion; the duration router sent it to micro")
check("an interval on nothing is a spurious FP", tax["fp_spurious"] == 1)
check("the three FP causes sum to the FP total", tax["fp"] == 3)

# A miss that was touched, versus a miss that was never touched. The interval is
# 6 frames inside a 14-frame truth (IoU 0.43); 7 frames would be IoU exactly 0.5
# and would score as a hit instead -- see the boundary pinned above.
tax2 = error_taxonomy([(104, 109, 107, 8.0)], [(100, 113), (500, 513)], [], 0.5)
check("a miss that was overlapped is a geometry FN", tax2["fn_geometry"] == 1)
check("a miss that was never overlapped is a blind FN", tax2["fn_blind"] == 1)
check("the two FN causes sum to the FN total", tax2["fn"] == 2)
check("blind share is reported", tax2["fn_blind_share"] == 0.5)
check("no intervals makes every miss blind",
      error_taxonomy([], [(1, 10)], [], 0.5)["fn_blind"] == 1)
check("an empty problem does not divide by zero",
      error_taxonomy([], [], [], 0.5)["fn_blind_share"] == 0.0)

# The distinction the taxonomy exists to force: two runs with identical TP/FN
# totals but opposite causes must not read the same.
spray = error_taxonomy([(96, 104, 100, 5.0), (108, 118, 113, 4.0)],
                       [(100, 113)], [], 0.5)
silent = error_taxonomy([(700, 712, 706, 5.0)], [(100, 113)], [], 0.5)
check("both runs miss the single event", spray["fn"] == silent["fn"] == 1)
check("but only one of them ever touched it",
      spray["fn_geometry"] == 1 and silent["fn_blind"] == 1,
      "identical FN counts, different fixes: one needs a decoder, the other "
      "needs a different signal")


print("\n[diagnostics: the assembled report]")


class _Event:
    def __init__(self, on, off, kind):
        self._i, self.kind = (on, off), kind

    @property
    def interval(self):
        return self._i


class _Video:
    def __init__(self, key, subject, events):
        self.video_key, self.subject, self.events = key, subject, events

    def micro_events(self):
        return [e for e in self.events if e.kind == "micro"]


videos = [
    _Video("v1", "s01", [_Event(50, 63, "micro"), _Event(150, 200, "macro")]),
    _Video("v2", "s02", [_Event(80, 93, "micro")]),
    _Video("v3", "s03", [_Event(40, 60, "macro")]),
]
curves = {"v1": synth_curve(300, [(50, 63)], seed=11),
          "v2": synth_curve(300, [(80, 93)], seed=12),
          "v3": synth_curve(300, [(40, 60)], seed=13)}
ivs = {"v1": [(50, 63, 56, 9.0)], "v2": [(80, 85, 83, 8.0)]}
rep = localisation_report(videos, curves, ivs, tau_hi=3.5, iou_threshold=0.5)

check("the report covers only videos with micro-expression truth",
      rep["n_videos_with_micro_truth"] == 2,
      "v3 is macro-only and must be excluded, per the reporting scope")
check("the report states its scope", "micro" in rep["scope"])
check("the micro event count excludes macro events",
      rep["n_micro_truth_events"] == 2)
check("layer 1 is reported", "layer_1_signal" in rep)
check("layer 2 is reported", "layer_2_threshold" in rep)
check("layer 3 is reported", "layer_3_extent" in rep)
check("layer 4 is reported", "layer_4_filtering" in rep)
check("layer 1 carries a plain-language verdict",
      isinstance(rep["layer_1_signal"]["verdict"], str)
      and len(rep["layer_1_signal"]["verdict"]) > 10)
check("layer 3 separates overlap from threshold hits",
      rep["layer_3_extent"]["n_found_any_overlap"] == 2
      and rep["layer_3_extent"]["n_hit_at_threshold"] == 1,
      "v1 matches exactly and hits; v2 overlaps at ratio 6/14 and does not -- "
      "distinguishing those two is the whole point of the layer")
check("per-video detail is keyed by video", set(rep["per_video"]) == {"v1", "v2"})
check("per-video detail names the subject",
      rep["per_video"]["v1"]["subject"] == "s01")
check("per-video detail can be suppressed",
      "per_video" not in localisation_report(videos, curves, ivs,
                                             include_per_video=False))

import json  # noqa: E402
try:
    json.dumps(rep)
    check("the report is JSON-serialisable", True)
except (TypeError, ValueError) as exc:
    check("the report is JSON-serialisable", False, str(exc))

empty = localisation_report([], {}, {})
check("an empty corpus reports zero videos rather than raising",
      empty["n_videos_with_micro_truth"] == 0)
anti_curves = {
    "v1": np.where(np.isin(np.arange(300), np.arange(50, 64)), 1.0, 10.0),
    "v2": np.where(np.isin(np.arange(300), np.arange(80, 94)), 1.0, 10.0),
}
anti_rep = localisation_report(videos, anti_curves, ivs)
check("an anti-correlated corpus earns the anti-informative verdict",
      "anti-informative" in anti_rep["layer_1_signal"]["verdict"],
      "the instrument has to be able to deliver bad news, or a run of it "
      "reads as a clean bill of health")
check("the anti-correlated corpus reports a pooled AUC below 0.5",
      anti_rep["layer_1_signal"]["pooled_auc"] < 0.5,
      str(anti_rep["layer_1_signal"]["pooled_auc"]))


# ---------------------------------------------------------------------------
print()
if FAIL:
    print(f"{PASS} passed, {FAIL} FAILED")
    raise SystemExit(1)
print(f"{PASS}/{PASS} checks passed")
