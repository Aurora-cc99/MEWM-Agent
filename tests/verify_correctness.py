"""Rigour audit: check the implementation against the paper's formal definitions.

The paper's prose can be loose in places; the code cannot be. This script verifies the
mathematical and structural properties the framework's claims depend on, independently of
the unit tests -- it recomputes each quantity from its definition and compares.

Run:  python tests/verify_correctness.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _try_effort(model: str, effort: str) -> bool:
    """True when the model correctly refuses an unavailable tier."""
    from mewm.llm.registry import resolve as _resolve
    try:
        _resolve(model).validate_effort(effort)
        return False
    except ValueError:
        return True


FAILURES: list[tuple[str, str]] = []
CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append((name, detail))


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
print("\n[eq. 3] measurement triple")
# ---------------------------------------------------------------------------
from mewm.engines.v1_motion import roi_motion

flow = np.zeros((8, 8, 2), np.float32)
flow[..., 0] = 3.0
flow[..., 1] = -4.0                       # magnitude 5, pointing up-right
m, theta, c = roi_motion(flow, (0, 0, 8, 8))
check("m equals the mean vector norm", approx(m, 5.0, 1e-5), f"m={m}")
check("c equals 1 for a perfectly aligned field", approx(c, 1.0, 1e-6), f"c={c}")
check("theta measures counter-clockwise from image-right",
      approx(theta, math.degrees(math.atan2(4.0, 3.0)), 1e-3), f"theta={theta}")

# c is a ratio of sums, so it must be invariant to a global scaling of the field.
m2, _t2, c2 = roi_motion(flow * 7.0, (0, 0, 8, 8))
check("c is scale invariant", approx(c, c2, 1e-6), f"{c} vs {c2}")
check("m scales linearly with the field", approx(m2, 7.0 * m, 1e-4), f"{m2} vs {7*m}")

# Opposing halves: magnitudes cancel in the resultant but not in the sum of norms.
opposed = np.zeros((8, 8, 2), np.float32)
opposed[:4, :, 0] = 1.0
opposed[4:, :, 0] = -1.0
_m3, _t3, c3 = roi_motion(opposed, (0, 0, 8, 8))
check("c equals 0 for exactly opposed motion", approx(c3, 0.0, 1e-6), f"c={c3}")

# Proposition B.1: E[c] = O(n^-1/2) under uniformly random directions.
rng = np.random.default_rng(0)
ratios = []
for n_side in (8, 16, 32):
    angles = rng.uniform(0, 2 * math.pi, (n_side, n_side))
    field = np.stack([np.cos(angles), np.sin(angles)], -1).astype(np.float32)
    _m4, _t4, c4 = roi_motion(field, (0, 0, n_side, n_side))
    ratios.append(c4 * math.sqrt(n_side * n_side))
check("c * sqrt(n) is bounded across sizes (proposition B.1)",
      max(ratios) < 4.0, f"c*sqrt(n) = {[round(r, 2) for r in ratios]}")

# ---------------------------------------------------------------------------
print("\n[eq. 4 / 3.2.2] slot routing")
# ---------------------------------------------------------------------------
from mewm.engines.v2_slots import ROUTING_MASK
from mewm.knowledge.au_anatomy import (
    AU_ROI_PRIOR, K_SLOTS, N_ROI, ROI_INDEX, SLOT_AUS, SLOT_INDEX, regions_of,
)

check("routing mask is (K, n_roi)", ROUTING_MASK.shape == (K_SLOTS, N_ROI),
      str(ROUTING_MASK.shape))
consistent = all(
    ROUTING_MASK[SLOT_INDEX[au], ROI_INDEX[roi] - 1] == 1.0
    for au in SLOT_AUS for roi in regions_of(au)
)
check("every AU-region pair in K_AU is routed", consistent)
no_extra = all(
    (ROUTING_MASK[SLOT_INDEX[au], ROI_INDEX[roi] - 1] == 1.0)
    == ((au, roi) in AU_ROI_PRIOR)
    for au in SLOT_AUS for roi in ROI_INDEX
)
check("no slot reads a region outside its own AU", no_extra)

# ---------------------------------------------------------------------------
print("\n[eq. 6 / appendix B.5] sequential error decomposition")
# ---------------------------------------------------------------------------
from mewm.engines.m2_spotting import ErrorDecomposer

rng = np.random.default_rng(3)
n = 600
delta = np.abs(rng.normal(0.05, 0.01, n))
head = np.zeros((n, 6))
delta[200:230] += 0.8
head[200:230, 0] = 1.0
slot_err = np.zeros((n, K_SLOTS))
gate = np.zeros((n, K_SLOTS))
for t in range(400, 412):
    amp = 0.3 * math.exp(-((t - 405) ** 2) / 12)
    delta[t] += amp
    slot_err[t, SLOT_INDEX["AU4"]] = amp
    gate[t, SLOT_INDEX["AU4"]] = 1.0

result = ErrorDecomposer(fps=30.0).decompose(delta, head, None, slot_err, gate)
total_parts = result.delta_scene + result.delta_physio + result.delta_expr
check("the three components never exceed the total error",
      bool(np.all(total_parts <= result.delta_total + 1e-6)),
      f"max excess {float((total_parts - result.delta_total).max()):.4g}")
check("every component is non-negative",
      bool(np.all(result.delta_scene >= -1e-9) and np.all(result.delta_physio >= -1e-9)
           and np.all(result.delta_expr >= -1e-9)))
check("the expressive term is confined to gated AU regions",
      bool(np.all(result.delta_expr[gate.sum(axis=1) == 0] <= 1e-9)),
      "expressive mass leaked outside the coherence gate")
check("per-slot attribution sums to the expressive term",
      bool(np.allclose(result.per_slot.sum(axis=1), result.delta_expr, atol=1e-6)))
check("head-motion burst is absorbed by the scene term",
      float(result.delta_expr[200:230].max()) < 0.05,
      f"expr peak in the burst = {float(result.delta_expr[200:230].max()):.4g}")

# ---------------------------------------------------------------------------
print("\n[eq. 7] detection statistic")
# ---------------------------------------------------------------------------
from mewm.engines.m2_spotting import robust_normalise

constant = np.full(400, 0.42)
s_const = robust_normalise(constant, 300)
check("a constant series yields a finite, zero-centred statistic",
      bool(np.all(np.isfinite(s_const))) and abs(float(s_const.max())) < 1e-6,
      f"max={float(s_const.max())}")

sparse = np.zeros(400)
sparse[200] = 1.0
s_sparse = robust_normalise(sparse, 300)
check("a sparse series does not diverge (scale floor)",
      bool(np.all(np.isfinite(s_sparse))) and float(s_sparse.max()) < 1e3,
      f"max={float(s_sparse.max())}")
check("the statistic peaks at the spike", int(np.argmax(s_sparse)) == 200)

# Trailing-window causality: a future spike must not move an earlier value.
base = np.abs(rng.normal(0.1, 0.02, 400))
altered = base.copy()
altered[300] += 5.0
check("the statistic is causal (a future spike cannot change the past)",
      bool(np.allclose(robust_normalise(base, 100)[:300],
                       robust_normalise(altered, 100)[:300], atol=1e-9)))

# ---------------------------------------------------------------------------
print("\n[eq. 2] proposal TP criterion, with a configurable threshold")
# ---------------------------------------------------------------------------
from mewm.eval.metrics import evaluate_proposals, iou, sweep_iou_threshold

check("IoU is symmetric", approx(iou((0, 9), (5, 14)), iou((5, 14), (0, 9))))
check("IoU of identical intervals is 1", approx(iou((0, 9), (0, 9)), 1.0))
check("IoU of disjoint intervals is 0", approx(iou((0, 9), (20, 29)), 0.0))
# 10-frame intervals overlapping in 5 frames: 5 / 15.
check("IoU matches the closed-form value", approx(iou((0, 9), (5, 14)), 5 / 15, 1e-12))

strict = evaluate_proposals([(1000, 1003)], [(1000, 1011)], ["disgust"], ["disgust"],
                            iou_threshold=0.5, affective_rescue=False)
rescued = evaluate_proposals([(1000, 1003)], [(1000, 1011)], ["disgust"], ["disgust"],
                             iou_threshold=0.5, affective_rescue=True)
check("affective rescue is off when disabled", strict.tp == 0)
check("affective rescue counts a correct label at low IoU", rescued.tp == 1)
check("rescue_gain isolates the rescue contribution", rescued.rescue_gain > 0)

# Lowering the threshold can only ever turn non-TPs into TPs.
low = evaluate_proposals([(1000, 1003)], [(1000, 1011)], iou_threshold=0.2)
high = evaluate_proposals([(1000, 1003)], [(1000, 1011)], iou_threshold=0.9)
check("a lower IoU threshold is monotonically more permissive",
      low.tp_strict >= high.tp_strict, f"{low.tp_strict} vs {high.tp_strict}")
rows = sweep_iou_threshold([([(1000, 1003)], [(1000, 1011)], ["disgust"], ["disgust"])],
                           (0.1, 0.5, 0.9))
check("the sweep reports strict F1 monotonically non-increasing in the threshold",
      all(rows[i]["f1_strict"] >= rows[i + 1]["f1_strict"] for i in range(len(rows) - 1)),
      str([r["f1_strict"] for r in rows]))

# ---------------------------------------------------------------------------
print("\n[eq. 10 / appendix C.2] masking necessity")
# ---------------------------------------------------------------------------
from mewm.engines.m3_primitives import RolloutService
from mewm.engines.v3_latent import BeliefState

service = RolloutService()
T = 14
observed = np.zeros((T, K_SLOTS))
for i in range(T):
    ramp = min(1.0, i / 5.0) * max(0.0, 1.0 - max(0, i - 8) / 6.0)
    observed[i, SLOT_INDEX["AU4"]] = 0.85 * ramp
    observed[i, SLOT_INDEX["AU9"]] = 0.48 * ramp
candidates = ["disgust", "anger", "fear", "happiness"]

active = service.mask(observed, ["AU9"], candidates, cid="v")
inactive = service.mask(observed, ["AU25"], candidates, cid="v")
check("MNI is non-negative (it is a KL divergence)",
      active.mni >= 0 and inactive.mni >= 0, f"{active.mni}, {inactive.mni}")
check("MNI of an unobserved AU is ~0 (hallucination detector)",
      inactive.mni < 0.01, f"MNI(AU25)={inactive.mni}")
check("MNI of a contributing AU exceeds it", active.mni > inactive.mni)

belief = BeliefState.uniform(candidates)
belief.update({"disgust": 2.0, "anger": 0.4})
check("KL(p||p) = 0", approx(belief.kl_to(belief), 0.0, 1e-9))
other = belief.copy()
other.update({"fear": 3.0})
check("KL is asymmetric, as a divergence should be",
      not approx(belief.kl_to(other), other.kl_to(belief), 1e-6))
check("belief probabilities sum to 1",
      approx(float(belief.probabilities.sum()), 1.0, 1e-9))

# ---------------------------------------------------------------------------
print("\n[eq. 9 / appendix C.5] CFS and the graph weight")
# ---------------------------------------------------------------------------
from mewm.agents.structure import harmonic_mean, lagged_cross_correlation

cfs = service.counterfactual_consistency(observed, candidates, cid="v")
check("CFS lies in [0, 1]", all(0.0 <= v <= 1.0 for v in cfs.values()), str(cfs))
check("the worst hypothesis scores 0 by construction",
      approx(min(cfs.values()), 0.0, 1e-9), str(cfs))

check("harmonic mean is 0 when either source is 0", approx(harmonic_mean(0.0, 0.9), 0.0))
check("harmonic mean of equal values is that value",
      approx(harmonic_mean(0.6, 0.6), 0.6, 1e-9))
check("harmonic mean never exceeds the arithmetic mean",
      harmonic_mean(0.2, 0.9) <= (0.2 + 0.9) / 2 + 1e-9)

lead = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0, 0], float)
lag = np.array([0, 0, 0, 0, 1, 2, 3, 2, 1, 0], float)
polarity, shift, peak = lagged_cross_correlation(lead, lag)
check("cross-correlation recovers a positive lag for a leading signal",
      shift > 0 and polarity == "+", f"lag={shift} polarity={polarity}")
check("cross-correlation peak is bounded by 1", peak <= 1.0 + 1e-9, f"peak={peak}")

# ---------------------------------------------------------------------------
print("\n[eq. 11 / appendix F.3] reward")
# ---------------------------------------------------------------------------
from mewm.training.rewards import (
    CompositeReward, curriculum_weights, reward_au, reward_causal, reward_emotion,
    reward_temporal,
)

for progress in (0.0, 0.13, 0.37, 0.5, 0.62, 0.88, 1.0):
    weights = curriculum_weights(progress)
    check(f"curriculum weights sum to 1 at progress {progress}",
          approx(sum(weights.values()), 1.0, 1e-9), str(sum(weights.values())))
    check(f"curriculum weights are non-negative at progress {progress}",
          all(v >= 0 for v in weights.values()))

for component, args in (
    ("R_AU", (["AU4"], ["AU4", "AU9"])),
    ("R_emo", ("disgust", "anger")),
):
    score = (reward_au(*args)[0] if component == "R_AU" else reward_emotion(*args)[0])
    check(f"{component} lies in [0, 1]", 0.0 <= score <= 1.0, str(score))

check("R_AU is 1 for an exact set match", approx(reward_au(["AU4", "AU9"], ["AU4", "AU9"])[0], 1.0))
check("R_emo is 1 for an exact emotion match",
      approx(reward_emotion("disgust", "disgust", "negative", "negative")[0], 1.0))
check("R_emo gives a near-miss more credit than a sign error",
      reward_emotion("anger", "disgust")[0] > reward_emotion("happiness", "disgust")[0])
check("R_causal lies in [0, 1]",
      0.0 <= reward_causal(1.0, {"AU4": 5.0}, 0.0)[0] <= 1.0)
check("R_temp honours a custom IoU threshold",
      reward_temporal((0, 5), (0, 9), iou_threshold=0.9)[1]["tp"] is False
      and reward_temporal((0, 5), (0, 9), iou_threshold=0.3)[1]["tp"] is True)

reward = CompositeReward()
breakdown = reward.score(
    {"fine_label": "disgust", "coarse_label": "negative", "k_crit": ["AU4"],
     "interval": [1000, 1011], "P": {"a": 1}, "M": {"a": 1}, "C": {"a": 1},
     "MC": {"a": 1}, "es": {"disgust": 1.0}, "dc": {"disgust": 1.0}, "refs": []},
    {"fine": "disgust", "coarse": "negative", "aus": ["AU4"],
     "interval": [1000, 1011]},
    judge={"dc": 1.0, "mni": {"AU4": 5.0}, "graph_edit_distance": 0.0})
check("the composite reward lies in [0, 1]", 0.0 <= breakdown.total <= 1.0,
      str(breakdown.total))
check("a perfect output approaches the reward ceiling", breakdown.total > 0.9,
      str(breakdown.total))

# ---------------------------------------------------------------------------
print("\n[3.4.1 / D.5] evidence ordering and gates")
# ---------------------------------------------------------------------------
from mewm.schemas import ContractError, Evidence, EvidenceChain, EvidenceLevel

chain = EvidenceChain("p")
motion = chain.add(Evidence.create("P", "region moved"))
au = chain.add(Evidence.create("A", "AU4 active", refs=[motion.eid]))
emotion = chain.add(Evidence.create("R", "disgust", refs=[au.eid, motion.eid]))
check("the reference graph is a DAG by level",
      all(chain.get(r).level < e.level for e in chain for r in e.refs))

for source, ref in (("P", au.eid), ("A", emotion.eid), ("A", au.eid)):
    try:
        chain.add(Evidence.create(source, "illegal", refs=[ref]))
        check(f"{source} citing a higher/equal level is rejected", False)
    except ContractError:
        check(f"{source} citing a higher/equal level is rejected", True)

from mewm.schemas import ChallengeRecord
try:
    ChallengeRecord.create("evidence_gap", "bare assertion", refs=["x"],
                           analysis_report_id="")
    check("a challenge without an analysis report is rejected", False)
except ContractError:
    check("a challenge without an analysis report is rejected", True)

from mewm.schemas import FINAL_VERDICT_STEP
check("challenge grades are monotone non-positive",
      all(v <= 0 for v in FINAL_VERDICT_STEP.values()), str(FINAL_VERDICT_STEP))
check("an upheld challenge costs more than a partial one",
      FINAL_VERDICT_STEP["upheld"] < FINAL_VERDICT_STEP["partial"] < 0
      or FINAL_VERDICT_STEP["upheld"] < FINAL_VERDICT_STEP["partial"])

# ---------------------------------------------------------------------------
print("\n[3.4.4] ES / DC and label mapping")
# ---------------------------------------------------------------------------
from mewm.agents.reasoning import challenge_grade, fuse_confidence, joint_score
from mewm.knowledge.emotion_prototypes import (
    FINE_EMOTIONS, coarse_of, evidence_sufficiency, labels_consistent, normalise_scores,
)

for emotion in FINE_EMOTIONS:
    value = evidence_sufficiency(emotion, ["AU4", "AU7", "AU9"], ["AU12"])
    check(f"ES({emotion}) lies in [0, 1]", 0.0 <= value <= 1.0, str(value))
    check(f"the coarse mapping of {emotion} is self-consistent",
          labels_consistent(emotion, coarse_of(emotion)))

normalised = normalise_scores({"a": -3.0, "b": 0.0, "c": 5.0})
check("DC normalisation maps to [0, 1] with the max at 1",
      approx(max(normalised.values()), 1.0) and approx(min(normalised.values()), 0.0),
      str(normalised))

joint = joint_score({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, alpha=0.5)
check("the joint score is the stated convex combination",
      approx(joint["a"], 0.5) and approx(joint["b"], 0.5), str(joint))

confidence, terms = fuse_confidence(1.0, 1.0, 1.0, 1.0)
check("confidence fusion is bounded by 1 with unit inputs",
      approx(confidence, 1.0, 1e-6), str(confidence))
check("confidence fusion is bounded below by 0",
      approx(fuse_confidence(0.0, 0.0, 0.0, 0.0)[0], 0.0, 1e-9))

from mewm.schemas import ChallengeRecord as CR
upheld = CR.create("evidence_gap", "x", refs=["a"], analysis_report_id="r")
upheld.set_final("upheld")
check("an upheld challenge lowers the challenge factor",
      challenge_grade([upheld]) < 1.0, str(challenge_grade([upheld])))

# ---------------------------------------------------------------------------
print("\n[appendix B.3] slow-variable absorption")
# ---------------------------------------------------------------------------
from mewm.config import RepresentationConfig
from mewm.engines.v3_latent import SlowStateTracker

tracker = SlowStateTracker(dim=16, config=RepresentationConfig())
rng = np.random.default_rng(1)
for t in range(80):
    tracker.update(rng.normal(0, 0.01, 16), t, n_samples=30)
gain = tracker.absorption_ratio()
check("the steady-state Kalman gain is small (h << 1, proposition B.3)",
      gain < 0.25, f"gain={gain:.4f}")
check("the slow covariance converges", tracker.sigma < 1e-3, f"sigma={tracker.sigma:.3g}")

# ---------------------------------------------------------------------------
print("\n[model registry] transport and heterogeneity")
# ---------------------------------------------------------------------------
from mewm.llm.registry import (
    MODELS, PROVIDERS, heterogeneous, list_models, resolve, transport_of,
)

check("every model's provider is registered",
      all(spec.provider in PROVIDERS for spec in MODELS.values()))
check("every open-weight model declares a HF repo",
      all(spec.hf_repo for spec in list_models(open_only=True)))
check("aliases resolve to a registered id",
      all(resolve(alias).model_id in MODELS
          for spec in MODELS.values() for alias in spec.aliases))
check("the DeepSeek mixed-case spelling resolves to the served id",
      resolve("deepseekV4-Flash-Vision-Exp").model_id == "deepseek-v4-flash-vision-exp")
check("the failing gemini preview id is aliased onto a working one",
      resolve("gemini-3-pro-preview").model_id == "gemini-3-pro")
check("heterogeneity is judged per provider, not per id",
      not heterogeneous("gemini-3-pro", "gemini-3.1-pro")
      and heterogeneous("claude-sonnet-5", "gpt-5.6-sol"))
check("effort validation rejects a tier the model does not offer",
      _try_effort("claude-sonnet-5", "ultra"))
check("effort validation accepts a declared tier",
      resolve("claude-sonnet-5").validate_effort("max") == "max")
check("friendly effort spellings fold to the internal code",
      resolve("gpt-5.6-sol").validate_effort("Extra high") == "xhigh")

# ---------------------------------------------------------------------------
print("\n[data] frame / flow alignment")
# ---------------------------------------------------------------------------
from mewm.data.paths import VideoPaths, max_micro_frames

paths = VideoPaths("casme_sq", "s15/x", flow_gap=7)
check("flow frame N derives from (N - k, N)", paths.flow_source_pair(100) == (93, 100))
check("frame and flow filenames agree for the same index",
      paths.frame(703).name == paths.flow(703).name == "img703.jpg")
check("the flow tree is rooted under pre_datasets",
      "pre_datasets" in str(paths.flow_dir) and "pre_datasets" not in str(paths.frame_dir))
# The ceiling is now a UNIFORM 200 frames for every dataset (MICRO_CEILING_FRAMES,
# user directive 2026-08-31: no per-dataset empirical ceilings). The previous
# per-dataset "longest annotated micro" values (17 for casme_sq, 101 for samm)
# routed truth-covering hysteresis spans to the macro channel (13 of 52 casme_sq
# truths covered at IoU > 0.5 yet discarded) -- see SpottingConfig docstring.
check("the micro ceiling is the uniform 200 frames for every dataset",
      max_micro_frames("casme_sq") == 200 and max_micro_frames("samm") == 200,
      f"{max_micro_frames('casme_sq')}, {max_micro_frames('samm')}")
check("an explicit seconds override still follows seconds x fps, same as before",
      max_micro_frames("casme_sq", 0.5) == 15 and max_micro_frames("samm", 0.5) == 100,
      f"{max_micro_frames('casme_sq', 0.5)}, {max_micro_frames('samm', 0.5)}")


# ---------------------------------------------------------------------------
print("\n[training] the LOSO protocol's arithmetic")
# ---------------------------------------------------------------------------
from mewm.config import TrainingConfig
from mewm.training.diagnostics import classify_group, pass_at_k
from mewm.training.grpo import GRPOConfig, GroupRollout, PolicySample, admit_prompt
from mewm.training.rewards import RewardBreakdown

# pass@k recomputed from 1 - C(n-c, k) / C(n, k), independently of the implementation.
_worst = 0.0
for _n in range(1, 13):
    for _c in range(_n + 1):
        for _k in range(1, _n + 1):
            if _c == 0:
                _ref = 0.0
            elif _n - _c < _k:
                _ref = 1.0
            else:
                _ref = 1.0 - math.comb(_n - _c, _k) / math.comb(_n, _k)
            _worst = max(_worst, abs(pass_at_k(_n, _c, _k) - _ref))
check("pass@k equals the unbiased estimator on every (n, c, k)", _worst < 1e-9,
      f"max deviation {_worst:.2e}")
check("pass@k is monotone non-decreasing in k, so a bare 'pass@k > pass@1' is vacuous",
      all(pass_at_k(16, 4, k) <= pass_at_k(16, 4, k + 1) + 1e-12 for k in range(1, 16)))
check("pass@1 equals the plain empirical rate", approx(pass_at_k(16, 4, 1), 0.25))


def _rollout(rewards):
    samples = [PolicySample(text="", product={}) for _ in rewards]
    for sample, value in zip(samples, rewards):
        sample.reward = RewardBreakdown(total=float(value))
    return GroupRollout("p", "prompt", samples)


# A degenerate group has zero advantage; this is why prompt admission exists at all.
check("a group whose rewards are all equal yields an identically zero advantage",
      all(abs(a) < 1e-12 for a in _rollout([0.7] * 8).compute_advantages()))
check("a group with spread yields non-zero, zero-mean advantages",
      (lambda a: max(abs(x) for x in a) > 0.1 and abs(sum(a)) < 1e-3)(
          _rollout([0.1, 0.4, 0.6, 0.9]).compute_advantages()))
check("a degenerate group at the floor is not admitted to an RL step",
      admit_prompt([0.0] * 8)[0] is False)
check("a degenerate group at the ceiling is not admitted either",
      admit_prompt([1.0] * 8)[0] is False)
check("a group with usable spread is admitted", admit_prompt([0.1, 0.5, 0.9, 0.3])[0])

# The three-way reward verdict; all-high is the branch the original plan omitted.
check("a spread reward group is trainable",
      classify_group([0.1, 0.4, 0.6, 0.9]) == "spread")
check("a uniformly low reward group means SFT is undertrained",
      classify_group([0.01, 0.02, 0.03, 0.02]) == "all_low")
check("a uniformly high reward group means saturation, not readiness",
      classify_group([0.97, 0.98, 0.99, 0.98]) == "all_high")

_training = TrainingConfig()
check("SFT runs at most 100 epochs and RL 1000 steps, as configured",
      _training.sft_max_epochs == 100 and _training.rl_total_steps == 1000,
      f"{_training.sft_max_epochs}, {_training.rl_total_steps}")
check("the RL driver inherits its step count from the training config",
      GRPOConfig.from_training(_training).total_steps == 1000)
check("the RL admission floor is the same constant the gate judges spread with",
      approx(GRPOConfig.from_training(_training).admit_std_min,
             _training.reward_std_min))
check("n_val_subjects = 0, so the whole non-test pool trains (in-sample diagnostics)",
      _training.n_val_subjects == 0)
check("the default policy is an open-weight checkpoint (hosted APIs expose no gradients)",
      any(k in _training.policy_model.lower() for k in ("qwen", "intern", "llava")),
      _training.policy_model)


# ---------------------------------------------------------------------------
print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
if FAILURES:
    print("\nfailures:")
    for name, detail in FAILURES:
        print(f"  {name}: {detail}")
raise SystemExit(1 if FAILURES else 0)
