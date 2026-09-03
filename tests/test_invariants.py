"""Tests for the framework's structural invariants.

These target the properties the architecture's claims rest on, not incidental behaviour:
evidence-chain acyclicity, the visibility projection, the gate rules, the disjunctive
salience test, error decomposition, and the counterfactual primitives. A regression in
any of them would invalidate a claim in the paper rather than merely degrade a number.

Run with:  python -m pytest tests/test_invariants.py -v
       or:  python tests/test_invariants.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mewm.config import load_config
from mewm.engines.m1_dynamics import AnalyticDynamics
from mewm.engines.m2_spotting import Spotter, alignment_auc, robust_normalise
from mewm.engines.m3_primitives import RolloutService, dtw_distance
from mewm.engines.m4_scheduler import Scheduler, build_signal
from mewm.engines.v1_motion import MotionFrontEnd, roi_motion
from mewm.engines.v2_slots import ROUTING_MASK, analytic_slot_readout, phase_profile
from mewm.engines.v3_latent import BeliefState, SlowStateTracker
from mewm.engines.v4_token_regulator import TokenRegulator, build_context_tokens
from mewm.eval.metrics import evaluate_proposals, iou
from mewm.knowledge.au_anatomy import K_SLOTS, N_ROI, SLOT_AUS, SLOT_INDEX, regions_of
from mewm.knowledge.emotion_prototypes import evidence_sufficiency, labels_consistent
from mewm.orchestration.gates import DualGate
from mewm.orchestration.state import (
    MEWMState, PHASE_A_ENCODE, PHASE_C_CRITIC, PHASE_P_VERIFY, PHASE_R_NARRATE, project,
)
from mewm.schemas import (
    CandidateInterval, ContractError, ErrorRecord, Evidence, EvidenceChain,
    EvidenceLevel, ROIMeasurement, Verdict, VideoMeta,
)
from mewm.training.rewards import CompositeReward, curriculum_weights, reward_causal


# ---------------------------------------------------------------------------
# Evidence chain -- the causal ordering guarantee
# ---------------------------------------------------------------------------


def test_evidence_chain_rejects_upward_reference():
    """An entry may not cite one at its own level or above (rule R1)."""
    chain = EvidenceChain("p01")
    motion = chain.add(Evidence.create("P", "brow moved 0.41px"))
    au = chain.add(Evidence.create("A", "AU4 active", refs=[motion.eid]))

    # AU -> motion is legal; motion -> AU must not be.
    bad = Evidence.create("P", "region moved because of AU4", refs=[au.eid])
    try:
        chain.add(bad)
        raise AssertionError("a motion entry citing an AU entry must be rejected")
    except ContractError:
        pass

    # Same-level citation is equally illegal.
    same = Evidence.create("A", "AU7 active", refs=[au.eid])
    try:
        chain.add(same)
        raise AssertionError("same-level citation must be rejected")
    except ContractError:
        pass


def test_evidence_revision_preserves_history():
    """Revision emits a new entry and marks the old one; it never edits in place."""
    chain = EvidenceChain("p01")
    original = chain.add(Evidence.create("A", "AU4 active at 0.62"))
    replacement = original.revise("AU4 active at 0.78")
    chain.add(replacement)
    assert chain.get(original.eid).status == "revised"
    assert replacement.supersedes == original.eid
    assert len(chain) == 2, "the superseded entry must be retained"


def test_evidence_levels_ordered():
    assert EvidenceLevel.MOTION < EvidenceLevel.AU < EvidenceLevel.EMOTION < EvidenceLevel.VERIFICATION


# ---------------------------------------------------------------------------
# Visibility projection
# ---------------------------------------------------------------------------


def _demo_state() -> MEWMState:
    state = MEWMState(video_meta=VideoMeta("v1", "casme_sq", "/p", 30.0, 2000, "s15"))
    state.proposals = [CandidateInterval("p01", 1000, 1011, 1005, 6.2, {"AU4": 0.6})]
    state.error_record = ErrorRecord("v1", 0, s_curve=[0.1] * 2000,
                                     delta_expr=[0.01] * 2000)
    state.verdicts["p01"] = Verdict("p01", "negative", "disgust", 0.8)
    return state


def test_critic_cannot_see_verdict():
    """A critic that knows the answer constructs challenges that lead to it."""
    projection = project(_demo_state(), PHASE_C_CRITIC, "p01")
    assert "verdict" not in projection.fields
    assert "verdict" in projection.withheld


def test_whole_curve_access_is_restricted():
    """Only the scan and narration phases may read the whole error curve."""
    state = _demo_state()
    assert "error_record" not in project(state, PHASE_P_VERIFY, "p01").fields
    assert "error_record" not in project(state, PHASE_A_ENCODE, "p01").fields
    assert "error_record" in project(state, PHASE_R_NARRATE, "p01").fields


def test_projection_is_pure():
    """Projection must not mutate the state it reads."""
    state = _demo_state()
    before = state.to_dict()
    project(state, PHASE_C_CRITIC, "p01")
    assert state.to_dict() == before


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_r2_blocks_au_and_emotion_in_motion_product():
    gate = DualGate()
    product = {"motion_evidence": [{
        "roi": "brow", "magnitude_px": 0.41, "direction_deg": 265.0,
        "coherence": 0.91, "salient": True, "note": "AU4 suggests disgust",
    }]}
    outcome = gate.evaluate(PHASE_P_VERIFY, product, EvidenceChain("p01"),
                            ban_au=True, ban_emotion=True)
    assert not outcome.passed
    assert "R2" in outcome.failed_rules


def test_r2_exempts_the_attribution_field():
    """The proposal contract requires AU-keyed attribution; R2 must not fire on it."""
    gate = DualGate()
    product = {
        "proposals": [{"cid": "p01", "interval": [1000, 1011], "apex": 1005,
                       "peak_S": 6.2, "attribution": {"AU4": 0.6, "AU7": 0.3},
                       "physio_overlap": False}],
        "curve_summary": "one excursion above baseline",
    }
    outcome = gate.evaluate("P.scan", product, EvidenceChain("p01"),
                            ban_au=True, ban_emotion=True)
    assert "R2" not in outcome.failed_rules, outcome.consistency.questions


def test_r3_catches_a_number_that_does_not_recompute():
    gate = DualGate()
    product = {"motion_evidence": [{
        "roi": "brow", "magnitude_px": 0.41, "direction_deg": 265.0,
        "coherence": 0.91, "salient": True}]}
    outcome = gate.evaluate(PHASE_P_VERIFY, product, EvidenceChain("p01"),
                            ban_au=True, ban_emotion=True,
                            recompute={"coherence": 0.55})
    assert not outcome.passed and "R3" in outcome.failed_rules


def test_r4_requires_conflicts_to_be_registered():
    from mewm.schemas import OpenQuestion
    gate = DualGate()
    product = {"active_aus": ["AU4", "AU12"], "fits": [{"au": "AU4"}],
               "slot_agreement": 0.9}

    outcome = gate.evaluate(PHASE_A_ENCODE, product, EvidenceChain("p01"),
                            ban_emotion=True, active_aus=["AU4", "AU12"])
    assert "R4" in outcome.failed_rules, "AU4/AU12 are antagonistic"

    registered = [OpenQuestion.create("conflict", "AU4 and AU12 co-active", cid="p01")]
    outcome = gate.evaluate(PHASE_A_ENCODE, product, EvidenceChain("p01"),
                            ban_emotion=True, active_aus=["AU4", "AU12"],
                            open_questions=registered)
    assert "R4" not in outcome.failed_rules


def test_r5_enforces_label_mapping():
    gate = DualGate()
    outcome = gate.evaluate(
        "R.adjudicate",
        {"confidence": 0.8, "fusion_terms": {"q_ev": 1.0}, "suppression": "none"},
        EvidenceChain("p01"), verdict=Verdict("p01", "positive", "disgust", 0.8),
    )
    assert "R5" in outcome.failed_rules
    assert not labels_consistent("disgust", "positive")


# ---------------------------------------------------------------------------
# V1 -- measurement
# ---------------------------------------------------------------------------


def test_coherence_separates_signal_from_noise():
    """Coherent sub-pixel motion stays near 1; random directions decay as n^-1/2."""
    coherent = np.zeros((20, 20, 2), np.float32)
    coherent[..., 1] = -0.05                       # sub-pixel, all one direction
    _m, _theta, c_signal = roi_motion(coherent, (0, 0, 20, 20))
    assert c_signal > 0.99

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.4, (20, 20, 2)).astype(np.float32)
    _m, _theta, c_noise = roi_motion(noise, (0, 0, 20, 20))
    assert c_noise < 0.15
    # Proposition B.1: E[c] = O(n^-1/2); with n = 400 that is about 0.05.
    assert c_noise < 4.0 / math.sqrt(400)


def test_salience_is_disjunctive():
    """A magnitude-only gate would delete the micro-expression signal."""
    front_end = MotionFrontEnd()
    assert front_end.is_salient(0.02, 0.95), "coherent sub-pixel motion must be salient"
    assert front_end.is_salient(0.90, 0.02), "large motion must be salient"
    assert not front_end.is_salient(0.02, 0.05), "small and incoherent must not be"


# ---------------------------------------------------------------------------
# V2 -- slot routing
# ---------------------------------------------------------------------------


def test_slot_routing_is_anatomically_confined():
    """Each slot reads only its own AU's regions -- the premise of masking."""
    assert ROUTING_MASK.shape == (K_SLOTS, N_ROI)
    for au in SLOT_AUS:
        row = ROUTING_MASK[SLOT_INDEX[au]]
        assert row.sum() == len(regions_of(au)) > 0
        assert set(np.unique(row)) <= {0.0, 1.0}


def test_slot_readout_discriminates_by_direction():
    """A brow-down pull activates AU4, not AU12."""
    from mewm.knowledge.au_anatomy import ROI_ORDER
    measurements = [
        ROIMeasurement(i + 1, name, label,
                       0.30 if "brow" in name else 0.02, 265.0,
                       0.90 if "brow" in name else 0.05, True)
        for i, (name, label) in enumerate(ROI_ORDER)
    ]
    readout = analytic_slot_readout(measurements)
    assert readout["AU4"].activation > 0.4
    assert readout["AU12"].activation < 0.05


def test_phase_profile_recovers_onset_apex_offset():
    trajectory = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 0.6, 0.3, 0.1])
    frames = list(range(100, 108))
    t_on, t_apex, t_off, peak, rise, decay = phase_profile(trajectory, frames)
    assert t_on < t_apex < t_off
    assert peak == 0.9 and rise > 0 > decay


# ---------------------------------------------------------------------------
# V3 -- latent state
# ---------------------------------------------------------------------------


def test_slow_state_gain_converges_small():
    """A converged small gain is what makes h << 1 in proposition B.3."""
    tracker = SlowStateTracker(dim=16)
    rng = np.random.default_rng(0)
    for t in range(60):
        tracker.update(rng.normal(0, 0.01, 16), t, n_samples=30)
    assert tracker.absorption_ratio() < 0.2


def test_belief_kl_is_the_mni_quantity():
    belief = BeliefState.uniform(["disgust", "anger", "fear"])
    belief.update({"disgust": 2.0, "anger": 0.5, "fear": 0.1})
    assert belief.mean_label == "disgust"
    assert belief.kl_to(belief) < 1e-9, "KL to itself must vanish"

    shifted = belief.copy()
    shifted.update({"anger": 3.0})
    assert belief.kl_to(shifted) > 0.0


# ---------------------------------------------------------------------------
# V4 -- token regulation
# ---------------------------------------------------------------------------


def test_regulator_keeps_evidence_and_drops_scene_chatter():
    tokens = build_context_tokens(
        ["You are the structure agent."],
        ["Left Inner Eyebrow magnitude 0.410px coherence 0.91",
         "The room appears well lit and modern."],
        [(1287, "brow"), (1288, "background")],
    )
    regulated, report = TokenRegulator().regulate(tokens, "A.encode")
    by_content = {t.content or t.source: t for t in regulated}
    assert by_content["Left Inner Eyebrow magnitude 0.410px coherence 0.91"].admitted
    assert not by_content["The room appears well lit and modern."].admitted
    assert by_content["frame:1287@brow"].admitted
    assert report.n_instruction == 1, "instruction tokens are always kept"


# ---------------------------------------------------------------------------
# M2 -- decomposition and detection
# ---------------------------------------------------------------------------


def test_robust_normalise_is_bounded_on_sparse_input():
    """A sparse residual must not blow the statistic up through a zero MAD."""
    values = np.zeros(500)
    values[250] = 1.0
    s = robust_normalise(values, window=300)
    assert np.isfinite(s).all()
    assert s.max() < 1e3, "the scale floor must bound the statistic"
    assert s[250] == s.max()


def test_decomposition_suppresses_head_motion_and_blink():
    """The central claim of C2: interference is explained away, the event is not."""
    rng = np.random.default_rng(7)
    n = 1200
    delta = np.abs(rng.normal(0.05, 0.012, n))
    head = np.zeros((n, 6))

    delta[400:430] += 0.9                       # head motion burst
    head[400:430, 0] = 1.0
    head[400:430, 4] = 1.0
    delta[700:706] += 0.30                      # blink-like pulse

    slot_error = np.zeros((n, K_SLOTS))
    gate = np.zeros((n, K_SLOTS))
    for t in range(1000, 1012):                 # the micro-expression
        amplitude = 0.32 * math.exp(-((t - 1005) ** 2) / 12)
        delta[t] += amplitude
        slot_error[t, SLOT_INDEX["AU4"]] = amplitude * 0.6
        slot_error[t, SLOT_INDEX["AU7"]] = amplitude * 0.3
        slot_error[t, SLOT_INDEX["AU9"]] = amplitude * 0.1
        gate[t, [SLOT_INDEX["AU4"], SLOT_INDEX["AU7"], SLOT_INDEX["AU9"]]] = 1.0

    result = Spotter(fps=30.0).run("t", delta, 0, head_motion=head,
                                   slot_errors=slot_error, coherence_gate=gate)
    record = result.error_record

    assert record.s_at(415) < 1.0, "head motion must be absorbed by the scene term"
    assert record.s_at(702) < 1.0, "the blink must be absorbed by the physio term"
    assert record.s_at(1005) > 10.0, "the micro-expression must spike"

    # The raw-curve pass (proposal_raw_curve_frac) may add low-confidence spans of
    # its own (config 2026-09-02); the decomposition invariant is about *decoded*
    # proposals: exactly one carries the AU attribution, and it is the micro event.
    decoded = [p for p in result.proposals if p.attribution]
    assert len(decoded) == 1
    proposal = decoded[0]
    # The energy-contour trim (proposal_trim_fraction) tightens the hysteresis span
    # to the event's core (1002-1008 here), so the honest expectation is the
    # pipeline's own TP criterion (IoU >= 0.5), not the untrimmed span.
    assert proposal.iou((1000, 1011)) > 0.5
    assert proposal.apex == 1005
    assert proposal.top_aus(1) == ["AU4"]
    assert alignment_auc(record.s_curve, [(1000, 1011)]) > 0.95


def test_physio_pass_does_not_steal_the_expression():
    """A 1-D template matches a micro-expression bump; AU localisation separates them."""
    n = 400
    delta = np.full(n, 0.05)
    slot_error = np.zeros((n, K_SLOTS))
    gate = np.zeros((n, K_SLOTS))
    for t in range(200, 212):
        amplitude = 0.3 * math.exp(-((t - 205) ** 2) / 12)
        delta[t] += amplitude
        slot_error[t, SLOT_INDEX["AU4"]] = amplitude
        gate[t, SLOT_INDEX["AU4"]] = 1.0

    result = Spotter(fps=30.0).run("t", delta, 0, slot_errors=slot_error,
                                   coherence_gate=gate)
    for event in result.error_record.physio_events:
        assert not event.overlaps(200, 211), "physio must not claim the expression"


# ---------------------------------------------------------------------------
# M3 -- counterfactual primitives
# ---------------------------------------------------------------------------


def _disgust_trajectory(length: int = 14) -> np.ndarray:
    trajectory = np.zeros((length, K_SLOTS))
    for i in range(length):
        ramp = min(1.0, i / 5.0) * max(0.0, 1.0 - max(0, i - 8) / 6.0)
        trajectory[i, SLOT_INDEX["AU4"]] = 0.85 * ramp
        trajectory[i, SLOT_INDEX["AU7"]] = 0.62 * ramp
        trajectory[i, SLOT_INDEX["AU9"]] = 0.48 * ramp
    return trajectory


def test_mni_is_zero_for_an_absent_au():
    """An AU that was never active cannot influence the belief -- the hallucination test."""
    service = RolloutService()
    observed = _disgust_trajectory()
    candidates = ["disgust", "anger", "fear", "happiness"]

    present = service.mask(observed, ["AU9"], candidates, cid="t")
    absent = service.mask(observed, ["AU25"], candidates, cid="t")
    assert absent.mni < 0.01, "an inactive AU must have near-zero necessity"
    assert present.mni > absent.mni


def test_every_primitive_call_is_recorded():
    """Provenance: each call carries the engine version so results can be recomputed."""
    service = RolloutService()
    observed = _disgust_trajectory()
    service.score(observed, ["disgust", "anger"], cid="p01")
    service.rollout(observed, "disgust", cid="p01")
    service.mask(observed, ["AU4"], ["disgust", "anger"], cid="p01")

    records = service.records_for("p01")
    assert len(records) == 3
    assert {r.primitive for r in records} == {"score", "rollout", "mask"}
    assert all(r.model_version for r in records)


def test_dtw_is_symmetric_and_zero_on_identity():
    a = _disgust_trajectory()
    assert dtw_distance(a, a) < 1e-9
    b = _disgust_trajectory(12)
    assert abs(dtw_distance(a, b) - dtw_distance(b, a)) < 1e-9


# ---------------------------------------------------------------------------
# M4 -- routing
# ---------------------------------------------------------------------------


def test_routing_paths():
    scheduler = Scheduler()
    fast = scheduler.route(build_signal("p1", {"disgust": 0.8, "anger": 0.5},
                                        {"disgust": 5.0, "anger": 1.0}, 0.2))
    standard = scheduler.route(build_signal("p2", {"disgust": 0.6, "anger": 0.55},
                                            {"disgust": 2.0, "anger": 1.5}, 0.3))
    deep = scheduler.route(build_signal("p3", {"disgust": 0.9, "anger": 0.2},
                                        {"disgust": 9.0, "anger": 1.0}, 0.8))
    assert (fast.path, standard.path, deep.path) == ("fast", "standard", "deep")


def test_open_question_blocks_the_fast_path():
    """An unresolved question must never be skipped past."""
    signal = build_signal("p1", {"disgust": 0.8, "anger": 0.5},
                          {"disgust": 5.0, "anger": 1.0}, 0.2, open_questions=1)
    assert Scheduler().route(signal).path == "standard"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_affective_rescue_is_separable():
    """Eq. (2): a low-IoU but correctly-labelled proposal is a TP, and it is countable."""
    proposals = [(1000, 1003)]
    truths = [(1000, 1011)]
    assert iou(proposals[0], truths[0]) <= 0.5

    rescued = evaluate_proposals(proposals, truths, ["disgust"], ["disgust"])
    assert rescued.tp == 1 and rescued.tp_strict == 0
    assert rescued.n_rescued == 1 and rescued.rescue_gain > 0

    wrong = evaluate_proposals(proposals, truths, ["happiness"], ["disgust"])
    assert wrong.tp == 0, "wrong on both time and label is a false positive"


def test_greedy_matching_claims_each_truth_once():
    metrics = evaluate_proposals([(1000, 1011), (1001, 1010)], [(1000, 1011)])
    assert metrics.tp <= 1, "two proposals must not both claim one event"


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def test_mni_mean_penalises_au_padding():
    """Padding the critical set with low-necessity AUs must lower the reward."""
    tight, _ = reward_causal(0.8, {"AU4": 0.31, "AU9": 0.28}, 1.0)
    padded, _ = reward_causal(0.8, {"AU4": 0.31, "AU9": 0.28, "AU25": 0.001,
                                    "AU20": 0.002}, 1.0)
    assert padded < tight


def test_structure_term_penalises_graph_padding():
    close, _ = reward_causal(0.8, {"AU4": 0.3}, 0.0)
    far, _ = reward_causal(0.8, {"AU4": 0.3}, 8.0)
    assert far < close


def test_curriculum_moves_weight_to_causal_and_temporal():
    early, late = curriculum_weights(0.0), curriculum_weights(1.0)
    assert early["fmt"] > late["fmt"], "format is front-loaded"
    assert late["causal"] > early["causal"]
    assert late["temp"] > early["temp"]
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert abs(sum(curriculum_weights(progress).values()) - 1.0) < 1e-6


def test_composite_reward_is_bounded():
    reward = CompositeReward()
    breakdown = reward.score(
        {"fine_label": "disgust", "coarse_label": "negative", "k_crit": ["AU4", "AU9"],
         "interval": [1000, 1011], "P": {"a": "x"}, "M": {"a": "y"}, "C": {"a": "z"},
         "MC": {"a": "w"}, "es": {"disgust": 0.8}, "dc": {"disgust": 1.0}, "refs": []},
        {"fine": "disgust", "coarse": "negative", "aus": ["AU4", "AU9"],
         "interval": [1000, 1011]},
        judge={"dc": 1.0, "mni": {"AU4": 0.31, "AU9": 0.28},
               "graph_edit_distance": 0.0},
    )
    assert 0.0 <= breakdown.total <= 1.0
    assert breakdown.r_emo > 0.9 and breakdown.r_temp > 0.9


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


def test_evidence_sufficiency_ranks_the_right_hypothesis():
    """AU4 + AU7 active with AU9 weak should favour disgust over anger."""
    assert (evidence_sufficiency("disgust", ["AU4", "AU7"], ["AU9"])
            > evidence_sufficiency("anger", ["AU4", "AU7"], ["AU9"]))
    assert (evidence_sufficiency("disgust", ["AU4", "AU7"], ["AU9"])
            > evidence_sufficiency("happiness", ["AU4", "AU7"], ["AU9"]))


def test_contradictory_aus_reduce_sufficiency():
    clean = evidence_sufficiency("happiness", ["AU6", "AU12"])
    contaminated = evidence_sufficiency("happiness", ["AU6", "AU12", "AU4", "AU9"])
    assert contaminated < clean


# ---------------------------------------------------------------------------
# Data alignment
# ---------------------------------------------------------------------------


def test_flow_pairs_on_the_later_frame():
    """Flow frame N holds the motion of (N - k, N), so it aligns with video frame N."""
    from mewm.data.paths import VideoPaths
    paths = VideoPaths("casme_sq", "s15/demo", flow_gap=7)
    assert paths.flow_source_pair(699) == (692, 699)
    assert paths.frame(703).name == "img703.jpg"
    assert paths.flow(703).name == "img703.jpg"
    assert "pre_datasets" in str(paths.flow_dir)
    assert "dataset" in str(paths.frame_dir)


def test_question_classification_is_text_based():
    """Question indices shift with event count, so classification must read the text."""
    from mewm.data.qa_loader import classify_question, parse_segment_question
    assert classify_question(
        "Reason over the whole video: how many micro-expression events does it contain?"
    ) == "reason_full"
    assert classify_question(
        "How many micro-expression events appear in this video? "
        "Localize every micro-expression event in the video."
    ) == "localize_micro"
    assert classify_question("How many micro-expression events appear in this video?") == "count_micro"
    span = parse_segment_question(
        "In the 3-th expression event of this video (frames 699-707, apex 703): describe it.")
    assert span == {"index": 3, "onset": 699, "offset": 707, "apex": 703}


def test_detection_statistic_is_causal():
    """A future spike must not change an earlier score.

    Eq. (7) is defined on a trailing window; if any statistic (including the scale
    floor) were computed over the whole series, reported localisation accuracy would be
    optimistic in a way no downstream metric would surface.
    """
    rng = np.random.default_rng(11)
    base = np.abs(rng.normal(0.1, 0.02, 400))
    altered = base.copy()
    altered[300] += 5.0
    assert np.allclose(robust_normalise(base, 100)[:300],
                       robust_normalise(altered, 100)[:300], atol=1e-9)


def test_iou_threshold_is_configurable():
    """The eq. (2) threshold is a hyper-parameter and must actually change the verdict."""
    proposals, truths = [(1000, 1005)], [(1000, 1011)]
    overlap = iou(proposals[0], truths[0])
    assert 0.3 < overlap < 0.6, overlap

    permissive = evaluate_proposals(proposals, truths, iou_threshold=0.3)
    strict = evaluate_proposals(proposals, truths, iou_threshold=0.9)
    assert permissive.tp_strict == 1 and strict.tp_strict == 0
    assert permissive.iou_threshold == 0.3 and strict.iou_threshold == 0.9


def test_affective_rescue_can_be_disabled():
    on = evaluate_proposals([(1000, 1003)], [(1000, 1011)], ["disgust"], ["disgust"],
                            affective_rescue=True)
    off = evaluate_proposals([(1000, 1003)], [(1000, 1011)], ["disgust"], ["disgust"],
                             affective_rescue=False)
    assert on.tp == 1 and off.tp == 0


def test_reward_follows_the_configured_iou_threshold():
    """Training against a different threshold than evaluation optimises the wrong thing."""
    from mewm.training.rewards import reward_temporal
    lenient = reward_temporal((0, 5), (0, 9), iou_threshold=0.3)
    strict = reward_temporal((0, 5), (0, 9), iou_threshold=0.9)
    assert lenient[1]["tp"] is True and strict[1]["tp"] is False
    assert lenient[0] > strict[0]


def test_model_registry_resolves_the_served_ids():
    """Two ids in circulation are not what the endpoints accept; aliases must fix that."""
    from mewm.llm.registry import resolve
    assert resolve("deepseekV4-Flash-Vision-Exp").model_id == "deepseek-v4-flash-vision-exp"
    assert resolve("gemini-3-pro-preview").model_id == "gemini-3-pro"


def test_verification_heterogeneity_is_per_provider():
    """Two tiers of one family share training, so id inequality is not enough."""
    from mewm.llm.registry import heterogeneous
    assert heterogeneous("claude-sonnet-5", "gpt-5.6-sol")
    assert heterogeneous("claude-sonnet-5", "grok-4.5")
    assert not heterogeneous("gemini-3-pro", "gemini-3.1-pro")
    assert not heterogeneous("claude-sonnet-5", "claude-sonnet-5")


def test_every_registered_model_has_a_reachable_transport():
    from mewm.llm.registry import MODELS, PROVIDERS, TRANSPORTS
    for spec in MODELS.values():
        assert spec.provider in PROVIDERS, spec.model_id
        assert spec.transport in TRANSPORTS, spec.model_id
        if spec.open_weights:
            assert spec.hf_repo, f"{spec.model_id} has no HF repo"


def test_effort_validation_rejects_unavailable_tiers():
    from mewm.llm.registry import resolve
    assert resolve("claude-sonnet-5").validate_effort("max") == "max"
    assert resolve("gpt-5.6-sol").validate_effort("Extra high") == "xhigh"
    try:
        resolve("gpt-5.6-sol").validate_effort("max")
        raise AssertionError("gpt-5.6-sol does not offer the max tier")
    except ValueError:
        pass



# ---------------------------------------------------------------------------
# Answer composition
# ---------------------------------------------------------------------------


def _demo_graph():
    from mewm.schemas import AUDynGraph, AUEdge, AUNode
    graph = AUDynGraph(cid="p01")
    for au, t_on, peak in (("AU4", 699, 0.85), ("AU7", 701, 0.62), ("AU24", 703, 0.48)):
        graph.nodes[au] = AUNode(au=au, t_on=t_on, t_apex=t_on + 4, t_off=t_on + 8,
                                 peak=peak, rise_slope=0.2, decay_slope=-0.15)
    graph.edges = [
        AUEdge("AU4", "AU7", "+", 2, 66.7, 0.232, 0.6, 0.5),
        AUEdge("AU7", "AU24", "+", 2, 66.7, 0.254, 0.6, 0.55),
        AUEdge("AU4", "AU24", "+", 4, 133.3, 0.201, 0.6, 0.45),
    ]
    return graph


def test_graph_analytics_are_well_formed():
    from mewm.eval.graph_analytics import (
        acyclicity_score, betweenness_centrality, gcn_propagation, main_path,
        strongest_links,
    )
    graph = _demo_graph()

    centrality = betweenness_centrality(graph)
    assert set(centrality) == set(graph.nodes)
    assert all(0.0 <= v <= 1.0 for v in centrality.values()), centrality
    # This graph carries a direct AU4 -> AU24 edge, so the two-hop route through AU7 is
    # not a shortest path and every node correctly scores zero. Betweenness only rewards
    # a node that traffic must pass through.
    assert all(v == 0.0 for v in centrality.values()), centrality

    # Remove the shortcut and AU7 becomes the sole intermediate hop.
    chain = _demo_graph()
    chain.edges = [e for e in chain.edges if not (e.source == "AU4" and e.target == "AU24")]
    chain_centrality = betweenness_centrality(chain)
    assert max(chain_centrality, key=lambda k: chain_centrality[k]) == "AU7", chain_centrality
    assert chain_centrality["AU7"] > 0.0

    # This graph is a DAG, so the acyclicity functional must vanish.
    assert acyclicity_score(graph) < 1e-6

    links = strongest_links(graph, 3)
    assert links[0][2] >= links[-1][2], links

    path = main_path(graph, "disgust")
    assert path[0] == "AU4" and path[-1] == "Disgust", path

    propagation = gcn_propagation(graph)
    assert 0.0 <= propagation["consistency_gap"]
    assert len(propagation["top_nodes"]) == 2


def test_acyclicity_detects_a_cycle():
    from mewm.eval.graph_analytics import acyclicity_score
    from mewm.schemas import AUEdge
    graph = _demo_graph()
    graph.edges.append(AUEdge("AU24", "AU4", "+", 1, 33.0, 0.5, 0.5, 0.5))
    assert acyclicity_score(graph) > 1e-6, "a cycle must register"


def test_confidence_bands_split_by_necessity():
    from mewm.eval.graph_analytics import confidence_bands
    bands = confidence_bands({"AU4": 0.31, "AU7": 0.10, "AU25": 0.001})
    assert bands["high"] == ["AU4"] and bands["low"] == ["AU25"]


def test_composed_answer_contains_the_required_elements():
    """The answer must carry counting, localisation, labels, graph and CFI."""
    from mewm.eval.answer_composer import AnswerComposer
    from mewm.schemas import CandidateInterval, CausalCoT, ErrorRecord, Verdict, VideoMeta

    state = MEWMState(video_meta=VideoMeta("v", "casme_sq", "/p", 30.0, 1000, "s15"))
    proposal = CandidateInterval("p01", 699, 707, 703, 6.2, {"AU4": 0.6})
    state.proposals = [proposal]
    state.error_record = ErrorRecord("v", 0, s_curve=[0.2] * 1000)
    state.au_graphs["p01"] = _demo_graph()
    state.causal_cots["p01"] = CausalCoT(
        cid="p01", es={"disgust": 0.81, "anger": 0.63},
        dc={"disgust": 0.74, "anger": 0.41}, k_crit=["AU4", "AU24"],
        fine_label="disgust", coarse_label="negative")
    state.verdicts["p01"] = Verdict("p01", "negative", "disgust", 0.78,
                                    prototype_completeness=0.6)

    class _Result:
        pass

    result = _Result()
    result.state = state
    result.episodic = None

    composed = AnswerComposer().compose(
        result, annotated_events=[],
        cfi_by_cid={"p01": {"AU4": 0.31, "AU24": 0.02}})
    text = composed["final_answer"]

    for needle in ("micro-expression", "frames 699-707", "apex 703",
                   "coarse-grained: negative", "fine-grained: disgust",
                   "W-matrix summary", "Global main path", "DAG validation",
                   "Key causal node", "Authenticity score", "GCN-style validation",
                   "Counterfactual feature intervention", "Label recheck",
                   "AU-change CoT"):
        assert needle in text, f"missing {needle!r} from the composed answer"
    assert "23.3s-23.57s" in text or "23.3s" in text, text[:200]
    assert composed["n_detected"] == 1


def test_answer_supports_a_negative_result():
    """"Nothing found" is a claim and must cite the curve, not be an empty string."""
    from mewm.eval.answer_composer import AnswerComposer
    from mewm.schemas import ErrorRecord, VideoMeta

    state = MEWMState(video_meta=VideoMeta("v", "casme_sq", "/p", 30.0, 1000, "s15"))
    state.error_record = ErrorRecord("v", 0, s_curve=[0.2] * 1000)

    class _Result:
        pass

    result = _Result()
    result.state = state
    result.episodic = None

    composed = AnswerComposer().compose(result)
    assert composed["n_detected"] == 0
    assert "0 micro-expression" in composed["final_answer"]
    assert "baseline" in composed["final_answer"].lower()


def test_answer_discloses_degradation():
    """A run produced under fallback must say so in the answer itself."""
    from mewm.eval.answer_composer import AnswerComposer
    from mewm.schemas import CandidateInterval, ErrorRecord, Verdict, VideoMeta

    state = MEWMState(video_meta=VideoMeta("v", "casme_sq", "/p", 30.0, 1000, "s15"))
    state.proposals = [CandidateInterval("p01", 699, 707, 703, 6.2, {})]
    state.error_record = ErrorRecord("v", 0, s_curve=[0.2] * 1000)
    state.verdicts["p01"] = Verdict("p01", "negative", "disgust", 0.5)
    state.budget.degrade("A.encode@p01: gate never passed")

    class _Result:
        pass

    result = _Result()
    result.state = state
    result.episodic = None

    composed = AnswerComposer().compose(result)
    assert composed["degraded"] is True
    text = composed["final_answer"]
    # The caveat must survive, but in reader-facing language: no module names, no
    # "stage(s) degraded" internals.
    assert "provisional" in text
    assert "Reliability" in text
    assert "degraded" not in text.lower()
    assert "R.narrate" not in text and "A.encode" not in text



def test_suppression_field_is_constrained_to_its_enum():
    """Models answer this field with a paragraph; it must still land on a valid state."""
    from mewm.schemas import SUPPRESSION_STATES, coerce_suppression

    for state in SUPPRESSION_STATES:
        assert coerce_suppression(state) == (state, "")

    prose = ("fine label 'surprise' suppressed to coarse 'other': "
             "prototype_completeness=0.0 means the prototype is not established")
    resolved, kept = coerce_suppression(prose)
    # A label-level downgrade is not a C.4 facial-suppression finding.
    assert resolved == "none"
    assert kept == prose, "the reasoning must be preserved, not discarded"

    assert coerce_suppression("a masquerading social smile")[0] == "masked"
    assert coerce_suppression("neutralised residual trace")[0] == "neutralised"
    assert coerce_suppression(None) == ("none", "")


def test_model_emitted_numbers_are_coerced_not_crashed():
    """Prose in a numeric field must degrade to a dropped entry, never an exception."""
    from mewm.agents.base import coerce_float, coerce_float_map
    assert coerce_float_map({"a": 0.8, "b": "high, coherence 0.85", "c": "none"}) == {
        "a": 0.8, "b": 0.85}
    assert coerce_float("confidence is 0.47") == 0.47
    assert coerce_float(None) == 0.0
    assert coerce_float({"nested": 1}) == 0.0


def test_open_questions_survive_any_emitted_shape():
    """An open question is the record that something is unresolved; never drop it."""
    from mewm.schemas import OpenQuestion
    assert OpenQuestion.coerce("bare string", "p01").detail == "bare string"
    assert OpenQuestion.coerce({"kind": "k", "detail": "d"}, "p01").kind == "k"
    assert OpenQuestion.coerce({"text": "t"}, "p01").detail == "t"
    assert OpenQuestion.coerce(123, "p01").detail == "123"



def test_answer_carries_no_paper_citations_or_internals():
    """The summary must read as facial analysis, not as a tour of the implementation."""
    from mewm.eval.answer_format import scrub, validate_answer

    offending = (
        "This video contains 1 micro-expression event. The masking necessity index of "
        "eq. (10) and appendix C.4 apply under gate rule R5; the R.narrate phase did "
        "not complete because optical flow was read back from HSV-rendered images."
    )
    report = validate_answer(offending, require_sections=False)
    assert not report.ok
    assert report.citations, "eq./appendix/rule references must be caught"
    assert report.internals, "implementation details must be caught"

    cleaned = scrub(offending)
    assert validate_answer(cleaned, require_sections=False).ok, scrub(offending)
    for banned in ("appendix", "eq. (10)", "gate rule", "HSV", "R.narrate",
                   "masking necessity index"):
        assert banned.lower() not in cleaned.lower(), cleaned


def test_answer_format_requires_the_reference_sections_in_order():
    from mewm.eval.answer_format import REQUIRED_SECTIONS, validate_answer

    ordered = " ".join(marker for _name, marker in REQUIRED_SECTIONS)
    report = validate_answer(ordered)
    assert not report.missing_sections, report.missing_sections
    assert not report.out_of_order, report.out_of_order

    shuffled = " ".join(marker for _name, marker in reversed(REQUIRED_SECTIONS))
    assert validate_answer(shuffled).out_of_order

    # A genuine negative result needs no per-event sections.
    negative = "This video contains 0 micro-expression events detected by the framework."
    assert not validate_answer(negative).missing_sections


def test_composed_answer_passes_its_own_format_check():
    from mewm.eval.answer_composer import AnswerComposer
    from mewm.schemas import CandidateInterval, CausalCoT, ErrorRecord, Verdict, VideoMeta

    state = MEWMState(video_meta=VideoMeta("v", "casme_sq", "/p", 30.0, 1000, "s16"))
    proposal = CandidateInterval("p01", 270, 278, 275, 6.2, {"AU4": 0.6})
    state.proposals = [proposal]
    state.error_record = ErrorRecord("v", 0, s_curve=[0.2] * 1000)
    state.au_graphs["p01"] = _demo_graph()
    state.causal_cots["p01"] = CausalCoT(
        cid="p01", es={"disgust": 0.81}, dc={"disgust": 0.74},
        k_crit=["AU4", "AU24"], fine_label="disgust", coarse_label="negative")
    state.verdicts["p01"] = Verdict("p01", "negative", "disgust", 0.78,
                                    prototype_completeness=0.6)

    class _Result:
        pass

    result = _Result()
    result.state = state
    result.episodic = None

    composed = AnswerComposer().compose(
        result, annotated_events=[], cfi_by_cid={"p01": {"AU4": 0.31, "AU24": 0.02}},
        saturated_by_cid={"p01": True})
    assert composed["format_report"]["ok"], composed["format_report"]
    # The saturation caveat must survive, in domain language.
    assert "Reliability note" in composed["final_answer"]
    assert "HSV" not in composed["final_answer"]



# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed, failed = 0, []
    for name, test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("\nfailures:")
        for name, exc in failed:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
