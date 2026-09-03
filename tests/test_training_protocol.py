"""Tests for the LOSO training protocol: folds, the sufficiency gate, and augmented QA.

These target the properties that make the protocol a protocol rather than a loop:

* the split really is subject-disjoint, and a fold cannot read another fold's material;
* the four sufficiency judgements return the verdict the arithmetic implies, including the
  cases the original plan omitted (all-high rewards, a degenerate mid-band group);
* pass@k matches the closed-form estimator on hand-checkable inputs;
* augmented QA is byte-compatible with the reference 4-field jsonl;
* the candidate filter never spends its budget on a failing candidate;
* the removed DPO objective stays removed.

A regression in any of them silently corrupts a training run rather than crashing it,
which is why they are asserted rather than left to inspection.

Run with:  python -m pytest tests/test_training_protocol.py -v
       or:  python tests/test_training_protocol.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mewm.config import EvaluationConfig, TrainingConfig, load_config
from mewm.data.datasets import DatasetIndex, ExpressionEvent, LongVideo
from mewm.eval.pass_criteria import PassOutcome, evaluate_sample, stage_rates
from mewm.training import qa_augment
from mewm.training.candidate_filter import Candidate, group_rewards, select
from mewm.training.diagnostics import (
    assess, classify_group, format_alignment, headroom, loss_plateau, pass_at_k,
    reward_distribution,
)
from mewm.training.grpo import GRPOConfig, admit_prompt
from mewm.training.instruction_set import InstructionSetBuilder
from mewm.training.loso import (
    FoldSpec, PolicyNotTrainable, build_folds, require_open_weight, summarise,
)
from mewm.training.sft import DryRunSFT, SFTTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _video(subject: str, index: int) -> LongVideo:
    key = f"{subject}_v{index}"
    return LongVideo(
        dataset="casme_sq", video_key=key, subject=subject,
        folder_rel=f"{subject}/{key}", fps=200.0, flow_gap=4,
        frame_lo=1, frame_hi=1000,
        events=[ExpressionEvent(
            event_id=f"{key}_0", onset=100, apex=140, offset=180,
            fine_label="happiness", coarse_label="positive",
            aus=["AU6", "AU12"], expression_type="micro-expression",
            subject=subject, video_key=key, event_index=0,
        )],
    )


def _index(n_subjects: int = 5, per_subject: int = 3) -> DatasetIndex:
    videos = [_video(f"s{10 + s}", v)
              for s in range(n_subjects) for v in range(per_subject)]
    return DatasetIndex(dataset="casme_sq", videos=videos, fps=200.0, flow_gap=4)


def _outcome(passed: bool, format_ok: bool = True) -> PassOutcome:
    return PassOutcome(passed=passed, format_ok=format_ok,
                       temporal_ok=passed, label_ok=passed,
                       iou=0.7 if passed else 0.1,
                       reasons=[] if format_ok else ["output did not parse as JSON"])


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def test_every_subject_gets_exactly_one_fold():
    index = _index(n_subjects=5)
    folds = build_folds(index, TrainingConfig())
    assert len(folds) == 5
    assert sorted(f.test_subject for f in folds) == index.subjects()


def test_folds_are_subject_disjoint():
    for fold in build_folds(_index(), TrainingConfig()):
        fold.check_disjoint()   # raises on overlap
        assert fold.test_subject not in fold.train_subjects
        assert fold.test_subject not in fold.val_subjects


def test_the_training_pool_is_every_other_subject():
    """The user's protocol: N-1 subjects train, none are held back for validation."""
    index = _index(n_subjects=5)
    for fold in build_folds(index, TrainingConfig(n_val_subjects=0)):
        assert len(fold.train_subjects) == 4
        assert fold.val_subjects == []
        assert fold.in_sample is True


def test_a_validation_split_leaves_at_least_one_training_subject():
    index = _index(n_subjects=3)
    # Asking for 5 validation subjects out of a 2-subject pool must not empty the pool.
    for fold in build_folds(index, TrainingConfig(n_val_subjects=5)):
        assert len(fold.train_subjects) >= 1
        assert fold.in_sample is False


def test_validation_choice_is_reproducible_and_fold_local():
    index = _index(n_subjects=6)
    config = TrainingConfig(n_val_subjects=2)
    first = {f.name: f.val_subjects for f in build_folds(index, config)}
    second = {f.name: f.val_subjects for f in build_folds(index, config)}
    assert first == second
    # A fold's own choice must not change when other folds are not built.
    single = build_folds(index, config, subjects=["s13"])[0]
    assert single.val_subjects == first["s13"]


def test_test_videos_never_appear_in_the_pool():
    for fold in build_folds(_index(), TrainingConfig()):
        assert not (fold.pool_videos() & set(fold.test_videos))


def test_folds_can_be_restricted_to_a_subset():
    index = _index(n_subjects=5)
    folds = build_folds(index, TrainingConfig(), subjects=["s11", "s13"])
    assert [f.name for f in folds] == ["s11", "s13"]


def test_an_unknown_fold_subject_is_an_error():
    try:
        build_folds(_index(), TrainingConfig(), subjects=["nobody"])
    except KeyError:
        return
    raise AssertionError("an unknown subject must not be silently ignored")


def test_a_corrupt_fold_is_rejected():
    fold = FoldSpec(dataset="casme_sq", test_subject="s15",
                    train_subjects=["s15", "s16"],
                    train_videos=["s15_v0"], test_videos=["s15_v0"])
    try:
        fold.check_disjoint()
    except ValueError:
        return
    raise AssertionError("a subject on both sides of the split must raise")


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_the_instruction_set_honours_the_subject_filter():
    index = _index(n_subjects=4)
    fold = build_folds(index, TrainingConfig(), subjects=["s11"])[0]
    builder = InstructionSetBuilder()
    samples = builder.build_dataset(
        index.videos, None, include_subjects=fold.train_subjects)
    built = {s.video for s in samples}
    assert built and not (built & set(fold.test_videos))


def test_excluded_subjects_are_dropped_even_when_whitelisted():
    index = _index(n_subjects=3)
    builder = InstructionSetBuilder()
    samples = builder.build_dataset(
        index.videos, None, include_subjects=["s10", "s11"], exclude_subjects=["s11"])
    assert {s.video.split("_")[0] for s in samples} == {"s10"}


def test_a_cross_fold_augmented_file_is_refused():
    """The leak that outlives a run: fold s16's file loaded into fold s15.

    Exercised through ``load_augmented`` itself, with the path redirected at a temp file
    so the real ``Q-T-A`` tree is never touched. Simulating the check by hand would test
    the test rather than the loader.
    """
    pair = qa_augment.AugmentedPair(
        video_id="casme_sq_augs16_s16_v0_1", video="s16_v0",
        question="q", answer="a", dataset="casme_sq", fold="s16", subject="s16")
    original = qa_augment.augmented_jsonl
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aug.jsonl"
        path.write_text(json.dumps(pair.to_jsonl(), ensure_ascii=False) + "\n",
                        encoding="utf-8")
        qa_augment.augmented_jsonl = lambda dataset, fold: path
        try:
            # s16_v0 belongs to s16, which is not in fold s15's pool.
            try:
                qa_augment.load_augmented("casme_sq", "s15",
                                          allowed_videos={"s15_v0", "s15_v1"})
            except qa_augment.AugmentationError:
                pass
            else:
                raise AssertionError(
                    "loading another fold's augmented file must raise")
            # The same file is legitimate for the fold it was built for.
            rows = qa_augment.load_augmented("casme_sq", "s16",
                                             allowed_videos={"s16_v0"})
            assert len(rows) == 1 and rows[0]["video"] == "s16_v0"
        finally:
            qa_augment.augmented_jsonl = original


def test_a_malformed_augmented_row_is_refused_at_load():
    original = qa_augment.augmented_jsonl
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aug.jsonl"
        path.write_text(json.dumps(
            {"video_id": "x", "video": "s16_v0", "question": "q",
             "answer": "a", "reward": 0.9}) + "\n", encoding="utf-8")
        qa_augment.augmented_jsonl = lambda dataset, fold: path
        try:
            qa_augment.load_augmented("casme_sq", "s15")
        except qa_augment.AugmentationError:
            return
        finally:
            qa_augment.augmented_jsonl = original
    raise AssertionError("an extra field must be refused on the way in too")


def test_writing_a_pair_from_outside_the_pool_raises():
    pair = qa_augment.AugmentedPair(
        video_id="casme_sq_augs15_s16_v0_1", video="s16_v0",
        question="q", answer="a", dataset="casme_sq", fold="s15", subject="s16")
    try:
        qa_augment.write_augmented([pair], "casme_sq", "s15", allowed_videos={"s15_v0"})
    except qa_augment.AugmentationError:
        return
    raise AssertionError("a stray video must raise rather than be dropped")


# ---------------------------------------------------------------------------
# Augmented QA format
# ---------------------------------------------------------------------------


def test_augmented_pairs_carry_exactly_the_reference_fields():
    pair = qa_augment.AugmentedPair(
        video_id="casme_sq_augs15_s16_v0_1", video="s16_v0",
        question="q", answer="a", dataset="casme_sq", fold="s15", subject="s16")
    payload = pair.to_jsonl()
    assert tuple(payload) == qa_augment.QA_FIELDS
    qa_augment.validate_schema(payload)   # raises on any deviation


def test_an_extra_field_in_the_jsonl_is_rejected():
    payload = {"video_id": "x", "video": "v", "question": "q", "answer": "a",
               "reward": 0.9}
    try:
        qa_augment.validate_schema(payload)
    except qa_augment.AugmentationError:
        return
    raise AssertionError("provenance must live in the manifest, not the jsonl")


def test_reordered_fields_are_rejected():
    payload = {"video": "v", "video_id": "x", "question": "q", "answer": "a"}
    try:
        qa_augment.validate_schema(payload)
    except qa_augment.AugmentationError:
        return
    raise AssertionError("field order is part of the format contract")


def test_the_video_id_names_the_source_video_and_fold():
    vid = qa_augment.make_video_id("casme_sq", "s15", "s16_0102", 3)
    assert "s15" in vid and "s16_0102" in vid and vid.endswith("_3")


def test_the_manifest_records_which_video_each_pair_came_from():
    pair = qa_augment.AugmentedPair(
        video_id="casme_sq_augs15_s16_v0_1", video="s16_v0", question="q", answer="a",
        dataset="casme_sq", fold="s15", subject="s16", event_index=2,
        interval=(100, 180), reward=0.81)
    manifest = pair.to_manifest()
    assert manifest["video"] == "s16_v0"
    assert manifest["subject"] == "s16"
    assert list(manifest["interval"]) == [100, 180]


# ---------------------------------------------------------------------------
# 1a -- format alignment
# ---------------------------------------------------------------------------


def test_format_alignment_separates_parse_from_contract_failures():
    outcomes = [_outcome(True), _outcome(False, format_ok=False),
                PassOutcome(passed=False, format_ok=False,
                            reasons=["missing required field 'onset'"])]
    report = format_alignment(outcomes, threshold=0.95)
    assert report.parse_failures == 1
    assert report.contract_failures == 1
    assert not report.ok


def test_format_alignment_needs_data():
    assert not format_alignment([], threshold=0.95).ok


def test_format_alignment_passes_at_the_threshold():
    outcomes = [_outcome(True)] * 19 + [_outcome(False, format_ok=False)]
    assert format_alignment(outcomes, threshold=0.95).ok


# ---------------------------------------------------------------------------
# 1b -- plateau
# ---------------------------------------------------------------------------


def test_a_flat_curve_is_a_plateau():
    report = loss_plateau([0.5] * 40)
    assert report.ok and not report.still_descending


def test_a_descending_curve_is_not_a_plateau():
    curve = [1.0 - 0.01 * i for i in range(40)]
    report = loss_plateau(curve)
    assert not report.ok and report.still_descending


def test_an_oscillating_curve_is_not_a_plateau():
    """Flat on average but still swinging: the user's 'no large fluctuation' criterion."""
    curve = [0.5 + (0.2 if i % 2 else -0.2) for i in range(40)]
    report = loss_plateau(curve)
    assert not report.ok
    assert report.cv > report.max_cv


def test_a_short_curve_cannot_be_judged():
    assert not loss_plateau([1.0, 0.9]).ok


# ---------------------------------------------------------------------------
# 2a -- pass@k
# ---------------------------------------------------------------------------


def _pass_at_k_reference(n: int, c: int, k: int) -> float:
    """Definition, straight from combinatorics -- no algebraic rearrangement."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def test_pass_at_k_matches_the_closed_form():
    for n in range(1, 17):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                assert abs(pass_at_k(n, c, k) - _pass_at_k_reference(n, c, k)) < 1e-9, \
                    f"n={n} c={c} k={k}"


def test_pass_at_k_edge_cases():
    assert pass_at_k(16, 0, 8) == 0.0     # never passes
    assert pass_at_k(16, 16, 1) == 1.0    # always passes
    assert pass_at_k(8, 1, 8) == 1.0      # k covers the whole sample


def test_pass_at_k_is_monotone_in_k():
    """Which is why 'pass@k > pass@1' alone proves nothing -- it is a tautology."""
    for k in range(1, 16):
        assert pass_at_k(16, 4, k) <= pass_at_k(16, 4, k + 1) + 1e-12


def test_headroom_needs_a_gap_and_a_floor():
    # Passes 4 of 16 per prompt: pass@1 = 0.25, pass@8 is high -> real headroom.
    outcomes = {f"p{i}": [_outcome(True)] * 4 + [_outcome(False)] * 12
                for i in range(5)}
    report = headroom(outcomes, k=8, gap_min=0.15, floor=0.5)
    assert report.verdict == "rl_ready"
    assert report.gap > 0.15


def test_headroom_calls_a_policy_that_cannot_pass_undertrained():
    outcomes = {f"p{i}": [_outcome(False)] * 16 for i in range(5)}
    assert headroom(outcomes, k=8, floor=0.5).verdict == "undertrained"


def test_headroom_calls_a_policy_that_always_passes_saturated():
    outcomes = {f"p{i}": [_outcome(True)] * 16 for i in range(5)}
    report = headroom(outcomes, k=8, gap_min=0.15, floor=0.5)
    assert report.verdict == "saturated"
    assert report.gap < 0.15


def test_headroom_with_no_prompts_is_no_data():
    assert headroom({}, k=8).verdict == "no_data"


# ---------------------------------------------------------------------------
# 2b -- reward distribution
# ---------------------------------------------------------------------------


def test_a_spread_group_is_trainable():
    assert classify_group([0.1, 0.4, 0.6, 0.9]) == "spread"


def test_a_uniformly_low_group_means_undertrained():
    assert classify_group([0.02, 0.03, 0.01, 0.02]) == "all_low"


def test_a_uniformly_high_group_means_saturated():
    """The case the original plan omitted: all-high is as useless as all-low."""
    assert classify_group([0.97, 0.98, 0.99, 0.98]) == "all_high"


def test_a_tight_mid_band_group_counts_as_untrainable():
    """std below the floor, mean mid-range: zero advantage, so not 'spread'."""
    assert classify_group([0.5, 0.5, 0.5, 0.5]) != "spread"


def test_an_empty_group_is_not_trainable():
    assert classify_group([]) == "all_low"


def test_the_verdict_follows_the_majority_of_groups():
    groups = [[0.1, 0.4, 0.6, 0.9]] * 3 + [[0.0] * 4] * 2
    assert reward_distribution(groups).verdict == "rl_ready"


def test_a_pool_of_degenerate_groups_does_not_train():
    """Half at the floor, half at the ceiling: the pooled mean looks fine, nothing trains."""
    groups = [[0.0] * 4] * 3 + [[1.0] * 4] * 3
    report = reward_distribution(groups)
    assert report.n_spread == 0
    assert report.verdict != "rl_ready"


# ---------------------------------------------------------------------------
# The gate decision
# ---------------------------------------------------------------------------


def test_bad_format_blocks_rl_regardless_of_reward():
    report = assess(
        {f"p{i}": [_outcome(False, format_ok=False)] * 8 for i in range(4)},
        [[0.1, 0.5, 0.9, 0.4]] * 4, [0.5] * 40, TrainingConfig(), fold="s15")
    assert report.decision == "continue_sft"


def test_a_descending_loss_blocks_rl():
    report = assess(
        {f"p{i}": [_outcome(True)] * 4 + [_outcome(False)] * 12 for i in range(4)},
        [[0.1, 0.5, 0.9, 0.4]] * 4,
        [1.0 - 0.01 * i for i in range(40)], TrainingConfig(), fold="s15")
    assert report.decision == "continue_sft"


def test_all_low_rewards_send_the_run_back_to_sft():
    report = assess(
        {f"p{i}": [_outcome(False)] * 16 for i in range(4)},
        [[0.0] * 4] * 4, [0.5] * 40, TrainingConfig(), fold="s15")
    assert report.decision == "continue_sft"


def test_a_ready_policy_reaches_rl():
    report = assess(
        {f"p{i}": [_outcome(True)] * 4 + [_outcome(False)] * 12 for i in range(4)},
        [[0.1, 0.4, 0.6, 0.9]] * 4, [0.5] * 40, TrainingConfig(), fold="s15")
    assert report.decision == "rl_ready"


def test_every_in_sample_report_says_so():
    report = assess({}, [], [0.5] * 40, TrainingConfig(), fold="s15", in_sample=True)
    assert report.in_sample and report.measured_on == "training_pool"
    assert any("in-sample" in r for r in report.reasons())


def test_a_held_out_report_drops_the_caveat():
    report = assess({}, [], [0.5] * 40, TrainingConfig(), fold="s15", in_sample=False)
    assert report.measured_on == "held_out_subjects"
    assert not any("in-sample" in r for r in report.reasons())


# ---------------------------------------------------------------------------
# Candidate filter (the role DPO played)
# ---------------------------------------------------------------------------


def _candidate(prompt: str, reward: float, passed: bool, video: str = "s16_v0") -> Candidate:
    return Candidate(prompt_id=prompt, text="{}", product={"answer": "a"},
                     reward=reward, outcome=_outcome(passed),
                     dataset="casme_sq", video=video)


def test_only_passing_candidates_are_admitted():
    candidates = [_candidate("p0", 0.99, False), _candidate("p0", 0.70, True)]
    accepted, report = select(candidates, TrainingConfig(rft_accept_top_k=1,
                                                        rft_min_reward=0.6))
    assert [c.reward for c in accepted] == [0.70]
    assert report.n_accepted == 1
    assert report.n_candidates == 2


def test_the_reward_floor_is_enforced():
    candidates = [_candidate("p0", 0.30, True)]
    accepted, _ = select(candidates, TrainingConfig(rft_min_reward=0.6))
    assert accepted == []


def test_the_top_k_budget_is_per_prompt():
    candidates = ([_candidate("p0", 0.9 - 0.01 * i, True) for i in range(5)]
                  + [_candidate("p1", 0.9 - 0.01 * i, True) for i in range(5)])
    accepted, _ = select(candidates, TrainingConfig(rft_accept_top_k=2,
                                                    rft_min_reward=0.6))
    assert len(accepted) == 4
    assert {c.prompt_id for c in accepted} == {"p0", "p1"}


def test_group_rewards_are_grouped_by_prompt():
    candidates = [_candidate("p0", 0.5, True), _candidate("p0", 0.7, True),
                  _candidate("p1", 0.2, False)]
    groups = group_rewards(candidates)
    assert sorted(len(g) for g in groups) == [1, 2]


# ---------------------------------------------------------------------------
# GRPO prompt admission
# ---------------------------------------------------------------------------


def test_a_degenerate_group_at_the_floor_is_not_admitted():
    admitted, reason = admit_prompt([0.0, 0.0, 0.0, 0.0])
    assert not admitted and "floor" in reason


def test_a_degenerate_group_at_the_ceiling_is_not_admitted():
    """A group at the ceiling has zero advantage too -- it is not free progress."""
    admitted, reason = admit_prompt([1.0, 1.0, 1.0, 1.0])
    assert not admitted and "ceiling" in reason


def test_a_group_with_spread_is_admitted():
    admitted, _ = admit_prompt([0.1, 0.5, 0.9, 0.3])
    assert admitted


def test_a_single_sample_cannot_form_a_group():
    admitted, _ = admit_prompt([0.5])
    assert not admitted


def test_grpo_config_reads_the_training_config():
    training = TrainingConfig(rl_total_steps=1000, rl_group_size=8)
    config = GRPOConfig.from_training(training)
    assert config.total_steps == 1000
    assert config.group_size == 8
    assert config.admit_std_min == training.reward_std_min


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


def test_sft_treats_the_epoch_count_as_a_ceiling():
    trainer = SFTTrainer(DryRunSFT(), TrainingConfig(sft_max_epochs=100))
    outcome = trainer.fit([{"messages": [{"role": "user", "content": "q"},
                                         {"role": "assistant", "content": "a"}]}] * 8)
    assert outcome.max_epochs == 100
    assert outcome.selected_epoch <= 100


def test_sft_records_that_it_stopped_on_the_training_pool():
    trainer = SFTTrainer(DryRunSFT(), TrainingConfig())
    outcome = trainer.fit([{"messages": []}] * 4, in_sample_eval=True)
    assert outcome.evaluated_on == "training_pool"


def test_sft_with_no_samples_does_not_pretend_to_train():
    trainer = SFTTrainer(DryRunSFT(), TrainingConfig())
    outcome = trainer.fit([])
    assert outcome.selected_epoch == 0 and "no samples" in outcome.stop_reason


# ---------------------------------------------------------------------------
# Policy eligibility
# ---------------------------------------------------------------------------


def test_a_hosted_api_model_cannot_be_the_policy():
    for model in ("claude-sonnet-5", "gpt-5", "gemini-3-pro", "grok-4", "deepseek-chat"):
        try:
            require_open_weight(model)
        except PolicyNotTrainable:
            continue
        raise AssertionError(f"{model} exposes no gradients and must be rejected")


def test_an_open_weight_model_is_accepted():
    assert require_open_weight("Qwen3-VL-8B") == "Qwen3-VL-8B"
    assert require_open_weight("InternVL3-8B") == "InternVL3-8B"


def test_the_configured_default_policy_is_trainable():
    require_open_weight(load_config().training.policy_model)


# ---------------------------------------------------------------------------
# DPO removal
# ---------------------------------------------------------------------------


def test_the_dpo_objective_is_gone():
    import mewm.training.grpo as grpo
    for symbol in ("DPOTrainer", "DPOConfig", "PreferencePair", "build_preference_pairs"):
        assert not hasattr(grpo, symbol), f"{symbol} was removed and must stay removed"
    assert all("DPO" not in name for name in grpo.__all__)


#: The only modules allowed to say "DPO", and only to explain why there is no DPO stage.
DPO_EXPLAINERS = {"candidate_filter.py", "loso.py"}


def test_no_module_advertises_dpo_training():
    """No module may describe DPO as something this framework does.

    The word boundary matters: a plain substring search for ``DPO`` also matches
    ``ENDPOINT``, which produced a page of false hits when this was first checked by eye.
    """
    root = Path(__file__).resolve().parent.parent / "mewm"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in DPO_EXPLAINERS:
            continue
        for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"\bDPO\b", line):
                offenders.append(f"{path.name}:{line_no}: {line.strip()[:70]}")
    assert not offenders, "stale DPO wording: " + "; ".join(offenders[:5])


def test_the_explainers_frame_dpo_as_removed():
    root = Path(__file__).resolve().parent.parent / "mewm" / "training"
    for name in DPO_EXPLAINERS:
        text = (root / name).read_text(encoding="utf-8")
        if not re.search(r"\bDPO\b", text):
            continue
        assert re.search(r"no preference|used to do|would have|rather than|instead of",
                         text), f"{name} mentions DPO without saying it was removed"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_summary_keeps_the_in_sample_caveat():
    from mewm.training.loso import FoldResult

    folds = build_folds(_index(n_subjects=3), TrainingConfig(n_val_subjects=0))
    results = [FoldResult(fold=f, decision="rl_ready") for f in folds]
    summary = summarise(results)
    assert summary["in_sample_folds"] == 3
    assert "in-sample" in summary["caveat"]


def test_stage_rates_expose_the_bottleneck():
    outcomes = [_outcome(True)] * 2 + [
        PassOutcome(passed=False, format_ok=True, temporal_ok=False, label_ok=True)] * 2
    rates = stage_rates(outcomes)
    assert rates["format"] == 1.0
    assert rates["temporal"] == 0.5
    assert rates["joint"] == 0.5


# ---------------------------------------------------------------------------
# Findings from the external code review (see 代码核查反驳报告.md for the ones
# that were rejected). Each test below pins a defect the review found and that
# was confirmed against the source.
# ---------------------------------------------------------------------------


def test_the_fold_seed_survives_a_different_hash_salt():
    """P2: ``hash()`` is salted per process, so it cannot seed a reproducible split."""
    import subprocess
    import sys as _sys

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from mewm.training.loso import _fold_seed\n"
        "print(_fold_seed(20260824, 'casme_sq', 's15'))\n" % str(ROOT)
    )
    seeds = set()
    for salt in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        out = subprocess.run([_sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, check=True)
        seeds.add(out.stdout.strip())
    assert len(seeds) == 1, f"fold seed varied with PYTHONHASHSEED: {seeds}"

    # And the built-in it replaced really does vary, so this test is not vacuous.
    hashes = set()
    for salt in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        out = subprocess.run(
            [_sys.executable, "-c", "print(hash(('x', 'casme_sq', 's15')))"],
            capture_output=True, text=True, env=env, check=True)
        hashes.add(out.stdout.strip())
    assert len(hashes) > 1, "PYTHONHASHSEED had no effect; the guard proves nothing"


def test_a_prompt_without_a_video_key_stops_the_fold():
    """P8: an unattributable prompt used to slip past the pool-isolation check."""
    from mewm.training.loso import LOSORunner
    from mewm.training.sft import DryRunSFT

    index = _index(n_subjects=3)
    fold = build_folds(index, TrainingConfig())[0]
    runner = LOSORunner("casme_sq", DryRunSFT(), index=index)

    for bad in ({"id": "p1"}, {"id": "p1", "video": ""}, {"id": "p1", "video_id": "x"}):
        try:
            runner.run_fold(fold, [bad])
        except ValueError as exc:
            assert "no 'video' key" in str(exc), str(exc)
        else:
            raise AssertionError(f"{bad} was accepted without a video attribution")


def test_the_registry_outranks_the_closed_source_heuristic():
    """P12: a registered open-weight id is trainable whatever its name looks like."""
    from mewm.llm.registry import list_models
    from mewm.training.loso import PolicyNotTrainable, require_open_weight

    open_ids = [s.model_id for s in list_models(open_only=True)]
    hosted_ids = [s.model_id for s in list_models(hosted_only=True)]
    assert open_ids and hosted_ids, "registry has nothing to check against"

    for name in open_ids:
        require_open_weight(name)          # raises on failure
    for name in hosted_ids:
        try:
            require_open_weight(name)
        except PolicyNotTrainable as exc:
            assert "registered as a hosted model" in str(exc), str(exc)
        else:
            raise AssertionError(f"{name} is hosted but was accepted as a policy")

    # Unregistered ids still fall back to the substring heuristic.
    try:
        require_open_weight("gpt-9-turbo-unregistered")
    except PolicyNotTrainable:
        pass
    else:
        raise AssertionError("the heuristic fallback stopped firing")


def test_the_dead_rft_sampling_budget_is_gone():
    """P7: a config field nothing reads promises a stage-2 budget that does not exist."""
    import inspect as _inspect

    import mewm.config as config_module

    assert not hasattr(TrainingConfig(), "rft_samples_per_prompt")
    source = _inspect.getsource(config_module)
    assert "rft_samples_per_prompt" in source, (
        "the removal should stay documented where the field used to be")


def test_early_stopping_restores_the_selected_epoch_before_saving():
    """P3: the saved weights used to be the last epoch, not ``selected_epoch``."""
    from mewm.training.sft import DryRunSFT, SFTTrainer

    class Recording(DryRunSFT):
        """Loss descends for five epochs and then rises, so epoch 5 wins."""

        def __init__(self) -> None:
            super().__init__()
            self.epoch = 0
            self.weight = 0.0
            self.restored_to = None

        def step(self, batch, learning_rate):
            self.weight += 1.0
            return 1.0

        def evaluate(self, batch):
            self.epoch += 1
            return 1.0 - 0.1 * self.epoch if self.epoch <= 5 else 1.0

        def snapshot(self):
            return self.weight

        def restore(self, state):
            self.restored_to = state
            self.weight = state

    backend = Recording()
    trainer = SFTTrainer(backend, TrainingConfig(sft_max_epochs=20, sft_patience=3,
                                                 sft_batch_size=4))
    outcome = trainer.fit([{"messages": [{"role": "user", "content": "q"},
                                         {"role": "assistant", "content": "a"}]}])
    assert outcome.selected_epoch == 5, outcome.selected_epoch
    assert outcome.epochs_run > outcome.selected_epoch, "patience never elapsed"
    assert outcome.restored_selected_epoch, outcome.to_dict()
    assert backend.restored_to == 5.0, backend.restored_to
    assert outcome.to_dict()["restored_selected_epoch"] is True


def test_a_backend_that_cannot_snapshot_says_so():
    """The base backend returns None; the outcome must not claim a rollback happened."""
    from mewm.training.sft import DryRunSFT, SFTTrainer

    class Descending(DryRunSFT):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = 0

        def evaluate(self, batch):
            self.epoch += 1
            return 1.0 - 0.1 * self.epoch if self.epoch <= 5 else 1.0

    trainer = SFTTrainer(Descending(), TrainingConfig(sft_max_epochs=20, sft_patience=3,
                                                     sft_batch_size=4))
    outcome = trainer.fit([{"messages": [{"role": "user", "content": "q"},
                                         {"role": "assistant", "content": "a"}]}])
    assert outcome.selected_epoch < outcome.epochs_run
    assert not outcome.restored_selected_epoch
    assert "cannot snapshot" in outcome.stop_reason, outcome.stop_reason


def test_the_monitored_loss_covers_the_whole_monitor_set():
    """P9: it used to read only the first ``sft_batch_size`` samples, every epoch."""
    from mewm.training.sft import DryRunSFT, SFTTrainer

    seen: List[str] = []

    class Watching(DryRunSFT):
        def evaluate(self, batch):
            seen.extend(s["messages"][0]["content"] for s in batch)
            return 0.5

    samples = [{"messages": [{"role": "user", "content": f"q{i}"},
                             {"role": "assistant", "content": "a"}]}
               for i in range(10)]
    trainer = SFTTrainer(Watching(), TrainingConfig(sft_max_epochs=1, sft_batch_size=4))
    trainer.fit(samples)
    assert set(seen) == {f"q{i}" for i in range(10)}, sorted(set(seen))


def test_the_grpo_ratio_is_taken_against_the_sampling_policy():
    """P1: ``ref_logprob`` was never measured, so the ratio was ``exp(logprob - 0)``."""
    import inspect as _inspect

    from mewm.training.grpo import LoRABackend, PolicySample

    assert not PolicySample("t", {}).logprobs_measured, (
        "an unset log-probability must be distinguishable from a measured 0.0")

    sample_src = _inspect.getsource(LoRABackend.sample)
    assert "self.reference_model" in sample_src, (
        "the reference model must be consulted where the samples are drawn")
    assert "logprobs_measured=True" in sample_src

    update_src = _inspect.getsource(LoRABackend.update)
    assert "logprobs - old" in update_src, (
        "the importance ratio must be against the sampling policy, not the reference")
    assert "sample.token_logprobs" in update_src, (
        "the ratio's denominator must be the sampling policy's own token log-probs")
    assert "logprobs_measured" in update_src, "the update must refuse unmeasured samples"

    lp_src = _inspect.getsource(LoRABackend._completion_token_logprobs)
    assert "len(prompt_ids)" in lp_src, (
        "the prompt must be masked out of the log-probability")
    assert not hasattr(LoRABackend, "_sequence_logprob"), (
        "the whole-sequence log-probability was replaced and must stay replaced")


def test_the_kl_estimator_cannot_overflow_at_sequence_scale():
    """K1: k3 is exponential in its argument, so it must be fed *per-token* deltas.

    The previous implementation formed ``delta`` from whole-sequence log-probabilities,
    which run to order 10^3 for a normal completion. ``exp(700)`` is 1e304 and ``exp(750)``
    is ``inf``; one such step makes the loss non-finite and the run is over. This pins
    both halves of the fix: the delta comes from a per-token vector, and it is saturated.
    """
    import inspect as _inspect
    import math

    from mewm.training.grpo import KL_DELTA_CLAMP, LoRABackend, PolicySample

    assert 0 < KL_DELTA_CLAMP <= 40, (
        f"the clamp must sit where k3 is still finite; exp({KL_DELTA_CLAMP}) is not")
    assert math.isfinite(math.exp(KL_DELTA_CLAMP) - KL_DELTA_CLAMP - 1.0)

    update_src = _inspect.getsource(LoRABackend.update)
    assert "KL_DELTA_CLAMP" in update_src, "the k3 delta must be saturated"
    assert "ref - logprobs" in update_src, (
        "the KL delta must be per-token, not a difference of sequence totals")
    assert "kl_clamped_tokens" in update_src, (
        "a clamped KL term is a silent cap unless it is counted and reported")

    sample = PolicySample("t", {})
    assert sample.token_logprobs == [] and sample.ref_token_logprobs == [], (
        "per-token log-probabilities must default to empty, not to a fabricated zero")


def test_one_tp_criterion_serves_reward_gate_and_evaluation():
    """K3: eq. (2) had three implementations and they had already drifted apart."""
    import inspect as _inspect

    from mewm.eval.metrics import Match, tp_decision
    from mewm.eval.pass_criteria import check_temporal
    from mewm.training.rewards import reward_temporal

    for function in (check_temporal, reward_temporal):
        assert "tp_decision" in _inspect.getsource(function), (
            f"{function.__name__} must defer to the shared criterion")
    assert "tp_decision" in _inspect.getsource(Match.decision.fget)

    # The drift that mattered: reward_temporal ignored rescue_min_iou entirely, so
    # raising it forked training from evaluation.
    signature = _inspect.signature(reward_temporal)
    assert "rescue_min_iou" in signature.parameters, (
        "the reward must honour the same rescue floor the evaluation does")

    # A zero-overlap proposal with the right label: rescued at the 0.0 default, refused
    # once the floor is raised -- and all three callers must agree on both answers.
    proposal, truth = (0, 10), (50, 60)
    for floor, expected in ((0.0, True), (0.2, False)):
        verdict = tp_decision(0.0, "happiness", "happiness", rescue_min_iou=floor)
        gate_ok, _, _, _ = check_temporal(
            proposal, truth, "happiness", "happiness",
            EvaluationConfig(rescue_min_iou=floor))
        _, detail = reward_temporal(proposal, truth, "happiness", "happiness",
                                    rescue_min_iou=floor)
        assert verdict.is_tp is expected
        assert gate_ok is expected, f"the gate disagrees at rescue_min_iou={floor}"
        assert detail["tp"] is expected, f"the reward disagrees at rescue_min_iou={floor}"


def test_an_out_of_vocabulary_label_cannot_rescue_against_a_different_one():
    """K3, follow-on: canonicalisation maps every unknown string to ``other``."""
    from mewm.eval.metrics import tp_decision

    assert not tp_decision(0.1, "banana", "unicorn").is_tp, (
        "two different out-of-vocabulary labels both canonicalise to 'other'; "
        "comparing there would rescue a proposal on a label it got wrong")
    assert tp_decision(0.1, "banana", "Banana").is_tp, (
        "a dataset-specific label must still be able to match itself")
    assert tp_decision(0.1, "happiness", "HAPPINESS ").is_tp, (
        "canonicalisation must still absorb case and whitespace")


def test_dropped_sft_samples_are_counted_once_each():
    """K9: ``_encode`` runs per batch per epoch, so a counter counted encodings."""
    import inspect as _inspect

    from mewm.training.sft import LoRASFTBackend

    assert isinstance(LoRASFTBackend.n_dropped, property), (
        "n_dropped must be derived from the set of dropped samples, not incremented")
    source = _inspect.getsource(LoRASFTBackend._encode)
    assert "self._dropped.add" in source, "a drop must record which sample was dropped"
    assert "self.n_dropped += 1" not in source, "the call-counting bug must stay fixed"

    key = LoRASFTBackend._sample_key
    sample = {"messages": [{"role": "user", "content": "x"}]}
    assert key(sample) == key(dict(sample)), (
        "the same sample encoded twice must produce the same identity")
    assert key({"id": "a", **sample}) != key({"id": "b", **sample}), (
        "two declared ids must stay distinct")


def test_the_calibrated_thresholds_are_recorded_on_the_fold():
    """K7: 'calibrated on the training fold only' was an unauditable claim."""
    import inspect as _inspect

    from mewm.training.loso import FoldResult, FoldSpec, LOSORunner

    result = FoldResult(fold=FoldSpec(dataset="d", test_subject="s1",
                                      train_subjects=["s2"], train_videos=["v2"],
                                      test_videos=["v1"]))
    payload = result.to_dict()
    assert "calibration" in payload, "the fold must say which thresholds it ran under"
    assert payload["calibration"]["status"] == "config defaults, not fitted", (
        "an uncalibrated fold must say so rather than look calibrated")

    source = _inspect.getsource(LOSORunner._calibrate)
    assert "excluded_test_videos" in source, (
        "the audit needs the held-out list, not just the fitted thresholds")
    assert "_pool_videos" in source, "calibration must draw only on the training pool"


def test_lodo_folds_hold_out_a_whole_corpus():
    """K10: P2 asks for unseen-subject *and* unseen-domain numbers; only LOSO existed."""
    from mewm.data.datasets import DatasetIndex, LongVideo, lodo_folds

    def video(dataset: str, key: str) -> LongVideo:
        return LongVideo(dataset=dataset, video_key=key, subject="s1",
                         folder_rel=key, fps=30.0, flow_gap=1)

    a = DatasetIndex(dataset="alpha", videos=[video("alpha", "a1"), video("alpha", "a2")])
    b = DatasetIndex(dataset="beta", videos=[video("beta", "b1")])

    folds = lodo_folds([a, b])
    assert [name for name, _, _ in folds] == ["alpha", "beta"]
    for held_out, train, test in folds:
        assert all(v.dataset == held_out for v in test)
        assert all(v.dataset != held_out for v in train), (
            "the held-out corpus must not appear in its own training side")

    try:
        lodo_folds([a])
    except ValueError:
        pass
    else:
        raise AssertionError("one dataset cannot make a leave-one-dataset-out split")

    try:
        lodo_folds([a, DatasetIndex(dataset="alpha", videos=[])])
    except ValueError:
        pass
    else:
        raise AssertionError("two indices with one name would split a corpus against itself")


def test_the_step_record_separates_the_five_reward_components():
    """K4: ablation (c) needs per-component curves; only the scalar total was logged."""
    from mewm.training.grpo import GroupRollout, GRPOTrainer, PolicySample
    from mewm.training.rewards import RewardBreakdown

    def sample(total: float, temp: float) -> PolicySample:
        return PolicySample("t", {}, reward=RewardBreakdown(
            r_au=0.5, r_emo=0.4, r_fmt=1.0, r_causal=0.3, r_temp=temp, total=total,
            weights={"au": 0.2, "emo": 0.2, "fmt": 0.2, "causal": 0.2, "temp": 0.2}))

    rollout = GroupRollout("p1", "prompt", samples=[sample(0.8, 0.9), sample(0.2, 0.1)])
    stats = GRPOTrainer._reward_stats([rollout])

    assert set(stats["components"]) == set(GRPOTrainer.REWARD_COMPONENTS)
    assert stats["components"]["r_temp"]["mean"] == 0.5, (
        "the temporal component must be averaged in its own right")
    assert stats["components"]["r_fmt"]["std"] == 0.0, (
        "a component that never varies must show it, not be folded into the total")
    assert stats["component_weights"]["temp"] == 0.2, (
        "a component mean is uninterpretable without the weight it earned")
    assert stats["n_unscored"] == 0


def test_the_report_marks_what_it_could_not_compute():
    """K2/K5/K6: the metrics existed; nothing called them, and nothing said so."""
    from mewm.eval.report import build_report

    report = build_report("casme_sq", [], [])
    for section in ("layer1_rollout_quality", "layer2_p1_localisation",
                    "causal_reliability", "narrative", "p4_propagation"):
        body = report[section]
        assert body.get("status") == "unavailable", (
            f"{section} must not report a number it could not compute")
        assert body.get("reason"), f"{section} must say why it is unavailable"


def test_the_critic_persists_its_belief_flip_test():
    """K5: the flips were computed, shown to the model, and then dropped."""
    import inspect as _inspect

    from mewm.agents.critic import CriticAgent

    analysis_src = _inspect.getsource(CriticAgent.run_analysis)
    assert '"flips": flips' in analysis_src, "the flip test must be computed"
    for method in (CriticAgent.parse, CriticAgent.fallback):
        assert '"flips"' in _inspect.getsource(method), (
            f"{method.__name__} must carry the flips through; rho_flip has no other "
            f"source and cannot be recovered from a finished run without it")


def test_the_rollout_metrics_decay_with_the_horizon():
    """K6: layer 1 reported one of its three numbers because two were never written."""
    import numpy as np

    from mewm.engines.m1_dynamics import AnalyticDynamics
    from mewm.eval.rollout_metrics import (
        counterfactual_structure, rollout_prediction_error)
    from mewm.knowledge.au_anatomy import K_SLOTS

    rng = np.random.default_rng(0)
    sequences = [np.clip(np.cumsum(rng.normal(0, 0.05, (40, K_SLOTS)), axis=0), 0, 1)
                 for _ in range(3)]
    dynamics = AnalyticDynamics()

    curve = rollout_prediction_error(sequences, dynamics, k_steps=5)
    assert len(curve.mse) == 5 and len(curve.cosine) == 5
    assert curve.mse[-1] > curve.mse[0], (
        "prediction error must grow with the horizon; a flat curve means the rollout "
        "is not actually being extended")
    assert curve.decay > 0

    structure = counterfactual_structure([s[:5] for s in sequences], dynamics)
    assert len(structure.divergence) == len(structure.emotions) == 8
    assert all(structure.divergence[i][i] == 0.0 for i in range(8)), (
        "a hypothesis must be at zero distance from itself")
    assert structure.mean_divergence > 0, (
        "zero divergence would mean conditioning does nothing and every downstream "
        "counterfactual primitive is comparing a trajectory with itself")
    assert -1.0 <= structure.arousal_rank_correlation <= 1.0


def test_an_unscored_candidate_pool_is_reported():
    """P5: an injected sampler that forgets ``reward`` pins the gate at continue_sft."""
    import inspect as _inspect

    from mewm.training.loso import LOSORunner

    source = _inspect.getsource(LOSORunner.gate)
    assert "not any(c.reward for c in candidates)" in source
    assert "LOGGER.warning" in source


# ---------------------------------------------------------------------------


def test_fixed_answer_instructions_are_excluded_from_augmentation():
    """The reference QA set is mostly questions the annotation answers exactly."""
    from mewm.training.rl_prompts import (
        KIND_DETERMINISTIC, KIND_EVENT_REASONING, KIND_VIDEO_REASONING,
        classify_question, parse_event_anchor)

    fixed = [
        "How many expression events appear in this video?",
        "How many micro-expression events appear in this video? Localize every event.",
        "How many macro-expression events appear in this video?",
        "What distinct action units appear in this video?",
        "What is the expression type of the 2-th expression event in this video?",
    ]
    for question in fixed:
        kind, reason = classify_question(question)
        assert kind == KIND_DETERMINISTIC, (
            f"{question!r} is answered exactly by the annotation; sampling a paraphrase "
            f"can only match the reference or be wrong")
        assert reason, "an exclusion must carry the reason it was excluded"

    kind, _ = classify_question(
        "In the 3-th expression event of this video (frames 699-707, apex 703): "
        "Describe the person's face and infer the likely emotional state.")
    assert kind == KIND_EVENT_REASONING
    kind, _ = classify_question(
        "Reason over the whole video: how many micro-expression events does it contain?")
    assert kind == KIND_VIDEO_REASONING

    # An unrecognised template is treated as fixed-answer, not admitted by default: an
    # unknown question with an unknown truth would be scored against an empty reference.
    assert classify_question("What colour is the wall?")[0] == KIND_DETERMINISTIC

    anchor = parse_event_anchor(
        "In the 3-th expression event of this video (frames 699-707, apex 703): x")
    assert anchor == {"ordinal": 3, "onset": 699, "offset": 707, "apex": 703}


def test_an_admitted_prompt_never_carries_the_answer_it_is_scored_against():
    """The truth travels beside the prompt, not inside it."""
    from mewm.data.datasets import ExpressionEvent, LongVideo
    from mewm.training.rl_prompts import build_rl_prompts, unavailable_evidence

    event = ExpressionEvent(event_id="e1", onset=100, apex=105, offset=110,
                            expression_type="micro", fine_label="disgust",
                            coarse_label="negative", aus=["AU4", "AU7"],
                            subject="s01", video_key="v1")
    video = LongVideo(dataset="d", video_key="v1", subject="s01", folder_rel="v1",
                      fps=30.0, flow_gap=1, events=[event])
    rows = [{"video_id": "d_v1_1", "video": "v1",
             "question": "In the 1-th expression event of this video "
                         "(frames 100-110, apex 105): Describe the face.",
             "answer": "disgust, AU4 then AU7"}]

    prompts, ledger = build_rl_prompts(
        "d", [video], rows, {"v1": unavailable_evidence("v1", "not spotted")})
    assert len(prompts) == 1 and len(ledger) == 1 and ledger[0].admitted

    prompt = prompts[0]
    assert prompt["truth"]["fine"] == "disgust", "the scorer needs the label"
    assert prompt["truth"]["aus"] == ["AU4", "AU7"]

    visible = prompt["evidence_text"] + prompt["question"]
    for leaked in ("disgust", "negative", "AU4", "AU7"):
        assert leaked not in visible, (
            f"{leaked!r} is ground truth and reached the text the policy sees")
    assert "reference" not in prompt, "the reference answer must not travel in the prompt"
    assert "unavailable" in prompt["evidence_text"], (
        "a missing perceptual channel must be stated, not shown as an empty block")


def test_a_question_whose_event_is_not_annotated_is_refused():
    """Admitting it would score every component against an empty truth."""
    from mewm.data.datasets import ExpressionEvent, LongVideo
    from mewm.training.rl_prompts import build_rl_prompts

    event = ExpressionEvent(event_id="e1", onset=100, apex=105, offset=110,
                            fine_label="disgust", coarse_label="negative",
                            subject="s01", video_key="v1")
    video = LongVideo(dataset="d", video_key="v1", subject="s01", folder_rel="v1",
                      fps=30.0, flow_gap=1, events=[event])
    rows = [{"video_id": "x", "video": "v1",
             "question": "In the 9-th expression event of this video "
                         "(frames 900-910, apex 905): Describe the face."}]

    prompts, ledger = build_rl_prompts("d", [video], rows)
    assert prompts == []
    assert not ledger[0].admitted and "no annotated event" in ledger[0].reason


def test_the_whole_video_criterion_reuses_the_single_tp_decision():
    """A video-level answer has no single interval, but it must not get a new criterion."""
    import inspect as _inspect

    from mewm.eval.pass_criteria import evaluate_video_sample

    source = _inspect.getsource(evaluate_video_sample)
    assert "tp_decision" in source, (
        "the video-level verdict must be built from the same eq. (2) criterion the "
        "proposal-level one uses, or the two will drift exactly as K3 described")

    truth = {"events": [{"interval": [100, 112], "fine": "disgust",
                         "coarse": "negative"}]}
    good = {"n_micro": 1, "answer": "One brief tightening around the brow and lids.",
            "events": [{"interval": [100, 110], "fine_label": "disgust",
                        "coarse_label": "negative"}]}
    assert evaluate_video_sample(good, truth, required_fields=("answer",)).passed

    over = dict(good, n_micro=3)
    outcome = evaluate_video_sample(over, truth, required_fields=("answer",))
    assert not outcome.passed and any("claims 3" in r for r in outcome.reasons)

    wrong_place = {"n_micro": 1, "answer": "ok",
                   "events": [{"interval": [900, 910], "fine_label": "disgust",
                               "coarse_label": "negative"}]}
    # At the paper's default rescue_min_iou of 0.0 this PASSES, and that is the criterion
    # working as specified rather than a bug: eq. (2)'s second clause rescues a zero-IoU
    # proposal whose fine label matches. It is worth pinning because the clause is much
    # more permissive at video level than at proposal level -- a whole-video answer with
    # the right count and the right labels passes wherever it puts the events. The fix is
    # the rescue floor, not a second criterion; forking the criterion here is precisely
    # the drift K3 documented.
    assert evaluate_video_sample(wrong_place, truth, required_fields=("answer",)).passed
    floored = EvaluationConfig(rescue_min_iou=0.2)
    assert not evaluate_video_sample(wrong_place, truth, floored, ("answer",)).passed, (
        "raising the rescue floor must close the label-only pass at video level exactly "
        "as it does at proposal level")
    assert evaluate_video_sample(good, truth, floored, ("answer",)).passed, (
        "the floor must not cost a genuinely well-localised answer")


def test_claiming_no_micro_expression_is_a_passable_answer():
    """Most CAS(ME)^2 long videos contain none; that answer must be scoreable."""
    from mewm.training.api_sampler import VIDEO_REQUIRED_FIELDS
    from mewm.eval.pass_criteria import evaluate_video_sample

    assert "events" not in VIDEO_REQUIRED_FIELDS, (
        "check_format reads an empty list as a missing field, so requiring 'events' "
        "would fail every correct 'this video contains none' answer")

    product = {"n_micro": 0, "events": [],
               "answer": "This video contains 0 micro-expression events."}
    outcome = evaluate_video_sample(product, {"events": []},
                                    required_fields=VIDEO_REQUIRED_FIELDS)
    assert outcome.passed, outcome.reasons

    # ...and claiming one where there is none must still fail.
    claimed = {"n_micro": 1, "answer": "x",
               "events": [{"interval": [10, 20], "fine_label": "fear",
                           "coarse_label": "negative"}]}
    assert not evaluate_video_sample(claimed, {"events": []},
                                     required_fields=VIDEO_REQUIRED_FIELDS).passed


def test_an_unavailable_reward_component_is_named_not_scored_as_zero():
    """R_causal needs a subgraph this sweep never builds."""
    from mewm.training.api_sampler import (
        COMPUTABLE_WITHOUT_JUDGE, NEEDS_JUDGE, APIPolicySampler)

    sampler = APIPolicySampler(caller=lambda *a, **k: None)
    note = sampler.renormalisation()

    assert note["unavailable_components"] == list(NEEDS_JUDGE)
    assert set(note["available_components"]) == set(COMPUTABLE_WITHOUT_JUDGE)
    assert note["available_weight_mass"] < note["total_weight_mass"], (
        "if the two were equal the renormalisation would be a no-op and the note a lie")
    assert note["reason"] and note["acceptance_score"], (
        "a renormalisation applied without being stated is exactly the silent-default "
        "failure the report discipline exists to prevent")

    available, total = sampler.weight_mass()
    assert abs(total - available - sampler.reward.config.w_causal) < 1e-9, (
        "the withheld mass must be exactly the causal weight")


def test_a_generation_that_does_not_parse_is_still_a_candidate():
    """Dropping it would make the pass rate a rate over generations that parsed."""
    from types import SimpleNamespace

    from mewm.training.api_sampler import APIPolicySampler, extract_product

    assert extract_product("no json here at all") is None
    assert extract_product('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_product('Sure! {"a": 2} hope that helps') == {"a": 2}

    sampler = APIPolicySampler(
        caller=lambda *a, **k: SimpleNamespace(text="I cannot answer.", latency_s=0.1))
    candidates = sampler({"id": "p1", "question": "q", "video": "v1", "kind":
                          "event_reasoning", "evidence_text": "e", "truth": {}}, 2)

    assert len(candidates) == 2, "an unparsable draw must still occupy its slot"
    assert all(c.reward == 0.0 and not c.passed for c in candidates)
    assert sampler.stats.n_unparsable == 2
    assert all(c.outcome is not None for c in candidates), (
        "every candidate needs an outcome or the rejection ledger loses it")


def test_a_failed_api_call_is_recorded_rather_than_raised():
    """One bad call must not abandon a sweep that has already spent an hour perceiving."""
    from mewm.training.api_sampler import APIPolicySampler

    def explode(*_args, **_kwargs):
        raise RuntimeError("gateway timeout")

    sampler = APIPolicySampler(caller=explode)
    candidates = sampler({"id": "p1", "question": "q", "video": "v1",
                          "kind": "event_reasoning", "evidence_text": "e",
                          "truth": {}}, 2)

    assert len(candidates) == 2
    assert sampler.stats.n_failed_calls == 2
    assert sampler.stats.errors.get("RuntimeError") == 2, (
        "the error class must reach the manifest; a sweep that quietly lost half its "
        "draws to timeouts would report a low pass rate as a policy finding")
    assert all("gateway timeout" in str(c.reward_detail.get("error", ""))
               for c in candidates)


def test_the_sampling_budget_is_reported_not_applied_silently():
    """A cap on coverage that is not stated reads as full coverage."""
    import inspect as _inspect

    from mewm.training.qa_sweep import run_sweep

    source = _inspect.getsource(run_sweep)
    assert "n_left_unsampled_for_budget" in source, (
        "the number of admitted instructions left undrawn must reach the summary")
    assert "LOGGER.warning" in source


def test_the_per_fold_record_names_the_subjects_that_contributed_nothing():
    """A subject with no micro-expression is a fact about the corpus, not a gap."""
    import inspect as _inspect

    from mewm.training.qa_sweep import _write_fold_supplement, write_folds

    source = _inspect.getsource(_write_fold_supplement)
    for required in ("pool_subjects_contributing_nothing", "silent_subject_note",
                     "instruction_filtering", "scoring_caveat", "sampling_caveat",
                     "fold_isolation"):
        assert required in source, f"the per-subject record must carry {required}"

    assert "spec.pool_videos()" in _inspect.getsource(write_folds), (
        "the write path must re-check fold isolation against the fold's own pool")


def test_a_subject_never_augments_its_own_held_out_fold():
    """The one isolation guarantee that survives sampling once and projecting."""
    from mewm.training.qa_augment import AugmentationError, write_augmented, AugmentedPair

    pair = AugmentedPair(video_id="d_augs01_v1_1", video="v1", question="q", answer="a",
                         dataset="d", fold="s01", subject="s01")
    try:
        write_augmented([pair], "d", "s01", allowed_videos={"v2", "v3"})
    except AugmentationError as exc:
        assert "outside the training pool" in str(exc)
    else:
        raise AssertionError(
            "writing a pair from outside the fold's pool must raise, not drop the row: "
            "a silent drop leaves a smaller file and no sign the sampler was misconfigured")


def test_the_consolidated_file_lists_each_pair_once_with_its_source_qa(tmp_path, monkeypatch):
    """The corpus-level JSON references the original QA pair every augmented pair
    extends, and the strict-TP gate holds there too."""
    from mewm.training.qa_sweep import SweepResult, write_consolidated

    monkeypatch.setattr(qa_augment, "qa_dir", lambda dataset: tmp_path)

    index = _index(n_subjects=5)
    config = load_config()
    config.training.rft_accept_top_k = 5   # take the top-k cap out of the test

    video = index.videos[0]                # s10_v0, annotated event 100-180
    source_id = "casme_sq_train_s10_s10_v0_1"
    row = {
        "video_id": source_id,
        "video": video.video_key,
        "question": ("In the 1-th expression event of this video "
                     "(frames 100-180, apex 140), describe the face"),
        "answer": "the reference answer text",
    }

    def candidate(draw: int, reward: float, iou: float, rescued: bool = False) -> Candidate:
        return Candidate(
            prompt_id=source_id, text="t", dataset="casme_sq",
            video=video.video_key, event_index=0, interval=(110, 170),
            product={"answer": f"policy answer {draw}"}, reward=reward,
            reward_detail={"draw": draw},
            outcome=PassOutcome(passed=True, format_ok=True, temporal_ok=True,
                                label_ok=True, iou=iou, rescued=rescued))

    result = SweepResult(
        dataset="casme_sq", policy_model="gpt-5.4-mini",
        prompts=[{"id": source_id, "question": row["question"],
                  "video": video.video_key, "subject": "s10"}],
        candidates=[candidate(1, 0.8, 0.67),
                    candidate(2, 0.75, 0.62),
                    candidate(3, 0.9, 0.4, rescued=True)])

    report = write_consolidated(result, index, config, [row])

    payload = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
    assert report["n_pairs"] == 2, "the rescued label-only pass must not be augmented"
    assert payload["n_pairs_with_source_qa"] == 2
    assert payload["n_videos"] == 1
    for pair in payload["pairs"]:
        assert pair["video"] == video.video_key
        assert pair["question"] == row["question"]
        assert pair["source_qa"]["video_id"] == source_id, (
            "every consolidated pair must reference the original QA row it came from")
        assert pair["source_qa"]["answer"] == "the reference answer text"
        # Subject s10's video is in every fold's pool except fold s10 itself.
        assert sorted(pair["provenance"]["folds"]) == ["s11", "s12", "s13", "s14"]
    assert payload["videos"][video.video_key]["subject"] == "s10"
    assert payload["videos"][video.video_key]["n_pairs"] == 2
    assert payload["tp_gate"]["rescued_excluded"] is True


def test_a_sweep_can_resume_from_its_sampling_checkpoint(tmp_path):
    """Prompts whose draws are already on disk are re-used, not paid for twice."""
    from mewm.training.qa_sweep import sample_pool

    class FakeSampler:
        def __init__(self):
            self.calls: List[str] = []

        def __call__(self, prompt, k):
            self.calls.append(str(prompt["id"]))
            return [Candidate(
                prompt_id=str(prompt["id"]), text=f"t{i}", video=str(prompt["video"]),
                product={"answer": f"a{i}"}, reward=0.7, reward_detail={"draw": i},
                interval=(1, 2),
                outcome=PassOutcome(passed=True, format_ok=True, temporal_ok=True,
                                    label_ok=True, iou=0.8))
                for i in range(1, int(k) + 1)]

    prompts = [{"id": "p1", "question": "q", "video": "v1"},
               {"id": "p2", "question": "q", "video": "v2"}]
    path = tmp_path / "checkpoint.jsonl"

    first, n_restored = sample_pool(prompts, FakeSampler(), 2, workers=1,
                                    checkpoint_path=path)
    assert n_restored == 0 and len(first) == 4 and path.exists()

    second_sampler = FakeSampler()
    second, n_restored = sample_pool(prompts, second_sampler, 2, workers=1,
                                     checkpoint_path=path, resume=True)
    assert second_sampler.calls == [], "restored prompts must not be sampled again"
    assert n_restored == 2
    assert len(second) == 4
    assert second[0].product == {"answer": "a1"}, (
        "a restored candidate must carry its full product, not just its score")


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
