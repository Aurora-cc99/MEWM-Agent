"""The QA-augmentation sweep: one dataset, every subject, per-fold output.

1. **Perceive once per video.** Stages I-II (representation and prediction-error
   spotting) are deterministic and involve no LLM call, so they run once for the whole
   corpus and the resulting :class:`~mewm.training.rl_prompts.VideoEvidence` is shared by
   every prompt on that video and by every fold that video belongs to.

2. **Filter, then sample once per admitted instruction.** The reference QA set is split
   into the fixed-answer instructions (excluded, with the reason recorded) and the
   free-form reasoning instructions (admitted). Each admitted instruction is sampled ``k``
   times and every draw is scored.

3. **Project into folds.** A candidate drawn on subject *S*'s video is valid supervision
   for every LOSO fold except fold *S*. The write path re-checks that with the fold's own
   pool, so an off-pool pair raises rather than being written.

**Why sampling once is legitimate here, and where it is not.** In the real protocol,
stage 2 samples from the fold's *own* SFT-trained policy, so the samples differ per fold
and must be drawn per fold. This sweep samples from a frozen hosted model that has no
fold-specific state at all, which makes the draws fold-independent by construction --
sampling them 22 times would produce 22 statistically identical sets at 22 times the cost.
That is a property of running the augmentation with an external policy, not a shortcut
that survives into the trained-policy setting, and it is written into every manifest so a
reader cannot mistake this output for the trained-policy stage 2.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import MEWMConfig, load_config
from ..data.datasets import DatasetIndex, LongVideo
from ..eval.pass_criteria import PassOutcome
from . import qa_augment
from .api_sampler import APIPolicySampler, video_truth
from .candidate_filter import Candidate, FilterReport, select
from .loso import build_folds
from .rl_prompts import (
    KIND_VIDEO_REASONING, PromptLedgerEntry, VideoEvidence, build_rl_prompts,
    evidence_from_spotting, ledger_by_subject, unavailable_evidence,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: perception
# ---------------------------------------------------------------------------


def perceive(
    videos: Sequence[LongVideo],
    config: Optional[MEWMConfig] = None,
    max_frames: int = 0,
    stride: int = 1,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[str, VideoEvidence], Dict[str, Any]]:
    """Run stages I-II over every video and keep the evidence.

    A video whose frames cannot be read yields an *unavailable* evidence block rather
    than being dropped: its prompts still go out, labelled as having no perceptual
    channel, so the loss of coverage is visible in the output instead of showing up as a
    smaller corpus with no explanation.
    """
    from ..pipeline import run_representation, run_spotting

    config = config or load_config()
    evidence: Dict[str, VideoEvidence] = {}
    failures: Dict[str, str] = {}
    started = time.time()

    for position, video in enumerate(videos, start=1):
        if progress:
            progress(video.video_key, position, len(videos))
        try:
            representation = run_representation(video, config, max_frames=max_frames,
                                                stride=stride)
            if not len(representation):
                raise RuntimeError("no usable frames")
            spotting = run_spotting(video, representation, config)
            evidence[video.video_key] = evidence_from_spotting(
                video, representation, spotting)
        except Exception as exc:  # noqa: BLE001 - recorded per video, not fatal
            LOGGER.warning("%s: perception unavailable (%s)", video.video_key, exc)
            failures[video.video_key] = f"{type(exc).__name__}: {exc}"
            evidence[video.video_key] = unavailable_evidence(
                video.video_key, failures[video.video_key])

    summary = {
        "n_videos": len(videos),
        "n_with_evidence": sum(1 for e in evidence.values() if e.available),
        "n_unavailable": len(failures),
        "failures": failures,
        "elapsed_s": round(time.time() - started, 1),
    }
    LOGGER.info("perception: %d/%d videos measured in %.0fs",
                summary["n_with_evidence"], summary["n_videos"], summary["elapsed_s"])
    return evidence, summary


# ---------------------------------------------------------------------------
# Stage 2: sample and score
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Everything one dataset sweep produced."""

    dataset: str
    policy_model: str
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    ledger: List[PromptLedgerEntry] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    perception: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    renormalisation: Dict[str, Any] = field(default_factory=dict)
    folds: Dict[str, Any] = field(default_factory=dict)
    subjects: Dict[str, Any] = field(default_factory=dict)
    consolidated: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 2: sample and score
# ---------------------------------------------------------------------------


def _candidate_row(candidate: Candidate) -> Dict[str, Any]:
    """A candidate serialised well enough to rebuild it after a process died."""
    row = candidate.to_dict()
    row["text"] = candidate.text
    row["product"] = candidate.product
    return row


def _candidate_from_row(row: Dict[str, Any]) -> Candidate:
    return Candidate(
        prompt_id=str(row.get("prompt_id", "")),
        text=str(row.get("text", "")),
        product=dict(row.get("product") or {}),
        reward=float(row.get("reward", 0.0)),
        reward_detail=dict(row.get("reward_detail") or {}),
        dataset=str(row.get("dataset", "")),
        video=str(row.get("video", "")),
        event_index=int(row.get("event_index", 0)),
        interval=tuple(row.get("interval") or (0, 0)),
        outcome=PassOutcome(**row["outcome"]) if row.get("outcome") else None,
    )


def _read_checkpoint(path: Path) -> Dict[str, List[Candidate]]:
    """Prompt id -> its already-drawn candidates, tolerating a torn last line."""
    restored: Dict[str, List[Candidate]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(),
                                       start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            candidates = [_candidate_from_row(row)
                          for row in payload.get("candidates", [])]
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            LOGGER.warning("sampling checkpoint %s:%d unreadable (%s); that prompt "
                           "will be re-sampled", path, line_number, exc)
            continue
        if candidates:
            restored[str(payload.get("prompt_id", ""))] = candidates
    return restored


def _append_checkpoint(path: Path, prompt_id: str, candidates: Sequence[Candidate]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {"prompt_id": prompt_id,
             "candidates": [_candidate_row(c) for c in candidates]},
            ensure_ascii=False) + "\n")
        handle.flush()


def sample_pool(
    prompts: Sequence[Dict[str, Any]],
    sampler: APIPolicySampler,
    k: int,
    workers: int = 1,
    progress: Optional[Callable[[str, int, int], None]] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
) -> Tuple[List[Candidate], int]:
    """``k`` scored draws for every admitted prompt.
    """
    restored: Dict[str, List[Candidate]] = {}
    if checkpoint_path is not None:
        if resume and checkpoint_path.is_file():
            restored = _read_checkpoint(checkpoint_path)
            LOGGER.info("sampling checkpoint: %d prompt(s) restored from %s",
                        len(restored), checkpoint_path)
        else:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("", encoding="utf-8")

    n_restored = 0
    done = 0
    lock = threading.Lock()
    results: List[Optional[List[Candidate]]] = [None for _ in prompts]

    def finish(prompt_id: str, position: int, group: List[Candidate]) -> None:
        nonlocal done
        if group is not None:
            results[position] = group
        with lock:
            done += 1
            if progress:
                progress(prompt_id, done, len(prompts))

    def draw_one(prompt: Dict[str, Any], position: int) -> None:
        nonlocal n_restored
        prompt_id = str(prompt.get("id", ""))
        prior = restored.pop(prompt_id, None) if restored else None
        if prior is not None and len(prior) >= k:
            with lock:
                n_restored += 1
            finish(prompt_id, position, prior)
            return
        group = sampler(prompt, k)
        if checkpoint_path is not None:
            with lock:
                _append_checkpoint(checkpoint_path, prompt_id, group)
        finish(prompt_id, position, group)

    if workers <= 1:
        for position, prompt in enumerate(prompts):
            draw_one(prompt, position)
        return [c for group in results if group for c in group], n_restored

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: draw_one(pair[1], pair[0]), enumerate(prompts)))
    return [c for group in results if group for c in group], n_restored


# ---------------------------------------------------------------------------
# Stage 3: project into folds and write
# ---------------------------------------------------------------------------


def _subject_of(prompt_index: Dict[str, Dict[str, Any]], candidate: Candidate) -> str:
    entry = prompt_index.get(candidate.prompt_id, {})
    return str(entry.get("subject", ""))


def write_folds(
    result: SweepResult,
    index: DatasetIndex,
    config: MEWMConfig,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Select per fold and write each fold's augmented QA file and manifest."""
    prompt_index = {str(p["id"]): p for p in result.prompts}
    question_by_prompt = {str(p["id"]): str(p["question"]) for p in result.prompts}
    subject_by_video = {v.video_key: v.subject for v in index.videos}

    folds: Dict[str, Any] = {}
    for spec in build_folds(index, config.training):
        pool = spec.pool_videos()
        eligible = [c for c in result.candidates if c.video in pool]
        accepted, report = select(eligible, config.training)

        written: Dict[str, Any] = {}
        if accepted and not dry_run:
            pairs = qa_augment.build_pairs(
                accepted, result.dataset, spec.name, question_by_prompt,
                subject_by_video, result.policy_model)
            written = qa_augment.write_augmented(
                pairs, result.dataset, spec.name, pool)
            _write_fold_supplement(result, spec, report, accepted, prompt_index)

        folds[spec.name] = {
            "held_out_subject": spec.test_subject,
            "n_pool_videos": len(pool),
            "n_candidates_in_pool": len(eligible),
            "filter": report.to_dict(),
            "written": written,
        }
        LOGGER.info("fold %s: %d/%d candidate(s) accepted from %d pool video(s)",
                    spec.name, report.n_accepted, len(eligible), len(pool))
    return folds


def _write_fold_supplement(
    result: SweepResult,
    spec: Any,
    report: FilterReport,
    accepted: Sequence[Candidate],
    prompt_index: Dict[str, Dict[str, Any]],
) -> Path:
    """The per-subject filtering-and-augmentation record, beside the fold's jsonl.

    ``qa_augment.write_augmented`` writes the pairs and their provenance. This adds what
    it cannot know: which reference instructions were considered and rejected, which
    subjects the surviving supervision came from, and under what scoring caveat.
    """
    target = qa_augment.augmented_dir(result.dataset, spec.name)
    target.mkdir(parents=True, exist_ok=True)

    contributing: Dict[str, Dict[str, Any]] = {}
    for candidate in accepted:
        subject = _subject_of(prompt_index, candidate)
        bucket = contributing.setdefault(subject or "(unknown)", {
            "n_pairs": 0, "videos": set(), "rewards": []})
        bucket["n_pairs"] += 1
        bucket["videos"].add(candidate.video)
        bucket["rewards"].append(round(float(candidate.reward), 4))
    for bucket in contributing.values():
        bucket["videos"] = sorted(bucket["videos"])
        bucket["n_videos"] = len(bucket["videos"])
        bucket["mean_reward"] = (round(sum(bucket["rewards"]) / len(bucket["rewards"]), 4)
                                 if bucket["rewards"] else 0.0)
        bucket.pop("rewards")

    pool_subjects = sorted({_subject_of(prompt_index, c) for c in result.candidates
                            if c.video in spec.pool_videos()} - {""})
    silent = [s for s in pool_subjects if s not in contributing]

    payload = {
        "dataset": result.dataset,
        "fold": spec.name,
        "held_out_subject": spec.test_subject,
        "policy_model": result.policy_model,
        "instruction_filtering": {
            "note": (
                "The reference QA set mixes fixed-answer instructions with free-form "
                "reasoning ones. A fixed-answer instruction ('how many events', "
                "'localize every event', 'what is the type of the n-th event') is "
                "answered exactly by the annotation: a sampled paraphrase can only equal "
                "the reference or be wrong, so augmenting it adds no supervision and "
                "risks a poisoned row. Those are excluded by name. What is sampled is the "
                "free-form subset, where a different-but-correct answer exists."
            ),
            "by_subject": result.subjects.get("filtering", {}),
        },
        "augmentation": {
            "n_accepted_pairs": report.n_accepted,
            "n_candidates_considered": report.n_candidates,
            "acceptance_rule": {
                "min_reward": report.min_reward,
                "top_k_per_prompt": report.top_k,
                "criterion": (
                    "the shared eq. (2) true-positive criterion, via "
                    "mewm.eval.metrics.tp_decision -- the same function the training "
                    "reward and the final evaluation call"
                ),
            },
            "rejected_reasons": dict(report.rejected_reasons),
            "contributing_subjects": contributing,
            "pool_subjects_contributing_nothing": silent,
            "silent_subject_note": (
                "These subjects are in this fold's training pool but contributed no "
                "accepted pair. In CAS(ME)^2 most long videos contain no micro-expression "
                "at all, so a subject with no annotated micro event has nothing for the "
                "localisation criterion to accept. The subject is still training material "
                "for the rest of the pipeline; it simply adds no augmented QA row here."
            ),
        },
        "scoring_caveat": result.renormalisation,
        "sampling_caveat": (
            "Candidates were drawn once per instruction from a frozen hosted policy and "
            "projected into every fold whose training pool contains the source video. "
            "The policy has no fold-specific state, so per-fold redrawing would produce "
            "statistically identical sets. This is NOT the trained-policy stage 2, where "
            "the policy differs per fold and per-fold sampling is mandatory."
        ),
        "fold_isolation": (
            f"Every pair in this directory comes from a video in fold {spec.name}'s "
            f"training pool. No video of the held-out subject {spec.test_subject} "
            f"appears. This is enforced on write and re-checked on load."
        ),
    }

    path = target / f"{result.dataset}_qa_selection_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The corpus-level consolidated file
# ---------------------------------------------------------------------------


def write_consolidated(
    result: SweepResult,
    index: DatasetIndex,
    config: MEWMConfig,
    qa_rows: Sequence[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """One dataset-level JSON: every accepted pair once, beside its source QA pair.
    """
    rows = list(qa_rows)
    row_by_id = {str(r.get("video_id", "")): r for r in rows if r.get("video_id")}
    question_by_prompt = {str(p["id"]): str(p["question"]) for p in result.prompts}
    subject_by_video = {v.video_key: v.subject for v in index.videos}

    accepted, report = select(result.candidates, config.training)

    # Fold membership, keyed on exactly the tuple build_pairs carries into the pair.
    membership: Dict[Tuple[str, str, int, Tuple[int, int]], Set[str]] = {}
    for spec in build_folds(index, config.training):
        pool = spec.pool_videos()
        fold_accepted, _ = select(
            [c for c in result.candidates if c.video in pool], config.training)
        for candidate in fold_accepted:
            key = (candidate.video, question_by_prompt.get(candidate.prompt_id, ""),
                   int(getattr(candidate, "event_index", 0)),
                   tuple(getattr(candidate, "interval", ()) or ()))
            membership.setdefault(key, set()).add(spec.name)

    pairs = qa_augment.build_pairs(
        accepted, result.dataset, "all", question_by_prompt, subject_by_video,
        result.policy_model, tp_iou_threshold=config.evaluation.iou_threshold)

    entries: List[Dict[str, Any]] = []
    videos: Dict[str, Dict[str, Any]] = {}
    n_with_source = 0
    for pair in pairs:
        key = (pair.video, pair.question, pair.event_index, tuple(pair.interval))
        source_row = row_by_id.get(pair.source_video_id)
        if source_row is not None:
            n_with_source += 1
            source_qa = {
                "video_id": str(source_row.get("video_id", pair.source_video_id)),
                "video": str(source_row.get("video", pair.video)),
                "question": str(source_row.get("question", pair.question)),
                "answer": source_row.get("answer"),
            }
        else:
            source_qa = None
        entries.append({
            "video_id": pair.video_id,
            "video": pair.video,
            "question": pair.question,
            "answer": pair.answer,
            "source_qa": source_qa,
            "provenance": {
                "subject": pair.subject,
                "event_index": pair.event_index,
                "interval": list(pair.interval),
                "reward": pair.reward,
                "policy_model": pair.policy_model,
                "accepted_because": list(pair.accepted_because),
                "folds": sorted(membership.get(key, [])),
            },
        })
        bucket = videos.setdefault(pair.video, {
            "subject": pair.subject, "n_pairs": 0, "source_video_ids": []})
        bucket["n_pairs"] += 1
        if pair.source_video_id and pair.source_video_id not in bucket["source_video_ids"]:
            bucket["source_video_ids"].append(pair.source_video_id)

    payload = {
        "dataset": result.dataset,
        "policy_model": result.policy_model,
        "note": (
            "Corpus-level view of the TP-gated augmentation. Every accepted pair appears "
            "exactly once, beside the reference QA pair it was sampled from "
            "(`source_qa`). The per-fold directories hold the byte-compatible 4-field "
            "jsonl files that training loads; this file is the whole-dataset record."
        ),
        "tp_gate": {
            "strict_true_positive": True,
            "iou_threshold": config.evaluation.iou_threshold,
            "rescued_excluded": True,
            "criterion": (
                "the shared eq. (2) true-positive criterion, via "
                "mewm.eval.metrics.tp_decision -- the same function the training reward "
                "and the final evaluation call"
            ),
        },
        "sampling_caveat": (
            "Candidates were drawn once per instruction from a frozen hosted policy and "
            "projected into every fold whose training pool contains the source video. "
            "The policy has no fold-specific state, so per-fold redrawing would produce "
            "statistically identical sets. This is NOT the trained-policy stage 2, where "
            "the policy differs per fold and per-fold sampling is mandatory."
        ),
        "scoring_caveat": result.renormalisation,
        "n_pairs": len(entries),
        "n_videos": len(videos),
        "n_pairs_with_source_qa": n_with_source,
        "n_source_instructions": report.n_prompts_with_accept,
        "pairs": entries,
        "videos": videos,
    }

    path: Optional[Path] = None
    if not dry_run:
        path = qa_augment.consolidated_path(result.dataset, result.policy_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        LOGGER.info("%s: wrote %d consolidated augmented pair(s) to %s",
                    result.dataset, len(entries), path)

    return {"path": str(path) if path else None, "n_pairs": len(entries),
            "n_videos": len(videos), "n_pairs_with_source_qa": n_with_source,
            "n_source_instructions": report.n_prompts_with_accept}


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run_sweep(
    dataset: str,
    index: DatasetIndex,
    qa_rows: Iterable[Dict[str, Any]],
    config: Optional[MEWMConfig] = None,
    model: str = "claude-sonnet-5",
    k: int = 3,
    max_prompts: int = 0,
    max_workers: int = 4,
    stride: int = 1,
    max_frames: int = 0,
    dry_run: bool = False,
    resume: bool = False,
    progress: Optional[Callable[[str, str, int, int], None]] = None,
) -> SweepResult:
    """Perceive, filter, sample, select and write, for one dataset.
    """
    config = config or load_config()
    videos = list(index.videos)

    evidence, perception = perceive(
        videos, config, max_frames=max_frames, stride=stride,
        progress=(lambda key, i, n: progress("perceive", key, i, n)) if progress else None)

    rows = list(qa_rows)
    prompts, ledger = build_rl_prompts(dataset, videos, rows, evidence)

    # Attach the whole-video truth the video-level criterion needs. It is not part of the
    # prompt the policy sees -- it travels beside it, for the scorer.
    by_key = {v.video_key: v for v in videos}
    for prompt in prompts:
        if prompt["kind"] == KIND_VIDEO_REASONING:
            video = by_key.get(prompt["video"])
            if video is not None:
                prompt["truth"] = video_truth(video)

    dropped_for_budget = 0
    if max_prompts and len(prompts) > max_prompts:
        dropped_for_budget = len(prompts) - max_prompts
        LOGGER.warning(
            "sampling budget: keeping %d of %d admitted instruction(s); %d are being "
            "left unsampled and are reported as such, not as unaugmentable",
            max_prompts, len(prompts), dropped_for_budget)
        prompts = prompts[:max_prompts]

    sampler = APIPolicySampler(model=model, evaluation=config.evaluation,
                              reward_config=config.reward, max_workers=1)
    checkpoint_path = qa_augment.sampling_checkpoint_path(dataset, model) if resume \
        else None
    candidates, n_restored = sample_pool(
        prompts, sampler, k, workers=max_workers,
        progress=(lambda pid, i, n: progress("sample", pid, i, n)) if progress else None,
        checkpoint_path=checkpoint_path, resume=resume)

    result = SweepResult(
        dataset=dataset, policy_model=model, prompts=prompts, ledger=ledger,
        candidates=candidates, perception=perception,
        sampling={**sampler.stats.to_dict(), "k_per_prompt": k,
                  "n_admitted_instructions": len(prompts),
                  "n_left_unsampled_for_budget": dropped_for_budget,
                  "n_restored_from_checkpoint": n_restored},
        renormalisation=sampler.renormalisation(),
    )
    result.subjects = {"filtering": ledger_by_subject(ledger)}
    result.folds = write_folds(result, index, config, dry_run=dry_run)
    result.subjects["augmentation"] = _augmentation_by_subject(result)
    result.consolidated = write_consolidated(
        result, index, config, rows, dry_run=dry_run)
    if checkpoint_path is not None and not dry_run:
        checkpoint_path.unlink(missing_ok=True)
    return result


def _augmentation_by_subject(result: SweepResult) -> Dict[str, Any]:
    """Per-subject augmentation outcome, aggregated across folds."""
    prompt_index = {str(p["id"]): p for p in result.prompts}
    out: Dict[str, Any] = {}

    for candidate in result.candidates:
        subject = _subject_of(prompt_index, candidate) or "(unknown)"
        bucket = out.setdefault(subject, {
            "n_candidates_drawn": 0, "n_passed_criterion": 0, "n_parse_failures": 0,
            "reward_sum": 0.0, "videos": set(), "rejected_reasons": {},
        })
        bucket["n_candidates_drawn"] += 1
        bucket["videos"].add(candidate.video)
        bucket["reward_sum"] += float(candidate.reward)
        if not candidate.product:
            bucket["n_parse_failures"] += 1
        if candidate.passed:
            bucket["n_passed_criterion"] += 1
        elif candidate.outcome and candidate.outcome.reasons:
            reason = candidate.outcome.reasons[0]
            bucket["rejected_reasons"][reason] = (
                bucket["rejected_reasons"].get(reason, 0) + 1)

    for subject, bucket in out.items():
        drawn = bucket["n_candidates_drawn"]
        bucket["videos"] = sorted(bucket["videos"])
        bucket["n_videos"] = len(bucket["videos"])
        bucket["mean_reward"] = round(bucket.pop("reward_sum") / drawn, 4) if drawn else 0.0
        bucket["pass_rate"] = round(bucket["n_passed_criterion"] / drawn, 4) if drawn else 0.0
        # Truncate to the reasons that actually mattered; the full ledger is per fold.
        bucket["rejected_reasons"] = dict(sorted(
            bucket["rejected_reasons"].items(), key=lambda kv: -kv[1])[:6])
        bucket["appears_in_folds"] = sorted(
            name for name, body in result.folds.items()
            if body["held_out_subject"] != subject)
    return out


def sweep_summary(result: SweepResult) -> Dict[str, Any]:
    """The top-level record written beside the per-fold directories."""
    accepted_total = sum(f["filter"]["n_accepted"] for f in result.folds.values())
    return {
        "dataset": result.dataset,
        "policy_model": result.policy_model,
        "perception": result.perception,
        "instruction_filtering": {
            "n_reference_instructions": len(result.ledger),
            "n_admitted": sum(1 for e in result.ledger if e.admitted),
            "n_excluded": sum(1 for e in result.ledger if not e.admitted),
            "by_subject": result.subjects.get("filtering", {}),
        },
        "sampling": result.sampling,
        "scoring_caveat": result.renormalisation,
        "augmentation_by_subject": result.subjects.get("augmentation", {}),
        "folds": result.folds,
        "consolidated": result.consolidated,
        "totals": {
            "n_candidates": len(result.candidates),
            "n_accepted_pairs_across_folds": accepted_total,
            "n_folds": len(result.folds),
        },
    }


__all__ = ["SweepResult", "perceive", "sample_pool", "write_folds",
           "write_consolidated", "run_sweep", "sweep_summary"]
