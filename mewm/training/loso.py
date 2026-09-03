"""The LOSO training driver: the stage-0 -> 1 -> 2 -> 3 -> 4 chain, one run per dataset.

**Stage 0 is in the chain, not beside it.** Every later stage consumes the representation
the CLIP engine produces, so the ordering is not a convention -- it is a correctness
condition. Run as a separate command it is one forgotten invocation away from training a
policy against a stock CLIP while the report says "fine-tuned", which is silent, produces
plausible numbers and leaves no trace in the artefacts. So the checkpoint is fitted here,
saved under ``<clip.checkpoint_root>/<dataset>/fold_<subject>/``, re-asserted against the
held-out subject, and its path recorded on the fold result.

**Where augmentation comes from.** Two sources, both filtered by the same pool:
``qa_augment.load_augmented`` for the pairs stage 2 earned with *this fold's* policy, and
``qa_augment.load_consolidated`` for the corpus-level sweep ``mewm augment-qa`` drew once
from a frozen, fold-agnostic policy. Drawing the sweep once and projecting it per fold is
statistically identical to redrawing it per fold and costs N times less, which is what
lets augmentation live inside this driver rather than as a pipeline beside it.

**What the fold guarantees.** The held-out subject contributes nothing to any *training*
stage: not its videos, not its annotations, not its QA rows, not augmented QA written by
another fold, and not the CLIP weights. The augmented case is the easy one to get wrong,
because pairs live on disk and outlive the run that produced them -- so the pool's video
keys are passed down on both the write and the read path. It appears exactly once, in
stage 4, under inference only: test-time augmentation takes k draws and no gradient step,
so the subject still reaches no loss term.

**What it does not guarantee.** With the configured default ``n_val_subjects = 0`` the gate
is measured on the training pool itself. A plateau then means the pool has been memorised,
not that the policy has converged, and pass@1 is optimistic. Every report carries
``in_sample=True`` and ``measured_on="training_pool"`` so this cannot be read as a held-out
number. Set ``training.n_val_subjects >= 1`` to carve a validation split out of the pool
instead; the code path is the same and only the fold construction changes.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, TrainingConfig, load_config
from ..data.datasets import DatasetIndex, LongVideo, load_dataset
from ..data.qa_loader import QASet, load_qa_set
from ..eval.pass_criteria import PassOutcome
from . import qa_augment
from .candidate_filter import Candidate, FilterReport, group_rewards, score_candidates, select
from .diagnostics import SufficiencyReport, assess
from .grpo import GRPOConfig, GRPOTrainer, PolicyBackend
from .instruction_set import InstructionSetBuilder
from .sft import SFTBackend, SFTOutcome, SFTTrainer, build_sft_samples

LOGGER = logging.getLogger(__name__)

#: Hosted-API model ids that cannot be a policy. They expose no gradients, so they can
#: serve as frozen critics or inference baselines but not as a target of SFT or RL.
CLOSED_SOURCE_HINTS = (
    "gpt-", "o1", "o3", "claude-", "gemini-", "grok-", "deepseek-chat",
    "deepseekv", "qwen-max", "qwen-plus", "doubao", "glm-4-plus",
)


class PolicyNotTrainable(ValueError):
    """Raised when a closed-source id is passed where a trainable policy is required."""


class ClipEngineUnavailable(RuntimeError):
    """Raised when stage 0 cannot produce a fine-tuned CLIP engine for a fold.

    Fatal rather than a warning under ``training.clip_required``: falling back to the
    analytic front end would still finish the fold and still print metrics, but they
    would be metrics for an engine the report does not name.
    """


def require_open_weight(policy_model: str) -> str:
    """Reject a hosted-API id as a training target, with the reason.
    """
    lowered = policy_model.strip().lower()
    if not lowered:
        raise PolicyNotTrainable("no policy model given")

    try:
        from ..llm.registry import resolve
        spec = resolve(policy_model)
    except Exception:  # unregistered id, or no registry available
        spec = None
    if spec is not None:
        if not spec.open_weights:
            raise PolicyNotTrainable(
                f"{policy_model!r} is registered as a hosted model "
                f"(provider {spec.provider!r}, open_weights=False). SFT and GRPO both "
                f"need gradients, which the HTTP endpoints do not expose, so it cannot "
                f"be the policy. Use an open-weight checkpoint (the default is "
                f"{TrainingConfig().policy_model}) and keep the hosted model as a frozen "
                f"critic or an inference baseline."
            )
        # Registered and open-weight: trust the flag over the spelling. A local
        # checkpoint may legitimately have "gpt" in its name.
        return policy_model

    for hint in CLOSED_SOURCE_HINTS:
        if hint in lowered:
            raise PolicyNotTrainable(
                f"{policy_model!r} looks like a hosted-API model. SFT and GRPO both need "
                f"gradients, which the HTTP endpoints do not expose, so it cannot be the "
                f"policy. Use an open-weight checkpoint (the default is "
                f"{TrainingConfig().policy_model}) and keep the hosted model as a frozen "
                f"critic or an inference baseline."
            )
    return policy_model


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


@dataclass
class FoldSpec:
    """One LOSO fold: which subjects and videos belong to which side."""

    dataset: str
    #: The held-out subject; also the fold's name on disk.
    test_subject: str
    train_subjects: List[str] = field(default_factory=list)
    val_subjects: List[str] = field(default_factory=list)
    train_videos: List[str] = field(default_factory=list)
    val_videos: List[str] = field(default_factory=list)
    test_videos: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.test_subject

    @property
    def in_sample(self) -> bool:
        """True when there is no validation split, so the gate reads the training pool."""
        return not self.val_subjects

    def pool_videos(self) -> set:
        """Every video a training stage of this fold may touch."""
        return set(self.train_videos) | set(self.val_videos)

    def check_disjoint(self) -> None:
        """Assert the split really is subject- and video-disjoint."""
        subjects = [set(self.train_subjects), set(self.val_subjects), {self.test_subject}]
        for i, left in enumerate(subjects):
            for right in subjects[i + 1:]:
                overlap = left & right
                if overlap:
                    raise ValueError(
                        f"fold {self.name}: subjects {sorted(overlap)} appear on both "
                        f"sides of the split")
        video_overlap = self.pool_videos() & set(self.test_videos)
        if video_overlap:
            raise ValueError(
                f"fold {self.name}: videos {sorted(video_overlap)[:5]} are in both the "
                f"training pool and the test set")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset, "test_subject": self.test_subject,
            "n_train_subjects": len(self.train_subjects),
            "n_val_subjects": len(self.val_subjects),
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "n_train_videos": len(self.train_videos),
            "n_val_videos": len(self.val_videos),
            "n_test_videos": len(self.test_videos),
            "in_sample_gate": self.in_sample,
        }


def _fold_seed(fold_seed: int, dataset: str, test_subject: str) -> int:
    """A process-stable seed for one fold's validation draw.
    """
    digest = hashlib.blake2b(
        f"{fold_seed}:{dataset}:{test_subject}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build_folds(
    index: DatasetIndex,
    config: Optional[TrainingConfig] = None,
    subjects: Optional[Sequence[str]] = None,
) -> List[FoldSpec]:
    """Construct every fold for one dataset.
    """
    config = config or TrainingConfig()
    all_subjects = index.subjects()
    if subjects is not None:
        wanted = {str(s) for s in subjects}
        unknown = wanted - set(all_subjects)
        if unknown:
            raise KeyError(
                f"{index.dataset}: unknown subject(s) {sorted(unknown)}; "
                f"known are {all_subjects}")
        fold_subjects = [s for s in all_subjects if s in wanted]
    else:
        fold_subjects = list(all_subjects)

    folds: List[FoldSpec] = []
    for test_subject in fold_subjects:
        pool = [s for s in all_subjects if s != test_subject]
        n_val = max(0, min(config.n_val_subjects, len(pool) - 1))
        if n_val != config.n_val_subjects:
            LOGGER.warning(
                "fold %s: asked for %d validation subject(s) but the pool holds %d; "
                "using %d so at least one subject still trains",
                test_subject, config.n_val_subjects, len(pool), n_val)

        if n_val:
            # Seeded per fold, so fold s16's validation choice does not depend on
            # whether fold s15 was run first -- and seeded from a *stable* digest, not
            # from ``hash()``. Python salts string hashing per process, so the built-in
            # would pick a different validation subject on every run and quietly break
            # the reproducibility this function promises one paragraph above.
            rng = np.random.default_rng(_fold_seed(
                config.fold_seed, index.dataset, test_subject))
            picked = rng.choice(len(pool), size=n_val, replace=False)
            val_subjects = sorted(pool[int(i)] for i in picked)
        else:
            val_subjects = []
        train_subjects = [s for s in pool if s not in set(val_subjects)]

        def keys(subject_list: Sequence[str]) -> List[str]:
            out: List[str] = []
            for subject in subject_list:
                out.extend(v.video_key for v in index.by_subject(subject))
            return sorted(out)

        fold = FoldSpec(
            dataset=index.dataset, test_subject=test_subject,
            train_subjects=train_subjects, val_subjects=val_subjects,
            train_videos=keys(train_subjects), val_videos=keys(val_subjects),
            test_videos=keys([test_subject]),
        )
        fold.check_disjoint()
        folds.append(fold)
    return folds


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Everything one fold produced, in the order it was produced."""

    fold: FoldSpec
    #: Stage 0: the fine-tuned CLIP engine every later stage reads through.
    clip: Dict[str, Any] = field(default_factory=dict)
    sft_rounds: List[Dict[str, Any]] = field(default_factory=list)
    gate_reports: List[Dict[str, Any]] = field(default_factory=list)
    rft: Dict[str, Any] = field(default_factory=dict)
    augmented: Dict[str, Any] = field(default_factory=dict)
    rl_history: List[Dict[str, Any]] = field(default_factory=list)
    test: Dict[str, Any] = field(default_factory=dict)
    #: ``rl_ready`` / ``continue_sft`` / ``saturated`` -- the last gate decision.
    decision: str = ""
    #: Set when RL was skipped, with the reason.
    rl_skipped: str = ""
    #: The thresholds this fold ran under, which pool they were fitted on, and which
    #: videos were withheld. Empty when the runner was not asked to calibrate, in which
    #: case the fold inherits the config defaults -- and says so.
    calibration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold.to_dict(),
            "clip": self.clip or {"status": "not run"},
            "calibration": self.calibration or {"status": "config defaults, not fitted"},
            "sft_rounds": self.sft_rounds,
            "gate_reports": self.gate_reports,
            "rft": self.rft,
            "augmented": self.augmented,
            "rl_steps": len(self.rl_history),
            "rl_final": self.rl_history[-1] if self.rl_history else {},
            "rl_skipped": self.rl_skipped,
            "test": self.test,
            "decision": self.decision,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


#: Draws candidates for one prompt. Returns raw candidates; the runner scores them.
SamplerFn = Callable[[Dict[str, Any], int], List[Candidate]]
#: Evaluates the current policy on a set of videos. Returns whatever the caller reports.
EvaluatorFn = Callable[[FoldSpec, Sequence[str]], Dict[str, Any]]


class LOSORunner:
    """Chains the stages for one dataset's folds.

    The runner owns the *order* and the *isolation*; it owns neither the parameters nor
    the sampling. Both come in as callables so a fixture can drive the whole chain with no
    GPU, which is the only way the fold arithmetic and the gate logic get tested at all.
    """

    def __init__(
        self,
        dataset: str,
        sft_backend: SFTBackend,
        policy_backend: Optional[PolicyBackend] = None,
        sampler: Optional[SamplerFn] = None,
        evaluator: Optional[EvaluatorFn] = None,
        mewm_config: Optional[MEWMConfig] = None,
        index: Optional[DatasetIndex] = None,
        output_root: Optional[Path | str] = None,
        calibrate: bool = False,
        clip_stride: int = 1,
        clip_max_frames: int = 0,
        device: str = "cuda",
    ) -> None:
        self.dataset = dataset
        self.mewm_config = mewm_config or load_config()
        self.config: TrainingConfig = self.mewm_config.training
        require_open_weight(self.config.policy_model)

        self.sft_backend = sft_backend
        self.policy_backend = policy_backend
        self.sampler = sampler
        self.evaluator = evaluator
        self.index = index or load_dataset(dataset)
        self.output_root = Path(output_root) if output_root else None
        self.builder = InstructionSetBuilder(seed=self.config.fold_seed)
        #: Fit ``(tau_hi, tau_lo)`` per fold rather than using the config defaults. Off
        #: by default because it runs the representation and spotting engines over the
        #: whole training pool, which is expensive; on, it is what makes the protocol's
        #: "thresholds calibrated on the training fold only" claim auditable.
        self.calibrate = calibrate
        #: Frame sampling for the stage-0 CLIP dataset. Passed through unchanged; the
        #: same values must be used for the pool and the held-out scoring or the two
        #: AUCs describe different temporal resolutions.
        self.clip_stride = clip_stride
        self.clip_max_frames = clip_max_frames
        self.device = device

    # -- stage 0: the CLIP engine -------------------------------------------

    def run_clip(self, fold: FoldSpec) -> Dict[str, Any]:
        """Fine-tune (or reuse) this fold's CLIP localiser and prove it never saw the
        held-out subject.
        """
        from .clip_localiser import (
            CLIPLocaliserCheckpoint, build_clip_dataset, checkpoint_path,
            evaluate_clip_localiser, train_clip_localiser, write_fold_report,
        )

        clip_config = self.mewm_config.clip
        target = checkpoint_path(self.dataset, fold.name, clip_config)
        reused = False

        if self.config.clip_reuse_checkpoint and target.is_file():
            checkpoint = CLIPLocaliserCheckpoint.load(target)
            reused = True
        else:
            pool = self._pool_videos(fold)
            samples = build_clip_dataset(pool, self.mewm_config,
                                         stride=self.clip_stride,
                                         max_frames=self.clip_max_frames)
            if not samples:
                raise ClipEngineUnavailable(
                    f"fold {fold.name}: no trainable video in the training pool, so the "
                    f"CLIP engine cannot be fitted. Every later stage would read a stock "
                    f"CLIP while the report claims a fine-tuned one.")
            checkpoint = train_clip_localiser(samples, clip_config, fold_name=fold.name)

        # Before anything downstream reads it. ``assert_excludes`` raises on a breach.
        checkpoint.assert_excludes([fold.test_subject])

        test_videos = [v for v in self.index.videos
                       if v.video_key in set(fold.test_videos)]
        test_samples = build_clip_dataset(test_videos, self.mewm_config,
                                          stride=self.clip_stride,
                                          max_frames=self.clip_max_frames)
        report = (evaluate_clip_localiser(checkpoint, test_samples, clip_config,
                                          device=self.device)
                  if test_samples
                  else {"note": "held-out subject has no micro-annotated video"})
        # Writes the weights *and* the held-out report next to them, so the checkpoint
        # a later run reuses is never separated from the evidence about what it is.
        saved = write_fold_report(checkpoint, report, self.dataset, fold.name,
                                  clip_config)
        return {
            "status": "reused" if reused else "fine-tuned",
            "checkpoint": str(saved),
            "fold_name": checkpoint.fold_name,
            "train_subjects": list(checkpoint.train_subjects),
            "val_subjects": list(checkpoint.val_subjects),
            "excluded_subject": fold.test_subject,
            "train_metrics": dict(checkpoint.metrics),
            "held_out_auc": report.get("pooled_mean_auc"),
            "held_out_videos": report.get("n_videos", 0),
        }

    # -- calibration --------------------------------------------------------

    def _calibrate(self, fold: FoldSpec) -> Dict[str, Any]:
        """Fit the detection thresholds on this fold's training pool, and record proof.
        """
        from .pretrain import calibrate_detection_thresholds

        pool_videos = self._pool_videos(fold)
        if not pool_videos:
            return {"status": "no pool videos", "thresholds": {}}
        used = sorted(v.video_key for v in pool_videos)
        leaked = sorted(set(used) & set(fold.test_videos))
        if leaked:
            # Unreachable through ``_pool_videos`` -- which is exactly why it is checked
            # here, because a silent breach contaminates every threshold downstream.
            raise RuntimeError(
                f"fold {fold.name}: calibration drew on held-out video(s) {leaked[:5]}")
        thresholds = calibrate_detection_thresholds(pool_videos, self.mewm_config)
        return {
            "status": "calibrated",
            "thresholds": thresholds,
            "n_videos_offered": len(used),
            "excluded_test_videos": sorted(fold.test_videos),
            "defaults": {"tau_hi": self.mewm_config.spotting.tau_hi,
                         "tau_lo": self.mewm_config.spotting.tau_lo},
        }

    # -- data ---------------------------------------------------------------

    def _qa(self, fold: FoldSpec, with_augmented: bool) -> Optional[QASet]:
        """The fold's QA set, optionally including its own augmented pairs.

        ``allowed_videos`` is the pool, so an augmented file built for a different fold
        raises here rather than silently supplying held-out material.
        """
        return load_qa_set(
            self.dataset,
            augmented_fold=fold.name if with_augmented else None,
            allowed_videos=sorted(fold.pool_videos()) if with_augmented else None,
        )

    def _pool_videos(self, fold: FoldSpec) -> List[LongVideo]:
        keys = fold.pool_videos()
        return [v for v in self.index.videos if v.video_key in keys]

    def _sft_samples(self, fold: FoldSpec, with_augmented: bool) -> List[Dict[str, Any]]:
        qa = self._qa(fold, with_augmented)
        subjects = list(fold.train_subjects) + list(fold.val_subjects)
        instruction_samples = self.builder.build_dataset(
            self.index.videos, qa, include_subjects=subjects)
        augmented = []
        if with_augmented and qa is not None:
            augmented = [
                {"video_id": i.video_id, "video": i.video,
                 "question": i.question, "answer": i.answer}
                for i in qa.augmented_items()
            ]
        augmented.extend(self._consolidated_augmented(fold, seen=augmented))
        return build_sft_samples(instruction_samples, augmented=augmented)

    def _consolidated_augmented(
        self, fold: FoldSpec, seen: Sequence[Dict[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        """The corpus-level sweep, projected onto this fold's pool.
        """
        if not self.config.use_consolidated_augmented:
            return []
        model = self.config.consolidated_augmented_model
        try:
            rows = qa_augment.load_consolidated(
                self.dataset, model, fold.pool_videos(), fold=fold.name)
        except qa_augment.AugmentationError as exc:
            LOGGER.error("fold %s: corpus-level augmented pool is malformed (%s); "
                         "training on the fold-local pairs alone", fold.name, exc)
            return []
        known = {str(row.get("video_id", "")) for row in seen}
        fresh = [row for row in rows if str(row.get("video_id", "")) not in known]
        if rows and len(fresh) < len(rows):
            LOGGER.info("fold %s: %d consolidated pair(s) already present from stage 2",
                        fold.name, len(rows) - len(fresh))
        return fresh

    # -- stages -------------------------------------------------------------

    def run_sft(
        self, fold: FoldSpec, round_index: int, with_augmented: bool = False,
    ) -> Tuple[SFTOutcome, List[Dict[str, Any]]]:
        """Stage 1 (and stage 2's refit): ``sft_max_epochs`` as a ceiling, early stop."""
        samples = self._sft_samples(fold, with_augmented)
        trainer = SFTTrainer(self.sft_backend, self.config, self.mewm_config,
                            seed=self.config.fold_seed + round_index)
        output_dir = None
        if self.output_root:
            output_dir = self.output_root / self.dataset / f"fold_{fold.name}" / f"sft_{round_index}"
        outcome = trainer.fit(samples, output_dir=output_dir,
                              in_sample_eval=fold.in_sample)
        return outcome, samples

    def sample_candidates(
        self, prompts: Sequence[Dict[str, Any]], n: int,
    ) -> List[Candidate]:
        """Draw ``n`` candidates per prompt through the injected sampler."""
        if not prompts:
            LOGGER.warning("no prompts to sample; the gate will have no data to judge")
            return []
        if self.sampler is None:
            raise RuntimeError(
                "no sampler supplied: stages 2 and the gate both need the policy's own "
                "generations, which the runner does not produce itself")
        candidates: List[Candidate] = []
        for entry in prompts:
            candidates.extend(self.sampler(entry, n))
        return candidates

    def gate(
        self,
        fold: FoldSpec,
        round_index: int,
        losses: Sequence[float],
        prompts: Sequence[Dict[str, Any]],
    ) -> Tuple[SufficiencyReport, List[Candidate]]:
        """The four sufficiency judgements, on ``pass_at_k_samples`` draws per prompt."""
        candidates = self.sample_candidates(prompts, self.config.pass_at_k_samples)
        truth_by_prompt = {p.get("id", ""): p.get("truth", {}) for p in prompts}
        score_candidates(candidates, truth_by_prompt,
                         self.mewm_config.evaluation)

        # ``score_candidates`` attaches the pass outcome; ``reward`` is the injected
        # sampler's job. A sampler that forgets it leaves every reward at 0.0, which the
        # spread diagnostic reads as a genuine all-low pool -- so the gate returns
        # ``continue_sft`` on every round of every fold and the run never reaches RL, with
        # nothing in the report to distinguish that from a policy that really is
        # undertrained. Name it instead of ranking noise.
        if candidates and not any(c.reward for c in candidates):
            LOGGER.warning(
                "fold %s round %d: all %d candidate(s) carry reward 0.0. The sampler is "
                "responsible for scoring with CompositeReward; an unscored pool makes "
                "the reward-spread diagnostic read all-low regardless of the policy.",
                fold.name, round_index, len(candidates))

        outcomes_by_prompt: Dict[str, List[PassOutcome]] = {}
        for candidate in candidates:
            if candidate.outcome is not None:
                outcomes_by_prompt.setdefault(candidate.prompt_id, []).append(
                    candidate.outcome)

        report = assess(
            outcomes_by_prompt, group_rewards(candidates), losses,
            self.config, fold=fold.name, round_index=round_index,
            in_sample=fold.in_sample,
        )
        return report, candidates

    def run_rft(
        self,
        fold: FoldSpec,
        candidates: Sequence[Candidate],
        prompts: Sequence[Dict[str, Any]],
    ) -> Tuple[FilterReport, Dict[str, Any]]:
        """Stage 2: keep the passing candidates, write them back as QA, refit on them.
        """
        accepted, report = select(candidates, self.config)
        question_by_prompt = {
            p.get("id", ""): p.get("question", p.get("prompt", "")) for p in prompts
        }
        subject_by_video = {v.video_key: v.subject for v in self.index.videos}

        written: Dict[str, Any] = {}
        if self.config.write_augmented_qa and accepted:
            pairs = qa_augment.build_pairs(
                accepted, self.dataset, fold.name, question_by_prompt,
                subject_by_video, self.config.policy_model)
            written = qa_augment.write_augmented(
                pairs, self.dataset, fold.name, fold.pool_videos())
        return report, written

    def run_rl(
        self, fold: FoldSpec, prompts: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 3: ``rl_total_steps`` GRPO optimisation steps on the pool's prompts."""
        if self.policy_backend is None:
            raise RuntimeError("no policy backend supplied; cannot run stage 3")
        trainer = GRPOTrainer(
            self.policy_backend,
            config=GRPOConfig.from_training(self.config),
            mewm_config=self.mewm_config,
        )
        history = trainer.train(prompts)
        if self.output_root:
            target = (self.output_root / self.dataset / f"fold_{fold.name}"
                      / "rl_history.json")
            trainer.save_history(target)
        return history

    # -- fold ---------------------------------------------------------------

    def run_test(
        self, fold: FoldSpec, test_prompts: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Stage 4: the one pass over the held-out subject, optionally with TTA.

        **Why this may touch the test set at all.** Test-time augmentation draws the
        policy ``tta_samples`` times per prompt and keeps the best-scoring draw. No
        gradient is taken, no optimiser is stepped, and nothing generated here is
        written back as training data -- ``run_rft`` is the only writer of augmented QA
        and it runs before this, on the pool, with the pool's own leak guard. So the
        held-out subject stays out of every loss term, which is the property LOSO
        actually claims; it does not claim the policy may only be run once.

        **What it costs, stated rather than hidden.** Best-of-k with a reward that reads
        ground truth would be oracle selection, so selection uses the *reward*, and
        pass@1 -- the first draw, no selection -- is reported beside it every time. A
        reader comparing against a pass@1 baseline needs the pass@1 column to exist; an
        aggregated number presented alone silently changes the inference budget.
        """
        result: Dict[str, Any] = {}
        if self.evaluator is not None:
            result["evaluator"] = self.evaluator(fold, fold.test_videos)

        k = int(self.config.tta_samples)
        if not self.config.test_time_augmentation or k <= 1 or not test_prompts:
            result["tta"] = {
                "status": "off" if not self.config.test_time_augmentation
                else ("no test prompts supplied" if not test_prompts
                      else f"tta_samples = {k}; nothing to aggregate"),
                "draws_per_prompt": 1,
            }
            return result

        # Mirror of the pool check in ``run_fold``, pointed the other way: a *pool*
        # prompt reaching the test stage would report training material as held-out.
        allowed = set(fold.test_videos)
        stray = sorted({p.get("video", "") for p in test_prompts} - allowed)
        if stray:
            raise ValueError(
                f"fold {fold.name}: {len(stray)} test prompt(s) are not from the "
                f"held-out subject: {stray[:5]}")

        draws = self.sample_candidates(test_prompts, k)
        truth_by_prompt = {p.get("id", ""): p.get("truth", {}) for p in test_prompts}
        score_candidates(draws, truth_by_prompt, self.mewm_config.evaluation)

        by_prompt: Dict[str, List[Candidate]] = {}
        for candidate in draws:
            by_prompt.setdefault(candidate.prompt_id, []).append(candidate)

        first_pass = selected_pass = 0
        for group in by_prompt.values():
            if group[0].outcome is not None and group[0].outcome.passed:
                first_pass += 1
            best = max(group, key=lambda c: c.reward)
            if best.outcome is not None and best.outcome.passed:
                selected_pass += 1

        n = len(by_prompt) or 1
        result["tta"] = {
            "status": "applied",
            "draws_per_prompt": k,
            "n_prompts": len(by_prompt),
            "pass_at_1": first_pass / n,
            "pass_selected": selected_pass / n,
            "selection": "highest CompositeReward among the k draws",
            "gradient_steps": 0,
            "caveat": (
                "inference-time only: the held-out subject entered no loss term, no "
                "optimiser step and no augmented-QA file. Not comparable with a "
                "single-draw baseline without stating the k."
            ),
        }
        return result

    def run_fold(
        self, fold: FoldSpec, prompts: Sequence[Dict[str, Any]],
        test_prompts: Sequence[Dict[str, Any]] = (),
    ) -> FoldResult:
        """The whole chain for one fold.

        ``prompts`` must already be restricted to the fold's training pool; the runner
        checks it rather than trusting it, because a stray test-subject prompt here would
        contaminate every downstream stage at once.
        """
        fold.check_disjoint()
        result = FoldResult(fold=fold)

        pool = fold.pool_videos()
        # A prompt with no ``video`` key used to pass this check: the empty string was
        # subtracted out along with the pool, so a builder that spelled the field
        # ``video_id`` silently disabled the isolation guard for every prompt it made.
        # An unattributable prompt is not evidence of isolation, it is the absence of
        # evidence, and the runner is here precisely so it does not have to trust the
        # caller on this point.
        unattributed = [i for i, p in enumerate(prompts) if not p.get("video")]
        if unattributed:
            raise ValueError(
                f"fold {fold.name}: {len(unattributed)} prompt(s) carry no 'video' key "
                f"(first at index {unattributed[:5]}); the pool-isolation check cannot "
                f"run, so the fold refuses to start rather than assume they are clean")
        stray = sorted({p["video"] for p in prompts} - pool)
        if stray:
            raise ValueError(
                f"fold {fold.name}: {len(stray)} prompt(s) come from videos outside the "
                f"training pool: {stray[:5]}")

        candidates: List[Candidate] = []
        report: Optional[SufficiencyReport] = None
        if self.config.clip_finetune:
            # Stage 0, before calibration: the thresholds are fitted on the detection
            # statistic, and that statistic comes out of this engine. Calibrating first
            # would tune them to a front end the run then replaces.
            try:
                result.clip = self.run_clip(fold)
            except Exception as exc:  # noqa: BLE001 - re-raised below when required
                if self.config.clip_required:
                    raise ClipEngineUnavailable(
                        f"fold {fold.name}: stage 0 failed ({exc}). Set "
                        f"training.clip_required = false to continue on the analytic "
                        f"front end, but then the fold's numbers are not CLIP numbers."
                    ) from exc
                LOGGER.error("fold %s: stage 0 failed (%s); continuing on the analytic "
                             "front end because clip_required is false", fold.name, exc)
                result.clip = {"status": "failed", "error": str(exc)}
        else:
            result.clip = {"status": "disabled (training.clip_finetune = false)"}

        if self.calibrate:
            # Before anything trains: the thresholds every downstream stage inherits are
            # fitted here, on the pool, and the evidence goes on the result.
            result.calibration = self._calibrate(fold)
        with_augmented = self.config.use_augmented_qa and bool(
            qa_augment.load_augmented(self.dataset, fold.name, pool))

        for round_index in range(max(1, self.config.max_sft_rounds)):
            outcome, samples = self.run_sft(fold, round_index, with_augmented)
            result.sft_rounds.append(outcome.to_dict())

            report, candidates = self.gate(
                fold, round_index, outcome.epoch_losses, prompts)
            result.gate_reports.append(report.to_dict())
            result.decision = report.decision

            if report.decision != "continue_sft":
                break
            LOGGER.info("fold %s round %d: gate says continue_sft (%s)",
                        fold.name, round_index, "; ".join(report.reasons()))
        else:
            LOGGER.warning(
                "fold %s: gate still says continue_sft after %d round(s); proceeding to "
                "stage 2 anyway so the run terminates, but the RL result on this fold is "
                "not comparable with a fold that converged",
                fold.name, self.config.max_sft_rounds)

        if candidates:
            filter_report, written = self.run_rft(fold, candidates, prompts)
            result.rft = filter_report.to_dict()
            result.augmented = written
            if written:
                # Refit on the strictly larger set before RL, so stage 3 starts from the
                # policy that has already seen its own accepted generations.
                outcome, _ = self.run_sft(fold, len(result.sft_rounds), with_augmented=True)
                result.sft_rounds.append(outcome.to_dict())

        if result.decision == "rl_ready":
            result.rl_history = self.run_rl(fold, prompts)
        else:
            result.rl_skipped = (
                f"gate decision was {result.decision!r}, not 'rl_ready': "
                + "; ".join(report.reasons() if report else ["no gate report"])
            )
            LOGGER.warning("fold %s: skipping stage 3 -- %s", fold.name, result.rl_skipped)

        if self.evaluator is not None or test_prompts:
            result.test = self.run_test(fold, test_prompts)
        return result

    def run(
        self,
        prompts_for: Callable[[FoldSpec], Sequence[Dict[str, Any]]],
        subjects: Optional[Sequence[str]] = None,
        test_prompts_for: Optional[
            Callable[[FoldSpec], Sequence[Dict[str, Any]]]] = None,
    ) -> List[FoldResult]:
        """Run every requested fold and return the per-fold results.
        """
        folds = build_folds(self.index, self.config, subjects)
        LOGGER.info("%s: %d fold(s), %d subject(s) total",
                    self.dataset, len(folds), len(self.index.subjects()))
        results = []
        for fold in folds:
            LOGGER.info("=== fold %s: %d train / %d val / %d test video(s) ===",
                        fold.name, len(fold.train_videos), len(fold.val_videos),
                        len(fold.test_videos))
            test_prompts = test_prompts_for(fold) if test_prompts_for else ()
            results.append(self.run_fold(fold, prompts_for(fold), test_prompts))
        return results


def summarise(results: Sequence[FoldResult]) -> Dict[str, Any]:
    """Aggregate across folds, keeping the in-sample caveat attached."""
    if not results:
        return {"folds": 0}
    decisions: Dict[str, int] = {}
    for result in results:
        decisions[result.decision] = decisions.get(result.decision, 0) + 1
    in_sample = [r for r in results if r.fold.in_sample]

    # -- stage 0 ------------------------------------------------------------
    clip_status: Dict[str, int] = {}
    clip_aucs: List[float] = []
    for result in results:
        clip_status[result.clip.get("status", "not run")] = (
            clip_status.get(result.clip.get("status", "not run"), 0) + 1)
        auc = result.clip.get("held_out_auc")
        if isinstance(auc, (int, float)) and np.isfinite(auc):
            clip_aucs.append(float(auc))

    # -- stage 4 ------------------------------------------------------------
    # Averaged over the folds that actually produced the number, and the count is
    # carried alongside: a mean over 3 of 15 folds is not a LOSO result, and the only
    # way a reader can tell is if the denominator travels with the mean.
    tta = [r.test.get("tta", {}) for r in results]
    applied = [t for t in tta if t.get("status") == "applied"]
    tta_summary: Dict[str, Any] = {
        "folds_with_tta": len(applied),
        "draws_per_prompt": applied[0].get("draws_per_prompt") if applied else 1,
    }
    if applied:
        tta_summary["pass_at_1"] = float(
            np.mean([t.get("pass_at_1", 0.0) for t in applied]))
        tta_summary["pass_selected"] = float(
            np.mean([t.get("pass_selected", 0.0) for t in applied]))
        tta_summary["delta"] = tta_summary["pass_selected"] - tta_summary["pass_at_1"]
        tta_summary["gradient_steps"] = 0
    elif tta:
        tta_summary["status"] = tta[0].get("status", "off")

    return {
        "folds": len(results),
        "decisions": decisions,
        "clip_engine": {
            "status": clip_status,
            "held_out_auc_mean": float(np.mean(clip_aucs)) if clip_aucs else None,
            "folds_scored": len(clip_aucs),
            "note": ("every fold's policy was trained against the fine-tuned CLIP "
                     "checkpoint recorded in folds[].clip.checkpoint"),
        },
        "rl_ran": sum(1 for r in results if r.rl_history),
        "rl_skipped": sum(1 for r in results if r.rl_skipped),
        "augmented_pairs": sum(int(r.augmented.get("n_pairs", 0)) for r in results),
        "test_time_augmentation": tta_summary,
        "in_sample_folds": len(in_sample),
        "caveat": (
            f"{len(in_sample)}/{len(results)} fold(s) gated on the training pool "
            f"(n_val_subjects = 0); those diagnostics are in-sample."
        ) if in_sample else "",
    }


def write_report(results: Sequence[FoldResult], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarise(results),
        "folds": [r.to_dict() for r in results],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


__all__ = [
    "CLOSED_SOURCE_HINTS", "PolicyNotTrainable", "ClipEngineUnavailable",
    "require_open_weight",
    "FoldSpec", "build_folds", "FoldResult", "LOSORunner", "summarise", "write_report",
]
