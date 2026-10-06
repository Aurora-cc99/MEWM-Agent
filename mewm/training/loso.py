"""Leave-one-subject-out training and evaluation loop."""

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

CLOSED_SOURCE_HINTS = (
    "gpt-", "o1", "o3", "claude-", "gemini-", "grok-", "deepseek-chat",
    "deepseekv", "qwen-max", "qwen-plus", "doubao", "glm-4-plus",
)


class PolicyNotTrainable(ValueError):


    ...
class ClipEngineUnavailable(RuntimeError):

    ...
class SoftNetEngineUnavailable(RuntimeError):


    ...
def require_open_weight(policy_model: str) -> str:
    lowered = policy_model.strip().lower()
    if not lowered:
        raise PolicyNotTrainable("no policy model given")

    try:
        from ..llm.registry import resolve
        spec = resolve(policy_model)
    except Exception:
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


@dataclass
class FoldSpec:

    dataset: str
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
        return not self.val_subjects

    def pool_videos(self) -> set:
        return set(self.train_videos) | set(self.val_videos)

    def check_disjoint(self) -> None:
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
    digest = hashlib.blake2b(
        f"{fold_seed}:{dataset}:{test_subject}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build_folds(
    index: DatasetIndex,
    config: Optional[TrainingConfig] = None,
    subjects: Optional[Sequence[str]] = None,
) -> List[FoldSpec]:
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


@dataclass
class FoldResult:

    fold: FoldSpec
    clip: Dict[str, Any] = field(default_factory=dict)
    softnet: Dict[str, Any] = field(default_factory=dict)
    sft_rounds: List[Dict[str, Any]] = field(default_factory=list)
    gate_reports: List[Dict[str, Any]] = field(default_factory=list)
    rft: Dict[str, Any] = field(default_factory=dict)
    augmented: Dict[str, Any] = field(default_factory=dict)
    rl_history: List[Dict[str, Any]] = field(default_factory=list)
    test: Dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    rl_skipped: str = ""
    calibration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold.to_dict(),
            "clip": self.clip or {"status": "not run"},
            "softnet": self.softnet or {"status": "not run"},
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


SamplerFn = Callable[[Dict[str, Any], int], List[Candidate]]
EvaluatorFn = Callable[[FoldSpec, Sequence[str]], Dict[str, Any]]


class LOSORunner:

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
        self.calibrate = calibrate
        self.clip_stride = clip_stride
        self.clip_max_frames = clip_max_frames
        self.device = device


    def run_clip(self, fold: FoldSpec) -> Dict[str, Any]:
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


    def run_softnet(self, fold: FoldSpec) -> Dict[str, Any]:
        from ..config import RUNS_ROOT
        from ..engines.softnet_spotter import (SoftNetCheckpoint, build_feature_caches,
                                               train_fold)

        checkpoint_root = (RUNS_ROOT / "softnet_spotter" / self.dataset /
                          f"fold_{fold.test_subject}")
        target = checkpoint_root / "softnet_spotter.npz"
        feature_root = RUNS_ROOT / "softnet_features" / self.dataset
        reused = False

        if self.config.softnet_reuse_checkpoint and target.is_file():
            checkpoint = SoftNetCheckpoint.load(target)
            reused = True
        else:
            pool = self._pool_videos(fold)
            caches = build_feature_caches(pool, feature_root)
            if not caches:
                raise SoftNetEngineUnavailable(
                    f"fold {fold.name}: no trainable video in the training pool")
            intervals = {v.video_key: [e.interval for e in v.events if e.is_micro]
                        for v in pool}
            subjects_by_video = {v.video_key: str(v.subject) for v in pool}
            durations = [e.duration for v in pool for e in v.events if e.is_micro]
            k = max(2, int(round(float(np.mean(durations)) / 2))) if durations else 7
            checkpoint = train_fold(caches, intervals, fold_name=fold.test_subject, k=k,
                                    epochs=self.config.softnet_epochs, device=self.device,
                                    subjects_by_video=subjects_by_video)
            checkpoint.save(target)

        if checkpoint.fold != fold.test_subject:
            raise SoftNetEngineUnavailable(
                f"fold {fold.name}: loaded checkpoint at {target} was trained for "
                f"fold {checkpoint.fold!r}, not the held-out subject "
                f"{fold.test_subject!r} -- refusing to use it.")

        return {
            "status": "reused" if reused else "fine-tuned",
            "checkpoint": str(target),
            "fold": checkpoint.fold,
            "k": checkpoint.k,
        }


    def _calibrate(self, fold: FoldSpec) -> Dict[str, Any]:
        from .pretrain import calibrate_detection_thresholds

        pool_videos = self._pool_videos(fold)
        if not pool_videos:
            return {"status": "no pool videos", "thresholds": {}}
        used = sorted(v.video_key for v in pool_videos)
        leaked = sorted(set(used) & set(fold.test_videos))
        if leaked:
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


    def _qa(self, fold: FoldSpec, with_augmented: bool) -> Optional[QASet]:
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


    def run_sft(
        self, fold: FoldSpec, round_index: int, with_augmented: bool = False,
    ) -> Tuple[SFTOutcome, List[Dict[str, Any]]]:
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
        candidates = self.sample_candidates(prompts, self.config.pass_at_k_samples)
        truth_by_prompt = {p.get("id", ""): p.get("truth", {}) for p in prompts}
        score_candidates(candidates, truth_by_prompt,
                         self.mewm_config.evaluation)

        if candidates and not any(c.reward for c in candidates):
            LOGGER.warning(
                "...",
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
        if self.policy_backend is None:
            raise RuntimeError("......")
        trainer = GRPOTrainer(
            self.policy_backend,
            config=GRPOConfig.from_training(self.config),
            mewm_config=self.mewm_config,
        )
        target = None
        if self.output_root:
            target = (self.output_root / self.dataset / f"fold_{fold.name}"
                      / "rl_history.json")
        history = trainer.train(prompts, history_path=target)
        if target is not None:
            trainer.save_history(target)
        return history


    def run_test(
        self, fold: FoldSpec, test_prompts: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
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
        fold.check_disjoint()
        result = FoldResult(fold=fold)

        pool = fold.pool_videos()
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
            try:
                result.clip = self.run_clip(fold)
            except Exception as exc:
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
    if not results:
        return {"folds": 0}
    decisions: Dict[str, int] = {}
    for result in results:
        decisions[result.decision] = decisions.get(result.decision, 0) + 1
    in_sample = [r for r in results if r.fold.in_sample]

    clip_status: Dict[str, int] = {}
    clip_aucs: List[float] = []
    for result in results:
        clip_status[result.clip.get("status", "not run")] = (
            clip_status.get(result.clip.get("status", "not run"), 0) + 1)
        auc = result.clip.get("held_out_auc")
        if isinstance(auc, (int, float)) and np.isfinite(auc):
            clip_aucs.append(float(auc))

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
