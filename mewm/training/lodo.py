"""Leave-one-dataset-out cross-corpus generalisation training loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, TrainingConfig, load_config
from ..data.datasets import DatasetIndex, LongVideo, load_dataset
from ..data.paths import DATASETS
from .grpo import GRPOConfig, GRPOTrainer, PolicyBackend
from .loso import ClipEngineUnavailable, FoldSpec, LOSORunner, require_open_weight
from .mappo import MAPPOConfig, MAPPOTrainer, AgentPolicyBackend, ROLES
from .sft import SFTBackend, SFTOutcome, SFTTrainer

LOGGER = logging.getLogger(__name__)

_POOL_SENTINEL = "__lodo_full_pool__"


@dataclass
class DatasetFoldSpec:

    test_dataset: str
    train_datasets: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.test_dataset

    def check_disjoint(self) -> None:
        if self.test_dataset in self.train_datasets:
            raise ValueError(
                f"LODO fold {self.name}: the held-out dataset {self.test_dataset!r} "
                f"also appears in train_datasets {self.train_datasets}")

    def to_dict(self) -> Dict[str, Any]:
        return {"test_dataset": self.test_dataset,
                "train_datasets": list(self.train_datasets)}


def build_dataset_folds(
    datasets: Optional[Sequence[str]] = None,
    held_out: Optional[Sequence[str]] = None,
) -> List[DatasetFoldSpec]:
    pool = list(datasets) if datasets is not None else list(DATASETS)
    wanted = set(held_out) if held_out is not None else set(pool)
    unknown = wanted - set(pool)
    if unknown:
        raise KeyError(f"unknown dataset(s) {sorted(unknown)}; known are {pool}")

    folds: List[DatasetFoldSpec] = []
    for test_dataset in pool:
        if test_dataset not in wanted:
            continue
        fold = DatasetFoldSpec(
            test_dataset=test_dataset,
            train_datasets=[d for d in pool if d != test_dataset],
        )
        fold.check_disjoint()
        folds.append(fold)
    return folds


def _full_pool_fold(index: DatasetIndex) -> FoldSpec:
    subjects = index.subjects()
    if _POOL_SENTINEL in subjects:
        raise RuntimeError(
            f"{index.dataset}: a real subject is named {_POOL_SENTINEL!r}, which "
            f"collides with the LODO full-pool sentinel; rename the sentinel")
    all_videos = sorted(v.video_key for v in index.videos)
    fold = FoldSpec(
        dataset=index.dataset, test_subject=_POOL_SENTINEL,
        train_subjects=list(subjects), val_subjects=[],
        train_videos=all_videos, val_videos=[], test_videos=[],
    )
    fold.check_disjoint()
    return fold


@dataclass
class DatasetFoldResult:
    fold: DatasetFoldSpec
    clip: Dict[str, Any] = field(default_factory=dict)
    sft: Dict[str, Any] = field(default_factory=dict)
    rft: Dict[str, Any] = field(default_factory=dict)
    sft_after_rft: Dict[str, Any] = field(default_factory=dict)
    rl_history: List[Dict[str, Any]] = field(default_factory=list)
    rl_skipped: str = ""
    test: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold.to_dict(), "clip": self.clip, "sft": self.sft,
            "rft": self.rft, "sft_after_rft": self.sft_after_rft,
            "rl_history": self.rl_history, "rl_skipped": self.rl_skipped,
            "test": self.test,
        }


RftFn = Callable[[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]],
                 Tuple[Dict[str, Any], List[Dict[str, Any]]]]
EvaluatorFn = Callable[[DatasetFoldSpec, Sequence[Dict[str, Any]]], Dict[str, Any]]


class LODORunner:

    def __init__(
        self,
        sft_backend: SFTBackend,
        rl_algorithm: str = "grpo",
        policy_backend: Optional[PolicyBackend] = None,
        mappo_backends: Optional[Dict[str, AgentPolicyBackend]] = None,
        rft_fn: Optional[RftFn] = None,
        evaluator: Optional[EvaluatorFn] = None,
        mewm_config: Optional[MEWMConfig] = None,
        indices: Optional[Dict[str, DatasetIndex]] = None,
        clip_stride: int = 1,
        clip_max_frames: int = 0,
        device: str = "cuda",
    ) -> None:
        if rl_algorithm not in ("grpo", "mappo"):
            raise ValueError(f"rl_algorithm must be 'grpo' or 'mappo', got {rl_algorithm!r}")
        if rl_algorithm == "grpo" and policy_backend is None:
            LOGGER.warning("rl_algorithm='grpo' but no policy_backend supplied; "
                           "stage 3 will be skipped on every fold")
        if rl_algorithm == "mappo" and mappo_backends is None:
            LOGGER.warning("rl_algorithm='mappo' but no mappo_backends supplied; "
                           "stage 3 will be skipped on every fold")

        self.mewm_config = mewm_config or load_config()
        self.config: TrainingConfig = self.mewm_config.training
        require_open_weight(self.config.policy_model)

        self.sft_backend = sft_backend
        self.rl_algorithm = rl_algorithm
        self.policy_backend = policy_backend
        self.mappo_backends = mappo_backends
        self.rft_fn = rft_fn
        self.evaluator = evaluator
        self.indices = indices or {d: load_dataset(d) for d in DATASETS}
        self.clip_stride = clip_stride
        self.clip_max_frames = clip_max_frames
        self.device = device


    def run_clip_pool(self, dataset: str) -> Dict[str, Any]:
        index = self.indices[dataset]
        runner = LOSORunner(
            dataset=dataset, sft_backend=self.sft_backend, mewm_config=self.mewm_config,
            index=index, clip_stride=self.clip_stride,
            clip_max_frames=self.clip_max_frames, device=self.device,
        )
        return runner.run_clip(_full_pool_fold(index))


    def run_sft(self, samples: Sequence[Dict[str, Any]]) -> SFTOutcome:
        trainer = SFTTrainer(self.sft_backend, self.config, self.mewm_config)
        return trainer.fit(list(samples), eval_samples=(), in_sample_eval=True)


    def run_rft(self, samples: Sequence[Dict[str, Any]],
               candidates: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.rft_fn is None:
            return {"status": "skipped", "reason": "no rft_fn supplied"}, []
        return self.rft_fn(samples, candidates)


    def run_rl(self, prompts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.rl_algorithm == "mappo":
            if self.mappo_backends is None:
                return []
            trainer = MAPPOTrainer(
                self.mappo_backends, config=MAPPOConfig.from_training(self.config),
                mewm_config=self.mewm_config,
            )
            batches = [prompts[i:i + trainer.config.max_candidates_per_step]
                      for i in range(0, len(prompts), max(1, trainer.config.max_candidates_per_step))]
            return trainer.train(batches[: trainer.config.total_steps])
        if self.policy_backend is None:
            return []
        trainer = GRPOTrainer(
            self.policy_backend, config=GRPOConfig.from_training(self.config),
            mewm_config=self.mewm_config,
        )
        batches = [prompts[i:i + trainer.config.max_prompts_per_step]
                  for i in range(0, len(prompts), max(1, trainer.config.max_prompts_per_step))]
        history = []
        for step_index, batch in enumerate(batches[: trainer.config.total_steps]):
            progress = step_index / max(1, trainer.config.total_steps - 1)
            history.append(trainer.step(batch, progress))
        return history


    def run_fold(
        self,
        fold: DatasetFoldSpec,
        prompts_by_dataset: Dict[str, Sequence[Dict[str, Any]]],
        candidates_by_dataset: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
        test_prompts: Sequence[Dict[str, Any]] = (),
    ) -> DatasetFoldResult:
        fold.check_disjoint()
        result = DatasetFoldResult(fold=fold)

        pooled_prompts: List[Dict[str, Any]] = []
        for dataset in fold.train_datasets:
            pooled_prompts.extend(prompts_by_dataset.get(dataset, []))
        stray = sorted({p.get("dataset") for p in pooled_prompts
                       if p.get("dataset") == fold.test_dataset})
        if stray:
            raise ValueError(
                f"LODO fold {fold.name}: prompts tagged with the held-out dataset "
                f"{fold.test_dataset!r} were included in the training pool")

        if self.config.clip_finetune:
            for dataset in fold.train_datasets:
                try:
                    result.clip[dataset] = self.run_clip_pool(dataset)
                except Exception as exc:
                    if self.config.clip_required:
                        raise ClipEngineUnavailable(
                            f"LODO fold {fold.name}: stage 0 failed for training "
                            f"dataset {dataset} ({exc}). Set training.clip_required = "
                            f"false to continue on the analytic front end."
                        ) from exc
                    LOGGER.error("LODO fold %s: stage 0 failed for %s (%s); continuing "
                                "on the analytic front end", fold.name, dataset, exc)
                    result.clip[dataset] = {"status": "failed", "error": str(exc)}
        else:
            result.clip = {"status": "disabled (training.clip_finetune = false)"}

        outcome = self.run_sft(pooled_prompts)
        result.sft = outcome.to_dict()

        pooled_candidates: List[Dict[str, Any]] = []
        for dataset in fold.train_datasets:
            pooled_candidates.extend((candidates_by_dataset or {}).get(dataset, []))
        if pooled_candidates:
            rft_report, accepted = self.run_rft(pooled_prompts, pooled_candidates)
            result.rft = rft_report
            if accepted:
                outcome2 = self.run_sft(pooled_prompts + accepted)
                result.sft_after_rft = outcome2.to_dict()

        rl_history = self.run_rl(pooled_prompts)
        if rl_history:
            result.rl_history = rl_history
        else:
            result.rl_skipped = (
                "no policy_backend/mappo_backends supplied, or the pooled prompt pool "
                "was empty"
            )

        if self.evaluator is not None:
            result.test = self.evaluator(fold, test_prompts)
        return result

    def run(
        self,
        prompts_for: Callable[[str], Sequence[Dict[str, Any]]],
        held_out: Optional[Sequence[str]] = None,
        candidates_for: Optional[Callable[[str], Sequence[Dict[str, Any]]]] = None,
        test_prompts_for: Optional[Callable[[DatasetFoldSpec], Sequence[Dict[str, Any]]]] = None,
    ) -> List[DatasetFoldResult]:
        folds = build_dataset_folds(list(self.indices), held_out)
        LOGGER.info("LODO: %d fold(s) over %s", len(folds), list(self.indices))
        results = []
        for fold in folds:
            LOGGER.info("=== LODO fold %s: train on %s ===", fold.name, fold.train_datasets)
            prompts_by_dataset = {d: prompts_for(d) for d in fold.train_datasets}
            candidates_by_dataset = ({d: candidates_for(d) for d in fold.train_datasets}
                                     if candidates_for else None)
            test_prompts = test_prompts_for(fold) if test_prompts_for else ()
            results.append(self.run_fold(fold, prompts_by_dataset, candidates_by_dataset,
                                         test_prompts))
        return results


def summarise(results: Sequence[DatasetFoldResult]) -> Dict[str, Any]:
    if not results:
        return {"folds": 0}
    clip_status: Dict[str, int] = {}
    for result in results:
        for _, clip_result in result.clip.items():
            status = clip_result.get("status", "not run") if isinstance(clip_result, dict) else "unknown"
            clip_status[status] = clip_status.get(status, 0) + 1
    return {
        "protocol": "lodo",
        "folds": len(results),
        "held_out_datasets": [r.fold.test_dataset for r in results],
        "clip_engine_status": clip_status,
        "rl_ran": sum(1 for r in results if r.rl_history),
        "rl_skipped": sum(1 for r in results if r.rl_skipped),
        "note": "",
    }


def write_report(results: Sequence[DatasetFoldResult], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summarise(results), "folds": [r.to_dict() for r in results]}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


__all__ = [
    "DatasetFoldSpec", "build_dataset_folds", "DatasetFoldResult", "LODORunner",
    "summarise", "write_report",
]
