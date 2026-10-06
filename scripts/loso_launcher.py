"""LOSO sweep launcher: orchestrates leave-one-subject-out training runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("loso_launcher")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




def _apply_lora(model: Any, lora: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=(lora.target_modules or "all-linear")))
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("LoRA applied (r=%d, alpha=%d, %s): %d/%d trainable parameter(s) "
                "(%.2f%%)", lora.r, lora.alpha, lora.target_modules, trainable, total,
                100.0 * trainable / total)
    return model


def _load_resumed_adapter(model: Any, adapter_dir: Path) -> Any:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("adapter resumed from %s: %d/%d trainable parameter(s) (%.2f%%)",
                adapter_dir, trainable, total, 100.0 * trainable / total)
    return model


def _load_fresh_base(spec: Any, quantization: str, device_map: str = "auto") -> Any:
    from mewm.llm.local_models import load_local_model, unload

    unload(spec.model_id)
    return load_local_model(spec, quantization=quantization, device_map=device_map)


def _load_frozen_reference(spec: Any, dtype: str = "bfloat16") -> Any:
    import torch
    import transformers

    from mewm.llm.local_models import ensure_weights

    source = str(ensure_weights(spec))
    if spec.vision:
        loader = (getattr(transformers, "AutoModelForImageTextToText", None)
                  or getattr(transformers, "AutoModelForVision2Seq", None)
                  or transformers.AutoModelForCausalLM)
    else:
        loader = transformers.AutoModelForCausalLM
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)
    LOGGER.info("loading frozen KL-anchor reference for %s on CPU (%s)...",
                spec.model_id, dtype)
    started = time.time()
    model = loader.from_pretrained(source, torch_dtype=torch_dtype, device_map="cpu",
                                   trust_remote_code=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    LOGGER.info("reference loaded in %.0fs", time.time() - started)
    return model


def _render_prompt(entry: Dict[str, Any], loaded: Any, spec: Any) -> str:
    from mewm.llm.local_models import (
        THINKING_DIRECTIVE, _NO_THINKING_KWARG, _apply_chat_template, thinking_mode)
    from mewm.training.api_sampler import SYSTEM_PROMPT, build_user_prompt

    user_prompt = build_user_prompt(entry)
    mode = thinking_mode(spec)
    want_thinking = mode != ""
    effective_system = SYSTEM_PROMPT
    if want_thinking and (mode == "prompt" or spec.model_id in _NO_THINKING_KWARG):
        effective_system = f"{THINKING_DIRECTIVE}\n\n{SYSTEM_PROMPT}"
    messages = [{"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt}]
    try:
        return _apply_chat_template(loaded.processor, messages, spec,
                                    want_thinking and mode == "template")
    except Exception as exc:
        LOGGER.warning("%s: chat template failed (%s); falling back to a plain "
                       "system+user join", spec.model_id, exc)
        return f"{effective_system}\n\n{user_prompt}"




class LocalPolicySampler:

    def __init__(self, policy_backend: Any, evaluation: Any, reward_config: Any,
                 temperature: float = 0.7) -> None:
        from mewm.training.rewards import CompositeReward

        self.policy_backend = policy_backend
        self.evaluation = evaluation
        self.reward = CompositeReward(reward_config, evaluation)
        self.temperature = temperature
        self.n_calls = 0
        self.n_failed_calls = 0
        self.n_unparsable = 0
        self.n_candidates = 0

    def _weight_mass(self) -> Any:
        from mewm.training.api_sampler import COMPUTABLE_WITHOUT_JUDGE

        w = self.reward.config
        weights = {"au": w.w_au, "emo": w.w_emo, "fmt": w.w_fmt,
                   "causal": w.w_causal, "temp": w.w_temp}
        total = sum(weights.values())
        available = sum(weights[k] for k in COMPUTABLE_WITHOUT_JUDGE)
        return available, total

    def __call__(self, prompt: Dict[str, Any], n: int) -> List[Any]:
        from mewm.eval.pass_criteria import evaluate_sample
        from mewm.training.api_sampler import EVENT_REQUIRED_FIELDS, extract_product
        from mewm.training.candidate_filter import Candidate

        n = max(1, int(n))
        text = prompt.get("prompt") or ""
        if not text:
            LOGGER.error("prompt %s carries no rendered 'prompt' text; call "
                        "_render_prompt on every entry before sampling",
                        prompt.get("id"))
        self.n_calls += 1
        try:
            samples = self.policy_backend.sample(text, n, self.temperature)
        except Exception as exc:
            LOGGER.error("sampling failed for prompt %s: %s", prompt.get("id"), exc)
            self.n_failed_calls += 1
            samples = []

        available, _ = self._weight_mass()
        truth = prompt.get("truth", {}) or {}
        candidates: List[Any] = []
        for draw in range(1, n + 1):
            sample = samples[draw - 1] if draw - 1 < len(samples) else None
            candidate = Candidate(
                prompt_id=str(prompt.get("id", "")), text="", product={},
                dataset=str(prompt.get("dataset", "")),
                video=str(prompt.get("video", "")),
                event_index=int((prompt.get("anchor") or {}).get("ordinal", 0)))
            if sample is None:
                candidate.reward_detail = {"error": "no generation returned",
                                           "draw": draw}
                candidates.append(candidate)
                continue
            candidate.text = sample.text
            product = sample.product or extract_product(sample.text) or {}
            if not product:
                self.n_unparsable += 1
                candidate.reward_detail = {
                    "error": "completion did not contain a JSON object", "draw": draw}
                candidates.append(candidate)
                continue
            candidate.product = product
            interval = product.get("interval")
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                try:
                    candidate.interval = (int(interval[0]), int(interval[1]))
                except (TypeError, ValueError):
                    candidate.interval = (0, 0)
            breakdown = self.reward.score(product, truth)
            candidate.reward = (round(float(breakdown.total) / available, 5)
                               if available else 0.0)
            candidate.reward_detail = {
                "draw": draw, "raw_total": breakdown.total,
                "renormalised_over": round(available, 4),
                "components": {"r_au": breakdown.r_au, "r_emo": breakdown.r_emo,
                               "r_fmt": breakdown.r_fmt, "r_temp": breakdown.r_temp},
                "detail": breakdown.detail,
            }
            candidates.append(candidate)

        for candidate in candidates:
            candidate.outcome = evaluate_sample(
                candidate.product, truth, self.evaluation, EVENT_REQUIRED_FIELDS)
        self.n_candidates += len(candidates)
        return candidates

    def stats(self) -> Dict[str, Any]:
        return {"n_calls": self.n_calls, "n_failed_calls": self.n_failed_calls,
                "n_unparsable": self.n_unparsable, "n_candidates": self.n_candidates}




def run_fold_full(runner: Any, fold: Any, prompts: Sequence[Dict[str, Any]],
                  test_prompts: Sequence[Dict[str, Any]], reference_model: Any,
                  fold_dir: Path, checkpoint: Optional[Any] = None) -> Any:
    from mewm.training import qa_augment
    from mewm.training.checkpoint import FoldCheckpoint, save_checkpoint
    from mewm.training.loso import (ClipEngineUnavailable, FoldResult,
                                     SoftNetEngineUnavailable)

    fold.check_disjoint()
    result = FoldResult(fold=fold)
    pool = fold.pool_videos()

    unattributed = [i for i, p in enumerate(prompts) if not p.get("video")]
    if unattributed:
        raise ValueError(
            f"fold {fold.name}: {len(unattributed)} prompt(s) carry no 'video' key "
            f"(first at index {unattributed[:5]}); refusing to start")
    stray = sorted({p["video"] for p in prompts} - pool)
    if stray:
        raise ValueError(
            f"fold {fold.name}: {len(stray)} prompt(s) come from videos outside the "
            f"training pool: {stray[:5]}")

    ckpt = checkpoint or FoldCheckpoint(fold_name=fold.name)
    resumed_past_sft = ckpt.stage != "sft"
    resumed_past_rl = ckpt.stage == "test"

    if runner.config.clip_finetune:
        try:
            result.clip = runner.run_clip(fold)
        except Exception as exc:
            if runner.config.clip_required:
                raise ClipEngineUnavailable(
                    f"fold {fold.name}: stage 0 failed ({exc}). Set "
                    f"training.clip_required = false to continue on the analytic "
                    f"front end.") from exc
            LOGGER.error("fold %s: stage 0 failed (%s); continuing on the analytic "
                        "front end because clip_required is false", fold.name, exc)
            result.clip = {"status": "failed", "error": str(exc)}
    else:
        result.clip = {"status": "disabled (training.clip_finetune = false)"}

    if runner.config.softnet_finetune:
        try:
            result.softnet = runner.run_softnet(fold)
        except Exception as exc:
            if runner.config.softnet_required:
                raise SoftNetEngineUnavailable(
                    f"fold {fold.name}: stage 0b failed ({exc}). Set "
                    f"training.softnet_required = false to continue with the "
                    f"CLIP branch alone.") from exc
            LOGGER.error("fold %s: stage 0b failed (%s); continuing on the CLIP "
                        "branch alone because softnet_required is false", fold.name, exc)
            result.softnet = {"status": "failed", "error": str(exc)}
    else:
        result.softnet = {"status": "disabled (training.softnet_finetune = false)"}

    if runner.calibrate:
        result.calibration = runner._calibrate(fold)

    with_augmented = runner.config.use_augmented_qa and bool(
        qa_augment.load_augmented(runner.dataset, fold.name, pool))

    report = None
    if not resumed_past_sft:
        candidates: List[Any] = []
        max_round_on_disk = -1
        for round_index in range(max(1, runner.config.max_sft_rounds)):
            if (fold_dir / f"sft_{round_index}" / "sft_outcome.json").is_file():
                max_round_on_disk = round_index

        for round_index in range(max(1, runner.config.max_sft_rounds)):
            sft_dir = fold_dir / f"sft_{round_index}"
            outcome_file = sft_dir / "sft_outcome.json"
            already_done = outcome_file.is_file()
            if already_done:
                outcome_dict = json.loads(outcome_file.read_text(encoding="utf-8"))
                LOGGER.info(
                    "fold %s: stage 1 round %d already on disk (stop_reason=%s) -- "
                    "reusing, not retraining", fold.name, round_index,
                    outcome_dict.get("stop_reason"))
            else:
                outcome, _samples = runner.run_sft(fold, round_index, with_augmented)
                outcome_dict = outcome.to_dict()
                LOGGER.info(
                    "fold %s: stage 1 round %d done (%d sample(s), stop_reason=%s)",
                    fold.name, round_index, outcome_dict.get("n_samples"),
                    outcome_dict.get("stop_reason"))
            result.sft_rounds.append(outcome_dict)

            if already_done and round_index < max_round_on_disk:
                LOGGER.info(
                    "fold %s: round %d superseded by round %d already on disk -- "
                    "skipping its gate re-evaluation", fold.name, round_index,
                    max_round_on_disk)
                continue

            report, candidates = runner.gate(fold, round_index,
                                             outcome_dict.get("epoch_losses", []),
                                             prompts)
            result.gate_reports.append(report.to_dict())
            result.decision = report.decision
            LOGGER.info("fold %s: gate round %d -> %s", fold.name, round_index,
                        report.decision)
            if report.decision != "continue_sft":
                break
            LOGGER.info("fold %s round %d: gate says continue_sft (%s)",
                        fold.name, round_index, "; ".join(report.reasons()))
        else:
            LOGGER.warning(
                "fold %s: gate still says continue_sft after %d round(s); proceeding to "
                "stage 2 anyway so the fold terminates", fold.name,
                runner.config.max_sft_rounds)

        if candidates:
            filter_report, written = runner.run_rft(fold, candidates, prompts)
            result.rft = filter_report.to_dict()
            result.augmented = written
            LOGGER.info("fold %s: stage 2 (RFT) accepted %s", fold.name,
                        filter_report.to_dict().get("n_accepted"))
            if written:
                refit_round = len(result.sft_rounds)
                refit_dir = fold_dir / f"sft_{refit_round}"
                refit_outcome_file = refit_dir / "sft_outcome.json"
                if refit_outcome_file.is_file():
                    outcome_dict = json.loads(
                        refit_outcome_file.read_text(encoding="utf-8"))
                    LOGGER.info("fold %s: refit already on disk -- reusing", fold.name)
                else:
                    outcome, _ = runner.run_sft(fold, refit_round, with_augmented=True)
                    outcome_dict = outcome.to_dict()
                    LOGGER.info("fold %s: refit on augmented QA done", fold.name)
                result.sft_rounds.append(outcome_dict)

        ckpt.sft_rounds = result.sft_rounds
        ckpt.gate_reports = result.gate_reports
        ckpt.rft = result.rft
        ckpt.augmented = result.augmented
        ckpt.decision = result.decision
        if result.decision == "rl_ready":
            ckpt.stage = "rl"
        else:
            result.rl_skipped = (
                f"gate decision was {result.decision!r}, not 'rl_ready': "
                + "; ".join(report.reasons() if report else ["no gate report"]))
            ckpt.rl_skipped = result.rl_skipped
            ckpt.stage = "test"
        save_checkpoint(fold_dir, ckpt)
    else:
        result.sft_rounds = list(ckpt.sft_rounds)
        result.gate_reports = list(ckpt.gate_reports)
        result.rft = dict(ckpt.rft)
        result.augmented = dict(ckpt.augmented)
        result.decision = ckpt.decision
        result.rl_skipped = ckpt.rl_skipped
        LOGGER.info("fold %s: stage 1/2 (SFT+gate+RFT) already complete (decision=%s) "
                    "-- resuming further along", fold.name, ckpt.decision)

    if result.decision == "rl_ready":
        rl_adapter_dir = fold_dir / "rl_adapter"
        rl_already_done = (rl_adapter_dir / "adapter_config.json").is_file()
        if not resumed_past_rl and not rl_already_done:
            if reference_model is not None:
                runner.policy_backend.reference_model = reference_model
                LOGGER.info("fold %s: KL anchor attached (frozen pretrained base, CPU)",
                            fold.name)
            else:
                LOGGER.warning("fold %s: no KL anchor -- stage 3 runs unregularised "
                              "(--kl-anchor none)", fold.name)
            LOGGER.info("fold %s: starting stage 3 (GRPO, %d step(s))", fold.name,
                        runner.config.rl_total_steps)
            result.rl_history = runner.run_rl(fold, prompts)
            LOGGER.info("fold %s: stage 3 done, %d step(s)", fold.name,
                        len(result.rl_history))
            saved = runner.policy_backend.save(rl_adapter_dir)
            LOGGER.info("fold %s: post-RL adapter -> %s", fold.name, saved)
            ckpt.stage = "test"
            save_checkpoint(fold_dir, ckpt)
        else:
            if not resumed_past_rl:
                LOGGER.info(
                    "fold %s: stage 3 (RL) adapter already on disk from a prior "
                    "attempt (checkpoint.json did not record it) -- reusing, not "
                    "retraining", fold.name)
                ckpt.stage = "test"
                save_checkpoint(fold_dir, ckpt)
            history_file = fold_dir / "rl_history.json"
            if history_file.is_file():
                result.rl_history = json.loads(history_file.read_text(encoding="utf-8"))
            LOGGER.info("fold %s: stage 3 (RL) already complete (%d step(s)) -- "
                        "resuming at stage 4", fold.name, len(result.rl_history))
    elif not resumed_past_sft:
        LOGGER.warning("fold %s: skipping stage 3 -- %s", fold.name, result.rl_skipped)
    else:
        LOGGER.info("fold %s: stage 3 was already skipped in a prior attempt -- %s",
                    fold.name, result.rl_skipped)

    LOGGER.info("fold %s: stage 4 (held-out test-time augmentation scoring)",
                fold.name)
    result.test = runner.run_test(fold, test_prompts)
    return result




def merge_adapter(spec: Any, adapter_dir: Path, merged_dir: Path) -> Path:
    import torch
    from peft import PeftModel

    loaded = _load_fresh_base(spec, quantization="none", device_map="cpu")
    model = PeftModel.from_pretrained(loaded.model, str(adapter_dir))
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    loaded.processor.save_pretrained(merged_dir)
    LOGGER.info("merged checkpoint -> %s", merged_dir)
    return merged_dir


def _weights_env_var(policy: str) -> str:
    from mewm.llm.registry import resolve

    spec = resolve(policy)
    return ("MEWM_LOCAL_WEIGHTS_" + spec.model_id.upper()
            .replace("-", "_").replace(".", "_"))


def run_inference(policy: str, jobs: Sequence[tuple], test_runs: Path,
                  workers: int, dataset: str) -> None:
    from mewm.data.paths import max_proposals_of

    env_var = _weights_env_var(policy)
    max_proposals = max_proposals_of(dataset)

    def _one(job: tuple) -> str:
        video_id, merged_dir = job
        env = dict(os.environ)
        env[env_var] = str(merged_dir)
        subprocess.run(
            [sys.executable, "-m", "mewm.cli.main", "run",
             "--dataset", dataset, "--video", video_id,
             "--model", policy, "--backend", "local",
             "--output", str(test_runs),
             "--max-proposals", str(max_proposals)],
            cwd=REPO_ROOT, env=env, check=True)
        return video_id

    LOGGER.info("inference: %d held-out video(s), %d worker(s)", len(jobs), workers)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_one, job): job for job in jobs}
        for future in as_completed(futures):
            LOGGER.info("inference done: %s", future.result())


def final_metrics(dataset: str, test_runs: Path, run_id: str) -> dict:
    qa_file = REPO_ROOT / "Q-T-A" / dataset / f"{dataset}_me_lvqa_gpt-5-6-sol.jsonl"
    cmd = [sys.executable, "-m", "mewm.cli.main", "final-metrics",
           "--dataset", dataset, "--runs", str(test_runs),
           "--run-id", run_id, "--protocol", "loso", "--mode", "api"]
    if qa_file.is_file():
        cmd += ["--qa-file", str(qa_file)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"final-metrics failed:\n{proc.stderr}")
    summary_path = REPO_ROOT / "runs" / run_id / "final_metrics_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _u1(block: dict, key: str) -> str:
    value = block.get(key)
    return "n/a" if value is None else f"{value:.4f}"


def print_headlines(policy: str, dataset: str, summary: dict) -> None:
    overall = summary.get("overall", {})
    strict = overall.get("spotting", {}).get("interval", {}).get("strict_iou", {})
    unweighted = overall.get("spotting", {}).get("interval", {}).get("unweighted_type", {})
    counting = overall.get("spotting", {}).get("counting", {})
    au = overall.get("recognition", {}).get("action_units", {})
    emo = overall.get("recognition", {}).get("emotion", {})
    strs_block = overall.get("strs", {}).get("score", {})
    runtime = overall.get("runtime", {})

    print("\n" + "=" * 66)
    print(f"LOSO SFT->RFT->GRPO final results ({policy}, {dataset}, all subjects)")
    print("=" * 66)
    print(f"  Localisation F1 (IoU>0.5) {_u1(strict, 'f1')}   "
          f"(TP {strict.get('tp')} / FP {strict.get('fp')} / FN {strict.get('fn')})")
    print(f"  SpotUF1 / SpotUAR         {_u1(unweighted, 'spot_uf1')} / "
          f"{_u1(unweighted, 'spot_uar')}")
    for quantity in ("expression", "micro", "macro"):
        block = counting.get(quantity, {})
        if isinstance(block, dict) and block.get("status") == "ok":
            print(f"  MAE / RMSE ({quantity:10s})     {_u1(block, 'mae')} / "
                  f"{_u1(block, 'rmse')}")
    print(f"  F1_AU / Jaccard_AU        {_u1(au, 'f1_au')} / {_u1(au, 'jaccard_au')}")
    fine = emo.get("fine", {}).get("megc", {})
    coarse = emo.get("coarse", {}).get("megc", {})
    print(f"  RegUF1 / RegUAR (fine)    {_u1(fine, 'reg_uf1')} / {_u1(fine, 'reg_uar')}")
    print(f"  RegUF1 / RegUAR (coarse)  {_u1(coarse, 'reg_uf1')} / "
          f"{_u1(coarse, 'reg_uar')}")
    print(f"  STRS = F1_s x F1_a        {_u1(strs_block, 'strs')}   "
          f"(F1_s {_u1(strs_block, 'f1_spot')} x F1_a {_u1(strs_block, 'f1_analysis')})")
    print(f"  (n_videos {overall.get('n_videos')} / n_subjects "
          f"{overall.get('n_subjects')} / n_events {overall.get('n_events')})")
    if runtime:
        print(f"  total wall time           {runtime.get('total_wall_hms', 'n/a')}")
        print(f"  throughput                "
              f"{runtime.get('throughput_videos_per_hour', 'n/a')} video/h")
        per_video = runtime.get("per_video_seconds", {})
        print(f"  per-video time (mean/med) "
              f"{per_video.get('mean', 'n/a')}s / {per_video.get('median', 'n/a')}s")
    print("=" * 66)




def _reference_qa_rows(dataset: str) -> List[Dict[str, Any]]:
    from mewm.cli.main import _reference_qa
    from mewm.data.paths import qa_dir

    source = _reference_qa(dataset)
    if source is None or not Path(source).is_file():
        raise FileNotFoundError(
            f"{dataset}: no reference QA jsonl found under {qa_dir(dataset)}. Build "
            f"one first (see `mewm build-instructions` / `mewm augment-qa`).")
    rows = []
    with open(source, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    LOGGER.info("reference QA: %s (%d instruction(s))", source, len(rows))
    return rows


def build_global_prompts(dataset: str, index: Any, config: Any, loaded: Any,
                         spec: Any, max_frames: int, stride: int) -> List[Dict[str, Any]]:
    from mewm.training.qa_sweep import perceive
    from mewm.training.rl_prompts import KIND_EVENT_REASONING, build_rl_prompts

    def progress(key: str, position: int, total: int) -> None:
        if position % 10 == 0 or position == total:
            print(f"  [perceive] {position}/{total} {key}", flush=True)

    evidence, summary = perceive(index.videos, config, max_frames=max_frames,
                                 stride=stride, progress=progress)
    LOGGER.info("perception: %s", summary)

    rows = _reference_qa_rows(dataset)
    prompts, ledger = build_rl_prompts(dataset, index.videos, rows, evidence,
                                      include_kinds=(KIND_EVENT_REASONING,))
    if not prompts:
        raise RuntimeError(
            f"{dataset}: 0 event-anchored prompts were built from {len(rows)} "
            f"reference QA row(s); nothing to train the gate/RFT/GRPO stages on. "
            f"Check the QA build matches this dataset's annotated events.")
    LOGGER.info("%s: %d event-anchored prompt(s) admitted (of %d reference row(s))",
                dataset, len(prompts), len(rows))

    for entry in prompts:
        entry["prompt"] = _render_prompt(entry, loaded, spec)
    return prompts




def _clear_fold_artifacts(fold_dir: Path) -> None:
    import shutil

    from mewm.training.checkpoint import checkpoint_path

    if not fold_dir.is_dir():
        return
    removed = []
    ckpt_file = checkpoint_path(fold_dir)
    if ckpt_file.is_file():
        ckpt_file.unlink()
        removed.append(ckpt_file.name)
    for pattern in ("sft_*", "rl_adapter", "rl_history.json", "final_adapter", "merged"):
        for path in fold_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)
    if removed:
        LOGGER.info("fold %s: --force, cleared %d stale artefact(s): %s",
                    fold_dir.name, len(removed), ", ".join(sorted(removed)))


def run_one_subject(subject: str, args: argparse.Namespace, config: Any, index: Any,
                    spec: Any, lora: Any, all_prompts: List[Dict[str, Any]],
                    reference_model: Any) -> tuple:
    from mewm.training.checkpoint import (clear_checkpoint, load_checkpoint,
                                          resume_adapter_dir)
    from mewm.training.grpo import LoRABackend
    from mewm.training.loso import LOSORunner, build_folds
    from mewm.training.sft import LoRASFTBackend

    dataset = args.dataset
    output_root = Path(args.output)
    fold_dir = output_root / dataset / f"fold_{subject}"
    result_path = fold_dir / "fold_result.json"
    if result_path.is_file() and not args.force:
        LOGGER.info("fold %s: fold_result.json already present, skipping (--force to "
                    "redo)", subject)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        merged_dir = fold_dir / "merged"
        final_dir = fold_dir / "final_adapter"
        if not (merged_dir / "config.json").is_file():
            if final_dir.is_dir():
                LOGGER.info("fold %s: merged/ was freed after a previous "
                           "inference pass -- rebuilding it from final_adapter/ "
                           "so Stage 5 can run again", subject)
                merge_adapter(spec, final_dir, merged_dir)
            else:
                LOGGER.warning("fold %s: neither merged/ nor final_adapter/ "
                              "present -- cannot run inference for this "
                              "already-trained fold", subject)
                return payload, None
        return payload, merged_dir

    checkpoint = None
    resume_dir = None
    if args.force:
        _clear_fold_artifacts(fold_dir)
    elif not getattr(args, "no_resume", False):
        checkpoint = load_checkpoint(fold_dir)
        resume_dir = resume_adapter_dir(fold_dir)
        if checkpoint is not None or resume_dir is not None:
            LOGGER.info("fold %s: resuming a previous attempt (stage=%s, adapter=%s)",
                       subject, checkpoint.stage if checkpoint else "sft", resume_dir)

    fold = build_folds(index, config.training, subjects=[subject])[0]
    pool = fold.pool_videos()
    test_videos = set(fold.test_videos)
    fold_prompts = [p for p in all_prompts if p.get("video") in pool]
    fold_test_prompts = [p for p in all_prompts if p.get("video") in test_videos]
    LOGGER.info("fold %s: %d train/val video(s), %d test video(s), %d pool prompt(s), "
               "%d test prompt(s)", subject, len(fold.train_videos) + len(fold.val_videos),
               len(fold.test_videos), len(fold_prompts), len(fold_test_prompts))

    loaded = _load_fresh_base(spec, quantization=lora.quantization, device_map="auto")
    if resume_dir is not None:
        model = _load_resumed_adapter(loaded.model, resume_dir)
    else:
        model = _apply_lora(loaded.model, lora)
    tokenizer = loaded.processor.tokenizer

    sft_backend = LoRASFTBackend(model=model, tokenizer=tokenizer,
                                 max_length=lora.max_length,
                                 prop_weight=config.training.sft_prop_weight)
    policy_backend = LoRABackend(model=model, tokenizer=tokenizer,
                                 reference_model=None, device=args.device)
    sampler = LocalPolicySampler(policy_backend, config.evaluation, config.reward,
                                 temperature=config.training.rl_temperature)

    runner = LOSORunner(dataset, sft_backend, policy_backend=policy_backend,
                        sampler=sampler, evaluator=None, mewm_config=config,
                        index=index, output_root=output_root,
                        calibrate=args.calibrate, clip_stride=args.clip_stride,
                        clip_max_frames=args.clip_max_frames, device=args.device)

    started = time.time()
    result = run_fold_full(runner, fold, fold_prompts, fold_test_prompts,
                           reference_model if args.kl_anchor != "none" else None,
                           fold_dir=fold_dir, checkpoint=checkpoint)
    elapsed = time.time() - started
    LOGGER.info("fold %s: training done in %.0fs (sampler stats: %s)", subject,
               elapsed, sampler.stats())

    final_dir = fold_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    LOGGER.info("fold %s: final adapter (post SFT+RFT+GRPO) -> %s", subject, final_dir)

    payload = {"summary": {"decision": result.decision, "elapsed_s": round(elapsed, 1)},
              "folds": [result.to_dict()]}
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    clear_checkpoint(fold_dir)

    from mewm.llm.local_models import unload
    unload(spec.model_id)
    import gc
    del model, sft_backend, policy_backend, loaded
    gc.collect()

    merged_dir = fold_dir / "merged"
    if not (merged_dir / "config.json").is_file():
        merge_adapter(spec, final_dir, merged_dir)
    else:
        LOGGER.info("fold %s: merged checkpoint present, skipping merge", subject)

    return payload["folds"][0], merged_dir




def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["casme_sq", "samm"])
    parser.add_argument("--policy", required=True,
                        choices=["Qwen2.5-VL-7B", "Qwen3-VL-8B", "GLM-4.1V-9B-Thinking"])
    parser.add_argument("--folds", default="",
                        help="comma-separated subjects (default: every subject, "
                             "full LOSO)")
    parser.add_argument("--output", default="",
                        help="default: runs/loso/<dataset>/<policy>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="")
    parser.add_argument("--calibrate", action="store_true",
                        help="fit (tau_hi, tau_lo) per fold instead of the config "
                             "defaults (expensive: reruns representation+spotting "
                             "over the training pool)")
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--clip-max-frames", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="perception frame cap per video, 0 = no cap")
    parser.add_argument("--stride", type=int, default=1,
                        help="perception frame stride")
    parser.add_argument("--kl-anchor", default="cpu", choices=["cpu", "gpu", "none"],
                        help="GRPO KL anchor placement (default: cpu -- see the "
                             "module docstring for why a 32GB card cannot hold two "
                             "live copies)")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--inference-workers", type=int, default=4)
    parser.add_argument("--skip-inference", action="store_true",
                        help="stop after training + merge; skip the production "
                             "pipeline pass over held-out videos")
    parser.add_argument("--skip-final-metrics", action="store_true")
    parser.add_argument("--run-id", default="",
                        help="default: <dataset>_<policy>_loso")
    parser.add_argument("--force", action="store_true",
                        help="redo folds whose fold_result.json already exists, "
                             "clearing any partial checkpoint/adapter first")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any on-disk checkpoint/partial adapter and "
                             "start every requested fold from a fresh LoRA, without "
                             "clearing its directory first (unlike --force, this "
                             "still leaves old sft_N/rl_adapter artefacts on disk, "
                             "just unread -- mainly for debugging the resume path "
                             "itself)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan (folds, memory budget, output layout); "
                             "load nothing, train nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    os.chdir(REPO_ROOT)

    from mewm.config import SFTLoraConfig, load_config
    from mewm.data.datasets import load_dataset
    from mewm.llm.registry import resolve
    from mewm.training.loso import PolicyNotTrainable, require_open_weight

    config = load_config(args.config or None)
    config.training.policy_model = args.policy
    try:
        require_open_weight(config.training.policy_model)
    except PolicyNotTrainable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    spec = resolve(args.policy)
    lora = SFTLoraConfig.for_policy(args.policy, dataset=args.dataset)

    output = Path(args.output) if args.output else REPO_ROOT / "runs" / "loso" / args.dataset / args.policy
    run_id = args.run_id or f"{args.dataset}_{args.policy}_loso"
    test_runs = output / "test_runs"

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    subjects = ([s.strip() for s in args.folds.split(",") if s.strip()]
               if args.folds else sorted(index.subjects()))

    if args.dry_run:
        per_fold = len(index.videos) // max(1, len(index.subjects()))
        print(f"dataset          {args.dataset} ({len(index.videos)} video(s), "
              f"{len(index.subjects())} subject(s))")
        print(f"policy           {args.policy} ({lora.quantization}, LoRA "
              f"r{lora.r}/a{lora.alpha}, {lora.target_modules}, bs {lora.batch_size}, "
              f"lr {lora.learning_rate}, {lora.epochs} epoch(s)/round)")
        print(f"folds            {len(subjects)} subject(s) (LOSO): {subjects}")
        print(f"stages           SFT -> gate -> RFT(+refit) -> GRPO "
              f"({config.training.rl_total_steps} step(s)) -> held-out TTA -> "
              f"merge -> production inference")
        print(f"KL anchor        {args.kl_anchor}")
        print(f"memory budget    bf16 base ~18GB + LoRA r{lora.r} "
              f"(~1-1.5% params) + AdamW + activations, ~20-23GB VRAM"
              + (" + ~18GB CPU RAM (frozen reference)"
                 if args.kl_anchor == "cpu" else ""))
        print(f"inference        ~{per_fold} held-out video(s)/fold, "
              f"{args.inference_workers} parallel worker(s)")
        print(f"outputs          {output}/fold_<subject>/{{fold_result.json,"
              f"final_adapter,merged (freed once that fold's inference is done)}} + "
              f"{test_runs} + runs/{run_id}/final_metrics_summary.json")
        return 0

    from mewm.llm.local_models import load_local_model

    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("loading base %s once to build the shared RL prompt set...",
               args.policy)
    loaded_for_prompts = load_local_model(spec, quantization=lora.quantization)
    all_prompts = build_global_prompts(args.dataset, index, config,
                                       loaded_for_prompts, spec,
                                       max_frames=args.max_frames, stride=args.stride)
    from mewm.llm.local_models import unload
    unload(spec.model_id)
    del loaded_for_prompts
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    reference_model = (_load_frozen_reference(spec) if args.kl_anchor != "none"
                       else None)
    if args.kl_anchor == "gpu" and reference_model is not None:
        reference_model = reference_model.to(args.device)

    fold_results: List[Dict[str, Any]] = []
    summary_path = output / f"{args.dataset}_loso.json"
    for subject in subjects:
        LOGGER.info("=== fold %s (%d/%d) ===", subject, subjects.index(subject) + 1,
                   len(subjects))
        result_dict, merged_dir = run_one_subject(
            subject, args, config, index, spec, lora, all_prompts, reference_model)
        fold_results.append(result_dict)
        summary_path.write_text(
            json.dumps({"folds": fold_results}, ensure_ascii=False, indent=1),
            encoding="utf-8")

        if merged_dir is not None and not args.skip_inference:
            test_videos = [v.video_key for v in index.videos if str(v.subject) == subject]
            jobs = [(video_id, merged_dir) for video_id in test_videos]
            run_inference(args.policy, jobs, test_runs, args.inference_workers,
                          dataset=args.dataset)
            all_done = all((test_runs / f"{args.dataset}_{v}" / "summary.json").is_file()
                           for v in test_videos)
            if all_done:
                import shutil
                shutil.rmtree(merged_dir, ignore_errors=True)
                LOGGER.info("fold %s: inference complete for all %d held-out "
                           "video(s), freed %s", subject, len(test_videos), merged_dir)
            else:
                LOGGER.warning(
                    "fold %s: not every held-out video has a summary.json yet -- "
                    "keeping %s on disk so a later run of this launcher can retry "
                    "the missing one(s) (run_one_subject sees this subject's "
                    "fold_result.json and hands the same merged_dir back without "
                    "retraining)", subject, merged_dir)

    print(f"per-fold results written -> {summary_path}")

    if args.skip_inference:
        print("--skip-inference: stopping after training + merge.")
        return 0

    if args.skip_final_metrics:
        print("--skip-final-metrics: stopping after inference.")
        return 0

    summary = final_metrics(args.dataset, test_runs, run_id)
    print_headlines(args.policy, args.dataset, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
