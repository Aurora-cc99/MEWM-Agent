"""CLI entry point: run, calibrate, and models sub-commands."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import RUNS_ROOT, load_config, run_dir
from ..data.datasets import DatasetIndex, LongVideo, load_dataset
from ..data.paths import DATASETS, available_datasets, find_qa_runs
from ..data.qa_loader import load_qa_set

_FEATURE_ROOT_REL = "softnet_features"
_CHECKPOINT_ROOT_REL = "softnet_spotter"

LOGGER = logging.getLogger("mewm")


def _setup_logging(verbose: bool, extra_log_dir: Optional[Path] = None) -> None:

    from ..config import RUNS_ROOT

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    for directory in {RUNS_ROOT, extra_log_dir} - {None}:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(directory / "framework.log",
                                                 encoding="utf-8"))
        except OSError as exc:
            print(f"warning: could not open framework.log under {directory}: {exc}",
                  file=sys.stderr)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _select_videos(index: DatasetIndex, subject: str = "", video: str = "",
                   limit: int = 0) -> List[LongVideo]:
    videos = list(index.videos)
    if subject:
        videos = [v for v in videos if v.subject == subject]
    if video:
        videos = [v for v in videos if v.video_key == video]
    if limit:
        videos = videos[:limit]
    return videos


def _dump(payload: Any, path: Optional[Path] = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"written: {path}")
    else:
        print(text)


def cmd_datasets(args: argparse.Namespace) -> int:
    report: Dict[str, Any] = {}
    for name in DATASETS:
        entry: Dict[str, Any] = {}
        try:
            index = load_dataset(name, limit_videos=args.limit)
            entry["annotations"] = index.stats()
        except Exception as exc:
            entry["annotations"] = {"error": str(exc)}

        runs = find_qa_runs(name)
        qa = load_qa_set(name, with_full=True) if runs else None
        entry["qa"] = qa.stats() if qa else "pending (no QA build yet)"
        entry["qa_runs"] = [r.name for r in runs]

        try:
            index = load_dataset(name, limit_videos=1)
            if index.videos:
                entry["flow_coverage"] = index.videos[0].paths.coverage()
        except Exception as exc:
            entry["flow_coverage"] = {"error": str(exc)}
        report[name] = entry

    _dump(report, Path(args.output) if args.output else None)
    return 0


def cmd_spot(args: argparse.Namespace) -> int:
    from ..engines.m2_spotting import alignment_auc
    from ..eval.metrics import evaluate_proposals
    from ..pipeline import apply_clip_activations, run_representation, run_spotting

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    if not videos:
        print(f"no videos matched in {args.dataset}", file=sys.stderr)
        return 1

    spotter_cache: Dict[str, Any] = {}
    results = []
    for video in videos:
        LOGGER.info("spotting %s (%d frames)", video.video_id, video.n_frames)
        representation = run_representation(video, config, max_frames=args.max_frames,
                                            stride=args.stride)
        if not len(representation):
            LOGGER.warning("%s: no usable frames", video.video_id)
            continue
        clip_spotter = _clip_spotter_for(video, args.dataset, config,
                                         getattr(args, "clip_localiser", "auto"),
                                         spotter_cache)
        external_curve = None
        if clip_spotter is not None:
            clip_out = clip_spotter.infer(video, representation)
            external_curve = clip_out["curve"]
            if config.clip.replace_activations:
                apply_clip_activations(representation, clip_out["activations"])
        spotting = run_spotting(video, representation, config,
                                external_curve=external_curve)

        truths = [e.interval for e in video.micro_events()]
        record = spotting.error_record
        auc = alignment_auc(
            record.s_curve,
            [(a - record.t_start, b - record.t_start) for a, b in truths],
        ) if truths else 0.0
        metrics = evaluate_proposals(
            [(p.t_on, p.t_off) for p in spotting.proposals], truths,
            iou_threshold=(args.iou_threshold if args.iou_threshold is not None
                           else config.evaluation.iou_threshold),
            affective_rescue=config.evaluation.affective_rescue)

        results.append({
            "video": video.video_id,
            "frames_processed": len(representation),
            "localiser": "clip_supervised" if external_curve is not None else "analytic",
            "spotting": spotting.summary(),
            "ground_truth": [list(t) for t in truths],
            "proposals": [[p.t_on, p.t_off, p.apex, p.peak_S] for p in spotting.proposals],
            "alignment_auc": auc,
            "proposal_metrics": metrics.to_dict(),
        })
        print(f"  {video.video_id}: {len(spotting.proposals)} proposals, "
              f"AUC {auc}, F1 {metrics.f1}")

    _dump({"dataset": args.dataset, "n_videos": len(results), "results": results},
          Path(args.output) if args.output else None)
    return 0


def _apply_model_overrides(config, args: argparse.Namespace) -> List[str]:
    from ..llm.registry import UnknownModelError, heterogeneous, resolve

    warnings: List[str] = []
    shorthand = getattr(args, "model", "")
    if shorthand:
        for field_name in ("reasoning_model", "perception_model", "structure_model",
                           "critic_model"):
            setattr(config.llm, field_name, shorthand)
    for flag, field_name in (
        ("reasoning_model", "reasoning_model"), ("perception_model", "perception_model"),
        ("structure_model", "structure_model"), ("critic_model", "critic_model"),
        ("reasoning_effort", "reasoning_effort"), ("critic_effort", "critic_effort"),
    ):
        value = getattr(args, flag, "")
        if value:
            setattr(config.llm, field_name, value)

    for role, model in (("reasoning", config.llm.reasoning_model),
                        ("perception", config.llm.perception_model),
                        ("structure", config.llm.structure_model),
                        ("critic", config.llm.critic_model)):
        try:
            resolve(model)
        except UnknownModelError as exc:
            warnings.append(f"{role}: {exc}")

    if not heterogeneous(config.llm.critic_model, config.llm.reasoning_model):
        warnings.append(
            f"critic ({config.llm.critic_model}) and reasoner "
            f"({config.llm.reasoning_model}) share a base or provider family. "
            "results should be reported as unverified."
        )
    return warnings


def _apply_backend_overrides(config, args: argparse.Namespace) -> Optional[str]:

    from ..llm.client import BackendMismatch, backend_for

    shorthand = getattr(args, "backend", "")
    if shorthand:
        for field_name in ("reasoning_backend", "perception_backend",
                           "structure_backend", "critic_backend"):
            setattr(config.llm, field_name, shorthand)
    for flag in ("reasoning_backend", "perception_backend",
                 "structure_backend", "critic_backend"):
        value = getattr(args, flag, "")
        if value:
            setattr(config.llm, flag, value)
    if getattr(args, "fallback_to_api", False):
        config.backends.fallback_to_api = True

    for role, model, backend in (
        ("reasoning", config.llm.reasoning_model, config.llm.reasoning_backend),
        ("perception", config.llm.perception_model, config.llm.perception_backend),
        ("structure", config.llm.structure_model, config.llm.structure_backend),
        ("critic", config.llm.critic_model, config.llm.critic_backend),
    ):
        try:
            backend_for(model, backend)
        except BackendMismatch as exc:
            return f"{role}: {exc}"
        except Exception:
            continue
    return None


def _clip_spotter_for(video, dataset: str, config, choice: str,
                      cache: Dict[str, Any]) -> Optional[Any]:

    if choice == "off" or not config.clip.enabled:
        return None
    from ..training.clip_localiser import TrainedCLIPSpotter, find_checkpoint

    if choice == "auto":
        path = find_checkpoint(dataset, str(video.subject), config.clip)
        if path is None:
            return None
    else:
        path = Path(choice)
        if not path.is_file():
            raise FileNotFoundError(f"CLIP localiser checkpoint not found: {path}")
    key = str(path)
    if key not in cache:
        LOGGER.info("loading CLIP localiser checkpoint %s on %s", path,
                    config.clip.device)
        cache[key] = TrainedCLIPSpotter.from_path(path, config=config.clip,
                                                 device=config.clip.device)
    return cache[key]


def cmd_run(args: argparse.Namespace) -> int:
    from ..llm.client import StubClient, backend_manifest, reset_backend_manifest
    from ..llm.registry import credentials_available

    config = load_config(args.config)
    warnings = _apply_model_overrides(config, args)
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)
    backend_error = _apply_backend_overrides(config, args)
    if backend_error:
        print(f"backend mismatch: {backend_error}", file=sys.stderr)
        return 2

    client = None
    if args.stub:
        client = StubClient({"default": {}})
        LOGGER.info("using the deterministic stub client (no network)")
    else:
        missing = [
            f"{role}={model}"
            for role, model in (("R", config.llm.reasoning_model),
                                ("P", config.llm.perception_model),
                                ("A", config.llm.structure_model),
                                ("C", config.llm.critic_model))
            if not credentials_available(model)
        ]
        if missing:
            print("no credentials for: " + ", ".join(missing)
                  + "\nSet the provider variables (see .env.example), pick different "
                    "models with --reasoning-model / --critic-model, or pass --stub.",
                  file=sys.stderr)
            return 2

    from ..pipeline import MEWMPipeline

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    if not videos:
        print(f"no videos matched in {args.dataset}", file=sys.stderr)
        return 1

    qa = load_qa_set(args.dataset) if not args.no_qa else None
    spotter_cache: Dict[str, Any] = {}
    outputs = []
    for video in videos:
        question = ""
        if qa is not None:
            item = qa.triple_task(video.video_key)
            if item is not None:
                question = item.question
        clip_spotter = _clip_spotter_for(video, args.dataset, config,
                                         args.clip_localiser, spotter_cache)
        if clip_spotter is None and args.clip_localiser == "auto":
            LOGGER.warning(
                "%s:",
                video.video_id, args.dataset, video.subject, args.dataset)
        target = run_dir(video.video_id, Path(args.output) if args.output else None)
        if (target / "summary.json").is_file() and not getattr(args, "force", False):
            print(f"  {video.video_id}: already complete; skipping (pass --force to redo)")
            continue
        reset_backend_manifest()
        pipeline = MEWMPipeline.build(config=config, client=client, lang=args.lang,
                                      checkpoint_dir=target,
                                      clip_spotter=clip_spotter)
        softnet_proposals = _softnet_proposals_for(
            video, args.dataset, args.softnet_spotter)
        result = pipeline.run(
            video, question=question, max_frames=args.max_frames,
            stride=args.stride, max_proposals=args.max_proposals,
            use_annotated_proposals=args.use_gt_proposals,
            softnet_proposals=softnet_proposals or None,
            iou_threshold=args.iou_threshold,
        )

        answer = result.answer()
        _dump(answer, target / "answer.json")
        (target / "answer.txt").write_text(answer["answer"], encoding="utf-8")
        summary = result.summary()
        summary["localiser"] = ("clip_supervised" if clip_spotter is not None
                                else "analytic")
        manifest = backend_manifest(reset=True)
        summary["backend_manifest_counts"] = _manifest_counts(manifest)
        _dump({"video_id": video.video_id, "calls": manifest},
              target / "backend_manifest.json")
        _dump(summary, target / "summary.json")
        result.episodic.save(target / "episodic.json")
        _dump(result.state.to_dict(), target / "state.json")
        outputs.append(summary)
        print(f"  {video.video_id}: {len(result.state.proposals)} proposals, "
              f"{result.state.budget.llm_calls_used} LLM calls, "
              f"{len(result.state.budget.degradations)} degradations")
        print(f"    answer -> {target / 'answer.txt'}")
        if args.print_answer:
            print()
            print(answer["answer"])
            print()

    _dump({"dataset": args.dataset, "runs": outputs})
    return 0


def _softnet_proposals_for(video: LongVideo, dataset: str, choice: str):
    if choice == "off":
        return []
    from ..engines.softnet_spotter import (SoftNetCheckpoint, SoftNetFeatureCache,
                                           SoftNetSpotter, extract_flow_tensor)
    from ..engines.v1_motion import MotionFrontEnd

    if choice == "auto":
        path = RUNS_ROOT / _CHECKPOINT_ROOT_REL / dataset / f"fold_{video.subject}" / "softnet_spotter.npz"
        if not path.is_file():
            return []
    else:
        path = Path(choice)
        if not path.is_file():
            raise FileNotFoundError(f"softnet checkpoint not found: {path}")
    feature_path = RUNS_ROOT / _FEATURE_ROOT_REL / dataset / f"{video.video_key}.npz"
    if feature_path.is_file():
        cache = SoftNetFeatureCache.load(feature_path)
    else:
        pairs = video.paths.aligned_pairs()
        landmarks = _first_frame_landmarks(video)
        tensors, frames = [], []
        for pair in pairs:
            flow = MotionFrontEnd.load_flow(
                MotionFrontEnd.prefer_raw_flow(pair.flow_path))
            if flow is None:
                continue
            tensors.append(extract_flow_tensor(flow, landmarks))
            frames.append(pair.t)
        if not tensors:
            LOGGER.warning("%s: no decodable flow for softnet features", video.video_id)
            return []
        cache = SoftNetFeatureCache(video=video.video_key,
                                    frames=np.asarray(frames, dtype=np.int64),
                                    tensors=np.asarray(tensors, dtype=np.float16),
                                    k=video.flow_gap)
        cache.save(feature_path)
    spotter = SoftNetSpotter(SoftNetCheckpoint.load(path))
    return spotter.propose(cache, frame_offset=int(cache.frames[0]))


def _manifest_counts(manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    fallbacks = 0
    for entry in manifest:
        key = f"{entry.get('role')}:{entry.get('model')}:{entry.get('backend')}"
        counts[key] = counts.get(key, 0) + 1
        fallbacks += int(bool(entry.get("fallback")))
    return {"calls": counts, "fallbacks": fallbacks}


def _first_frame_landmarks(video: LongVideo):
    from ..engines.v1_motion import detect_landmarks
    first = video.paths.frame(video.frame_lo)
    if not first.is_file():
        for index in range(video.frame_lo, min(video.frame_hi + 1, video.frame_lo + 30)):
            candidate = video.paths.frame(index)
            if candidate.is_file():
                first = candidate
                break
    if not first.is_file():
        return None
    try:
        return detect_landmarks(first)
    except Exception:
        return None


def cmd_build_softnet_features(args: argparse.Namespace) -> int:

    from ..data.datasets import load_dataset
    from ..engines.softnet_spotter import SoftNetFeatureCache, extract_flow_tensor
    from ..engines.v1_motion import MotionFrontEnd

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    root = Path(args.output) if args.output else RUNS_ROOT
    cache_root = root / _FEATURE_ROOT_REL / args.dataset
    for video in videos:
        pairs = video.paths.aligned_pairs()
        if not pairs:
            print(f"  {video.video_id}: no flow pairs; skipping")
            continue
        landmarks = _first_frame_landmarks(video)
        tensors, frames = [], []
        for pair in pairs:
            flow = MotionFrontEnd.load_flow(
                MotionFrontEnd.prefer_raw_flow(pair.flow_path))
            if flow is None:
                continue
            tensors.append(extract_flow_tensor(flow, landmarks))
            frames.append(pair.t)
        if not tensors:
            print(f"  {video.video_id}: no decodable flow; skipping")
            continue
        cache = SoftNetFeatureCache(video=video.video_key,
                                    frames=np.asarray(frames, dtype=np.int64),
                                    tensors=np.asarray(tensors, dtype=np.float16),
                                    k=video.flow_gap)
        target = cache_root / f"{video.video_key}.npz"
        cache.save(target)
        print(f"  {video.video_id}: {len(tensors)} frames -> {target}")
    return 0


def cmd_train_softnet(args: argparse.Namespace) -> int:
    from ..data.datasets import load_dataset
    from ..data.paths import DATASET_FPS
    from ..engines.softnet_spotter import (SoftNetFeatureCache, SoftNetCheckpoint,
                                           train_fold)

    index = load_dataset(args.dataset)
    subject = args.subject or args.video
    if not subject:
        print("--subject (the held-out subject) is required", file=sys.stderr)
        return 2
    pool = [v for v in index.videos if v.subject != subject]
    if not pool:
        print(f"no training pool for held-out subject {subject}", file=sys.stderr)
        return 1
    root = Path(args.output) if args.output else RUNS_ROOT
    feature_root = root / _FEATURE_ROOT_REL / args.dataset
    caches, intervals, subjects_by_video = [], {}, {}
    for video in pool:
        path = feature_root / f"{video.video_key}.npz"
        if not path.is_file():
            print(f"  missing features for {video.video_id}; run "
                  f"build-softnet-features first")
            continue
        cache = SoftNetFeatureCache.load(path)
        caches.append(cache)
        intervals[video.video_key] = [e.interval for e in video.events
                                      if e.is_micro]
        subjects_by_video[video.video_key] = str(video.subject)
    if not caches:
        print("no feature caches loaded", file=sys.stderr)
        return 1
    durations = [e.duration for video in pool for e in video.events if e.is_micro]
    k = max(2, int(round(float(np.mean(durations)) / 2))) if durations else 7
    print(f"  fold {subject}: pool {len(caches)} videos, k={k}")
    checkpoint = train_fold(caches, intervals, fold_name=subject, k=k,
                            epochs=args.epochs, device=args.device,
                            subjects_by_video=subjects_by_video)
    target = root / _CHECKPOINT_ROOT_REL / args.dataset / f"fold_{subject}" / "softnet_spotter.npz"
    checkpoint.save(target)
    print(f"  checkpoint -> {target}")
    return 0


def cmd_qa_interrogate(args: argparse.Namespace) -> int:
    import json as _json

    from ..data.datasets import load_dataset
    from ..eval.subject_report import write_qa_records
    from ..llm.client import call_model, credentials_available
    from ..pipeline import _merge_proposal_sets, apply_clip_activations, run_representation, run_spotting
    from ..qa.interrogate import integrate, interrogate_video_gated, load_jsonl_qa

    config = load_config(args.config)
    model = args.model or config.llm.reasoning_model
    effort = args.reasoning_effort or "high"
    if not credentials_available(model):
        print(f"no credentials for: {model}", file=sys.stderr)
        return 2

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    if not videos:
        print(f"no videos matched in {args.dataset}", file=sys.stderr)
        return 1

    by_video = load_jsonl_qa(args.qa_file)
    root = Path(args.output) if args.output else Path(".")
    run_id = getattr(args, "run_id", "") or root.name or "qa_interrogate"
    records_root = root.parent if getattr(args, "run_id", "") or root.name else root
    gate_on = not getattr(args, "no_existence_gate", False)
    spotter_cache: Dict[str, Any] = {}
    outputs = []
    for video in videos:
        items = by_video.get(video.video_key, [])
        if not items:
            print(f"  {video.video_id}: no QA rows in the jsonl; skipping")
            continue
        micro = [e for e in video.events if e.is_micro]
        if not micro and not args.ask_all:
            print(f"  {video.video_id}: no annotated micro-expressions; skipping "
                  f"({len(items)} questions) -- pass --ask-all to force")
            continue

        n_tp_proposals = 0
        if gate_on:
            representation = run_representation(video, config, max_frames=args.max_frames,
                                                stride=args.stride)
            proposals: List[Any] = []
            truths = [e.interval for e in micro]
            iou_threshold = (args.iou_threshold if args.iou_threshold is not None
                            else config.evaluation.iou_threshold)
            if len(representation):
                clip_spotter = _clip_spotter_for(video, args.dataset, config,
                                                  args.clip_localiser, spotter_cache)
                external_curve = None
                if clip_spotter is not None:
                    clip_out = clip_spotter.infer(video, representation)
                    external_curve = clip_out["curve"]
                    if config.clip.replace_activations:
                        apply_clip_activations(representation, clip_out["activations"])
                spotting = run_spotting(video, representation, config,
                                        external_curve=external_curve)
                softnet_proposals = _softnet_proposals_for(
                    video, args.dataset, args.softnet_spotter)
                proposals, _fusion_stats = _merge_proposal_sets(
                    spotting.proposals, softnet_proposals, truths,
                    iou_threshold=iou_threshold)
            n_tp_proposals = len(proposals)
            print(f"  {video.video_id}: {n_tp_proposals} TP proposal(s) (ground-"
                  f"truth-gated union of both branches, IoU > {iou_threshold}) "
                  f"vs {len(truths)} annotated micro event(s)")

        records, decision, asked_items, asked_predictions = interrogate_video_gated(
            video, items, call_model, model, n_tp_proposals=n_tp_proposals,
            reasoning_effort=effort, gate_on_existence=gate_on,
        )
        write_qa_records(records, run_id=run_id, dataset=args.dataset,
                         subject=str(video.subject), video_id=video.video_id,
                         run_root=records_root)

        out_dir = root / video.video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "qa_predictions.json").write_text(
            _json.dumps({"video_id": video.video_id, "model": model,
                         "n_questions": len(asked_predictions),
                         "existence_decision": decision.to_dict(),
                         "predictions": [p.to_dict() for p in asked_predictions]},
                        ensure_ascii=False, indent=1), encoding="utf-8")
        integrated = integrate(asked_items, asked_predictions)
        _dump({"video_id": video.video_id, "model": model, **integrated},
              out_dir / "qa_integrated.json")
        outputs.append({"video_id": video.video_id, "n_questions": len(asked_predictions),
                        "n_gated_off": sum(1 for r in records if r.gated),
                        "existence": decision.to_dict(),
                        "count_error": integrated.get("count_error"),
                        "emotion_accuracy": integrated.get("emotion_accuracy"),
                        "au_f1_mean": integrated.get("au_f1_mean"),
                        "n_failed": integrated.get("n_failed", 0)})
        print(f"  {video.video_id}: {len(asked_predictions)} asked / "
              f"{sum(1 for r in records if r.gated)} gated off -> "
              f"count_err={integrated.get('count_error')}, "
              f"emo_acc={integrated.get('emotion_accuracy')}, "
              f"au_f1={integrated.get('au_f1_mean')}, "
              f"failed={integrated.get('n_failed', 0)}")
    _dump({"dataset": args.dataset, "model": model, "qa_file": args.qa_file,
           "existence_gate": gate_on, "n_videos": len(outputs), "per_video": outputs})
    return 0


def cmd_final_metrics(args: argparse.Namespace) -> int:
    from ..eval.report import load_runs
    from ..eval.subject_report import (aggregate_final_metrics, collect_metric_inputs,
                                       write_summary)
    from ..qa.interrogate import load_jsonl_qa

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    if not videos:
        print(f"no videos matched in {args.dataset}", file=sys.stderr)
        return 1
    wanted = {v.video_id for v in videos}

    runs = load_runs(Path(args.runs) if args.runs else RUNS_ROOT, wanted)
    if not runs:
        print("no saved runs found; run the pipeline first", file=sys.stderr)
        return 1

    qa_by_id = None
    if args.qa_file:
        by_key = load_jsonl_qa(args.qa_file)
        qa_by_id = {v.video_id: by_key.get(v.video_key, []) for v in videos}

    iou_threshold = (args.iou_threshold if args.iou_threshold is not None
                     else config.evaluation.iou_threshold)
    inputs = collect_metric_inputs(runs, videos, qa_by_id, config)
    print(f"{len(inputs)} run(s) collected for the MEGC summary "
          f"(IoU threshold {iou_threshold})")
    summary = aggregate_final_metrics(
        inputs, run_id=args.run_id, protocol=args.protocol, mode=args.mode,
        iou_threshold=iou_threshold)
    json_path, md_path = write_summary(summary, run_id=args.run_id,
                                       run_root=Path(args.output) if args.output else None)
    print(f"written: {json_path}")
    print(f"written: {md_path}")
    overall = summary.get("overall", {})
    strict = overall.get("spotting", {}).get("interval", {}).get("strict_iou", {})
    strs_block = overall.get("strs", {}).get("score", {})
    print(f"overall: F1_s {strict.get('f1')}, "
          f"F1_a(TP) {strs_block.get('f1_analysis')}, "
          f"STRS {strs_block.get('strs')}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from ..eval.iou_histogram import save_distribution, truth_best_ious
    from ..eval.metrics import aggregate_proposal_metrics, evaluate_proposals

    config = load_config(args.config)
    threshold = (args.iou_threshold if args.iou_threshold is not None
                 else config.evaluation.iou_threshold)
    rescue = config.evaluation.affective_rescue and not args.no_rescue

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = {v.video_id: v for v in _select_videos(index, args.subject, args.video, args.limit)}
    root = Path(args.runs) if args.runs else RUNS_ROOT

    per_video, details = [], []
    truth_rows: List[Dict[str, object]] = []
    truth_ious: List[float] = []
    for video_id, video in videos.items():
        answer_path = root / video_id / "answer.json"
        if not answer_path.is_file():
            LOGGER.warning("no saved run for %s; skipping", video_id)
            continue
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        proposals = [(int(p["onset"]), int(p["offset"]))
                     for p in answer.get("part1_proposals", [])]
        labels = [str(a.get("fine_label", ""))
                  for a in answer.get("part2_analysis", [])]
        events = video.micro_events()
        metrics = evaluate_proposals(
            proposals, [e.interval for e in events], labels,
            [e.fine_label for e in events],
            iou_threshold=threshold, affective_rescue=rescue,
            rescue_min_iou=config.evaluation.rescue_min_iou,
        )
        per_video.append(metrics)
        details.append({"video": video_id, **metrics.to_dict()})
        best = truth_best_ious(proposals, [e.interval for e in events])
        for event, value in zip(events, best):
            truth_rows.append({
                "video": video_id, "onset": event.onset, "offset": event.offset,
                "apex": event.apex, "fine_label": event.fine_label,
                "best_iou": value, "is_tp": int(value > threshold),
            })
            truth_ious.append(value)

    if not per_video:
        print("no runs found to evaluate", file=sys.stderr)
        return 1

    total = aggregate_proposal_metrics(per_video)
    payload = {"dataset": args.dataset, "n_videos": len(per_video),
               "iou_threshold": threshold, "affective_rescue": rescue,
               "aggregate": total.to_dict(), "per_video": details}
    _dump(payload, Path(args.output) if args.output else None)
    print(f"IoU threshold {threshold}  |  rescue {'on' if rescue else 'off'}")
    print(f"F1 (eq. 2)  {total.f1}   |   F1 (strict IoU)  {total.f1_strict}   "
          f"|   rescue gain  {total.rescue_gain}")

    save_distribution(root, truth_rows, truth_ious, threshold=threshold)
    n_tp = sum(1 for v in truth_ious if v > threshold)
    print(f"IoU distribution: {len(truth_ious)} truths, "
          f"{n_tp} above {threshold}, "
          f"{sum(1 for v in truth_ious if v <= 0.0)} at zero -> "
          f"{root / 'iou_distribution.json'} / .csv / .png")
    print(f"IoU distribution (TP only, n={n_tp}) -> "
          f"{root / 'iou_distribution_tp.json'} / .csv / .png")
    return 0


def _layer1_metrics(videos, config, k_steps: int) -> Dict[str, Any]:
    import numpy as np

    from ..engines.m1_dynamics import AnalyticDynamics
    from ..engines.m2_spotting import alignment_auc
    from ..eval.rollout_metrics import counterfactual_structure, rollout_prediction_error
    from ..pipeline import run_representation, run_spotting

    dynamics = AnalyticDynamics(config.dynamics)
    sequences, contexts, aucs = [], [], []
    for video in videos:
        try:
            representation = run_representation(video, config)
            spotting = run_spotting(video, representation, config, None)
        except Exception as exc:
            LOGGER.warning("layer-1 skipped %s: %s", video.video_id, exc)
            continue
        slots = np.asarray(representation.slot_activations, dtype=np.float64)
        if slots.size:
            sequences.append(slots)
            contexts.append(slots[: max(2, min(8, slots.shape[0]))])
        record = spotting.error_record
        events = video.micro_events()
        if events:
            aucs.append(alignment_auc(
                record.s_curve,
                [(e.onset - record.t_start, e.offset - record.t_start) for e in events]))

    if not sequences:
        return {"status": "unavailable",
                "reason": "no video yielded a slot trajectory",
                "remedy": "check that the frames and flow for these videos are readable"}
    return {
        "status": "ok",
        "n_videos": len(sequences),
        "alignment_auc": (round(float(np.mean(aucs)), 4) if aucs else None),
        "prediction_error": rollout_prediction_error(
            sequences, dynamics, k_steps=k_steps).to_dict(),
        "counterfactual_structure": counterfactual_structure(
            contexts, dynamics).to_dict(),
    }


def cmd_report(args: argparse.Namespace) -> int:
    from ..eval.report import build_report, format_report, load_runs
    from ..qa.interrogate import load_jsonl_qa

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    wanted = {v.video_id for v in videos}

    runs = load_runs(Path(args.runs) if args.runs else RUNS_ROOT, wanted)
    if not runs:
        print("no saved runs found; run the pipeline first", file=sys.stderr)
        return 1
    upper = load_runs(Path(args.gt_runs), wanted) if args.gt_runs else []

    references = None
    if args.references:
        references = json.loads(Path(args.references).read_text(encoding="utf-8"))

    qa_by_id = None
    if args.qa_file:
        by_key = load_jsonl_qa(args.qa_file)
        qa_by_id = {v.video_id: by_key.get(v.video_key, []) for v in videos}
        matched = sum(1 for rows in qa_by_id.values() if rows)
        print(f"qa reference file: {args.qa_file} ({matched} of {len(videos)} "
              f"selected video(s) have rows)")

    layer1 = None
    if args.layer1:
        layer1 = _layer1_metrics(videos, config, args.k_steps)

    report = build_report(args.dataset, runs, videos, config,
                          references=references, layer1=layer1,
                          upper_bound_runs=upper, qa=qa_by_id)
    if args.output:
        _dump(report, Path(args.output))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(format_report(report))
    return 0


def cmd_compare_p4(args: argparse.Namespace) -> int:
    from ..eval.report import compare_p4, load_runs

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    wanted = {v.video_id for v in videos}

    own = load_runs(Path(args.runs) if args.runs else RUNS_ROOT, wanted)
    upper = load_runs(Path(args.gt_runs), wanted)
    if not own or not upper:
        print("compare-p4 needs both run sets; produce the upper bound with "
              "`run --use-gt-proposals --output <dir>`", file=sys.stderr)
        return 1

    result = compare_p4(own, upper, {v.video_id: v.micro_events() for v in videos}, config)
    _dump(result, Path(args.output) if args.output else None)
    if result.get("status") != "ok":
        print(f"unavailable: {result.get('reason')}", file=sys.stderr)
        return 1
    print(f"{result['n_paired']} video(s) paired")
    for stratum, body in result["by_stratum"].items():
        print(f"  boundary error {stratum:<10} n={body['n']:<4} "
              f"mean delta-F1 {body['mean_delta_f1']}")
    return 0


def cmd_build_instructions(args: argparse.Namespace) -> int:
    from ..training.instruction_set import InstructionSetBuilder, write_jsonl

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    qa = load_qa_set(args.dataset)

    builder = InstructionSetBuilder(lang=args.lang)
    samples = builder.build_dataset(videos, qa, limit=args.limit)
    if not samples:
        print("no samples built (no annotated micro-expression events)", file=sys.stderr)
        return 1

    target = Path(args.output) if args.output else (
        RUNS_ROOT / "instructions" / f"{args.dataset}_sft.jsonl")
    write_jsonl(samples, target)
    report = builder.report()
    report.update({"n_samples": len(samples), "path": str(target)})
    _dump(report)
    return 0


def cmd_folds(args: argparse.Namespace) -> int:
    from ..training.loso import build_folds

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    subjects = args.folds.split(",") if args.folds else None
    folds = build_folds(index, config.training, subjects)

    print(f"{args.dataset}: {len(folds)} fold(s) of {len(index.subjects())} subject(s)")
    for fold in folds:
        gate = "in-sample gate" if fold.in_sample else f"val={','.join(fold.val_subjects)}"
        print(f"  {fold.name:>8}  train {len(fold.train_videos):>4} video(s)  "
              f"test {len(fold.test_videos):>3}  [{gate}]")
    if folds and folds[0].in_sample:
        print("\nn_val_subjects = 0: the sufficiency gate is measured on the training "
              "pool, so its numbers are in-sample.")
    return 0


def cmd_sft(args: argparse.Namespace) -> int:
    from ..training.loso import build_folds, require_open_weight
    from ..training.sft import DryRunSFT, SFTTrainer, build_sft_samples
    from ..training.instruction_set import InstructionSetBuilder

    config = load_config(args.config)
    training = config.training
    if args.policy:
        training.policy_model = args.policy
    try:
        require_open_weight(training.policy_model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    folds = build_folds(index, training, [args.fold] if args.fold else None)
    if not folds:
        print(f"no folds for {args.dataset}", file=sys.stderr)
        return 1
    fold = folds[0]

    qa = load_qa_set(args.dataset,
                     augmented_fold=fold.name if training.use_augmented_qa else None,
                     allowed_videos=sorted(fold.pool_videos())
                     if training.use_augmented_qa else None)
    builder = InstructionSetBuilder(lang=args.lang, seed=training.fold_seed)
    pool_subjects = list(fold.train_subjects) + list(fold.val_subjects)
    instruction_samples = builder.build_dataset(
        index.videos, qa, include_subjects=pool_subjects)
    augmented = [
        {"video_id": i.video_id, "video": i.video, "question": i.question,
         "answer": i.answer}
        for i in (qa.augmented_items() if qa else [])
    ]
    samples = build_sft_samples(instruction_samples, augmented=augmented)
    if not samples:
        print("no SFT samples built", file=sys.stderr)
        return 1

    if args.dry_run:
        backend = DryRunSFT()
    else:
        try:
            import torch
        except ImportError:
            print("PyTorch is not installed, so there is nothing to run SFT on. Install "
                  "torch/transformers/peft, or use --dry-run to check the schedule, the "
                  "fold and the sample count without parameters.", file=sys.stderr)
            return 2
        print("`sft` builds the training set and drives the loop, but it does not "
              "construct the policy: LoRASFTBackend takes an already-built model and "
              "tokenizer so this package stays independent of any serving stack. Build "
              "them in your launcher and call mewm.training.sft.SFTTrainer directly, or "
              "use --dry-run here.", file=sys.stderr)
        return 2

    trainer = SFTTrainer(backend, training, config, seed=training.fold_seed)
    target = Path(args.output) if args.output else (
        RUNS_ROOT / "sft" / args.dataset / f"fold_{fold.name}")
    outcome = trainer.fit(samples, output_dir=target, in_sample_eval=fold.in_sample)
    _dump({"fold": fold.to_dict(), "n_chats": len(samples),
           "n_augmented": len(augmented), **outcome.to_dict()})
    return 0


def cmd_sft_report(args: argparse.Namespace) -> int:
    from ..training.diagnostics import loss_plateau, pass_at_k

    config = load_config(args.config)
    training = config.training
    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    losses = payload.get("epoch_losses") or payload.get("curve") or []

    plateau = loss_plateau(losses, training.plateau_window,
                           training.plateau_max_slope, training.plateau_max_cv)
    report = {
        "run": str(args.run),
        "measured_on": payload.get("evaluated_on", "unknown"),
        "plateau": plateau.to_dict(),
    }
    if args.n and args.c:
        report["pass_at_k"] = {
            "pass@1": round(pass_at_k(args.n, args.c, 1), 4),
            f"pass@{training.pass_k}": round(pass_at_k(args.n, args.c, training.pass_k), 4),
        }
    _dump(report)
    if payload.get("evaluated_on") == "training_pool":
        print("\nThese figures are in-sample: the stopping epoch was selected on the "
              "same pool the model was fitted to.", file=sys.stderr)
    return 0


def cmd_build_augmented_qa(args: argparse.Namespace) -> int:
    from ..training.loso import build_folds
    from ..training.qa_augment import (
        AugmentationError, augmented_jsonl, discover_folds, load_augmented,
    )

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)

    folds = discover_folds(args.dataset) if not args.fold else [args.fold]
    if not folds:
        print(f"{args.dataset}: no augmented QA on disk yet. It is written by stage 2 "
              f"of `loso`, not by this command.")
        return 0

    specs = {f.name: f for f in build_folds(index, config.training)}
    rows = []
    for fold_name in folds:
        spec = specs.get(fold_name)
        allowed = spec.pool_videos() if spec else None
        try:
            pairs = load_augmented(args.dataset, fold_name, allowed)
        except AugmentationError as exc:
            print(f"  {fold_name}: INVALID -- {exc}", file=sys.stderr)
            return 1
        rows.append({"fold": fold_name, "n_pairs": len(pairs),
                     "n_videos": len({p["video"] for p in pairs}),
                     "path": str(augmented_jsonl(args.dataset, fold_name)),
                     "leak_checked": allowed is not None})
    _dump({"dataset": args.dataset, "folds": rows})
    return 0


def cmd_augment_qa(args: argparse.Namespace) -> int:
    import json as _json

    from ..data.paths import qa_dir
    from ..llm.registry import credentials_available
    from ..training.qa_sweep import run_sweep, sweep_summary

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)

    if not credentials_available(args.policy_model):
        print(f"no credentials for {args.policy_model}. This command samples a real "
              f"policy; it will not fabricate generations.", file=sys.stderr)
        return 1

    source = Path(args.qa_file) if args.qa_file else _reference_qa(args.dataset)
    if source is None or not source.is_file():
        print(f"{args.dataset}: no reference QA jsonl found under {qa_dir(args.dataset)}. "
              f"Pass one with --qa-file.", file=sys.stderr)
        return 1

    with open(source, "r", encoding="utf-8") as handle:
        qa_rows = [_json.loads(line) for line in handle if line.strip()]
    print(f"reference QA: {source} ({len(qa_rows)} instruction(s))")

    def progress(stage: str, key: str, position: int, total: int) -> None:
        if stage == "perceive" or position % 10 == 0 or position == total:
            print(f"  [{stage}] {position}/{total} {key}", flush=True)

    result = run_sweep(
        args.dataset, index, qa_rows, config,
        model=args.policy_model, k=args.k, max_prompts=args.max_prompts,
        max_workers=args.max_workers, stride=args.stride, max_frames=args.max_frames,
        dry_run=args.dry_run, resume=getattr(args, "resume", False),
        progress=progress,
    )

    summary = sweep_summary(result)
    target = qa_dir(args.dataset) / "augmented"
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{args.dataset}_augmentation_summary_{args.policy_model}.json"
        path.write_text(_json.dumps(summary, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"written: {path}")

    consolidated = summary.get("consolidated", {})
    if consolidated.get("path"):
        print(f"written: {consolidated['path']} ({consolidated['n_pairs']} pair(s) "
              f"across {consolidated['n_videos']} video(s), "
              f"{consolidated['n_pairs_with_source_qa']} with source QA)")

    totals = summary["totals"]
    filtering = summary["instruction_filtering"]
    written_rows = sum((f.get("written") or {}).get("n_pairs", 0)
                       for f in summary.get("folds", {}).values())
    print(f"\n{args.dataset}: {filtering['n_admitted']}/"
          f"{filtering['n_reference_instructions']} instruction(s) admitted, "
          f"{totals['n_candidates']} candidate(s) drawn, "
          f"{written_rows} strict-TP augmented row(s) written across "
          f"{totals['n_folds']} fold(s) "
          f"({totals['n_accepted_pairs_across_folds']} accepted before the TP gate)")

    if getattr(args, "with_megc_metrics", False) and not args.dry_run:
        print(f"\n{args.dataset}: scoring the MEGC metric suite over all "
              f"{len(qa_rows)} reference instruction(s)")
        _run_megc_channel(args, config, index, qa_rows, target)

    _dump(summary, Path(args.output) if args.output else None)
    return 0


def _perception_cache_root(dataset: str, disabled: bool) -> Optional[Path]:
    from ..data.paths import qa_dir

    if disabled:
        return None
    return qa_dir(dataset) / "perception_cache"


def _run_megc_channel(args, config, index, qa_rows, target: Path) -> dict:
    from ..training.qa_eval import run_evaluation

    target.mkdir(parents=True, exist_ok=True)
    output = target / f"{args.dataset}_megc_metrics_{args.policy_model}.json"
    report = run_evaluation(
        args.dataset, list(index.videos), qa_rows, output, config=config,
        model=args.policy_model,
        cache_root=_perception_cache_root(
            args.dataset, getattr(args, "no_perception_cache", False)),
        max_frames=args.max_frames, stride=args.stride,
        max_workers=getattr(args, "max_workers", 8),
        reasoning_effort=getattr(args, "reasoning_effort", ""),
        iou_threshold=getattr(args, "iou_threshold", 0.5),
        max_questions=getattr(args, "max_questions", 0),
        include_per_subject=not getattr(args, "no_per_subject", False),
    )
    print(f"written: {output}")
    _print_megc_summary(report)
    return report


def _print_megc_summary(report: dict) -> None:
    metrics = report.get("metrics", {})
    loc = metrics.get("localisation", {})
    rec = metrics.get("recognition", {})
    whole = metrics.get("whole_video", {})

    def get(block: dict, *keys, default="n/a"):
        for key in keys:
            if not isinstance(block, dict):
                return default
            block = block.get(key, {})
        return block if not isinstance(block, dict) else default

    print("\n  localisation")
    engine = loc.get("interval", {}).get("engine_proposals", {})
    for expression_type in ("micro_expression", "macro_expression", "pooled_both_types"):
        block = engine.get(expression_type, {})
        if not isinstance(block, dict) or block.get("status") != "ok":
            print(f"    engine {expression_type:<18} unavailable "
                  f"({block.get('reason', 'n/a') if isinstance(block, dict) else 'n/a'})")
            continue
        marker = " <- headline" if engine.get("headline") == expression_type else ""
        print(f"    engine {expression_type:<18} @IoU>=0.5: "
              f"F1 {get(block, 'strict_iou', 'f1')}  "
              f"P {get(block, 'strict_iou', 'precision')}  "
              f"R {get(block, 'strict_iou', 'recall')}  "
              f"(TP {get(block, 'strict_iou', 'tp')} "
              f"FP {get(block, 'strict_iou', 'fp')} "
              f"FN {get(block, 'strict_iou', 'fn')}){marker}")
        spot = block.get("unweighted_type", {})
        if spot.get("status") == "ok":
            print(f"      SpotUF1 {spot.get('spot_uf1')}  SpotUAR {spot.get('spot_uar')} "
                  f"[on matched pairs]")
    types = loc.get("expression_type_unweighted", {})
    print(f"    SpotUF1 {types.get('spot_uf1', 'n/a')}  "
          f"SpotUAR {types.get('spot_uar', 'n/a')}  [ME/MaE type question]")
    counting = loc.get("counting", {})
    for quantity in ("expression", "micro", "macro"):
        entry = counting.get(quantity, {})
        print(f"    {quantity:<11} MAE {entry.get('mae', 'n/a')}  "
              f"RMSE {entry.get('rmse', 'n/a')}")

    print("  recognition")
    au = rec.get("action_units", {})
    print(f"    F1_AU {au.get('f1_au', 'n/a')}  "
          f"Jaccard_AU {au.get('jaccard_au', 'n/a')}")
    emotion = rec.get("event", {}).get("emotion", {})
    for granularity in ("fine", "coarse"):
        for vocabulary in ("megc", "repo"):
            entry = emotion.get(granularity, {}).get(vocabulary, {})
            if entry.get("status") == "ok":
                print(f"    RegUF1 {entry.get('reg_uf1')}  "
                      f"RegUAR {entry.get('reg_uar')}  "
                      f"[{granularity}/{vocabulary}, {entry.get('macro_divisor')} "
                      f"class(es) with support]")
    text = rec.get("event", {}).get("text", {})
    print(f"    BLEU {text.get('bleu', 'n/a')}  ROUGE-1 {text.get('rouge_1', 'n/a')}")

    print("  whole video (spot-then-recognise)")
    strs_block = whole.get("strs", {})
    print(f"    STRS {strs_block.get('score', 'n/a')} "
          f"= F1_s {strs_block.get('f1_spotting', 'n/a')} "
          f"x F1_a {strs_block.get('f1_analysis', 'n/a')}")
    whole_text = whole.get("text", {})
    print(f"    BLEU {whole_text.get('bleu', 'n/a')}  "
          f"ROUGE-1 {whole_text.get('rouge_1', 'n/a')}")


def cmd_evaluate_megc(args: argparse.Namespace) -> int:
    import json as _json

    from ..data.paths import qa_dir
    from ..llm.registry import credentials_available

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)

    if not credentials_available(args.policy_model):
        print(f"no credentials for {args.policy_model}. This command samples a real "
              f"policy; it will not fabricate answers.", file=sys.stderr)
        return 1

    source = Path(args.qa_file) if args.qa_file else _reference_qa(args.dataset)
    if source is None or not source.is_file():
        print(f"{args.dataset}: no reference QA jsonl found under {qa_dir(args.dataset)}. "
              f"Pass one with --qa-file.", file=sys.stderr)
        return 1

    with open(source, "r", encoding="utf-8") as handle:
        qa_rows = [_json.loads(line) for line in handle if line.strip()]
    print(f"reference QA: {source} ({len(qa_rows)} instruction(s))")

    target = qa_dir(args.dataset) / "augmented"
    report = _run_megc_channel(args, config, index, qa_rows, target)
    _dump(report.get("metrics", {}), Path(args.output) if args.output else None)
    return 0


def _reference_qa(dataset: str) -> Optional[Path]:
    from ..data.paths import find_qa_runs, qa_dir

    root = qa_dir(dataset)
    flat = sorted(root.glob(f"{dataset}_me_lvqa_*.jsonl"))
    if flat:
        return flat[0]
    for run in find_qa_runs(dataset):
        found = sorted(run.glob("*.jsonl"))
        if found:
            return found[0]
    return None


def cmd_loso(args: argparse.Namespace) -> int:
    from ..training.loso import (
        LOSORunner, PolicyNotTrainable, require_open_weight, write_report, summarise,
    )
    from ..training.sft import DryRunSFT

    config = load_config(args.config)
    training = config.training
    if args.policy:
        training.policy_model = args.policy
    if args.sft_epochs:
        training.sft_max_epochs = args.sft_epochs
    if args.rl_steps:
        training.rl_total_steps = args.rl_steps
    if args.no_clip:
        training.clip_finetune = False
    if args.clip_retrain:
        training.clip_reuse_checkpoint = False
    if args.tta_samples:
        training.tta_samples = args.tta_samples
        training.test_time_augmentation = args.tta_samples > 1
    try:
        require_open_weight(training.policy_model)
    except PolicyNotTrainable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not args.dry_run:
        print("The LOSO driver needs a policy backend, a sampler and an evaluator; they "
              "are injected by the training entry point rather than constructed here. "
              "Run with --dry-run to exercise the fold arithmetic and the gate wiring, "
              "or drive `mewm.training.loso.LOSORunner` directly.\n",
              file=sys.stderr)
        return 2

    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    subjects = args.folds.split(",") if args.folds else None
    if training.clip_finetune:
        print("dry run: skipping stage 0. Drop --dry-run to fine-tune the CLIP engine.",
              file=sys.stderr)
        training.clip_finetune = False
    runner = LOSORunner(args.dataset, DryRunSFT(), mewm_config=config, index=index,
                        output_root=Path(args.output) if args.output else None,
                        calibrate=args.calibrate, clip_stride=args.clip_stride,
                        clip_max_frames=args.clip_max_frames, device=args.device)
    results = runner.run(lambda fold: [], subjects)

    _dump(summarise(results))
    if args.output:
        target = write_report(results, Path(args.output) / f"{args.dataset}_loso.json")
        print(f"report: {target}")
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from ..training.pretrain import (
        LongVideoWindows, PretrainConfig, WorldModelPretrainer, harvest_windows,
    )

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    if not videos:
        print(f"no videos matched in {args.dataset}", file=sys.stderr)
        return 1

    pretrain = PretrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    excluded = args.exclude_subjects.split(",") if args.exclude_subjects else []
    windows = harvest_windows(videos, config, pretrain, exclude_subjects=excluded)
    if not windows:
        print("no training windows harvested", file=sys.stderr)
        return 1

    trainer = WorldModelPretrainer(config, pretrain)
    target = Path(args.output) if args.output else (RUNS_ROOT / "stage0")
    trainer.fit(LongVideoWindows(windows, pretrain), output_dir=target)
    print(f"stage-0 checkpoint: {target}")
    return 0


def cmd_train_localiser(args: argparse.Namespace) -> int:
    import json as _json

    from ..training.localiser_supervised import (
        LocaliserTrainConfig, build_frame_dataset, evaluate_localiser, train_localiser,
    )

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = [v for v in _select_videos(index, args.subject, args.video, args.limit)
              if v.micro_events()]
    if not videos:
        print(f"no videos with micro-expression annotations in {args.dataset}",
              file=sys.stderr)
        return 1

    print(f"extracting frame features for {len(videos)} annotated video(s)...")
    samples = build_frame_dataset(
        videos, config, max_frames=args.max_frames, stride=args.stride)
    if not samples:
        print("no usable samples after feature extraction", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path(".runlogs") / "localiser" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted({s.subject for s in samples})
    folds = [f.strip() for f in args.folds.split(",") if f.strip()] or subjects
    train_config = LocaliserTrainConfig(epochs=args.epochs, device=args.device)

    reports = []
    for subject in folds:
        train = [s for s in samples if s.subject != subject]
        test = [s for s in samples if s.subject == subject]
        if not test or sum(s.n_positive for s in test) == 0:
            print(f"fold {subject}: no held-out positive frames, skipping")
            continue
        if len({s.subject for s in train}) < 2:
            print(f"fold {subject}: fewer than two training subjects, skipping")
            continue

        checkpoint = train_localiser(train, train_config, fold_name=f"loso_{subject}")
        path = checkpoint.save(out_dir / f"localiser_{subject}.pt")
        report = evaluate_localiser(checkpoint, test, device=args.device)
        print(f"fold {subject}: held-out AUC {report['pooled_mean_auc']:.4f} "
              f"(val {checkpoint.metrics['best_val_auc']:.4f}) -> {path.name}")
        reports.append({
            "fold": subject, "checkpoint": str(path),
            "val_auc": checkpoint.metrics["best_val_auc"],
            "test": report,
        })

    summary_path = out_dir / "folds.json"
    summary_path.write_text(_json.dumps({
        "dataset": args.dataset, "n_samples": len(samples),
        "n_subjects": len(subjects), "folds": reports,
    }, indent=2), encoding="utf-8")
    print(f"\n{len(reports)} fold(s) written to {out_dir}")
    print(f"summary: {summary_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from ..training.pretrain import calibrate_detection_thresholds

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    result = calibrate_detection_thresholds(videos, config)
    _dump(result, Path(args.output) if args.output else None)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from ..llm.registry import credentials_available, describe, list_models

    rows = []
    for spec in list_models(open_only=args.open_only, hosted_only=args.hosted_only):
        entry = spec.to_dict()
        entry["route"] = describe(spec.model_id)
        entry["ready"] = (credentials_available(spec.model_id) if not spec.open_weights
                          else None)
        rows.append(entry)

    if args.output:
        _dump({"models": rows}, Path(args.output))
        return 0

    print(f"{'model':32s} {'provider':10s} {'transport':10s} {'vision':7s} "
          f"{'efforts':22s} ready")
    print("-" * 100)
    for row in rows:
        efforts = ",".join(row["efforts"]) or "-"
        ready = ("local" if row["open_weights"]
                 else ("yes" if row["ready"] else "NO CREDENTIALS"))
        print(f"{row['model_id']:32s} {row['provider']:10s} {row['transport']:10s} "
              f"{str(row['vision']):7s} {efforts:22s} {ready}")
        if row["notes"]:
            print(f"{'':32s} note: {row['notes']}")
    print("\nSelect a model per role with --reasoning-model / --critic-model, or in "
          "configs/mewm_agent.yaml under llm:")
    return 0


def cmd_test_models(args: argparse.Namespace) -> int:
    from ..llm.registry import list_models, resolve
    from ..llm.client import ping

    if args.models:
        targets = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        targets = [s.model_id for s in list_models(hosted_only=True)]

    image = None
    if args.vision:
        image = args.image
        if not image:
            try:
                index = load_dataset("casme_sq", limit_videos=6)
                video = index.videos[0] if index.videos else None
                if video is not None:
                    candidate = video.paths.frame(video.frame_lo + 10)
                    image = str(candidate) if candidate.is_file() else None
            except Exception:
                image = None
        if not image:
            print("no probe image available; pass --image PATH", file=sys.stderr)
            return 1
        print(f"vision probe image: {image}")

    results = []
    for name in targets:
        spec = resolve(name) if args.vision else None
        if spec is not None and not spec.vision:
            print(f"  skipping {name} (text-only model)")
            continue
        print(f"  probing {name}{' +vision' if args.vision else ''} ...", flush=True)
        result = ping(name, timeout=args.timeout, image=image)
        results.append(result)
        if result["ok"]:
            print(f"    OK   served={result['served'] or result['resolved']!r} "
                  f"{result['latency_s']}s  {result['text']!r}")
        else:
            print(f"    FAIL {result['error']}")

    ok = sum(1 for r in results if r["ok"])
    print(f"\n{ok}/{len(results)} reachable")
    _dump({"results": results, "reachable": ok, "total": len(results)},
          Path(args.output) if args.output else None)
    return 0 if ok else 1


def cmd_local_models(args: argparse.Namespace) -> int:
    from ..llm.local_models import (
        WeightsUnavailableError, ensure_weights, load_local_model, local_status,
        manual_download_instructions,
    )
    from ..llm.registry import resolve

    if getattr(args, "prefetch", ""):
        spec = resolve(args.prefetch)
        if not spec.open_weights:
            print(f"{spec.model_id} is a hosted model; nothing to prefetch",
                  file=sys.stderr)
            return 1
        try:
            target = ensure_weights(spec)
            print(f"weights ready: {spec.model_id} -> {target}")
            return 0
        except WeightsUnavailableError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.download:
        spec = resolve(args.download)
        if not spec.open_weights:
            print(f"{spec.model_id} is a hosted model; nothing to download",
                  file=sys.stderr)
            return 1
        try:
            loaded = load_local_model(spec, dtype=args.dtype,
                                      device_map=args.device_map)
            print(f"ready: {spec.model_id} from {loaded.source} on {loaded.device}")
            return 0
        except WeightsUnavailableError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except Exception as exc:
            print(manual_download_instructions(spec, str(exc)), file=sys.stderr)
            return 2

    _dump(local_status(), Path(args.output) if args.output else None)
    return 0


def cmd_train_clip_localiser(args: argparse.Namespace) -> int:
    from ..training.clip_localiser import (
        build_clip_dataset, checkpoint_path, evaluate_clip_localiser,
        train_clip_localiser, train_state_path, write_fold_report,
    )
    from ..training.loso import build_folds

    config = load_config(args.config)
    if args.clip_weights:
        config.clip.weights_path = args.clip_weights
    if args.epochs:
        config.clip.epochs = args.epochs
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] \
        if args.subjects else None
    folds = build_folds(index, config.training, subjects)

    reports = []
    for fold in folds:
        target = checkpoint_path(args.dataset, fold.name, config.clip)
        report_path = target.parent / "test_report.json"
        if not args.force_retrain and target.is_file() and report_path.is_file():
            LOGGER.info("fold %s: checkpoint + test report already on disk, "
                       "skipping (pass --force-retrain to redo it)", fold.name)
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue

        LOGGER.info("=== CLIP localiser fold %s: %d pool / %d test video(s) ===",
                    fold.name, len(fold.pool_videos()), len(fold.test_videos))
        pool_keys = fold.pool_videos()
        pool = [v for v in index.videos if v.video_key in pool_keys]
        samples = build_clip_dataset(pool, config, stride=args.stride,
                                     max_frames=args.max_frames)
        if not samples:
            LOGGER.warning("fold %s: no trainable videos in the pool; skipping",
                           fold.name)
            continue
        state_path = train_state_path(args.dataset, fold.name, config.clip)
        checkpoint = train_clip_localiser(samples, config.clip, fold_name=fold.name,
                                          state_path=state_path)

        test_videos = [v for v in index.videos if v.video_key in set(fold.test_videos)]
        test_samples = build_clip_dataset(test_videos, config, stride=args.stride,
                                          max_frames=args.max_frames)
        report = (evaluate_clip_localiser(checkpoint, test_samples, config.clip)
                  if test_samples else {"note": "held-out subject has no "
                                                "micro-annotated videos"})
        report["fold"] = fold.name
        report["train_metrics"] = checkpoint.metrics
        target = write_fold_report(checkpoint, report, args.dataset, fold.name,
                                   config.clip)
        print(f"  fold {fold.name}: best val AUC "
              f"{checkpoint.metrics.get('best_val_auc'):.4f}, test pooled AUC "
              f"{report.get('pooled_mean_auc', float('nan'))}, checkpoint -> {target}")
        reports.append(report)

    _dump({"dataset": args.dataset, "folds": reports},
          Path(args.output) if args.output else None)
    return 0 if reports else 1


def cmd_sweep_iou(args: argparse.Namespace) -> int:
    from ..eval.metrics import sweep_iou_threshold

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = {v.video_id: v for v in _select_videos(index, args.subject, args.video,
                                                    args.limit)}
    root = Path(args.runs) if args.runs else RUNS_ROOT

    inputs = []
    for video_id, video in videos.items():
        answer_path = root / video_id / "answer.json"
        if not answer_path.is_file():
            continue
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        proposals = [(int(p["onset"]), int(p["offset"]))
                     for p in answer.get("part1_proposals", [])]
        labels = [str(a.get("fine_label", "")) for a in answer.get("part2_analysis", [])]
        events = video.micro_events()
        inputs.append((proposals, [e.interval for e in events], labels,
                       [e.fine_label for e in events]))

    if not inputs:
        print("no saved runs found to sweep", file=sys.stderr)
        return 1

    grid = ([float(x) for x in args.thresholds.split(",")] if args.thresholds
            else list(config.evaluation.iou_sweep))
    rows = sweep_iou_threshold(inputs, grid, config.evaluation.affective_rescue)

    print(f"{'IoU':>6s} {'P':>8s} {'R':>8s} {'F1':>8s} {'F1_strict':>10s} "
          f"{'rescue':>8s}")
    for row in rows:
        print(f"{row['iou_threshold']:6.2f} {row['precision']:8.4f} "
              f"{row['recall']:8.4f} {row['f1']:8.4f} {row['f1_strict']:10.4f} "
              f"{row['rescue_gain']:8.4f}")
    _dump({"dataset": args.dataset, "n_videos": len(inputs), "sweep": rows},
          Path(args.output) if args.output else None)
    return 0


def cmd_diagnose_localisation(args: argparse.Namespace) -> int:
    from ..eval.localisation_diagnostics import localisation_report
    from ..pipeline import run_representation, run_spotting

    config = load_config(args.config)
    index = load_dataset(args.dataset, limit_videos=args.limit_videos)
    videos = _select_videos(index, args.subject, args.video, args.limit)
    videos = [v for v in videos if v.micro_events()]
    if not videos:
        print(f"no videos with micro-expression annotations in {args.dataset}",
              file=sys.stderr)
        return 1

    if args.iou_threshold is not None:
        iou_threshold = float(args.iou_threshold)
    else:
        iou_threshold = config.evaluation.iou_threshold

    curves: Dict[str, Any] = {}
    offsets: Dict[str, int] = {}
    intervals: Dict[str, Any] = {}
    baseline: Dict[str, Any] = {}
    used: List[Any] = []

    for video in videos:
        LOGGER.info("diagnosing %s (%d frames)", video.video_id, video.n_frames)
        representation = run_representation(video, config, max_frames=args.max_frames,
                                            stride=args.stride)
        if not len(representation):
            LOGGER.warning("%s: no usable frames", video.video_id)
            continue
        spotting = run_spotting(video, representation, config)
        record = spotting.error_record

        key = video.video_key
        curves[key] = record.s_curve
        offsets[key] = int(record.t_start)
        intervals[key] = [(p.t_on, p.t_off, p.apex, p.peak_S)
                          for p in spotting.micro_intervals]
        baseline[key] = [(p.t_on, p.t_off, p.apex, p.peak_S)
                         for p in spotting.proposals]
        used.append(video)
        print(f"  {video.video_id}: {len(intervals[key])} intervals "
              f"({len(baseline[key])} raw proposals), "
              f"{len(video.micro_events())} micro events")

    if not used:
        print("no videos yielded a usable curve", file=sys.stderr)
        return 1

    report = localisation_report(
        used, curves, intervals, offsets=offsets,
        tau_hi=config.spotting.tau_hi, iou_threshold=iou_threshold,
        include_per_video=not args.no_per_video)
    report["decoder_enabled"] = bool(getattr(config.spotting, "localiser_enabled", False))
    if report["decoder_enabled"]:
        report["without_decoder"] = localisation_report(
            used, curves, baseline, offsets=offsets, tau_hi=config.spotting.tau_hi,
            iou_threshold=iou_threshold, include_per_video=False)

    print()
    print(f"scope: {report['scope']}, IoU >= {iou_threshold}")
    print(f"videos {report['n_videos_with_micro_truth']}, "
          f"micro events {report['n_micro_truth_events']}, "
          f"intervals {report['n_intervals']}")
    for layer in ("layer_1_signal", "layer_2_threshold", "layer_3_extent",
                  "layer_5_error_taxonomy"):
        block = report.get(layer)
        if not block:
            continue
        print(f"\n{layer}")
        for k, v in block.items():
            if k != "verdict":
                print(f"    {k:34s} {v}")
        print(f"  -> {block['verdict']}")
    filt = report.get("layer_4_filtering", {})
    if filt:
        print("\nlayer_4_filtering")
        print(f"    as-is           {filt['as_is']}")
        print(f"    oracle filter   {filt['oracle_filter']}")
        print(f"  -> a perfect proposal filter would reach F1 "
              f"{filt['oracle_filter']['f1']}; that bounds every reranking, "
              f"NMS and calibration scheme")
    if "without_decoder" in report:
        a = report["without_decoder"]["layer_4_filtering"]["as_is"]
        b = filt["as_is"]
        print(f"\ndecoder off -> F1 {a['f1']} (TP {a['tp']}, FP {a['fp']})")
        print(f"decoder on  -> F1 {b['f1']} (TP {b['tp']}, FP {b['fp']})")

    _dump(report, Path(args.output) if args.output else None)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..agents.base import available_roles
    from ..config import DATASET_ROOT, FLOW_ROOT, LOADED_ENV_FILE, PROJECT_ROOT, QTA_ROOT
    from ..llm.client import credentials_available, describe_route

    config = load_config(args.config)
    clip_weights = Path(config.clip.weights_path)
    report: Dict[str, Any] = {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "dataset_root": f"{DATASET_ROOT} ({'ok' if DATASET_ROOT.is_dir() else 'MISSING'})",
            "flow_root": f"{FLOW_ROOT} ({'ok' if FLOW_ROOT.is_dir() else 'MISSING'})",
            "qta_root": f"{QTA_ROOT} ({'ok' if QTA_ROOT.is_dir() else 'MISSING'})",
            "env_file": (str(LOADED_ENV_FILE) if LOADED_ENV_FILE
                        else "none found (see mewm.config._find_env_file search order; "
                             "set MEWM_ENV_FILE explicitly if credentials are elsewhere)"),
            "clip_weights": f"{clip_weights} "
                           f"({'ok' if (clip_weights / 'config.json').is_file() else 'MISSING'})",
        },
        "datasets_present": available_datasets(),
        "datasets_with_qa": available_datasets(require_qa=True),
        "roles": available_roles(),
    }

    packages = {}
    for name in ("numpy", "torch", "cv2", "openpyxl", "xlrd", "yaml", "scipy"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "present")
        except ImportError:
            packages[name] = "MISSING"
    report["packages"] = packages

    report["models"] = {
        role: {"requested": model, "resolved": _resolved_id(model),
               "backend": backend,
               "backend_ok": _backend_status(model, backend),
               "route": describe_route(model),
               "credentials": credentials_available(model)}
        for role, model, backend in (
            ("reasoning", config.llm.reasoning_model, config.llm.reasoning_backend),
            ("perception", config.llm.perception_model, config.llm.perception_backend),
            ("structure", config.llm.structure_model, config.llm.structure_backend),
            ("critic", config.llm.critic_model, config.llm.critic_backend),
        )
    }
    report["backends"] = {
        "fallback_to_api": config.backends.fallback_to_api,
        "local_quantization": config.backends.local_quantization,
        "prefetch": list(config.backends.prefetch),
    }
    report["clip_engine"] = _clip_status(config)
    from ..llm.registry import heterogeneous
    is_heterogeneous = heterogeneous(config.llm.critic_model,
                                     config.llm.reasoning_model)
    report["verification_heterogeneity"] = {
        "ok": is_heterogeneous,
        "critic": config.llm.critic_model,
        "reasoning": config.llm.reasoning_model,
        "note": ("critic and reasoner use different providers, as required"
                 if is_heterogeneous else
                 "WARNING: critic and reasoner share a base or provider family; "
                 "adversarial verification loses its independence (appendix D.1)"),
    }
    report["evaluation"] = {
        "iou_threshold": config.evaluation.iou_threshold,
        "affective_rescue": config.evaluation.affective_rescue,
        "iou_sweep": list(config.evaluation.iou_sweep),
    }
    _dump(report, Path(args.output) if args.output else None)
    return 0


def _resolved_id(model: str) -> str:
    from ..llm.registry import UnknownModelError, resolve
    try:
        return resolve(model).model_id
    except UnknownModelError:
        return f"UNREGISTERED ({model})"


def _backend_status(model: str, backend: str) -> str:
    from ..llm.client import BackendMismatch, backend_for
    try:
        backend_for(model, backend)
        return "ok"
    except BackendMismatch as exc:
        return f"MISMATCH: {exc}"
    except Exception as exc:
        return f"unresolvable ({exc})"


def _clip_status(config) -> Dict[str, Any]:
    from ..engines.clip_motion_engine import resolve_clip_weights

    status: Dict[str, Any] = {
        "enabled": config.clip.enabled,
        "weights_path": config.clip.weights_path,
        "vision_unfreeze_layers": config.clip.vision_unfreeze_layers,
        "text_unfreeze_layers": config.clip.text_unfreeze_layers,
    }
    try:
        weights = resolve_clip_weights(config.clip)
        status["weights_present"] = (weights / "config.json").is_file()
        status["weights_resolved"] = str(weights)
    except Exception:
        status["weights_present"] = False
    from ..config import PACKAGE_ROOT
    root = Path(config.clip.checkpoint_root)
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    status["trained_folds"] = sorted(
        str(p.parent.relative_to(root)) for p in root.glob("*/fold_*/clip_localiser.pt")
    ) if root.is_dir() else []
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mewm", description="MEWM-Agent: multi-agent emotional world model for "
                                 "micro-expression understanding in long videos")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", default=None, help="path to a YAML config overlay")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser, with_dataset: bool = True) -> None:
        if with_dataset:
            p.add_argument("--dataset", required=True, choices=list(DATASETS))
        p.add_argument("--subject", default="", help="restrict to one subject id")
        p.add_argument("--video", default="", help="restrict to one video key")
        p.add_argument("--limit", type=int, default=0, help="max videos to process")
        p.add_argument("--limit-videos", type=int, default=0,
                       help="max videos to load from the annotation file")
        p.add_argument("--output", default=None)

    def _model_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--reasoning-model", default="",
                       help="R-Agent base, e.g. claude-sonnet-5 / Qwen3-VL-8B")
        p.add_argument("--perception-model", default="")
        p.add_argument("--structure-model", default="")
        p.add_argument("--critic-model", default="",
                       help="C-Agent base; must differ in provider from the reasoner")
        p.add_argument("--model", default="",
                       help="shorthand: set every role to this model (drops critic "
                            "heterogeneity, so it warns)")
        p.add_argument("--reasoning-effort", default="",
                       help="tier for the R-Agent: low/medium/high/xhigh/max")
        p.add_argument("--critic-effort", default="")

    def _backend_flags(p: argparse.ArgumentParser) -> None:
        choices = ["hosted", "local"]
        p.add_argument("--backend", default="", choices=[""] + choices,
                       help="set every role's backend at once (hosted | local)")
        p.add_argument("--reasoning-backend", default="", choices=[""] + choices)
        p.add_argument("--perception-backend", default="", choices=[""] + choices)
        p.add_argument("--structure-backend", default="", choices=[""] + choices)
        p.add_argument("--critic-backend", default="", choices=[""] + choices)
        p.add_argument("--fallback-to-api", action="store_true",
                       help="allow a role whose local weights are missing to fall "
                            "back to the hosted default -- every fallback is logged "
                            "and recorded in backend_manifest.json")

    def _clip_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--clip-localiser", default="auto",
                       help="'auto' uses the fold checkpoint under "
                            "runs/clip_localiser/<dataset>/fold_<subject>/ when one "
                            "exists; 'off' forces the analytic curve; anything else "
                            "is an explicit checkpoint path")

    def _softnet_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--softnet-spotter", default="auto",
                       help="'auto' unions the SOFTNet peak proposals with the "
                            "prediction system's when a fold checkpoint exists under "
                            "runs/softnet_spotter/<dataset>/fold_<subject>/ "
                            "(blueprint phase 4); 'off' disables")

    p = sub.add_parser("datasets", help="report dataset and QA availability")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("spot", help="stages I-II only (no LLM calls)")
    _common(p)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=None)
    _clip_flag(p)
    p.set_defaults(func=cmd_spot)

    p = sub.add_parser(
        "diagnose-localisation",
        help="attribute a low localisation score to one pipeline layer "
             "(micro-expression events only, no LLM calls)")
    _common(p)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=None,
                   help="TP threshold; MEGC scores at 0.5")
    p.add_argument("--no-per-video", action="store_true",
                   help="pooled layers only, omitting the per-video breakdown")
    p.set_defaults(func=cmd_diagnose_localisation)

    p = sub.add_parser("run", help="the full six-stage pipeline")
    _common(p)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-proposals", type=int, default=0)
    p.add_argument("--iou-threshold", type=float, default=None,
                   help='')
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--stub", action="store_true", help="use the offline stub client")
    p.add_argument("--no-qa", action="store_true", help="do not read the question set")
    p.add_argument("--use-gt-proposals", action="store_true",
                   help="analyse the annotated intervals instead of detected ones "
                        "(protocol P4 upper bound; recorded in the output)")
    p.add_argument("--force", action="store_true",
                   help="re-run videos that already have a summary.json "
                        "(default: skip them, so a relaunch resumes where it stopped)")
    p.add_argument("--print-answer", action="store_true",
                   help="print the composed answer to stdout")
    _model_flags(p)
    _backend_flags(p)
    _clip_flag(p)
    _softnet_flag(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "build-softnet-features",
        help="build the per-video (u, v, epsilon) feature caches for SOFTNet spotting")
    _common(p)
    p.set_defaults(func=cmd_build_softnet_features)

    p = sub.add_parser(
        "train-softnet-spotter",
        help="train the SOFTNet peak scorer for one LOSO fold (held-out subject)")
    _common(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_train_softnet)

    p = sub.add_parser(
        "qa-interrogate",
        help="ask every ME-LVQA jsonl question of a video separately (gated on "
             "existence by default) and integrate the per-question predictions "
             "(count vote / emotion vote / AU-set F1)")
    _common(p)
    p.add_argument("--qa-file", required=True,
                   help="path to a *_me_lvqa_*.jsonl build")
    p.add_argument("--ask-all", action="store_true",
                   help="ask even when the video has no annotated micro-expressions")
    p.add_argument("--no-existence-gate", action="store_true",
                   help="ask every question unconditionally instead of gating the "
                        "rest of the video's questions on the first (existence) "
                        "question's answer / the localisation proposals")
    p.add_argument("--run-id", default="",
                   help="subdirectory name for qa_records.jsonl "
                        "(<output's parent>/<run-id>/<dataset>/<subject>/<video_id>/"
                        "qa_records.jsonl); defaults to --output's own directory name")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=None,
                   help="TP threshold for the existence gate; MEGC scores at 0.5")
    _clip_flag(p)
    _softnet_flag(p)
    _model_flags(p)
    _backend_flags(p)
    p.set_defaults(func=cmd_qa_interrogate)

    p = sub.add_parser(
        "final-metrics",
        help="the whole-dataset MEGC summary over saved runs "
             "(final_metrics_summary.json/.md: SpotUF1/SpotUAR, MAE/RMSE, F1AU/"
             "JaccardAU, RegUF1/RegUAR fine+coarse, BLEU/ROUGE-1, STRS)")
    _common(p)
    p.add_argument("--runs", default=None, help="directory holding the saved runs")
    p.add_argument("--qa-file", default="",
                   help="path to a *_me_lvqa_*.jsonl reference build: supplies the "
                        "reference answers the text metrics score against")
    p.add_argument("--run-id", default="final_metrics",
                   help="subdirectory name for final_metrics_summary.json/.md")
    p.add_argument("--protocol", default="loso", choices=["loso", "lodo"])
    p.add_argument("--mode", default="api", choices=["api", "open_weight"])
    p.add_argument("--iou-threshold", type=float, default=None,
                   help="TP threshold; MEGC scores at 0.5")
    p.set_defaults(func=cmd_final_metrics)

    p = sub.add_parser("evaluate", help="P1 proposal-level metrics over saved runs")
    _common(p)
    p.add_argument("--runs", default=None, help="directory holding the saved runs")
    p.add_argument("--iou-threshold", type=float, default=None,
                   help="override the eq. (2) IoU threshold (default 0.5)")
    p.add_argument("--no-rescue", action="store_true",
                   help="disable affective rescue and score pure localisation")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sweep-iou", help="recompute P1 across a grid of IoU thresholds")
    _common(p)
    p.add_argument("--runs", default=None)
    p.add_argument("--thresholds", default="",
                   help="comma-separated grid, e.g. 0.1,0.3,0.5,0.7")
    p.set_defaults(func=cmd_sweep_iou)

    p = sub.add_parser(
        "report", help="")
    _common(p)
    p.add_argument("--runs", default=None, help="directory holding the saved runs")
    p.add_argument("--gt-runs", default=None,
                   help="")
    p.add_argument("--references", default=None,
                   help="JSON mapping video_id -> reference narrative, for BLEU/ROUGE")
    p.add_argument("--qa-file", default="",
                   help="")
    p.add_argument("--layer1", action="store_true",
                   help="")
    p.add_argument("--k-steps", type=int, default=5,
                   help="rollout horizon for the layer-1 prediction-error curve")
    p.add_argument("--json", action="store_true",
                   help="emit the report as JSON instead of the text table")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "compare-p4",
        help="quantify how far localisation error propagates into the answer")
    _common(p)
    p.add_argument("--runs", default=None, help="self-proposed runs")
    p.add_argument("--gt-runs", required=True,
                   help="upper-bound runs, produced with --use-gt-proposals")
    p.set_defaults(func=cmd_compare_p4)

    p = sub.add_parser("models", help="list every registered model")
    p.add_argument("--open-only", action="store_true")
    p.add_argument("--hosted-only", action="store_true")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("test-models", help="probe each model endpoint once")
    p.add_argument("--models", default="", help="comma-separated ids; default = all hosted")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--vision", action="store_true",
                   help="probe with an image instead of text only")
    p.add_argument("--image", default="", help="explicit probe image for --vision")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_test_models)

    p = sub.add_parser("local-models", help="open-weight status, or pre-download one")
    p.add_argument("--download", default="", help="model id to fetch AND load now")
    p.add_argument("--prefetch", default="",
                   help="model id: resumable snapshot download only, no load "
                        "(方案 §5.5 prefetch)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_local_models)

    p = sub.add_parser(
        "train-clip-localiser",
        help="")
    _common(p)
    p.add_argument("--subjects", default="",
                   help="comma-separated held-out subject ids; default = every fold")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0,
                   help="override clip.epochs from the config")
    p.add_argument("--clip-weights", default="",
                   help="")
    p.add_argument("--force-retrain", action="store_true",
                   help="")
    p.set_defaults(func=cmd_train_clip_localiser)

    p = sub.add_parser("build-instructions", help="build the SFT instruction set")
    _common(p)
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.set_defaults(func=cmd_build_instructions)

    p = sub.add_parser("pretrain", help="")
    _common(p)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--exclude-subjects", default="",
                   help="")
    p.set_defaults(func=cmd_pretrain)

    p = sub.add_parser("calibrate", help="calibrate the detection thresholds")
    _common(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser(
        "train-localiser",
        help="supervised localiser per LOSO fold, fitted on ground-truth intervals")
    _common(p)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--folds", default="",
                   help="")
    p.add_argument("--out", default="",
                   help="")
    p.set_defaults(func=cmd_train_localiser)

    p = sub.add_parser("folds", help="")
    _common(p)
    p.add_argument("--folds", default="",
                   help="")
    p.set_defaults(func=cmd_folds)

    p = sub.add_parser("sft", help="stage-1 SFT on one LOSO fold's training pool")
    _common(p)
    p.add_argument("--fold", default="",
                   help="held-out subject id; default = the first fold")
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--policy", default="",
                   help="open-weight policy checkpoint; hosted-API ids are rejected")
    p.add_argument("--dry-run", action="store_true",
                   help="run the loop with no parameters, to check the schedule")
    p.set_defaults(func=cmd_sft)

    p = sub.add_parser("sft-report",
                       help="re-run the sufficiency judgements over a saved SFT run")
    p.add_argument("--run", required=True, help="path to an SFT outcome json")
    p.add_argument("--n", type=int, default=0, help="samples drawn per prompt")
    p.add_argument("--c", type=int, default=0, help="of those, how many passed")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_sft_report)

    p = sub.add_parser("build-augmented-qa",
                       help="inspect and leak-check a fold's augmented QA file")
    _common(p)
    p.add_argument("--fold", default="", help="held-out subject id; default = all on disk")
    p.set_defaults(func=cmd_build_augmented_qa)

    p = sub.add_parser("augment-qa",
                       help="filter the reference QA set and sample augmented pairs "
                            "for every LOSO fold")
    _common(p)
    p.add_argument("--policy-model", default="claude-sonnet-5",
                   help="hosted model that plays the policy for this sweep")
    p.add_argument("--k", type=int, default=3, help="draws per admitted instruction")
    p.add_argument("--max-prompts", type=int, default=0,
                   help="sampling budget; 0 = every admitted instruction. A budget is "
                        "reported in the summary, never applied silently")
    p.add_argument("--max-workers", type=int, default=4,
                   help="concurrent draws within one prompt")
    p.add_argument("--stride", type=int, default=1, help="frame stride for perception")
    p.add_argument("--max-frames", type=int, default=0, help="cap frames per video")
    p.add_argument("--qa-file", default="",
                   help="")
    p.add_argument("--dry-run", action="store_true",
                   help="")
    p.add_argument("--resume", action="store_true",
                   help="")
    p.add_argument("--with-megc-metrics", action="store_true",
                   help="")
    p.add_argument("--no-perception-cache", action="store_true",
                   help="")
    p.set_defaults(func=cmd_augment_qa)

    p = sub.add_parser("evaluate-megc",
                       help="")
    _common(p)
    p.add_argument("--policy-model", default="claude-sonnet-5",
                   help="")
    p.add_argument("--max-workers", type=int, default=8,
                   help="")
    p.add_argument("--stride", type=int, default=1, help="")
    p.add_argument("--max-frames", type=int, default=0, help="")
    p.add_argument("--qa-file", default="",
                   help="")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="")
    p.add_argument("--max-questions", type=int, default=0,
                   help="")
    p.add_argument("--reasoning-effort", default="",
                   help="")
    p.add_argument("--no-per-subject", action="store_true",
                   help="")
    p.add_argument("--no-perception-cache", action="store_true",
                   help="")
    p.set_defaults(func=cmd_evaluate_megc)

    p = sub.add_parser("loso", help="the full LOSO protocol, per fold")
    _common(p)
    p.add_argument("--folds", default="",
                   help="comma-separated held-out subject ids; default = every fold. "
                        "")
    p.add_argument("--policy", default="",
                   help="open-weight policy checkpoint; hosted-API ids are rejected")
    p.add_argument("--sft-epochs", type=int, default=0, help="override sft_max_epochs")
    p.add_argument("--rl-steps", type=int, default=0, help="override rl_total_steps")
    p.add_argument("--no-clip", action="store_true",
                   help="")
    p.add_argument("--clip-retrain", action="store_true",
                   help="re-fit the CLIP engine even when a fold checkpoint exists")
    p.add_argument("--clip-stride", type=int, default=1,
                   help="frame stride for the stage-0 CLIP dataset")
    p.add_argument("--clip-max-frames", type=int, default=0,
                   help="cap frames per video in stage 0; 0 = every frame")
    p.add_argument("--calibrate", action="store_true",
                   help="")
    p.add_argument("--tta-samples", type=int, default=0,
                   help="")
    p.add_argument("--device", default="cuda", help="device for stage 0")
    p.add_argument("--dry-run", action="store_true",
                   help="")
    p.set_defaults(func=cmd_loso)

    p = sub.add_parser("doctor", help="environment and credential check")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_hint = getattr(args, "output", None) or getattr(args, "runs", None)
    _setup_logging(getattr(args, "verbose", False),
                   extra_log_dir=Path(output_hint) if output_hint else None)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
