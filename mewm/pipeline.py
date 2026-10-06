"""Six-stage end-to-end pipeline that coordinates engines, agents, and memory."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MEWMConfig, load_config, run_dir
from .data.datasets import LongVideo
from .data.paths import max_micro_frames
from .data.qa_loader import TRIPLE_TASK_QUESTION_EN, TRIPLE_TASK_QUESTION_ZH
from .engines.m1_dynamics import AnalyticDynamics
from .engines.m2_spotting import Spotter, SpottingResult
from .engines.m3_primitives import RolloutService
from .engines.v1_motion import MotionFrontEnd, build_roi_boxes
from .engines.v2_slots import SlotBank, analytic_slot_readout
from .engines.v3_latent import LatentComposer
from .eval.metrics import greedy_match, iou as _interval_iou
from .knowledge.au_anatomy import K_SLOTS, ROI_INDEX, SLOT_AUS, SLOT_INDEX, regions_of
from .memory.store import CheckpointStore, EpisodicMemory
from .orchestration.orchestrator import Orchestrator, ProposalContext
from .orchestration.state import MEWMState
from .schemas import CandidateInterval, FrameState, LatentStream, ROIMeasurement

LOGGER = logging.getLogger(__name__)


@dataclass
class RepresentationOutput:
    stream: LatentStream
    slot_bank: SlotBank
    frames: List[int] = field(default_factory=list)
    head_motion: Optional[np.ndarray] = None
    slow_prediction: Optional[np.ndarray] = None
    slot_activations: Optional[np.ndarray] = None
    coherence_gate: Optional[np.ndarray] = None
    unavailable: List[Tuple[int, int]] = field(default_factory=list)
    low_confidence: List[Tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)


def _landmarks_for(frame_path: Path, cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    key = str(frame_path)
    if key in cache:
        return cache[key]
    from .engines.v1_motion import detect_landmarks
    points = detect_landmarks(frame_path)
    if points is not None:
        cache[key] = points
    return points


def run_representation(
    video: LongVideo,
    config: Optional[MEWMConfig] = None,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
    stride: int = 1,
    max_frames: int = 0,
    landmarks: Optional[np.ndarray] = None,
    landmark_interval: int = 15,
) -> RepresentationOutput:
    config = config or load_config()
    front_end = MotionFrontEnd(config.motion)
    composer = LatentComposer(config.representation)
    bank = SlotBank(video.video_id, video.fps)
    stream = LatentStream(video_id=video.video_id, fps=video.fps,
                          slot_order=list(SLOT_AUS))

    pairs = video.paths.aligned_pairs(t_start, t_end, step=max(1, stride))
    if max_frames:
        pairs = pairs[:max_frames]
    if not pairs:
        LOGGER.warning("%s: no aligned frame/flow pairs found", video.video_id)
        return RepresentationOutput(stream, bank)

    landmark_cache: Dict[str, np.ndarray] = {}
    frames: List[int] = []
    head_rows: List[np.ndarray] = []
    slow_rows: List[np.ndarray] = []
    activation_rows: List[np.ndarray] = []
    gate_rows: List[np.ndarray] = []
    unavailable: List[int] = []
    previous_landmarks: Optional[np.ndarray] = None
    last_landmarks: Optional[np.ndarray] = None
    last_landmark_frame: int = -10 ** 9

    for pair in pairs:
        flow = front_end.load_flow(front_end.prefer_raw_flow(pair.flow_path))
        if flow is None:
            unavailable.append(pair.t)
            continue

        points = landmarks
        if points is None:
            due = (last_landmarks is None
                   or (pair.t - last_landmark_frame) >= landmark_interval)
            if due:
                detected = _landmarks_for(pair.frame_path, landmark_cache)
                if detected is not None:
                    last_landmarks, last_landmark_frame = detected, pair.t
            points = last_landmarks

        if points is None:
            height, width = flow.shape[:2]
            points = _nominal_landmarks(width, height)
            unavailable.append(pair.t)

        measurements = front_end.measure(flow, points)
        head = front_end.head_motion(previous_landmarks, points) if previous_landmarks is not None else None
        previous_landmarks = points

        readout = analytic_slot_readout(measurements)
        activations = {au: readout[au].activation for au in SLOT_AUS}
        latent = composer.step(front_end.measurement_matrix(measurements), pair.t)

        frame_state = FrameState(
            t=pair.t, measurements=measurements,
            head_motion=(head.as_tuple() if head else (0.0, 0.0, 0.0, 1.0)),
            slot_activations=activations,
            z_slow=latent["z_slow"], z_fast=latent["z_fast"],
            z_belief=latent["z_belief"],
            available=pair.t not in unavailable,
        )
        stream.frames[pair.t] = frame_state
        bank.append(pair.t, activations)

        frames.append(pair.t)
        head_rows.append(head.as_regressor() if head else np.zeros(6))
        slow_rows.append(np.asarray(latent["z_slow"], dtype=np.float64)[:8])
        activation_rows.append(np.array([activations[au] for au in SLOT_AUS]))
        gate_rows.append(_coherence_gate_row(measurements, config.motion.c_min))

    stream.frame_lo = frames[0] if frames else 0
    stream.frame_hi = frames[-1] if frames else 0

    return RepresentationOutput(
        stream=stream, slot_bank=bank, frames=frames,
        head_motion=np.stack(head_rows) if head_rows else None,
        slow_prediction=np.stack(slow_rows) if slow_rows else None,
        slot_activations=np.stack(activation_rows) if activation_rows else None,
        coherence_gate=np.stack(gate_rows) if gate_rows else None,
        unavailable=_runs(unavailable),
        low_confidence=composer.low_confidence_spans(config.spotting.window),
    )


def _coherence_gate_row(measurements: Sequence[ROIMeasurement], c_min: float) -> np.ndarray:
    # indicator I[c_{r(k),t} >= c_min] per slot
    by_roi = {m.roi_name: m for m in measurements}
    row = np.zeros(K_SLOTS, dtype=np.float64)
    for au in SLOT_AUS:
        for roi in regions_of(au):
            measurement = by_roi.get(roi)
            if measurement is not None and measurement.coherence >= c_min:
                row[SLOT_INDEX[au]] = 1.0
                break
    return row


def _nominal_landmarks(width: int, height: int) -> np.ndarray:
    cx, cy = width / 2.0, height / 2.0
    scale = min(width, height) / 220.0
    base = np.array([
        [-70, -10], [-68, 8], [-64, 26], [-58, 44], [-48, 60], [-34, 74], [-18, 84],
        [0, 90], [18, 84], [34, 74], [48, 60], [58, 44], [64, 26], [68, 8], [70, -10],
        [70, -28], [68, -44],
        [-52, -46], [-40, -54], [-26, -56], [-12, -52], [-2, -46],
        [2, -46], [12, -52], [26, -56], [40, -54], [52, -46],
        [0, -30], [0, -18], [0, -6], [0, 6],
        [-12, 14], [-6, 16], [0, 18], [6, 16], [12, 14],
        [-40, -30], [-32, -34], [-22, -34], [-14, -28], [-22, -24], [-32, -24],
        [14, -28], [22, -34], [32, -34], [40, -30], [32, -24], [22, -24],
        [-24, 42], [-14, 36], [-6, 34], [0, 36], [6, 34], [14, 36], [24, 42],
        [14, 52], [6, 56], [0, 56], [-6, 56], [-14, 52],
        [-20, 42], [-6, 40], [0, 40], [6, 40], [20, 42], [6, 46], [0, 48], [-6, 46],
    ], dtype=np.float64)
    return base * scale + np.array([cx, cy])


def _runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(set(indices))
    runs, start, previous = [], ordered[0], ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _effective_spotting_config(config: MEWMConfig, dataset: str) -> Any:
    return config.spotting


def run_spotting(
    video: LongVideo,
    representation: RepresentationOutput,
    config: Optional[MEWMConfig] = None,
    dynamics: Optional[Any] = None,
    external_curve: Optional[np.ndarray] = None,
) -> SpottingResult:
    config = config or load_config()
    spotting_config = _effective_spotting_config(config, video.dataset)
    dynamics = dynamics or AnalyticDynamics(config.dynamics)
    activations = representation.slot_activations
    if activations is None or activations.shape[0] < 3:
        empty = np.zeros(max(1, len(representation.frames)))
        return Spotter(spotting_config, video.fps).run(video.video_id, empty, 0)

    n, k = activations.shape
    slot_error = np.zeros((n, k), dtype=np.float64)
    velocity = np.zeros(k)
    for t in range(1, n):
        predicted, _variance = dynamics.step(activations[t - 1], momentum=velocity)
        slot_error[t] = np.abs(activations[t] - predicted)
        velocity = 0.6 * velocity + 0.4 * (activations[t] - activations[t - 1])

    t_start = representation.frames[0] if representation.frames else 0

    if external_curve is not None:
        curve = np.asarray(external_curve, dtype=np.float64).reshape(-1)
        if curve.size != n:
            raise ValueError(
                f"external detection curve has {curve.size} frames but the "
                f"representation carries {n}")
        delta = curve - float(np.median(curve))
        return Spotter(spotting_config, video.fps).run(
            video_id=video.video_id, delta=delta, t_start=t_start,
            head_motion=None, slow_prediction=None,
            slot_errors=slot_error,
            coherence_gate=representation.coherence_gate,
            low_confidence_spans=representation.low_confidence,
        )

    delta = slot_error.sum(axis=1)

    return Spotter(spotting_config, video.fps).run(
        video_id=video.video_id, delta=delta, t_start=t_start,
        head_motion=representation.head_motion,
        slow_prediction=representation.slow_prediction,
        slot_errors=slot_error,
        coherence_gate=representation.coherence_gate,
        low_confidence_spans=representation.low_confidence,
    )


def apply_clip_activations(representation: RepresentationOutput,
                           activations: np.ndarray) -> None:
    acts = np.asarray(activations, dtype=np.float64)
    if representation.slot_activations is None or \
            acts.shape != representation.slot_activations.shape:
        raise ValueError("transition-head activations do not match the representation")
    representation.slot_activations = acts

    bank = representation.slot_bank
    for i, t in enumerate(representation.frames):
        named = {au: float(acts[i, j]) for j, au in enumerate(SLOT_AUS)}
        for au in SLOT_AUS:
            series = bank._activations[au]
            if i < len(series):
                series[i] = named[au]
        state = representation.stream.get(t)
        if state is not None:
            state.slot_activations = named


def _merge_proposal_sets(
    primary: Sequence[CandidateInterval],
    extra: Sequence[Tuple[int, int, int, float]],
    truths: Sequence[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Tuple[List[CandidateInterval], Dict[str, Any]]:
    pool: List[CandidateInterval] = list(primary)
    softnet_ids = set()
    for t_on, t_off, apex, peak in extra:
        candidate = CandidateInterval(
            cid="", t_on=t_on, t_off=t_off, apex=apex, peak_S=peak,
            attribution={}, physio_overlap=False, channel="micro",
            notes="softnet peak proposal")
        pool.append(candidate)
        softnet_ids.add(id(candidate))

    n_primary = len(primary)
    n_softnet = len(extra)

    matches = greedy_match([(c.t_on, c.t_off) for c in pool], list(truths),
                          iou_threshold=iou_threshold)

    merged: List[CandidateInterval] = []
    n_tp_primary = 0
    n_tp_softnet = 0
    for candidate, match in zip(pool, matches):
        if not match.is_tp_strict:
            continue
        merged.append(candidate)
        if id(candidate) in softnet_ids:
            n_tp_softnet += 1
        else:
            n_tp_primary += 1

    merged.sort(key=lambda proposal: proposal.t_on)
    for order, proposal in enumerate(merged):
        proposal.cid = f"p{order + 1:02d}"

    stats = {
        "n_primary": n_primary,
        "n_softnet": n_softnet,
        "n_pool": n_primary + n_softnet,
        "n_truths": len(truths),
        "iou_threshold": iou_threshold,
        "n_tp_primary": n_tp_primary,
        "n_tp_softnet": n_tp_softnet,
        "n_merged": len(merged),
        "n_truths_unmatched": len(truths) - len(merged),
        "primary_intervals": [[c.t_on, c.t_off] for c in primary],
        "softnet_intervals": [[t_on, t_off] for t_on, t_off, _apex, _peak in extra],
        "truth_intervals": [[int(a), int(b)] for a, b in truths],
    }

    if truths and not merged:
        best_iou_per_truth = []
        for truth in truths:
            best = 0.0
            for candidate in pool:
                best = max(best, _interval_iou((candidate.t_on, candidate.t_off), truth))
            best_iou_per_truth.append(round(best, 4))
        stats["best_iou_per_truth"] = best_iou_per_truth
    return merged, stats


@dataclass
class PipelineResult:
    video_id: str
    state: MEWMState
    representation: RepresentationOutput
    spotting: SpottingResult
    episodic: EpisodicMemory
    elapsed_s: float = 0.0
    annotated_events: List[Any] = field(default_factory=list)
    cfi_by_cid: Dict[str, Dict[str, float]] = field(default_factory=dict)
    measurements_by_cid: Dict[str, Any] = field(default_factory=dict)
    saturated_by_cid: Dict[str, bool] = field(default_factory=dict)
    proposals_from_annotation: bool = False
    proposal_fusion: Optional[Dict[str, Any]] = None

    def final_answer(self, lang: str = "en") -> Dict[str, Any]:
        from .eval.answer_composer import compose_answer
        return compose_answer(self, self.annotated_events, self.cfi_by_cid,
                              self.measurements_by_cid, self.saturated_by_cid,
                              lang=lang)

    def answer(self) -> Dict[str, Any]:
        proposals = [
            {"proposal_id": i + 1, "onset": p.t_on, "offset": p.t_off, "apex": p.apex}
            for i, p in enumerate(self.state.proposals)
        ]
        analysis = []
        for index, proposal in enumerate(self.state.proposals, start=1):
            verdict = self.state.verdicts.get(proposal.cid)
            graph = self.state.au_graphs.get(proposal.cid)
            cot = self.state.causal_cots.get(proposal.cid)
            analysis.append({
                "proposal_id": index,
                "cid": proposal.cid,
                "interval": [proposal.t_on, proposal.t_off],
                "apex": proposal.apex,
                "coarse_label": verdict.e_coarse if verdict else "",
                "fine_label": verdict.e_fine if verdict else "",
                "confidence": verdict.confidence if verdict else 0.0,
                "suppression": verdict.suppression if verdict else "none",
                "active_aus": graph.active_aus if graph else [],
                "weak_aus": graph.weak_aus if graph else [],
                "au_order": graph.onset_order() if graph else [],
                "au_cot": cot.C.get("a", "") if cot else "",
                "es": cot.es if cot else {},
                "dc": cot.dc if cot else {},
                "k_crit": cot.k_crit if cot else [],
                "path": self.state.path_choices.get(proposal.cid, ""),
            })
        composed = self.final_answer()
        return {
            "video_id": self.video_id,
            "answer": composed["final_answer"],
            "part1_proposals": proposals,
            "part2_analysis": analysis,
            "answer_sections": composed["sections"],
            "format_report": composed.get("format_report", {}),
            "global_narrative": self.state.narrative.text if self.state.narrative else "",
            "n_detected": composed["n_detected"],
            "n_annotated": composed["n_annotated"],
            "proposals_from_annotation": self.proposals_from_annotation,
            "n_llm_calls": self.state.budget.llm_calls_used,
            "degradations": list(self.state.budget.degradations),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frames_processed": len(self.representation),
            "spotting": self.spotting.summary(),
            "proposal_fusion": self.proposal_fusion,
            "n_verdicts": len(self.state.verdicts),
            "paths": dict(self.state.path_choices),
            "llm_calls": self.state.budget.llm_calls_used,
            "degradations": self.state.budget.degradations,
            "elapsed_s": round(self.elapsed_s, 2),
        }


class MEWMPipeline:
    def __init__(
        self,
        config: Optional[MEWMConfig] = None,
        agents: Optional[Dict[str, Any]] = None,
        dynamics: Optional[Any] = None,
        lang: str = "en",
        checkpoint_dir: Optional[Path] = None,
        clip_spotter: Optional[Any] = None,
    ) -> None:
        self.config = config or load_config()
        self.lang = lang
        self.dynamics = dynamics or AnalyticDynamics(self.config.dynamics)
        self.service = RolloutService(self.dynamics, self.config.dynamics)
        self.agents = agents or {}
        self.checkpoint_dir = checkpoint_dir
        self.clip_spotter = clip_spotter

    @classmethod
    def build(
        cls, config: Optional[MEWMConfig] = None, client: Optional[Any] = None,
        lang: str = "en", **kwargs: Any,
    ) -> "MEWMPipeline":
        from .agents.critic import CriticAgent
        from .agents.perception import PerceptionAgent
        from .agents.reasoning import ReasoningAgent
        from .agents.structure import StructureAgent

        config = config or load_config()
        shared = {"config": config, "client": client, "lang": lang}
        agents = {
            "P": PerceptionAgent(config.llm.perception_model, **shared),
            "A": StructureAgent(config.llm.structure_model, **shared),
            "R": ReasoningAgent(config.llm.reasoning_model, **shared),
            "C": CriticAgent(config.llm.critic_model, **shared),
        }
        return cls(config=config, agents=agents, lang=lang, **kwargs)

    def run(
        self,
        video: LongVideo,
        question: str = "",
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        stride: int = 1,
        max_frames: int = 0,
        max_proposals: int = 0,
        annotated_events: Optional[Sequence[Any]] = None,
        use_annotated_proposals: bool = False,
        softnet_proposals: Optional[Sequence[Tuple[int, int, int, float]]] = None,
        iou_threshold: Optional[float] = None,
    ) -> PipelineResult:
        started = time.time()
        question = question or (TRIPLE_TASK_QUESTION_ZH if self.lang == "zh"
                                else TRIPLE_TASK_QUESTION_EN)
        resolved_iou_threshold = (
            self.config.evaluation.iou_threshold if iou_threshold is None
            else iou_threshold)

        representation = run_representation(video, self.config, t_start, t_end,
                                            stride, max_frames)
        external_curve = None
        if self.clip_spotter is not None and len(representation):
            clip_out = self.clip_spotter.infer(video, representation)
            external_curve = clip_out["curve"]
            if self.config.clip.replace_activations:
                apply_clip_activations(representation, clip_out["activations"])
        spotting = run_spotting(video, representation, self.config, self.dynamics,
                                external_curve=external_curve)

        state = MEWMState(video_meta=video.to_meta(), question=question)
        state.error_record = spotting.error_record
        state.macro_intervals = list(spotting.macro_intervals)
        state.slow_state_log = []

        events = list(annotated_events) if annotated_events is not None else list(video.events)
        micro_events = [e for e in events if getattr(e, "is_micro", True)]
        truths = [e.interval for e in micro_events]
        fusion_stats: Optional[Dict[str, Any]] = None
        if use_annotated_proposals:
            proposals = [
                CandidateInterval(
                    cid=f"g{i + 1:02d}", t_on=event.onset, t_off=event.offset,
                    apex=event.apex, peak_S=spotting.error_record.s_at(event.apex),
                    attribution={}, physio_overlap=False, channel="micro",
                    confirmed=True, notes="interval supplied from annotation (P4 upper bound)",
                )
                for i, event in enumerate(micro_events)
            ]
            if softnet_proposals:
                proposals, fusion_stats = _merge_proposal_sets(
                    proposals, softnet_proposals, truths,
                    iou_threshold=resolved_iou_threshold)
        else:
            proposals, fusion_stats = _merge_proposal_sets(
                list(spotting.proposals), softnet_proposals or [], truths,
                iou_threshold=resolved_iou_threshold)
        if max_proposals:
            if fusion_stats is not None:
                fusion_stats["n_before_max_proposals_cap"] = len(proposals)
            proposals = sorted(proposals, key=lambda p: -p.peak_S)[:max_proposals]
            proposals.sort(key=lambda p: p.t_on)
            if fusion_stats is not None:
                fusion_stats["n_after_max_proposals_cap"] = len(proposals)
        state.proposals = proposals

        episodic = EpisodicMemory(video.to_meta(), self.config.memory)
        episodic.index_error_record(spotting.error_record)
        for proposal in proposals:
            episodic.add_proposal(proposal)

        checkpoint = None
        if self.checkpoint_dir is not None:
            checkpoint = CheckpointStore(Path(self.checkpoint_dir) / "checkpoints.sqlite")

        contexts = {
            proposal.cid: self._build_context(video, representation, spotting,
                                              episodic, proposal, state)
            for proposal in proposals
        }

        orchestrator = Orchestrator(
            agents=self.agents, service=self.service, config=self.config,
            episodic=episodic, checkpoint=checkpoint, lang=self.lang,
        )
        ceiling_seconds = _effective_spotting_config(self.config, video.dataset).max_micro_seconds
        micro_frame_ceiling = max_micro_frames(video.dataset, ceiling_seconds)
        orchestrator.run(state, contexts, candidates=proposals,
                         max_micro_frames=micro_frame_ceiling)

        if checkpoint is not None:
            checkpoint.close()

        cfi_by_cid: Dict[str, Dict[str, float]] = {}
        for proposal in proposals:
            context = contexts.get(proposal.cid)
            cot = state.causal_cots.get(proposal.cid)
            graph = state.au_graphs.get(proposal.cid)
            if context is None or context.slot_trajectory is None:
                continue
            claimed = list(cot.k_crit) if (cot and cot.k_crit) else (
                graph.active_aus[:4] if graph else [])
            if not claimed:
                continue
            candidates = self._answer_candidates(cot)
            try:
                cfi_by_cid[proposal.cid] = {
                    au: self.service.mask(context.slot_trajectory, [au], candidates,
                                          caller="answer", cid=proposal.cid).mni
                    for au in claimed
                }
            except Exception as exc:
                LOGGER.warning("CFI failed for %s: %s", proposal.cid, exc)

        return PipelineResult(
            video_id=video.video_id, state=state, representation=representation,
            spotting=spotting, episodic=episodic, elapsed_s=time.time() - started,
            annotated_events=events, cfi_by_cid=cfi_by_cid,
            measurements_by_cid={cid: c.measurements for cid, c in contexts.items()},
            saturated_by_cid={cid: c.coherence_saturated for cid, c in contexts.items()},
            proposals_from_annotation=use_annotated_proposals,
            proposal_fusion=fusion_stats,
        )

    @staticmethod
    def _answer_candidates(cot: Optional[Any]) -> List[str]:
        from .knowledge.emotion_prototypes import FINE_EMOTIONS
        if cot is not None and cot.es:
            return list(cot.es)
        return list(FINE_EMOTIONS[:4])

    def _build_context(
        self, video: LongVideo, representation: RepresentationOutput,
        spotting: SpottingResult, episodic: EpisodicMemory,
        proposal: CandidateInterval,
        state: Optional[MEWMState] = None,
    ) -> ProposalContext:
        from .agents.structure import build_reference_graph

        frames = [t for t in representation.frames if proposal.t_on <= t <= proposal.t_off]
        trajectory = representation.slot_bank.matrix(proposal.t_on, proposal.t_off)

        apex_state = representation.stream.get(proposal.apex)
        if apex_state is None:
            fallback_frames = sorted(
                (t for t in frames if t in representation.stream),
                key=lambda t: abs(t - proposal.apex),
            )
            if fallback_frames:
                apex_state = representation.stream.get(fallback_frames[0])
                LOGGER.warning(
                    "proposal %s: apex frame %d has no representation entry; "
                    "using nearest in-interval frame %d instead",
                    proposal.cid, proposal.apex, fallback_frames[0],
                )
            else:
                LOGGER.warning(
                    "proposal %s: no frame in interval [%d, %d] has a representation "
                    "entry (apex %d); P.verify/A.encode will receive zero motion "
                    "measurements for this proposal",
                    proposal.cid, proposal.t_on, proposal.t_off, proposal.apex,
                )
        measurements = apex_state.measurements if apex_state else []
        readout = {
            au: float(representation.slot_bank.peak(au, proposal.t_on, proposal.t_off))
            for au in SLOT_AUS
        }
        from .engines.v2_slots import SlotReadout, coherence_is_saturated, select_active_slots
        representation_cfg = self.config.representation
        preselected, preweak = select_active_slots(
            {au: SlotReadout(au, value, value, 0.0, 0.0) for au, value in readout.items()},
            threshold=representation_cfg.activation_threshold,
            max_active=representation_cfg.max_active_aus,
            relative_margin=representation_cfg.activation_relative_margin,
            weak_threshold=representation_cfg.weak_threshold,
        )
        saturated = coherence_is_saturated(measurements)

        trajectories = {
            au: representation.slot_bank.trajectory(au, proposal.t_on, proposal.t_off)
            for au in SLOT_AUS
        }
        reference_graph, graph_questions = build_reference_graph(
            trajectories, frames, cid=proposal.cid, fps=video.fps,
            interaction=self.dynamics, n_permutations=60,
            active_aus=preselected, weak_aus=preweak,
        )

        baseline = episodic.outside_proposal_summary()
        before = [b for b in baseline if b["interval"][1] < proposal.t_on]
        baseline_context = ""
        if before:
            last = before[-1]
            baseline_context = (
                f"frames {last['interval'][0]}-{last['interval'][1]} before this "
                f"proposal averaged S = {last.get('mean', 0.0)} "
                f"(max {last.get('max', 0.0)})"
            )

        image_paths = []
        for t in (proposal.t_on, proposal.apex, proposal.t_off):
            path = video.paths.frame(t)
            if path.is_file():
                image_paths.append(str(path))

        context = ProposalContext(
            cid=proposal.cid, interval=(proposal.t_on, proposal.t_off),
            measurements=measurements, slot_trajectory=trajectory,
            slot_readout=readout, frames=frames, reference_graph=reference_graph,
            image_paths=image_paths, baseline_context=baseline_context,
            preselected_active=preselected, preselected_weak=preweak,
            coherence_saturated=saturated,
        )
        if state is not None:
            for question in graph_questions:
                state.register_question(question)
        return context


__all__ = [
    "RepresentationOutput", "run_representation", "run_spotting",
    "apply_clip_activations", "PipelineResult", "MEWMPipeline",
]
