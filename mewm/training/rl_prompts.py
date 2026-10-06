"""RL prompt templates: system and user prompts for policy-training rollouts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..data.datasets import ExpressionEvent, LongVideo
from ..knowledge.au_anatomy import SLOT_AUS

LOGGER = logging.getLogger(__name__)

KIND_DETERMINISTIC = "deterministic"
KIND_EVENT_REASONING = "event_reasoning"
KIND_VIDEO_REASONING = "video_reasoning"

ELIGIBLE_KINDS = (KIND_EVENT_REASONING, KIND_VIDEO_REASONING)

_DETERMINISTIC_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("How many expression events", "event count is fixed by the annotation"),
    ("How many micro-expression events", "micro count is fixed by the annotation"),
    ("How many macro-expression events", "macro count is fixed by the annotation"),
    ("What distinct action units", "the AU inventory is fixed by the annotation"),
    ("What is the expression type", "the event type is fixed by the annotation"),
)

_EVENT_ANCHOR = re.compile(
    r"In the (\d+)-th expression event of this video "
    r"\(frames (\d+)-(\d+), apex (\d+)\)", re.IGNORECASE)


def classify_question(question: str) -> Tuple[str, str]:
    text = (question or "").strip()
    if not text:
        return KIND_DETERMINISTIC, "empty question"

    for prefix, reason in _DETERMINISTIC_PREFIXES:
        if text.startswith(prefix):
            return KIND_DETERMINISTIC, reason

    if _EVENT_ANCHOR.search(text):
        return KIND_EVENT_REASONING, "free-form reasoning anchored on one event"
    if text.startswith("Reason over the whole video"):
        return KIND_VIDEO_REASONING, "free-form reasoning over the whole video"

    return KIND_DETERMINISTIC, "unrecognised template; treated as fixed-answer"


def parse_event_anchor(question: str) -> Optional[Dict[str, int]]:
    match = _EVENT_ANCHOR.search(question or "")
    if not match:
        return None
    ordinal, onset, offset, apex = (int(g) for g in match.groups())
    return {"ordinal": ordinal, "onset": onset, "offset": offset, "apex": apex}


@dataclass
class VideoEvidence:

    video: str
    n_frames: int = 0
    fps: float = 30.0
    frame_offset: int = 0
    slot_order: Sequence[str] = field(default_factory=lambda: list(SLOT_AUS))
    activations: Optional[np.ndarray] = None
    proposals: List[Tuple[int, int, int, float]] = field(default_factory=list)
    macro_intervals: List[Tuple[int, int]] = field(default_factory=list)
    error_shares: Dict[str, float] = field(default_factory=dict)
    scene_r2: float = 0.0
    status: str = "ok"
    note: str = ""

    @property
    def available(self) -> bool:
        return self.status == "ok" and self.activations is not None

    def window_slots(self, onset: int, offset: int, top_k: int = 6) -> List[Dict[str, Any]]:
        if self.activations is None or self.activations.size == 0:
            return []
        lo = max(0, onset - self.frame_offset)
        hi = min(self.activations.shape[0] - 1, offset - self.frame_offset)
        if hi < lo:
            return []
        window = self.activations[lo:hi + 1]
        baseline = np.median(self.activations, axis=0)
        delta = window.mean(axis=0) - baseline
        order = np.argsort(-np.abs(delta))[:top_k]
        return [
            {
                "slot": self.slot_order[i] if i < len(self.slot_order) else f"slot{i}",
                "mean": round(float(window[:, i].mean()), 4),
                "delta_vs_video_median": round(float(delta[i]), 4),
                "peak": round(float(window[:, i].max()), 4),
            }
            for i in order
        ]

    def overlapping_proposals(self, onset: int, offset: int) -> List[Dict[str, Any]]:
        rows = []
        for t_on, t_off, apex, peak in self.proposals:
            lo, hi = max(t_on, onset), min(t_off, offset)
            inter = max(0, hi - lo + 1)
            if inter <= 0:
                continue
            union = (t_off - t_on + 1) + (offset - onset + 1) - inter
            rows.append({"interval": [t_on, t_off], "apex": apex,
                         "peak_S": round(float(peak), 3),
                         "iou_with_query": round(inter / union, 4) if union else 0.0})
        rows.sort(key=lambda r: -r["iou_with_query"])
        return rows[:5]


def evidence_from_spotting(
    video: LongVideo, representation: Any, spotting: Any,
) -> VideoEvidence:
    activations = getattr(representation, "slot_activations", None)
    frames = list(getattr(representation, "frames", []) or [])
    record = getattr(spotting, "error_record", None)
    offset = int(getattr(record, "t_start", 0) or 0)
    decomposition = getattr(spotting, "decomposition", None)

    micro = getattr(spotting, "micro_intervals", None)
    if micro is None:
        micro = getattr(spotting, "proposals", [])

    return VideoEvidence(
        video=video.video_key,
        n_frames=len(frames),
        fps=float(video.fps),
        frame_offset=offset,
        slot_order=list(getattr(representation, "slot_order", SLOT_AUS) or SLOT_AUS),
        activations=activations,
        proposals=[(int(p.t_on), int(p.t_off), int(p.apex), float(p.peak_S))
                   for p in micro],
        macro_intervals=[(int(p.t_on), int(p.t_off))
                         for p in getattr(spotting, "macro_intervals", [])],
        error_shares=dict(decomposition.shares()) if decomposition is not None else {},
        scene_r2=float(getattr(decomposition, "scene_r2", 0.0) or 0.0),
    )


def unavailable_evidence(video: str, reason: str) -> VideoEvidence:
    return VideoEvidence(video=video, status="unavailable", note=reason)


def _match_event(video: LongVideo, anchor: Dict[str, int]) -> Optional[ExpressionEvent]:
    target = (anchor["onset"], anchor["offset"])
    for event in video.events:
        if event.interval == target:
            return event
    best, best_iou = None, 0.0
    for event in video.events:
        overlap = event.iou(target)
        if overlap > best_iou:
            best, best_iou = event, overlap
    return best if best_iou >= 0.5 else None


def _truth_for_event(event: ExpressionEvent) -> Dict[str, Any]:
    return {
        "interval": list(event.interval),
        "apex": event.apex,
        "fine": event.fine_label,
        "coarse": event.coarse_label,
        "aus": list(event.aus),
        "type": event.expression_type,
    }


def _render_evidence(evidence: VideoEvidence, anchor: Optional[Dict[str, int]]) -> str:
    if not evidence.available:
        return ("PERCEPTUAL EVIDENCE: unavailable for this video "
                f"({evidence.note or 'no reason recorded'}). Answer from the question's "
                "stated window alone and say plainly that the slot evidence was missing; "
                "do not invent activations.")

    lines = [
        "PERCEPTUAL EVIDENCE (frozen representation + spotting engines, not ground truth):",
        f"- video {evidence.video}: {evidence.n_frames} measured frames at "
        f"{evidence.fps:g} fps",
        f"- prediction-error decomposition: {evidence.error_shares}, scene R^2 "
        f"{evidence.scene_r2:.3f}",
        f"- the spotter proposed {len(evidence.proposals)} micro-scale interval(s) and "
        f"{len(evidence.macro_intervals)} macro-scale interval(s) in this video",
    ]

    if anchor:
        slots = evidence.window_slots(anchor["onset"], anchor["offset"])
        overlapping = evidence.overlapping_proposals(anchor["onset"], anchor["offset"])
        lines.append(
            f"- slot activation over the queried window {anchor['onset']}-"
            f"{anchor['offset']}, as a deviation from this video's own median "
            f"(so a resting posture does not read as a movement):")
        for row in slots or [{"slot": "(no slots measured in this window)"}]:
            lines.append(f"    {row}")
        lines.append("- engine proposals overlapping the queried window:")
        for row in overlapping or [{"note": "none -- the spotter found no interval here"}]:
            lines.append(f"    {row}")
    else:
        strongest = sorted(evidence.proposals, key=lambda p: -p[3])[:8]
        lines.append("- strongest proposals by peak prediction error S:")
        for t_on, t_off, apex, peak in strongest or []:
            lines.append(f"    {{'interval': [{t_on}, {t_off}], 'apex': {apex}, "
                         f"'peak_S': {peak:.3f}}}")
        if not strongest:
            lines.append("    none -- the spotter proposed nothing in this video")
    return "\n".join(lines)


@dataclass
class PromptLedgerEntry:

    video_id: str
    video: str
    subject: str
    kind: str
    admitted: bool
    reason: str
    question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"video_id": self.video_id, "video": self.video, "subject": self.subject,
                "kind": self.kind, "admitted": self.admitted, "reason": self.reason,
                "question": self.question[:160]}


def build_rl_prompts(
    dataset: str,
    videos: Sequence[LongVideo],
    qa_rows: Iterable[Dict[str, Any]],
    evidence_by_video: Optional[Dict[str, VideoEvidence]] = None,
    include_kinds: Sequence[str] = ELIGIBLE_KINDS,
) -> Tuple[List[Dict[str, Any]], List[PromptLedgerEntry]]:
    evidence_by_video = evidence_by_video or {}
    by_key = {v.video_key: v for v in videos}
    allowed = set(include_kinds)

    prompts: List[Dict[str, Any]] = []
    ledger: List[PromptLedgerEntry] = []

    for row in qa_rows:
        video_key = str(row.get("video", ""))
        video = by_key.get(video_key)
        video_id = str(row.get("video_id", ""))
        question = str(row.get("question", ""))
        subject = video.subject if video else ""

        if video is None:
            ledger.append(PromptLedgerEntry(
                video_id, video_key, subject, "unknown", False,
                "video is not in this fold's pool", question))
            continue

        kind, reason = classify_question(question)
        if kind not in allowed:
            ledger.append(PromptLedgerEntry(
                video_id, video_key, subject, kind, False, reason, question))
            continue

        anchor = parse_event_anchor(question)
        truth: Dict[str, Any] = {}
        if kind == KIND_EVENT_REASONING:
            if anchor is None:
                ledger.append(PromptLedgerEntry(
                    video_id, video_key, subject, kind, False,
                    "event-anchored question whose window could not be parsed", question))
                continue
            event = _match_event(video, anchor)
            if event is None:
                ledger.append(PromptLedgerEntry(
                    video_id, video_key, subject, kind, False,
                    f"no annotated event matches the stated window "
                    f"{anchor['onset']}-{anchor['offset']}", question))
                continue
            truth = _truth_for_event(event)

        evidence = evidence_by_video.get(
            video_key, unavailable_evidence(video_key, "video not spotted in this run"))

        prompts.append({
            "id": video_id or f"{dataset}_{video_key}_{len(prompts)}",
            "question": question,
            "video": video_key,
            "subject": subject,
            "dataset": dataset,
            "kind": kind,
            "truth": truth,
            "evidence_text": _render_evidence(evidence, anchor),
            "evidence_status": evidence.status,
            "anchor": anchor or {},
            "n_events": len(video.events),
            "n_micro": len(video.micro_events()),
        })
        ledger.append(PromptLedgerEntry(
            video_id, video_key, subject, kind, True, reason, question))

    LOGGER.info("%s: admitted %d/%d reference instruction(s) for augmentation",
                dataset, len(prompts), len(ledger))
    return prompts, ledger


def ledger_by_subject(ledger: Sequence[PromptLedgerEntry]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for entry in ledger:
        bucket = out.setdefault(entry.subject or "(unknown)", {
            "n_reference_instructions": 0, "n_admitted": 0, "n_excluded": 0,
            "admitted_by_kind": {}, "excluded_by_reason": {}, "videos": set(),
        })
        bucket["n_reference_instructions"] += 1
        bucket["videos"].add(entry.video)
        if entry.admitted:
            bucket["n_admitted"] += 1
            bucket["admitted_by_kind"][entry.kind] = (
                bucket["admitted_by_kind"].get(entry.kind, 0) + 1)
        else:
            bucket["n_excluded"] += 1
            bucket["excluded_by_reason"][entry.reason] = (
                bucket["excluded_by_reason"].get(entry.reason, 0) + 1)
    for bucket in out.values():
        bucket["videos"] = sorted(bucket["videos"])
        bucket["n_videos"] = len(bucket["videos"])
    return out


__all__ = [
    "KIND_DETERMINISTIC", "KIND_EVENT_REASONING", "KIND_VIDEO_REASONING",
    "ELIGIBLE_KINDS", "VideoEvidence", "PromptLedgerEntry",
    "classify_question", "parse_event_anchor", "evidence_from_spotting",
    "unavailable_evidence", "build_rl_prompts", "ledger_by_subject",
]
