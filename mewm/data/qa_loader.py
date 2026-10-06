"""QA pair loader: reads and normalises question–answer sets per dataset split."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .paths import DATASETS, find_qa_runs, from_record_path

LOGGER = logging.getLogger(__name__)

QUESTION_TYPES: Tuple[str, ...] = (
    "count_expression", "count_micro", "count_macro", "au_set", "event_type",
    "localize_expression", "localize_micro", "localize_macro", "reason_full",
    "segment_analysis",
)

_SEGMENT_RE = re.compile(
    r"in the\s+(?P<idx>\d+)-th\s+expression event of this video\s*"
    r"\(frames\s+(?P<onset>\d+)-(?P<offset>\d+),\s*apex\s+(?P<apex>\d+)\)",
    re.IGNORECASE,
)

_QUESTION_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("reason_full", ("reason over the whole video",)),
    ("localize_micro", ("localize every micro-expression event",)),
    ("localize_macro", ("localize every macro-expression event",)),
    ("localize_expression", ("localize every event",)),
    ("au_set", ("what distinct action units",)),
    ("event_type", ("what is the expression type of",)),
    ("count_micro", ("how many micro-expression events",)),
    ("count_macro", ("how many macro-expression events",)),
    ("count_expression", ("how many expression events",)),
)


def classify_question(question: str) -> str:
    text = question or ""
    lowered = text.strip().lower()
    for qtype, needles in _QUESTION_RULES:
        if any(needle in lowered for needle in needles):
            return qtype
    if _SEGMENT_RE.search(text):
        return "segment_analysis"
    return "other"


def parse_segment_question(question: str) -> Optional[Dict[str, int]]:
    match = _SEGMENT_RE.search(question or "")
    if not match:
        return None
    return {
        "index": int(match.group("idx")),
        "onset": int(match.group("onset")),
        "offset": int(match.group("offset")),
        "apex": int(match.group("apex")),
    }


TRIPLE_TASK_QUESTION_EN = (
    "Reason over the whole video: how many micro-expression events does it contain, "
    "what coarse-grained and fine-grained emotion does each of them convey, and what "
    "happens inside every micro-expression segment?"
)
TRIPLE_TASK_QUESTION_ZH = (
    "请对输入的微表情视频进行微表情发生帧的定位；对定位到的每个片段分析其情感含义"
    "（给出粗粒度与细粒度标签）；并生成覆盖时间定位与情感分析全过程的完整描述。"
)

_LOCALISATION_RE = re.compile(
    r"(?P<idx>\d+)-th\s+(?P<kind>micro|macro)-expression:\s*frames\s+"
    r"(?P<onset>\d+)-(?P<offset>\d+)\s*\(apex\s+(?P<apex>\d+)\)",
    re.IGNORECASE,
)


@dataclass
class QAItem:

    video_id: str
    video: str
    question: str
    answer: Any
    qtype: str = ""
    index: int = 0

    @property
    def is_triple_task(self) -> bool:
        return self.qtype == "reason_full"

    @property
    def answer_text(self) -> str:
        return self.answer if isinstance(self.answer, str) else json.dumps(
            self.answer, ensure_ascii=False
        )

    def parse_localisations(self) -> List[Dict[str, int | str]]:
        out: List[Dict[str, int | str]] = []
        for match in _LOCALISATION_RE.finditer(self.answer_text):
            out.append({
                "index": int(match.group("idx")),
                "kind": f"{match.group('kind').lower()}-expression",
                "onset": int(match.group("onset")),
                "offset": int(match.group("offset")),
                "apex": int(match.group("apex")),
            })
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"video_id": self.video_id, "video": self.video,
                "question": self.question, "answer": self.answer, "qtype": self.qtype}


@dataclass
class MotionObservation:

    roi: int
    region_name: str
    direction_deg: float
    direction_label: str
    magnitude_px: float
    magnitude_class: str
    coherence: float
    coherence_label: str
    au_candidates: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MotionObservation":
        return cls(
            roi=int(payload.get("roi", 0)),
            region_name=str(payload.get("region_name", "")),
            direction_deg=float(payload.get("direction_deg", 0.0)),
            direction_label=str(payload.get("direction_label", "")),
            magnitude_px=float(payload.get("magnitude_px", 0.0)),
            magnitude_class=str(payload.get("magnitude_class", "")),
            coherence=float(payload.get("coherence", 0.0)),
            coherence_label=str(payload.get("coherence_label", "")),
            au_candidates=list(payload.get("au_candidates", [])),
        )

    def fitting_aus(self, level: str = "FIT") -> List[str]:
        order = {"FIT": 0, "PARTIAL": 1, "NO-FIT": 2}
        ceiling = order.get(level, 0)
        return [c["au"] for c in self.au_candidates
                if order.get(str(c.get("direction_fit", "NO-FIT")), 2) <= ceiling]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi": self.roi, "region_name": self.region_name,
            "au_candidates": self.au_candidates, "direction_deg": self.direction_deg,
            "direction_label": self.direction_label, "magnitude_px": self.magnitude_px,
            "magnitude_class": self.magnitude_class, "coherence": self.coherence,
            "coherence_label": self.coherence_label,
        }


@dataclass
class AUCorrelation:

    nodes: List[str] = field(default_factory=list)
    active_aus: List[str] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    main_path: List[str] = field(default_factory=list)
    matrix: List[List[float]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AUCorrelation":
        return cls(
            nodes=list(payload.get("nodes", [])),
            active_aus=list(payload.get("active_aus", [])),
            edges=list(payload.get("edges", [])),
            main_path=list(payload.get("main_path", [])),
            matrix=list(payload.get("matrix", [])),
        )

    def strongest_edges(self, n: int = 3) -> List[Dict[str, Any]]:
        return sorted(self.edges, key=lambda e: -float(e.get("weight", 0.0)))[:n]

    def terminal_emotion(self) -> str:
        if self.main_path and not str(self.main_path[-1]).upper().startswith("AU"):
            return str(self.main_path[-1]).lower()
        return ""

    def au_path(self) -> List[str]:
        return [p for p in self.main_path if str(p).upper().startswith("AU")]


@dataclass
class ObservationRecord:

    sample_id: str
    video_id: str
    video: str
    dataset: str
    expression_type: str
    coarse_label: str
    fine_label: str
    frame_span: Tuple[int, int, int]
    flow_gap_k: int
    annotated_aus: List[str] = field(default_factory=list)
    related_supplementary_aus: List[str] = field(default_factory=list)
    w_matrix: AUCorrelation = field(default_factory=AUCorrelation)
    static_description: str = ""
    dynamic_description: str = ""
    question: str = ""
    video_frames: List[str] = field(default_factory=list)
    flow_frames: List[Dict[str, Any]] = field(default_factory=list)
    interval_flow_frames: List[Dict[str, Any]] = field(default_factory=list)
    landmarks: List[List[int]] = field(default_factory=list)
    phase_observations: Dict[str, List[MotionObservation]] = field(default_factory=dict)


    @property
    def onset(self) -> int:
        return self.frame_span[0]

    @property
    def apex(self) -> int:
        return self.frame_span[1]

    @property
    def offset(self) -> int:
        return self.frame_span[2]

    @property
    def active_aus(self) -> List[str]:
        return self.w_matrix.active_aus or self.related_supplementary_aus

    def observations(self, phase: str = "AP_ON") -> List[MotionObservation]:
        return self.phase_observations.get(phase, [])

    def salient_rois(self, phase: str = "AP_ON", coherence_min: float = 0.6,
                     magnitude_min: float = 0.15) -> List[MotionObservation]:
        return [o for o in self.observations(phase)
                if o.magnitude_px >= magnitude_min or o.coherence >= coherence_min]

    def frame_paths(self) -> List[Path]:
        return [from_record_path(p) for p in self.video_frames]

    def flow_paths(self) -> List[Path]:
        return [from_record_path(str(f.get("flow_image", ""))) for f in self.flow_frames]

    def aligned_pairs(self) -> List[Dict[str, Any]]:
        out = []
        for item in self.interval_flow_frames:
            pair = item.get("frame_pair") or [0, 0]
            out.append({
                "t": int(item.get("flow_frame", pair[-1])),
                "flow_pair": [int(pair[0]), int(pair[-1])],
                "phase": item.get("phase", ""),
                "flow": str(item.get("flow_image", "")),
            })
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id, "video_id": self.video_id, "video": self.video,
            "dataset": self.dataset, "expression_type": self.expression_type,
            "coarse_label": self.coarse_label, "fine_label": self.fine_label,
            "frame_span": list(self.frame_span), "flow_gap_k": self.flow_gap_k,
            "annotated_aus": self.annotated_aus,
            "related_supplementary_aus": self.related_supplementary_aus,
            "active_aus": self.active_aus,
            "main_path": self.w_matrix.main_path,
            "edges": self.w_matrix.edges,
            "n_observations": {k: len(v) for k, v in self.phase_observations.items()},
        }


_PHASE_RE = re.compile(r"\b(AP_ON|ON_OFF|AP_OFF)\b\s*:\s*(\{.*?\})(?=\s+(?:AP_ON|ON_OFF|AP_OFF)\b\s*:|\s+B\.|\Z)",
                       re.DOTALL)


def parse_dynamic_description(text: str) -> Dict[str, List[MotionObservation]]:
    out: Dict[str, List[MotionObservation]] = {}
    if not text:
        return out
    part_a = text.split("B. W-matrix", 1)[0]
    for phase, blob in _PHASE_RE.findall(part_a):
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            LOGGER.debug("unparsable %s block in dynamic description", phase)
            continue
        out[phase] = [
            MotionObservation.from_dict(item)
            for item in payload.get("motion_observations", [])
        ]
    return out


def _record_from_full(payload: Dict[str, Any]) -> ObservationRecord:
    conversations = payload.get("conversations", []) or []
    human = next((c for c in conversations if c.get("from") == "human"), {})
    model = next((c for c in conversations if c.get("from") == "gpt"), {})
    dynamic = str(model.get("Dynamic description", "") or "")
    span = payload.get("frame_span") or [0, 0, 0]
    while len(span) < 3:
        span.append(span[-1] if span else 0)

    consistency = payload.get("au_consistency", {}) or {}
    return ObservationRecord(
        sample_id=str(payload.get("id", "")),
        video_id=str(payload.get("video_id", "")),
        video=str(payload.get("video", "")),
        dataset=str(payload.get("source", "")),
        expression_type=str(payload.get("expression_type", "")),
        coarse_label=str(payload.get("coarse_label", "")),
        fine_label=str(payload.get("fine_label", "")),
        frame_span=(int(span[0]), int(span[1]), int(span[2])),
        flow_gap_k=int(payload.get("flow_gap_k", 0) or 0),
        annotated_aus=list(payload.get("annotated_aus", []) or []),
        related_supplementary_aus=list(consistency.get("related_supplementary_aus", []) or []),
        w_matrix=AUCorrelation.from_dict(payload.get("w_matrix", {}) or {}),
        static_description=str(model.get("Static description", "") or ""),
        dynamic_description=dynamic,
        question=str(human.get("value", "") or ""),
        video_frames=list(payload.get("video_frames", []) or []),
        flow_frames=list(payload.get("flow_frames", []) or []),
        interval_flow_frames=list(payload.get("interval_flow_frames", []) or []),
        landmarks=list(payload.get("landmark", []) or []),
        phase_observations=parse_dynamic_description(dynamic),
    )


class QASet:

    def __init__(self, dataset: str, run_dir: Path,
                 items: Sequence[QAItem], records: Sequence[ObservationRecord],
                 augmented_ids: Optional[Sequence[str]] = None) -> None:
        self.dataset = dataset
        self.run_dir = run_dir
        self.items: List[QAItem] = list(items)
        self.records: List[ObservationRecord] = list(records)
        self.augmented_ids: set = set(augmented_ids or ())
        self._by_video: Dict[str, List[QAItem]] = {}
        for item in self.items:
            self._by_video.setdefault(item.video, []).append(item)
        self._records_by_video: Dict[str, List[ObservationRecord]] = {}
        for record in self.records:
            self._records_by_video.setdefault(record.video, []).append(record)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[QAItem]:
        return iter(self.items)


    def videos(self) -> List[str]:
        return sorted(self._by_video)

    def for_video(self, video: str) -> List[QAItem]:
        return self._by_video.get(video, [])

    def by_type(self, qtype: str) -> List[QAItem]:
        return [i for i in self.items if i.qtype == qtype]

    def augmented_items(self) -> List[QAItem]:
        return [i for i in self.items if i.video_id in self.augmented_ids]

    def annotated_items(self) -> List[QAItem]:
        return [i for i in self.items if i.video_id not in self.augmented_ids]

    def triple_task(self, video: str) -> Optional[QAItem]:
        return next((i for i in self.for_video(video) if i.is_triple_task), None)

    def segment_questions(self, video: str) -> List[Tuple[QAItem, Dict[str, int]]]:
        out: List[Tuple[QAItem, Dict[str, int]]] = []
        for item in self.for_video(video):
            if item.qtype != "segment_analysis":
                continue
            span = parse_segment_question(item.question)
            if span:
                out.append((item, span))
        out.sort(key=lambda pair: pair[1]["onset"])
        return out

    def ground_truth_localisations(self, video: str, micro_only: bool = True) -> List[Dict[str, Any]]:
        qtype = "localize_micro" if micro_only else "localize_expression"
        item = next((i for i in self.for_video(video) if i.qtype == qtype), None)
        return item.parse_localisations() if item else []


    def observation_targets(self, video: str) -> List[ObservationRecord]:
        return self._records_by_video.get(video, [])

    def stats(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for item in self.items:
            by_type[item.qtype] = by_type.get(item.qtype, 0) + 1
        micro = [r for r in self.records if r.expression_type == "micro-expression"]
        return {
            "dataset": self.dataset,
            "run_dir": self.run_dir.name,
            "questions": len(self.items),
            "augmented_questions": len(self.augmented_ids),
            "videos": len(self._by_video),
            "question_types": by_type,
            "observation_records": len(self.records),
            "micro_records": len(micro),
            "records_with_observations": sum(1 for r in self.records if r.phase_observations),
        }


def load_qa_set(
    dataset: str,
    run_dir: Optional[Path | str] = None,
    with_full: bool = True,
    prefer_model: str = "claude-sonnet-5",
    augmented_fold: Optional[str] = None,
    allowed_videos: Optional[Sequence[str]] = None,
) -> Optional[QASet]:

    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}")

    target = Path(run_dir) if run_dir is not None else _select_run(dataset, prefer_model)

    if target is None or not Path(target).is_dir():
        LOGGER.info("no QA build for %s yet (pending)", dataset)
        return None

    target = Path(target)
    jsonl = _pick_jsonl(target)
    if jsonl is None:
        LOGGER.warning("QA build %s has no .jsonl", target)
        return None

    items: List[QAItem] = []
    with open(jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("skipping malformed QA line in %s", jsonl.name)
                continue
            video_id = str(payload.get("video_id", ""))
            question = str(payload.get("question", ""))
            items.append(QAItem(
                video_id=video_id,
                video=str(payload.get("video", "")),
                question=question,
                answer=payload.get("answer"),
                qtype=classify_question(question),
                index=_suffix_index(video_id),
            ))

    records: List[ObservationRecord] = []
    if with_full:
        full = jsonl.with_name(jsonl.stem + "_full.json")
        if full.is_file():
            try:
                payload = json.loads(full.read_text(encoding="utf-8"))
                records = [_record_from_full(item) for item in payload]
            except (json.JSONDecodeError, OSError) as exc:
                LOGGER.warning("could not read %s: %s", full.name, exc)

    augmented_ids: List[str] = []
    if augmented_fold:
        from ..training.qa_augment import load_augmented

        allowed = set(allowed_videos) if allowed_videos is not None else None
        for payload in load_augmented(dataset, augmented_fold, allowed):
            video_id = str(payload["video_id"])
            question = str(payload["question"])
            items.append(QAItem(
                video_id=video_id,
                video=str(payload["video"]),
                question=question,
                answer=payload["answer"],
                qtype=classify_question(question),
                index=_suffix_index(video_id),
            ))
            augmented_ids.append(video_id)

    return QASet(dataset, target, items, records, augmented_ids)


def _select_run(dataset: str, prefer_model: str) -> Optional[Path]:
    runs = find_qa_runs(dataset)
    if not runs:
        return None

    def rank(path: Path) -> Tuple[int, int, str]:
        name = path.name
        model_hit = 0 if (prefer_model and prefer_model in name) else 1
        smoke = 1 if "smoke" in name else 0
        return (model_hit, smoke, name)

    return sorted(runs, key=rank)[0]


def _pick_jsonl(run_dir: Path) -> Optional[Path]:
    candidates = [c for c in sorted(run_dir.glob("*.jsonl")) if "checkpoint" not in c.name]
    if not candidates:
        return None
    named = [c for c in candidates if "me_lvqa" in c.name]
    return (named or candidates)[0]


def _suffix_index(video_id: str) -> int:
    tail = video_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def load_all_qa(with_full: bool = True, prefer_model: str = "claude-sonnet-5") -> Dict[str, QASet]:
    out: Dict[str, QASet] = {}
    for name in DATASETS:
        qa = load_qa_set(name, with_full=with_full, prefer_model=prefer_model)
        if qa is not None:
            out[name] = qa
    return out


__all__ = [
    "QUESTION_TYPES", "classify_question", "TRIPLE_TASK_QUESTION_EN", "TRIPLE_TASK_QUESTION_ZH",
    "QAItem", "MotionObservation", "AUCorrelation", "ObservationRecord",
    "parse_dynamic_description", "parse_segment_question", "QASet", "load_qa_set", "load_all_qa",
]
