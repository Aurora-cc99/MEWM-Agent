"""Dataset loaders for CAS(ME)², SAMM, CAS(ME)³, and 4D-ME."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..config import ensure_pre_process_importable
from ..knowledge.emotion_prototypes import FINE_TO_COARSE, coarse_of
from .paths import (
    DATASETS, VideoPaths, clip_rel_dir, dataset_frame_root, flow_gap_of, fps_of,
    max_micro_frames,
)

LOGGER = logging.getLogger(__name__)

MICRO = "micro-expression"
MACRO = "macro-expression"


@dataclass
class ExpressionEvent:

    event_id: str
    onset: int
    apex: int
    offset: int
    expression_type: str = MICRO
    fine_label: str = ""
    coarse_label: str = ""
    raw_label: str = ""
    aus: List[str] = field(default_factory=list)
    subject: str = ""
    video_key: str = ""
    event_index: int = 0

    @property
    def duration(self) -> int:
        return self.offset - self.onset + 1

    @property
    def is_micro(self) -> bool:
        return self.expression_type == MICRO

    @property
    def interval(self) -> Tuple[int, int]:
        return (self.onset, self.offset)

    def iou(self, span: Tuple[int, int]) -> float:
        lo, hi = max(self.onset, span[0]), min(self.offset, span[1])
        inter = max(0, hi - lo + 1)
        union = self.duration + (span[1] - span[0] + 1) - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id, "onset": self.onset, "apex": self.apex,
            "offset": self.offset, "type": self.expression_type,
            "fine": self.fine_label, "coarse": self.coarse_label,
            "aus": list(self.aus), "subject": self.subject,
        }


@dataclass
class LongVideo:

    dataset: str
    video_key: str
    subject: str
    folder_rel: str
    fps: float
    flow_gap: int
    frame_prefix: str = ""
    frame_ext: str = ".jpg"
    frame_digits: int = 0
    frame_lo: int = 0
    frame_hi: int = 0
    events: List[ExpressionEvent] = field(default_factory=list)

    _paths: Optional[VideoPaths] = field(default=None, repr=False, compare=False)

    @property
    def paths(self) -> VideoPaths:
        if self._paths is None:
            self._paths = VideoPaths(
                self.dataset, self.folder_rel, frame_prefix=self.frame_prefix,
                frame_ext=self.frame_ext, frame_digits=self.frame_digits,
                flow_gap=self.flow_gap,
            )
        return self._paths

    @property
    def video_id(self) -> str:
        return f"{self.dataset}_{self.video_key}"

    @property
    def n_frames(self) -> int:
        return max(0, self.frame_hi - self.frame_lo + 1)

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def max_micro_frames(self) -> int:
        return max_micro_frames(self.dataset)

    def micro_events(self) -> List[ExpressionEvent]:
        return [e for e in self.events if e.is_micro]

    def macro_events(self) -> List[ExpressionEvent]:
        return [e for e in self.events if not e.is_micro]

    def ground_truth_intervals(self, micro_only: bool = True) -> List[Tuple[int, int]]:
        events = self.micro_events() if micro_only else self.events
        return [e.interval for e in events]

    def to_meta(self):
        from ..schemas import VideoMeta
        return VideoMeta(
            video_id=self.video_id, dataset=self.dataset,
            path=str(self.paths.frame_dir), fps=self.fps, n_frames=self.n_frames,
            subject_id=self.subject, frame_lo=self.frame_lo, frame_hi=self.frame_hi,
            frame_prefix=self.frame_prefix, frame_ext=self.frame_ext,
            frame_digits=self.frame_digits,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset, "video_key": self.video_key,
            "subject": self.subject, "folder_rel": self.folder_rel, "fps": self.fps,
            "flow_gap": self.flow_gap, "frame_lo": self.frame_lo,
            "frame_hi": self.frame_hi, "n_frames": self.n_frames,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class DatasetIndex:

    dataset: str
    videos: List[LongVideo] = field(default_factory=list)
    fps: float = 30.0
    flow_gap: int = 1
    micro_mean_length: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.videos)

    def __iter__(self) -> Iterator[LongVideo]:
        return iter(self.videos)

    def by_key(self, video_key: str) -> Optional[LongVideo]:
        return next((v for v in self.videos if v.video_key == video_key), None)

    def subjects(self) -> List[str]:
        return sorted({v.subject for v in self.videos if v.subject})

    def by_subject(self, subject: str) -> List[LongVideo]:
        return [v for v in self.videos if v.subject == subject]

    def loso_folds(self) -> List[Tuple[str, List[LongVideo], List[LongVideo]]]:
        folds = []
        for subject in self.subjects():
            test = self.by_subject(subject)
            train = [v for v in self.videos if v.subject != subject]
            folds.append((subject, train, test))
        return folds

    def stats(self) -> Dict[str, object]:
        micro = sum(len(v.micro_events()) for v in self.videos)
        macro = sum(len(v.macro_events()) for v in self.videos)
        return {
            "dataset": self.dataset, "videos": len(self.videos),
            "subjects": len(self.subjects()), "micro_events": micro,
            "macro_events": macro, "fps": self.fps, "flow_gap": self.flow_gap,
            "micro_mean_length": round(self.micro_mean_length, 3),
            "total_frames": sum(v.n_frames for v in self.videos),
            "notes": list(self.notes),
        }


def lodo_folds(
    indices: Sequence["DatasetIndex"],
) -> List[Tuple[str, List[LongVideo], List[LongVideo]]]:
    named = [(index.dataset, index) for index in indices]
    duplicates = sorted({name for name, _ in named if
                         sum(1 for other, _ in named if other == name) > 1})
    if duplicates:
        raise ValueError(
            f"lodo_folds needs one index per dataset; {duplicates} appear more than once")
    if len(named) < 2:
        raise ValueError(
            f"leave-one-dataset-out needs at least two datasets, got "
            f"{[name for name, _ in named]}; use DatasetIndex.loso_folds for one")

    folds = []
    for held_out, index in named:
        test = list(index.videos)
        train = [v for other, source in named if other != held_out
                 for v in source.videos]
        folds.append((held_out, train, test))
    return folds


def _import_reference_loader():
    ensure_pre_process_importable()
    try:
        import me_datasets
        return me_datasets
    except Exception as exc:
        LOGGER.warning("pre_process/me_datasets.py unavailable (%s); "
                       "falling back to filesystem discovery", exc)
        return None


def _folder_rel_tail(dataset: str, folder_rel: str) -> str:
    from .paths import DATASET_FRAME_REL
    prefix = DATASET_FRAME_REL[dataset].strip("/") + "/"
    cleaned = folder_rel.replace("\\", "/").strip("/")
    return cleaned[len(prefix):] if cleaned.startswith(prefix) else cleaned


def _event_from_spec(spec, index: int) -> ExpressionEvent:
    fine = (getattr(spec, "fine_label", "") or "").strip().lower()
    coarse = (getattr(spec, "coarse_label", "") or "").strip().lower()
    if fine and not coarse:
        coarse = coarse_of(fine)
    aus_raw = getattr(spec, "au_annotation", "") or ""
    aus = _parse_aus(aus_raw)
    return ExpressionEvent(
        event_id=getattr(spec, "sample_id", "") or f"evt{index}",
        onset=int(getattr(spec, "onset", 0)),
        apex=int(getattr(spec, "apex", 0)),
        offset=int(getattr(spec, "offset", 0)),
        expression_type=getattr(spec, "expression_type", MICRO),
        fine_label=fine,
        coarse_label=coarse,
        raw_label=(getattr(spec, "raw_emotion_label", "") or ""),
        aus=aus,
        subject=getattr(spec, "subject", ""),
        video_key=getattr(spec, "video_key", ""),
        event_index=int(getattr(spec, "event_index", index)),
    )


def _parse_aus(raw: object) -> List[str]:
    if not raw:
        return []
    text = str(raw).replace("＋", "+").replace("，", ",")
    out: List[str] = []
    token = ""
    for ch in text + " ":
        if ch.isdigit():
            token += ch
        else:
            if token:
                number = int(token)
                if number < 50:
                    code = f"AU{number}"
                    if code not in out:
                        out.append(code)
                token = ""
    return sorted(out, key=lambda a: int(a[2:]))


def load_dataset(dataset: str, limit_videos: int = 0) -> DatasetIndex:

    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")

    module = _import_reference_loader()
    if module is None:
        return discover_dataset(dataset, limit_videos=limit_videos)

    try:
        bundle = module.load_dataset(dataset, limit_videos=limit_videos)
    except Exception as exc:
        LOGGER.warning("annotation load failed for %s (%s); "
                       "falling back to filesystem discovery", dataset, exc)
        index = discover_dataset(dataset, limit_videos=limit_videos)
        index.notes.append(f"annotation load failed: {exc}")
        return index

    gap = int(getattr(bundle, "gap", flow_gap_of(dataset)) or flow_gap_of(dataset))
    fps = float(getattr(bundle, "fps", fps_of(dataset)) or fps_of(dataset))

    videos: List[LongVideo] = []
    for spec in getattr(bundle, "videos", []):
        folder_rel = _folder_rel_tail(dataset, getattr(spec, "folder_rel", ""))
        video = LongVideo(
            dataset=dataset,
            video_key=getattr(spec, "video_key", "") or folder_rel.replace("/", "_"),
            subject=getattr(spec, "subject", ""),
            folder_rel=folder_rel,
            fps=float(getattr(spec, "fps", fps) or fps),
            flow_gap=gap,
            frame_prefix=getattr(spec, "frame_prefix", ""),
            frame_ext=getattr(spec, "frame_ext", ".jpg"),
            frame_digits=int(getattr(spec, "frame_digits", 0) or 0),
            frame_lo=int(getattr(spec, "frame_lo", 0) or 0),
            frame_hi=int(getattr(spec, "frame_hi", 0) or 0),
        )
        video.events = [
            _event_from_spec(event, i)
            for i, event in enumerate(getattr(spec, "events", []))
        ]
        videos.append(video)

    return DatasetIndex(
        dataset=dataset, videos=videos, fps=fps, flow_gap=gap,
        micro_mean_length=float(getattr(bundle, "micro_mean_length", 0.0) or 0.0),
        notes=list(getattr(bundle, "notes", [])),
    )


def discover_dataset(dataset: str, limit_videos: int = 0) -> DatasetIndex:

    root = dataset_frame_root(dataset)
    gap, fps = flow_gap_of(dataset), fps_of(dataset)
    videos: List[LongVideo] = []
    if not root.is_dir():
        return DatasetIndex(dataset=dataset, videos=[], fps=fps, flow_gap=gap,
                            notes=[f"frame root missing: {root}"])

    for clip_dir in _iter_clip_dirs(dataset, root):
        folder_rel = clip_dir.relative_to(root).as_posix()
        subject = folder_rel.split("/")[0]
        video_key = folder_rel.replace("/", "_")
        video = LongVideo(
            dataset=dataset, video_key=video_key, subject=subject,
            folder_rel=folder_rel, fps=fps, flow_gap=gap,
        )
        span = video.paths.scan_frame_range()
        if span is None:
            continue
        video.frame_lo, video.frame_hi, _ = span
        videos.append(video)
        if limit_videos and len(videos) >= limit_videos:
            break

    return DatasetIndex(dataset=dataset, videos=videos, fps=fps, flow_gap=gap,
                        notes=["filesystem discovery: no ground-truth annotations"])


def _iter_clip_dirs(dataset: str, root: Path) -> Iterator[Path]:
    if dataset == "samm":
        yield from (p for p in sorted(root.iterdir()) if p.is_dir())
        return
    if dataset == "casme3":
        for subject in sorted(p for p in root.iterdir() if p.is_dir()):
            for clip in sorted(p for p in subject.iterdir() if p.is_dir()):
                colour = clip / "color"
                if colour.is_dir():
                    yield colour
        return
    for subject in sorted(p for p in root.iterdir() if p.is_dir()):
        for clip in sorted(p for p in subject.iterdir() if p.is_dir()):
            yield clip


def load_all(limit_videos: int = 0, datasets: Optional[Sequence[str]] = None) -> Dict[str, DatasetIndex]:
    out: Dict[str, DatasetIndex] = {}
    for name in (datasets or DATASETS):
        if not dataset_frame_root(name).is_dir():
            LOGGER.info("skipping %s: frame root not present", name)
            continue
        out[name] = load_dataset(name, limit_videos=limit_videos)
    return out


def resolve_video(dataset: str, video_key: str) -> Optional[LongVideo]:
    index = load_dataset(dataset)
    return index.by_key(video_key)


__all__ = [
    "MICRO", "MACRO", "ExpressionEvent", "LongVideo", "DatasetIndex", "lodo_folds",
    "load_dataset", "discover_dataset", "load_all", "resolve_video",
]
