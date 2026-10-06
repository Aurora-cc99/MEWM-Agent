"""Dataset path resolution: maps dataset keys to raw-video and annotation dirs."""
from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import DATASET_ROOT, FLOW_ROOT, QTA_ROOT

DATASETS: Tuple[str, ...] = ("casme_sq", "samm", "casme3", "4dme")

DATASET_FPS: Dict[str, float] = {
    "casme_sq": 30.0, "samm": 200.0, "casme3": 30.0, "4dme": 60.0,
}

MICRO_CEILING_FRAMES: int = 200

DATASET_FRAME_REL: Dict[str, str] = {
    "casme_sq": "CASME_sq/rawpic_crop",
    "samm": "SAMMLV/SAMM_longvideos_crop",
    "casme3": "CASME3/part_A_split/part_A",
    "4dme": "4DME/long_gray_video/long gray video",
}

DATASET_LABEL_REL: Dict[str, str] = {
    "casme_sq": "CASME_sq/code_final.xlsx",
    "samm": "SAMMLV/SAMM_LongVideos_V3_Release.xlsx",
    "casme3": "CASME3/CAS(ME)3_part_A_v1.xls",
    "4dme": "4DME/Micro and Macro-expression labels.xlsx",
}

DATASET_FRAME_FORMAT: Dict[str, Tuple[str, int]] = {
    "casme_sq": ("img", 3),
    "samm": ("", 4),
    "casme3": ("", 0),
    "4dme": ("Frame_", 9),
}

DATASET_FLOW_GAP: Dict[str, int] = {
    "casme_sq": 7, "samm": 47, "casme3": 7, "4dme": 14,
}

DATASET_MAX_PROPOSALS: Dict[str, int] = {
    "casme_sq": 5, "samm": 5, "casme3": 0, "4dme": 0,
}

CASME3_MODALITY = "color"

QA_SUBDIR: Dict[str, str] = {
    "casme_sq": "casme_sq", "samm": "samm", "casme3": "casme3", "4dme": "4dme",
}


class DatasetPathError(FileNotFoundError):
    pass


def dataset_frame_root(dataset: str) -> Path:
    _check(dataset)
    return DATASET_ROOT / DATASET_FRAME_REL[dataset]


def dataset_flow_root(dataset: str) -> Path:
    _check(dataset)
    return FLOW_ROOT / DATASET_FRAME_REL[dataset]


def dataset_label_path(dataset: str) -> Path:
    _check(dataset)
    return DATASET_ROOT / DATASET_LABEL_REL[dataset]


def _check(dataset: str) -> None:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def fps_of(dataset: str) -> float:
    _check(dataset)
    return DATASET_FPS[dataset]


def flow_gap_of(dataset: str) -> int:
    _check(dataset)
    return DATASET_FLOW_GAP[dataset]


def max_proposals_of(dataset: str) -> int:
    _check(dataset)
    return DATASET_MAX_PROPOSALS[dataset]


def max_micro_frames(dataset: str, seconds: Optional[float] = None) -> int:
    _check(dataset)
    if seconds is None:
        return MICRO_CEILING_FRAMES
    return int(round(seconds * fps_of(dataset)))


def frame_name(dataset: str, index: int, ext: str = ".jpg",
               prefix: Optional[str] = None, digits: Optional[int] = None) -> str:
    _check(dataset)
    default_prefix, default_digits = DATASET_FRAME_FORMAT[dataset]
    prefix = default_prefix if prefix is None else prefix
    digits = default_digits if digits is None else digits
    body = f"{index:0{digits}d}" if digits > 0 else str(index)
    return f"{prefix}{body}{ext}"


def parse_frame_index(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def clip_rel_dir(dataset: str, subject: str, clip: str) -> str:
    _check(dataset)
    if dataset == "casme3":
        return f"{subject}/{clip}/{CASME3_MODALITY}"
    if dataset == "samm":
        return clip
    return f"{subject}/{clip}"


@dataclass(frozen=True)
class FramePair:

    t: int
    frame_path: Path
    flow_path: Path
    flow_pair: Tuple[int, int]
    frame_exists: bool
    flow_exists: bool

    @property
    def usable(self) -> bool:
        return self.frame_exists and self.flow_exists

    def to_dict(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "frame": str(self.frame_path),
            "flow": str(self.flow_path),
            "flow_pair": list(self.flow_pair),
            "usable": self.usable,
        }


class VideoPaths:

    def __init__(
        self,
        dataset: str,
        folder_rel: str,
        frame_prefix: Optional[str] = None,
        frame_ext: str = ".jpg",
        frame_digits: Optional[int] = None,
        flow_gap: Optional[int] = None,
    ) -> None:
        _check(dataset)
        self.dataset = dataset
        self.folder_rel = folder_rel.replace("\\", "/").strip("/")
        default_prefix, default_digits = DATASET_FRAME_FORMAT[dataset]
        self.frame_prefix = default_prefix if frame_prefix is None else frame_prefix
        self.frame_digits = default_digits if frame_digits is None else frame_digits
        self.frame_ext = frame_ext
        self.flow_gap = flow_gap if flow_gap is not None else flow_gap_of(dataset)
        self.fps = fps_of(dataset)


    @property
    def frame_dir(self) -> Path:
        return dataset_frame_root(self.dataset) / self.folder_rel

    @property
    def flow_dir(self) -> Path:
        return dataset_flow_root(self.dataset) / self.folder_rel


    def frame(self, index: int) -> Path:
        return self.frame_dir / frame_name(
            self.dataset, index, self.frame_ext, self.frame_prefix, self.frame_digits
        )

    def flow(self, index: int) -> Path:
        return self.flow_dir / frame_name(
            self.dataset, index, self.frame_ext, self.frame_prefix, self.frame_digits
        )

    def flow_source_pair(self, index: int) -> Tuple[int, int]:
        return (index - self.flow_gap, index)


    def scan_frame_range(self) -> Optional[Tuple[int, int, int]]:
        directory = self.frame_dir
        if not directory.is_dir():
            return None
        indices = [
            idx for idx in (parse_frame_index(p.name) for p in directory.iterdir()
                            if p.suffix.lower() == self.frame_ext.lower())
            if idx is not None
        ]
        if not indices:
            return None
        return min(indices), max(indices), len(indices)

    def scan_flow_range(self) -> Optional[Tuple[int, int, int]]:
        directory = self.flow_dir
        if not directory.is_dir():
            return None
        indices = [
            idx for idx in (parse_frame_index(p.name) for p in directory.iterdir()
                            if p.suffix.lower() == self.frame_ext.lower())
            if idx is not None
        ]
        if not indices:
            return None
        return min(indices), max(indices), len(indices)


    def pair(self, index: int, check: bool = True) -> FramePair:
        frame_path, flow_path = self.frame(index), self.flow(index)
        return FramePair(
            t=index,
            frame_path=frame_path,
            flow_path=flow_path,
            flow_pair=self.flow_source_pair(index),
            frame_exists=frame_path.is_file() if check else True,
            flow_exists=flow_path.is_file() if check else True,
        )

    def aligned_pairs(
        self,
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        step: int = 1,
        check: bool = True,
        require_flow: bool = True,
    ) -> List[FramePair]:
        span = self.scan_frame_range()
        if span is None:
            return []
        lo, hi, _ = span
        begin = lo + self.flow_gap if (t_start is None and require_flow) else (t_start if t_start is not None else lo)
        begin = max(begin, lo)
        finish = hi if t_end is None else min(t_end, hi)
        pairs = [self.pair(t, check=check) for t in range(begin, finish + 1, max(1, step))]
        return [p for p in pairs if p.usable] if (check and require_flow) else pairs

    def iter_pairs(self, t_start: Optional[int] = None, t_end: Optional[int] = None,
                   step: int = 1, check: bool = True) -> Iterator[FramePair]:
        span = self.scan_frame_range()
        if span is None:
            return
        lo, hi, _ = span
        begin = max(lo + self.flow_gap if t_start is None else t_start, lo)
        finish = hi if t_end is None else min(t_end, hi)
        for t in range(begin, finish + 1, max(1, step)):
            yield self.pair(t, check=check)

    def event_window(self, onset: int, offset: int, pad: int = 0) -> List[FramePair]:
        return self.aligned_pairs(onset - pad, offset + pad)

    def coverage(self) -> Dict[str, object]:
        frames = self.scan_frame_range()
        flows = self.scan_flow_range()
        if frames is None:
            return {"frames": None, "flow": None, "ratio": 0.0,
                    "note": f"frame directory missing: {self.frame_dir}"}
        expected = max(0, frames[2] - self.flow_gap)
        have = flows[2] if flows else 0
        return {
            "frames": {"lo": frames[0], "hi": frames[1], "count": frames[2]},
            "flow": ({"lo": flows[0], "hi": flows[1], "count": flows[2]} if flows else None),
            "expected_flow": expected,
            "ratio": round(have / expected, 4) if expected else 0.0,
            "flow_gap": self.flow_gap,
        }


def to_record_path(path: Path | str) -> str:
    from ..config import PROJECT_ROOT
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return resolved.as_posix()


def from_record_path(rel: str) -> Path:
    from ..config import PROJECT_ROOT
    candidate = Path(rel)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / rel)


def qa_dir(dataset: str) -> Path:
    _check(dataset)
    return QTA_ROOT / QA_SUBDIR[dataset]


def find_qa_runs(dataset: str) -> List[Path]:
    root = qa_dir(dataset)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*.jsonl"))]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def available_datasets(require_qa: bool = False) -> List[str]:
    out = []
    for name in DATASETS:
        if not dataset_frame_root(name).is_dir():
            continue
        if require_qa and not find_qa_runs(name):
            continue
        out.append(name)
    return out


__all__ = [
    "DATASETS", "DATASET_FPS", "DATASET_FRAME_REL", "DATASET_LABEL_REL",
    "DATASET_FRAME_FORMAT", "DATASET_FLOW_GAP", "DATASET_MAX_PROPOSALS",
    "CASME3_MODALITY", "QA_SUBDIR",
    "MICRO_CEILING_FRAMES",
    "DatasetPathError", "dataset_frame_root", "dataset_flow_root", "dataset_label_path",
    "fps_of", "flow_gap_of", "max_proposals_of", "max_micro_frames", "frame_name",
    "parse_frame_index",
    "clip_rel_dir", "FramePair", "VideoPaths", "to_record_path", "from_record_path",
    "qa_dir", "find_qa_runs", "available_datasets",
]
