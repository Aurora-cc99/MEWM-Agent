"""Dataset roots, frame naming and the video-frame <-> flow-frame correspondence.

**Alignment.**  A flow frame is named after the *later* frame of the pair it was
computed from: ``imgNNN.jpg`` under ``pre_datasets`` holds the flow of
``(NNN - k, NNN)``, where ``k`` is the per-dataset gap the front end used.  Video frame
``N`` therefore pairs with flow frame ``N`` directly, and the first ``k`` frames of a
video have no flow.  :class:`FramePair` makes that offset explicit rather than leaving
it to callers to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import DATASET_ROOT, FLOW_ROOT, QTA_ROOT

#: Canonical dataset identifiers.
DATASETS: Tuple[str, ...] = ("casme_sq", "samm", "casme3", "4dme")

#: Nominal capture rate.  Drives the 0.5 s micro-expression ceiling and ms-valued lags.
DATASET_FPS: Dict[str, float] = {
    "casme_sq": 30.0, "samm": 200.0, "casme3": 30.0, "4dme": 60.0,
}

#: Uniform micro-expression frame ceiling for the micro/macro routing, in frames,
#: identical for every dataset (user directive 2026-08-31: no per-dataset empirical
#: ceilings -- "不要设置微表情检测帧的上限，如果设置，则所有数据集都设置为200帧").
#: The previous per-dataset values (17 frames for casme_sq, 15 for samm, ...) were
#: each dataset's longest *annotated* micro-expression, and they misfired in practice:
#: a hysteresis span around a true event is typically wider than the annotation, so
#: spans covering the truth were routed to the macro channel and never scored
#: (measured 2026-08-31 on casme_sq: 13 of 52 truths covered by a span at IoU > 0.5,
#: all 13 discarded by the 17-frame ceiling). 200 frames (~6.7 s at 30 fps, ~3.3 s at
#: 60 fps) is far beyond any genuine micro-expression, so no real event is ever
#: re-routed; it only keeps genuinely long excursions (scene changes) in the macro
#: channel.
MICRO_CEILING_FRAMES: int = 200

#: Frame root of each dataset, relative to ``dataset/``.  The flow tree reuses these.
DATASET_FRAME_REL: Dict[str, str] = {
    "casme_sq": "CASME_sq/rawpic_crop",
    "samm": "SAMMLV/SAMM_longvideos_crop",
    "casme3": "CASME3/part_A_split/part_A",
    "4dme": "4DME/long_gray_video/long gray video",
}

#: Annotation workbooks, relative to ``dataset/``.
DATASET_LABEL_REL: Dict[str, str] = {
    "casme_sq": "CASME_sq/code_final.xlsx",
    "samm": "SAMMLV/SAMM_LongVideos_V3_Release.xlsx",
    "casme3": "CASME3/CAS(ME)3_part_A_v1.xls",
    "4dme": "4DME/Micro and Macro-expression labels.xlsx",
}

#: Frame filename shape per dataset: ``(prefix, zero-padded digits)``.
#: ``digits = 0`` means "no zero padding" (CAS(ME)^3 writes ``8.jpg``, not ``0008.jpg``).
DATASET_FRAME_FORMAT: Dict[str, Tuple[str, int]] = {
    "casme_sq": ("img", 3),      # img003.jpg
    "samm": ("", 4),             # 0003.jpg
    "casme3": ("", 0),           # 8.jpg
    "4dme": ("Frame_", 9),       # Frame_000001826.jpg
}

#: Fallback flow gap ``k`` in frames, used only when the annotation statistics are not
#: available.  The real value is derived per dataset by the front end as
#: ``round(mean micro-expression length / 2)`` and travels on ``DatasetBundle.gap``;
#: :class:`VideoPaths` takes it as a constructor argument so the derived value wins.
DATASET_FLOW_GAP: Dict[str, int] = {
    "casme_sq": 7, "samm": 47, "casme3": 7, "4dme": 14,
}

#: Sub-path inside a CAS(ME)^3 clip folder that holds the colour frames.
CASME3_MODALITY = "color"

#: Question/answer sets.  ``4dme`` is registered but not yet built; the loader reports
#: it as pending rather than failing, so the rest of the pipeline stays usable.
QA_SUBDIR: Dict[str, str] = {
    "casme_sq": "casme_sq", "samm": "samm", "casme3": "casme3", "4dme": "4dme",
}


class DatasetPathError(FileNotFoundError):
    """Raised when a required dataset path is absent."""


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def dataset_frame_root(dataset: str) -> Path:
    """Absolute frame root, e.g. ``.../dataset/CASME_sq/rawpic_crop``."""
    _check(dataset)
    return DATASET_ROOT / DATASET_FRAME_REL[dataset]


def dataset_flow_root(dataset: str) -> Path:
    """Absolute flow root -- the same relative path under ``pre_datasets``."""
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


def max_micro_frames(dataset: str, seconds: Optional[float] = None) -> int:
    """The micro/macro routing ceiling in frames (see ``p_agent_scan.md`` rule 4 --
    this is a ceiling weighed as evidence, not a blind cutoff).

    Uniform ``MICRO_CEILING_FRAMES`` (200) for every dataset by default. Pass
    ``seconds`` explicitly to override with a physical duration, converted through
    the dataset's own capture rate.
    """
    _check(dataset)
    if seconds is None:
        return MICRO_CEILING_FRAMES
    return int(round(seconds * fps_of(dataset)))


# ---------------------------------------------------------------------------
# Frame naming
# ---------------------------------------------------------------------------


def frame_name(dataset: str, index: int, ext: str = ".jpg",
               prefix: Optional[str] = None, digits: Optional[int] = None) -> str:
    """Render a frame filename in the dataset's own convention."""
    _check(dataset)
    default_prefix, default_digits = DATASET_FRAME_FORMAT[dataset]
    prefix = default_prefix if prefix is None else prefix
    digits = default_digits if digits is None else digits
    body = f"{index:0{digits}d}" if digits > 0 else str(index)
    return f"{prefix}{body}{ext}"


def parse_frame_index(filename: str) -> Optional[int]:
    """Recover the frame number from a filename; ``None`` when it has no digits."""
    stem = Path(filename).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def clip_rel_dir(dataset: str, subject: str, clip: str) -> str:
    """Relative directory of one long video inside its frame root.

    CAS(ME)^3 nests one more level for the modality (``spNO.1/a/color``); the others put
    the frames directly under the clip folder.
    """
    _check(dataset)
    if dataset == "casme3":
        return f"{subject}/{clip}/{CASME3_MODALITY}"
    if dataset == "samm":
        # SAMM long videos are flat: the clip folder already encodes the subject.
        return clip
    return f"{subject}/{clip}"


# ---------------------------------------------------------------------------
# Frame / flow pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePair:
    """One aligned ``(video frame, flow frame)`` observation at time ``t``.

    ``flow_pair`` records which two source frames the flow came from, so an agent that
    quotes a measurement can be traced back to the exact pixels it was computed on.
    """

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
    """Path resolver for one long video: frames, flow frames, and their alignment."""

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

    # -- directories --------------------------------------------------------

    @property
    def frame_dir(self) -> Path:
        return dataset_frame_root(self.dataset) / self.folder_rel

    @property
    def flow_dir(self) -> Path:
        return dataset_flow_root(self.dataset) / self.folder_rel

    # -- single files -------------------------------------------------------

    def frame(self, index: int) -> Path:
        return self.frame_dir / frame_name(
            self.dataset, index, self.frame_ext, self.frame_prefix, self.frame_digits
        )

    def flow(self, index: int) -> Path:
        """Flow frame ``index`` -- the motion arriving at frame ``index``."""
        return self.flow_dir / frame_name(
            self.dataset, index, self.frame_ext, self.frame_prefix, self.frame_digits
        )

    def flow_source_pair(self, index: int) -> Tuple[int, int]:
        """The ``(earlier, later)`` frames flow ``index`` was computed from."""
        return (index - self.flow_gap, index)

    # -- discovery ----------------------------------------------------------

    def scan_frame_range(self) -> Optional[Tuple[int, int, int]]:
        """``(lo, hi, count)`` of the frames present on disk."""
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

    # -- alignment ----------------------------------------------------------

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
        """Aligned observations over ``[t_start, t_end]``.

        Flow only exists from ``frame_lo + gap`` onward, so the default start is clamped
        there instead of silently yielding pairs whose flow file is missing.
        """
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
        """Aligned pairs covering one annotated event, optionally padded."""
        return self.aligned_pairs(onset - pad, offset + pad)

    def coverage(self) -> Dict[str, object]:
        """Diagnostic: how much of the frame range actually has flow beside it."""
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


# ---------------------------------------------------------------------------
# Relative-path helpers (records store repo-relative paths, as the QA sets do)
# ---------------------------------------------------------------------------


def to_record_path(path: Path | str) -> str:
    """Render an absolute path the way the QA sets do: relative to the project root."""
    from ..config import PROJECT_ROOT
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return resolved.as_posix()


def from_record_path(rel: str) -> Path:
    """Inverse of :func:`to_record_path`."""
    from ..config import PROJECT_ROOT
    candidate = Path(rel)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / rel)


def qa_dir(dataset: str) -> Path:
    _check(dataset)
    return QTA_ROOT / QA_SUBDIR[dataset]


def find_qa_runs(dataset: str) -> List[Path]:
    """Timestamped QA build directories, newest first."""
    root = qa_dir(dataset)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*.jsonl"))]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def available_datasets(require_qa: bool = False) -> List[str]:
    """Datasets whose frames (and optionally QA sets) are present on this machine."""
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
    "DATASET_FRAME_FORMAT", "DATASET_FLOW_GAP", "CASME3_MODALITY", "QA_SUBDIR",
    "MICRO_CEILING_FRAMES",
    "DatasetPathError", "dataset_frame_root", "dataset_flow_root", "dataset_label_path",
    "fps_of", "flow_gap_of", "max_micro_frames", "frame_name", "parse_frame_index",
    "clip_rel_dir", "FramePair", "VideoPaths", "to_record_path", "from_record_path",
    "qa_dir", "find_qa_runs", "available_datasets",
]
