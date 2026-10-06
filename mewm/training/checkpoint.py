"""Checkpoint save/load utilities for all training stages."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "checkpoint.json"


@dataclass
class FoldCheckpoint:

    fold_name: str
    stage: str = "sft"
    decision: str = ""
    sft_rounds: List[Dict[str, Any]] = field(default_factory=list)
    gate_reports: List[Dict[str, Any]] = field(default_factory=list)
    rft: Dict[str, Any] = field(default_factory=dict)
    augmented: Dict[str, Any] = field(default_factory=dict)
    rl_skipped: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "stage": self.stage,
            "decision": self.decision,
            "sft_rounds": self.sft_rounds,
            "gate_reports": self.gate_reports,
            "rft": self.rft,
            "augmented": self.augmented,
            "rl_skipped": self.rl_skipped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FoldCheckpoint":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def checkpoint_path(fold_dir: Path | str) -> Path:
    return Path(fold_dir) / CHECKPOINT_FILENAME


def load_checkpoint(fold_dir: Path | str) -> Optional[FoldCheckpoint]:
    path = checkpoint_path(fold_dir)
    if not path.is_file():
        return None
    try:
        return FoldCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        LOGGER.warning("fold %s: checkpoint.json unreadable (%s); starting fresh",
                        Path(fold_dir).name, exc)
        return None


def save_checkpoint(fold_dir: Path | str, checkpoint: FoldCheckpoint) -> Path:
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(fold_dir)
    path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return path


def clear_checkpoint(fold_dir: Path | str) -> None:
    path = checkpoint_path(fold_dir)
    if path.is_file():
        path.unlink()


def resume_adapter_dir(fold_dir: Path | str) -> Optional[Path]:
    fold_dir = Path(fold_dir)
    rl_dir = fold_dir / "rl_adapter"
    if (rl_dir / "adapter_config.json").is_file():
        return rl_dir
    best: Optional[Path] = None
    best_n = -1
    if fold_dir.is_dir():
        for candidate in fold_dir.glob("sft_*"):
            if not (candidate / "adapter_config.json").is_file():
                continue
            suffix = candidate.name.split("_", 1)[-1]
            if not suffix.isdigit():
                continue
            n = int(suffix)
            if n > best_n:
                best_n, best = n, candidate
    return best


__all__ = [
    "FoldCheckpoint",
    "checkpoint_path",
    "load_checkpoint",
    "save_checkpoint",
    "clear_checkpoint",
    "resume_adapter_dir",
]
