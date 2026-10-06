"""Perception feature cache to avoid redundant encoding across training steps."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, load_config
from ..data.datasets import LongVideo
from .rl_prompts import VideoEvidence

LOGGER = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1

_FINGERPRINTED_BLOCKS = ("motion", "representation", "dynamics", "spotting")


def fingerprint(config: MEWMConfig, max_frames: int, stride: int) -> str:
    payload = config.to_dict()
    relevant = {block: payload.get(block) for block in _FINGERPRINTED_BLOCKS}
    relevant["_sampling"] = {"max_frames": int(max_frames), "stride": int(stride)}
    relevant["_format"] = CACHE_FORMAT_VERSION
    blob = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class PerceptionCache:

    def __init__(self, root: Path, config: Optional[MEWMConfig] = None,
                 max_frames: int = 0, stride: int = 1) -> None:
        config = config or load_config()
        self.fingerprint = fingerprint(config, max_frames, stride)
        self.root = Path(root) / self.fingerprint
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path(self, video_key: str) -> Path:
        stem = hashlib.sha1(video_key.encode("utf-8")).hexdigest()[:20]
        return self.root / f"{stem}.npz"


    def get(self, video_key: str) -> Optional[VideoEvidence]:
        path = self._path(video_key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as handle:
                meta = json.loads(str(handle["meta"].item()))
                if meta.get("video") != video_key:
                    LOGGER.warning("perception cache: key mismatch at %s", path)
                    self.misses += 1
                    return None
                activations = handle["activations"] if "activations" in handle else None
                if activations is not None and activations.size == 0:
                    activations = None
        except Exception as exc:
            LOGGER.warning("perception cache: unreadable %s (%s)", path, exc)
            self.misses += 1
            return None

        self.hits += 1
        return VideoEvidence(
            video=video_key,
            n_frames=int(meta.get("n_frames", 0)),
            fps=float(meta.get("fps", 30.0)),
            frame_offset=int(meta.get("frame_offset", 0)),
            slot_order=list(meta.get("slot_order", [])),
            activations=activations,
            proposals=[tuple(p) for p in meta.get("proposals", [])],
            macro_intervals=[tuple(m) for m in meta.get("macro_intervals", [])],
            error_shares=dict(meta.get("error_shares", {})),
            scene_r2=float(meta.get("scene_r2", 0.0)),
            status=str(meta.get("status", "ok")),
            note=str(meta.get("note", "")),
        )


    def put(self, evidence: VideoEvidence) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = {
            "video": evidence.video, "n_frames": int(evidence.n_frames),
            "fps": float(evidence.fps), "frame_offset": int(evidence.frame_offset),
            "slot_order": list(evidence.slot_order),
            "proposals": [list(p) for p in evidence.proposals],
            "macro_intervals": [list(m) for m in evidence.macro_intervals],
            "error_shares": {k: float(v) for k, v in evidence.error_shares.items()},
            "scene_r2": float(evidence.scene_r2), "status": evidence.status,
            "note": evidence.note, "_format": CACHE_FORMAT_VERSION,
        }
        activations = evidence.activations
        if activations is None:
            activations = np.zeros((0, 0), dtype=np.float32)
        path = self._path(evidence.video)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), activations=activations)
        tmp.replace(path)
        self.writes += 1

    def summary(self) -> Dict[str, Any]:
        return {"fingerprint": self.fingerprint, "dir": str(self.root),
                "hits": self.hits, "misses": self.misses, "writes": self.writes}


def perceive_cached(
    videos: Sequence[LongVideo],
    cache_root: Optional[Path],
    config: Optional[MEWMConfig] = None,
    max_frames: int = 0,
    stride: int = 1,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[str, VideoEvidence], Dict[str, Any]]:
    from .qa_sweep import perceive

    if cache_root is None:
        evidence, summary = perceive(videos, config, max_frames, stride, progress)
        summary["cache"] = {"enabled": False}
        return evidence, summary

    cache = PerceptionCache(cache_root, config, max_frames, stride)
    evidence: Dict[str, VideoEvidence] = {}
    pending = []
    for video in videos:
        hit = cache.get(video.video_key)
        if hit is not None:
            evidence[video.video_key] = hit
        else:
            pending.append(video)

    fresh_summary: Dict[str, Any] = {}
    if pending:
        LOGGER.info("perception: %d/%d served from cache, %d to compute",
                    len(evidence), len(videos), len(pending))
        computed, fresh_summary = perceive(pending, config, max_frames, stride, progress)
        for key, value in computed.items():
            evidence[key] = value
            cache.put(value)

    failures = {k: e.note for k, e in evidence.items() if not e.available}
    summary = {
        "n_videos": len(videos),
        "n_with_evidence": sum(1 for e in evidence.values() if e.available),
        "n_unavailable": len(failures),
        "failures": failures,
        "elapsed_s": float(fresh_summary.get("elapsed_s", 0.0)),
        "cache": dict(cache.summary(), enabled=True, n_computed=len(pending)),
    }
    return evidence, summary


__all__ = ["CACHE_FORMAT_VERSION", "fingerprint", "PerceptionCache", "perceive_cached"]
