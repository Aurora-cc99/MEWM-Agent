"""V1 motion encoder: frame-level optical-flow and motion feature extraction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MotionConfig, ensure_pre_process_importable
from ..knowledge.au_anatomy import (
    INDEX_TO_ROI, N_ROI, ROI_INDEX, ROI_LABELS, ROI_NAMES, ROI_ORDER,
    candidate_entries,
)
from ..schemas import ROIMeasurement

LOGGER = logging.getLogger(__name__)

_DIRECTION_LABELS: Tuple[Tuple[float, str], ...] = (
    (22.5, "right"), (67.5, "up-right"), (112.5, "up"), (157.5, "up-left"),
    (202.5, "left"), (247.5, "down-left"), (292.5, "down"), (337.5, "down-right"),
)

def direction_label(angle_deg: float) -> str:
    angle = float(angle_deg) % 360.0
    for edge, label in _DIRECTION_LABELS:
        if angle < edge:
            return label
    return "right"

def magnitude_class(value: float) -> str:
    if value < 0.15:
        return "Micro"
    if value < 0.60:
        return "Moderate"
    return "Macro"

def coherence_label(value: float) -> str:
    if value < 0.45:
        return "Low"
    if value < 0.75:
        return "Medium"
    return "High"

def roi_motion(flow: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = box
    patch = flow[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if patch.size == 0:
        return 0.0, 0.0, 0.0
    vectors = patch.reshape(-1, 2).astype(np.float64)
    magnitudes = np.linalg.norm(vectors, axis=1)
    total = float(magnitudes.sum())
    mean_magnitude = float(magnitudes.mean())
    if mean_magnitude <= 1e-8 or total <= 1e-8:
        return 0.0, 0.0, 0.0
    resultant = vectors.sum(axis=0)
    coherence = float(np.linalg.norm(resultant) / (total + 1e-8))
    angle = float(np.degrees(np.arctan2(-resultant[1], resultant[0])) % 360.0)
    return mean_magnitude, angle, coherence

def coherence_noise_floor(n_pixels: int) -> float:
    return 1.0 / math.sqrt(max(1, n_pixels))

@dataclass
class HeadMotion:

    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    residual: float = 0.0

    @property
    def translation_norm(self) -> float:
        return math.hypot(self.dx, self.dy)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.dx, self.dy, self.rotation_deg, self.scale)

    def as_regressor(self) -> np.ndarray:
        return np.array([
            self.dx, self.dy, self.rotation_deg, self.scale - 1.0,
            self.translation_norm, abs(self.rotation_deg),
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {"dx": round(self.dx, 4), "dy": round(self.dy, 4),
                "rotation_deg": round(self.rotation_deg, 4),
                "scale": round(self.scale, 5), "residual": round(self.residual, 4)}

def estimate_head_motion(src: np.ndarray, dst: np.ndarray) -> HeadMotion:
    if src is None or dst is None or len(src) < 3 or len(src) != len(dst):
        return HeadMotion()
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    src_var = float((src_c ** 2).sum() / len(src))
    if src_var < 1e-9:
        return HeadMotion()

    covariance = (dst_c.T @ src_c) / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[1, 1] = -1.0
    rotation = u @ correction @ vt
    scale = float((singular * np.diag(correction)).sum() / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)
    residual = float(
        np.linalg.norm(dst - (scale * (src @ rotation.T) + translation), axis=1).mean()
    )
    return HeadMotion(
        dx=float(translation[0]), dy=float(translation[1]),
        rotation_deg=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        scale=scale, residual=residual,
    )

STABLE_LANDMARKS: Tuple[int, ...] = tuple(range(0, 17)) + (27, 28, 29, 30)

def stable_subset(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or len(points) < 31:
        return points
    return points[list(STABLE_LANDMARKS)]

def _fallback_roi_boxes(landmarks: np.ndarray, width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or len(points) < 68:
        return {name: (0, 0, 0, 0) for name in ROI_NAMES}

    left_eye, right_eye = points[36:42].mean(axis=0), points[42:48].mean(axis=0)
    inter_ocular = max(8.0, float(np.linalg.norm(right_eye - left_eye)))
    size = max(6, int(round(inter_ocular * 0.22)))

    anchors: Dict[str, np.ndarray] = {
        "left_eye_lower_left": points[41], "left_eye_lower_center": points[40],
        "left_eye_lower_right": points[40] * 0.5 + points[41] * 0.5,
        "right_eye_lower_left": points[46], "right_eye_lower_center": points[47],
        "right_eye_lower_right": points[46] * 0.5 + points[47] * 0.5,
        "left_eye_upper_left": points[37], "left_eye_upper_center": points[37] * 0.5 + points[38] * 0.5,
        "left_eye_upper_right": points[38],
        "right_eye_upper_left": points[43], "right_eye_upper_center": points[43] * 0.5 + points[44] * 0.5,
        "right_eye_upper_right": points[44],
        "left_outer_brow": points[18], "left_inner_brow": points[21],
        "right_outer_brow": points[25], "right_inner_brow": points[22],
        "left_nostril_wing": points[31], "right_nostril_wing": points[35],
        "left_mouth_corner": points[48], "right_mouth_corner": points[54],
        "upper_lip_left": points[50], "upper_lip_center": points[51], "upper_lip_right": points[52],
        "lower_lip_left": points[59], "lower_lip_center": points[57], "lower_lip_right": points[55],
        "chin": points[8], "left_cheek": points[2] * 0.4 + points[48] * 0.6,
        "right_cheek": points[14] * 0.4 + points[54] * 0.6,
    }

    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for name in ROI_NAMES:
        centre = anchors.get(name)
        if centre is None:
            boxes[name] = (0, 0, 0, 0)
            continue
        cx, cy = float(centre[0]), float(centre[1])
        x1 = int(max(0, min(width - 1, cx - size / 2)))
        y1 = int(max(0, min(height - 1, cy - size / 2)))
        x2 = int(max(x1 + 1, min(width, cx + size / 2)))
        y2 = int(max(y1 + 1, min(height, cy + size / 2)))
        boxes[name] = (x1, y1, x2, y2)
    return boxes

def build_roi_boxes(landmarks: np.ndarray, width: int, height: int,
                    me_flow=None) -> Dict[str, Tuple[int, int, int, int]]:
    try:
        ensure_pre_process_importable()
        from me_flow import build_roi_boxes as shared_boxes
        boxes = shared_boxes(np.asarray(landmarks), width, height)
        if boxes:
            return boxes
    except Exception as exc:
        LOGGER.debug("shared ROI box builder unavailable (%s); using fallback", exc)
    return _fallback_roi_boxes(landmarks, width, height)

_DLIB_STATE: Dict[str, Any] = {"detector": None, "predictor": None, "warned": False}

def _dlib_pair():
    if _DLIB_STATE["detector"] is not None:
        return _DLIB_STATE["detector"], _DLIB_STATE["predictor"]
    try:
        ensure_pre_process_importable()
        import dlib

        predictor_path = None
        try:
            import me_flow
            predictor_path = me_flow.configure_dlib_predictor()
        except Exception:
            import os
            env_path = os.environ.get("DLIB_PREDICTOR")
            predictor_path = Path(env_path) if env_path else None

        _DLIB_STATE["detector"] = dlib.get_frontal_face_detector()
        if predictor_path and Path(predictor_path).is_file():
            _DLIB_STATE["predictor"] = dlib.shape_predictor(str(predictor_path))
        elif not _DLIB_STATE["warned"]:
            LOGGER.warning("dlib 68-point predictor not found; ROI geometry will be "
                           "nominal and measurements only approximate")
            _DLIB_STATE["warned"] = True
    except Exception as exc:
        if not _DLIB_STATE["warned"]:
            LOGGER.warning("dlib unavailable (%s); ROI geometry will be nominal", exc)
            _DLIB_STATE["warned"] = True
    return _DLIB_STATE["detector"], _DLIB_STATE["predictor"]

def detect_landmarks(image_path: Path | str) -> Optional[np.ndarray]:
    detector, predictor = _dlib_pair()
    if detector is None or predictor is None:
        return None
    try:
        import cv2
        import dlib
    except ImportError:
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(grey, 1)
    if not faces:
        height, width = grey.shape[:2]
        faces = [dlib.rectangle(0, 0, width - 1, height - 1)]
    face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    shape = predictor(grey, face)
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                    dtype=np.float64)

class MotionFrontEnd:

    def __init__(self, config: Optional[MotionConfig] = None) -> None:
        self.config = config or MotionConfig()

    def measure(
        self,
        flow: np.ndarray,
        landmarks: np.ndarray,
        boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    ) -> List[ROIMeasurement]:
        height, width = flow.shape[:2]
        boxes = boxes or build_roi_boxes(landmarks, width, height)
        out: List[ROIMeasurement] = []
        for roi_name, roi_label in ROI_ORDER:
            box = boxes.get(roi_name, (0, 0, 0, 0))
            magnitude, angle, coherence = roi_motion(flow, box)
            out.append(ROIMeasurement(
                roi_index=ROI_INDEX[roi_name],
                roi_name=roi_name,
                roi_label=roi_label,
                magnitude_px=round(magnitude, 4),
                direction_deg=round(angle, 3),
                coherence=round(coherence, 4),
                salient=self.is_salient(magnitude, coherence),
                magnitude_class=magnitude_class(magnitude),
                direction_label=direction_label(angle),
                coherence_label=coherence_label(coherence),
            ))
        return out

    def is_salient(self, magnitude: float, coherence: float) -> bool:
        return bool(magnitude >= self.config.m_min or coherence >= self.config.c_min)

    def head_motion(self, landmarks_prev: np.ndarray, landmarks_now: np.ndarray) -> HeadMotion:
        return estimate_head_motion(stable_subset(landmarks_prev), stable_subset(landmarks_now))

    def measurement_matrix(self, measurements: Sequence[ROIMeasurement]) -> np.ndarray:
        rows = np.zeros((N_ROI, 4), dtype=np.float32)
        for measurement in measurements:
            radians = math.radians(measurement.direction_deg)
            rows[measurement.roi_index - 1] = (
                measurement.magnitude_px, math.sin(radians), math.cos(radians),
                measurement.coherence,
            )
        return rows

    def salient_measurements(self, measurements: Sequence[ROIMeasurement]) -> List[ROIMeasurement]:
        return [m for m in measurements if m.salient]

    def observation_payload(
        self, measurements: Sequence[ROIMeasurement], top_k: int = 8, salient_only: bool = True,
    ) -> Dict[str, object]:
        pool = self.salient_measurements(measurements) if salient_only else list(measurements)
        if not pool:
            pool = sorted(measurements, key=lambda m: -m.magnitude_px)[:top_k]
        pool = sorted(pool, key=lambda m: (-m.magnitude_px, m.roi_index))[:top_k]
        return {
            "motion_observations": [
                {
                    "roi": m.roi_index,
                    "region_name": m.roi_label,
                    "au_candidates": [
                        {"au": c["au"], "significance": c["significance"],
                         "direction_fit": c["direction_fit"]}
                        for c in candidate_entries(m.roi_name, m.direction_deg)
                    ],
                    "direction_deg": round(m.direction_deg, 3),
                    "direction_label": m.direction_label,
                    "magnitude_px": round(m.magnitude_px, 3),
                    "magnitude_class": m.magnitude_class,
                    "coherence": round(m.coherence, 3),
                    "coherence_label": m.coherence_label,
                }
                for m in pool
            ]
        }

    @staticmethod
    def load_flow(path: Path | str) -> Optional[np.ndarray]:
        path = Path(path)
        if not path.is_file():
            return None
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        try:
            import cv2
        except ImportError:
            LOGGER.warning("cv2 unavailable; cannot decode %s", path.name)
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        angle = np.deg2rad(hsv[..., 0] * 2.0)
        magnitude = hsv[..., 1] / 255.0
        flow = np.stack([magnitude * np.cos(angle), magnitude * np.sin(angle)], axis=-1)
        return flow.astype(np.float32)

    @staticmethod
    def flow_is_relative(path: Path | str) -> bool:
        return Path(path).suffix.lower() not in {".npy", ".flo"}

    @staticmethod
    def prefer_raw_flow(image_path: Path | str) -> Path:
        path = Path(image_path)
        raw = path.with_suffix(".npy")
        return raw if raw.is_file() else path

def summarise_measurements(measurements: Sequence[ROIMeasurement], lang: str = "en") -> str:
    from ..knowledge.au_anatomy import roi_label as localised
    rows = [m for m in measurements if m.salient] or list(measurements)[:6]
    rows = sorted(rows, key=lambda m: -m.magnitude_px)
    if lang == "zh":
        return "\n".join(
            f"- {localised(m.roi_name, 'zh')}(ROI{m.roi_index}): 幅度 {m.magnitude_px:.3f}px、"
            f"主方向 {m.direction_deg:.1f}°（{m.direction_label}）、一致性 {m.coherence:.3f}"
            f"（{m.coherence_label}）、显著={'是' if m.salient else '否'}"
            for m in rows
        )
    return "\n".join(
        f"- {m.roi_label} (ROI{m.roi_index}): magnitude {m.magnitude_px:.3f}px, "
        f"direction {m.direction_deg:.1f} deg ({m.direction_label}), "
        f"coherence {m.coherence:.3f} ({m.coherence_label}), salient={m.salient}"
        for m in rows
    )

__all__ = [
    "direction_label", "magnitude_class", "coherence_label", "roi_motion",
    "coherence_noise_floor", "HeadMotion", "estimate_head_motion", "STABLE_LANDMARKS",
    "stable_subset", "build_roi_boxes", "detect_landmarks", "MotionFrontEnd",
    "summarise_measurements",
]
