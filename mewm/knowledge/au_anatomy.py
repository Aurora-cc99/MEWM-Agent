"""AU anatomy knowledge base: facial muscle groups, AU co-occurrence priors."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..config import ensure_pre_process_importable

KB_VERSION = "K_AU-1.0.0"


ROI_ORDER: List[Tuple[str, str]] = [
    ("left_eye_lower_left", "Left Lower Eyelid (Left Side)"),
    ("left_eye_lower_center", "Left Lower Eyelid (Center)"),
    ("left_eye_lower_right", "Left Lower Eyelid (Right Side)"),
    ("right_eye_lower_left", "Right Lower Eyelid (Left Side)"),
    ("right_eye_lower_center", "Right Lower Eyelid (Center)"),
    ("right_eye_lower_right", "Right Lower Eyelid (Right Side)"),
    ("left_eye_upper_left", "Left Upper Eyelid (Left Side)"),
    ("left_eye_upper_center", "Left Upper Eyelid (Center)"),
    ("left_eye_upper_right", "Left Upper Eyelid (Right Side)"),
    ("right_eye_upper_left", "Right Upper Eyelid (Left Side)"),
    ("right_eye_upper_center", "Right Upper Eyelid (Center)"),
    ("right_eye_upper_right", "Right Upper Eyelid (Right Side)"),
    ("left_outer_brow", "Left Outer Eyebrow"),
    ("left_inner_brow", "Left Inner Eyebrow"),
    ("right_outer_brow", "Right Outer Eyebrow"),
    ("right_inner_brow", "Right Inner Eyebrow"),
    ("left_nostril_wing", "Left Nostril Wing"),
    ("right_nostril_wing", "Right Nostril Wing"),
    ("left_mouth_corner", "Left Mouth Corner"),
    ("right_mouth_corner", "Right Mouth Corner"),
    ("upper_lip_left", "Upper Lip (Left Side)"),
    ("upper_lip_center", "Upper Lip (Center)"),
    ("upper_lip_right", "Upper Lip (Right Side)"),
    ("lower_lip_left", "Lower Lip (Left Side)"),
    ("lower_lip_right", "Lower Lip (Right Side)"),
    ("lower_lip_center", "Lower Lip (Center)"),
    ("chin", "Chin"),
    ("left_cheek", "Left Cheek"),
    ("right_cheek", "Right Cheek"),
]

ROI_CANDIDATES: Dict[str, List[Tuple[str, str, List[float]]]] = {
    "left_eye_lower_left": [("AU6", "★", [90, 60, 120]), ("AU7", "◈", [90]), ("AU43", "○", [270])],
    "left_eye_lower_center": [("AU6", "★", [90]), ("AU7", "◈", [90]), ("AU43", "○", [270])],
    "left_eye_lower_right": [("AU6", "★", [90, 60, 120]), ("AU7", "◈", [90]), ("AU43", "○", [270])],
    "right_eye_lower_left": [("AU6", "★", [90, 60, 120]), ("AU7", "◈", [90]), ("AU43", "○", [270])],
    "right_eye_lower_center": [("AU6", "★", [90]), ("AU7", "◈", [90]), ("AU43", "○", [270])],
    "right_eye_lower_right": [("AU6", "★", [90, 60, 120]), ("AU7", "◈", [90]), ("AU43", "○", [270])],
    "left_eye_upper_left": [("AU5", "★", [90]), ("AU7", "◈", [270]), ("AU43", "○", [270])],
    "left_eye_upper_center": [("AU5", "★", [90]), ("AU7", "◈", [270]), ("AU43", "○", [270])],
    "left_eye_upper_right": [("AU5", "★", [90]), ("AU7", "◈", [270]), ("AU43", "○", [270])],
    "right_eye_upper_left": [("AU5", "★", [90]), ("AU7", "◈", [270]), ("AU43", "○", [270])],
    "right_eye_upper_center": [("AU5", "★", [90]), ("AU7", "◈", [270]), ("AU43", "○", [270])],
    "right_eye_upper_right": [("AU5", "★", [90]), ("AU7", "◈", [270]), ("AU43", "○", [270])],
    "left_outer_brow": [("AU2", "★", [90]), ("AU4", "◈", [270, 225, 315])],
    "left_inner_brow": [("AU1", "★", [90]), ("AU4", "★", [270, 225, 315])],
    "right_outer_brow": [("AU2", "★", [90]), ("AU4", "◈", [270, 225, 315])],
    "right_inner_brow": [("AU1", "★", [90]), ("AU4", "★", [270, 225, 315])],
    "left_nostril_wing": [("AU9", "◈", [90, 60, 120]), ("AU10", "◈", [90]), ("AU38", "○", [0, 180])],
    "right_nostril_wing": [("AU9", "◈", [90, 60, 120]), ("AU10", "◈", [90]), ("AU38", "○", [0, 180])],
    "left_mouth_corner": [("AU12", "★", [135]), ("AU15", "★", [225]), ("AU20", "◈", [180]), ("AU14", "○", [180])],
    "right_mouth_corner": [("AU12", "★", [45]), ("AU15", "★", [315]), ("AU20", "◈", [0]), ("AU14", "○", [0])],
    "upper_lip_left": [("AU10", "★", [90]), ("AU24", "◈", [0, 180]), ("AU23", "○", [0, 180])],
    "upper_lip_center": [("AU10", "★", [90]), ("AU24", "◈", [0, 180]), ("AU25", "○", [270])],
    "upper_lip_right": [("AU10", "★", [90]), ("AU24", "◈", [0, 180]), ("AU23", "○", [0, 180])],
    "lower_lip_left": [("AU17", "★", [90]), ("AU25", "◈", [270]), ("AU26", "○", [270])],
    "lower_lip_right": [("AU17", "★", [90]), ("AU25", "◈", [270]), ("AU26", "○", [270])],
    "lower_lip_center": [("AU17", "★", [90]), ("AU25", "◈", [270]), ("AU26", "★", [270])],
    "chin": [("AU17", "★", [90]), ("AU24", "◈", [0, 180]), ("AU26", "◈", [270])],
    "left_cheek": [("AU6", "★", [90, 45, 135]), ("AU12", "◈", [45, 135]), ("AU14", "○", [0, 180])],
    "right_cheek": [("AU6", "★", [90, 45, 135]), ("AU12", "◈", [45, 135]), ("AU14", "○", [0, 180])],
}

try:
    ensure_pre_process_importable()
    from me_facs_core import (
        ROI_ORDER as _SHARED_ROI_ORDER,
        ROI_CANDIDATES as _SHARED_ROI_CANDIDATES,
    )
    if _SHARED_ROI_ORDER and _SHARED_ROI_CANDIDATES:
        ROI_ORDER = list(_SHARED_ROI_ORDER)
        ROI_CANDIDATES = dict(_SHARED_ROI_CANDIDATES)
except Exception:
    pass


ROI_NAMES: List[str] = [name for name, _ in ROI_ORDER]
ROI_LABELS: Dict[str, str] = {name: label for name, label in ROI_ORDER}
ROI_INDEX: Dict[str, int] = {name: i for i, (name, _) in enumerate(ROI_ORDER, start=1)}
INDEX_TO_ROI: Dict[int, str] = {i: name for name, i in ROI_INDEX.items()}
N_ROI = len(ROI_ORDER)

ROI_LABELS_ZH: Dict[str, str] = {
    "left_eye_lower_left": "左下睑（左侧）", "left_eye_lower_center": "左下睑（中部）",
    "left_eye_lower_right": "左下睑（右侧）", "right_eye_lower_left": "右下睑（左侧）",
    "right_eye_lower_center": "右下睑（中部）", "right_eye_lower_right": "右下睑（右侧）",
    "left_eye_upper_left": "左上睑（左侧）", "left_eye_upper_center": "左上睑（中部）",
    "left_eye_upper_right": "左上睑（右侧）", "right_eye_upper_left": "右上睑（左侧）",
    "right_eye_upper_center": "右上睑（中部）", "right_eye_upper_right": "右上睑（右侧）",
    "left_outer_brow": "左外侧眉", "left_inner_brow": "左内侧眉",
    "right_outer_brow": "右外侧眉", "right_inner_brow": "右内侧眉",
    "left_nostril_wing": "左鼻翼", "right_nostril_wing": "右鼻翼",
    "left_mouth_corner": "左口角", "right_mouth_corner": "右口角",
    "upper_lip_left": "上唇（左）", "upper_lip_center": "上唇（中）",
    "upper_lip_right": "上唇（右）", "lower_lip_left": "下唇（左）",
    "lower_lip_right": "下唇（右）", "lower_lip_center": "下唇（中）",
    "chin": "下颏", "left_cheek": "左颊", "right_cheek": "右颊",
}


AU_ANATOMY: Dict[str, str] = {
    "AU1": "inner brow raise", "AU2": "outer brow raise", "AU4": "brow draw-down",
    "AU5": "upper-eyelid raise", "AU6": "cheek raise", "AU7": "lid tightening",
    "AU9": "nose wrinkle", "AU10": "upper-lip raise", "AU12": "lip-corner pull",
    "AU14": "mouth-corner tighten", "AU15": "lip-corner depression", "AU17": "chin raise",
    "AU20": "lip stretch", "AU23": "lip tightening", "AU24": "lip press",
    "AU25": "lip part", "AU26": "jaw drop", "AU38": "nostril flare",
    "AU43": "eye closure",
}

AU_ANATOMY_ZH: Dict[str, str] = {
    "AU1": "内眉上提", "AU2": "外眉上提", "AU4": "眉下压内聚", "AU5": "上睑提升",
    "AU6": "颊部上提", "AU7": "睑部收紧", "AU9": "鼻皱缩", "AU10": "上唇上提",
    "AU12": "口角上扬", "AU14": "口角收紧", "AU15": "口角下压", "AU17": "颏部上提",
    "AU20": "唇部横向拉伸", "AU23": "唇部收紧", "AU24": "唇部压紧", "AU25": "双唇分开",
    "AU26": "下颌下落", "AU38": "鼻翼扩张", "AU43": "闭眼",
}

SLOT_AUS: List[str] = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25",
]
K_SLOTS = len(SLOT_AUS)
SLOT_INDEX: Dict[str, int] = {au: i for i, au in enumerate(SLOT_AUS)}

ALL_ROI_AUS: List[str] = sorted(
    {au for cands in ROI_CANDIDATES.values() for au, _, _ in cands},
    key=lambda a: int(a[2:]),
)

SIGNIFICANCE_WEIGHTS: Dict[str, float] = {"★": 1.0, "◈": 0.72, "○": 0.45}
FIT_WEIGHTS: Dict[str, float] = {"FIT": 1.0, "PARTIAL": 0.62, "NO-FIT": 0.15}
MAGNITUDE_WEIGHTS: Dict[str, float] = {"Micro": 0.42, "Moderate": 0.72, "Macro": 1.0}
COHERENCE_WEIGHTS: Dict[str, float] = {"Low": 0.42, "Medium": 0.72, "High": 1.0}


def _build_au_regions() -> Dict[str, List[str]]:
    table: Dict[str, List[str]] = {}
    for roi, candidates in ROI_CANDIDATES.items():
        for au, _sig, _dirs in candidates:
            table.setdefault(au, []).append(roi)
    return {au: sorted(rois, key=lambda r: ROI_INDEX[r]) for au, rois in table.items()}


AU_REGIONS: Dict[str, List[str]] = _build_au_regions()

AU_ROI_PRIOR: Dict[Tuple[str, str], Tuple[str, List[float]]] = {
    (au, roi): (sig, dirs)
    for roi, cands in ROI_CANDIDATES.items()
    for au, sig, dirs in cands
}

SYMMETRIC_PAIRS: List[Tuple[str, str]] = [
    ("left_eye_lower_left", "right_eye_lower_right"),
    ("left_eye_lower_center", "right_eye_lower_center"),
    ("left_eye_upper_center", "right_eye_upper_center"),
    ("left_outer_brow", "right_outer_brow"),
    ("left_inner_brow", "right_inner_brow"),
    ("left_nostril_wing", "right_nostril_wing"),
    ("left_mouth_corner", "right_mouth_corner"),
    ("upper_lip_left", "upper_lip_right"),
    ("lower_lip_left", "lower_lip_right"),
    ("left_cheek", "right_cheek"),
]

AU_ANTAGONISTS: List[Tuple[str, str]] = [
    ("AU4", "AU1"), ("AU4", "AU2"), ("AU4", "AU12"), ("AU12", "AU15"),
    ("AU12", "AU24"), ("AU5", "AU7"), ("AU5", "AU43"), ("AU24", "AU25"),
    ("AU23", "AU25"), ("AU25", "AU24"), ("AU6", "AU4"),
]

AU_SYNERGISTS: List[Tuple[str, str]] = [
    ("AU1", "AU2"), ("AU1", "AU4"), ("AU4", "AU7"), ("AU4", "AU9"), ("AU7", "AU9"),
    ("AU9", "AU10"), ("AU6", "AU12"), ("AU12", "AU25"), ("AU23", "AU24"),
    ("AU17", "AU24"), ("AU5", "AU26"), ("AU2", "AU5"), ("AU15", "AU17"),
]

_ANTAGONIST_SET = {frozenset(p) for p in AU_ANTAGONISTS}
_SYNERGIST_SET = {frozenset(p) for p in AU_SYNERGISTS}

AMBIGUOUS_PAIRS = _ANTAGONIST_SET & _SYNERGIST_SET


def regions_of(au: str) -> List[str]:
    return AU_REGIONS.get(au, [])


def region_indices_of(au: str) -> List[int]:
    return [ROI_INDEX[r] for r in regions_of(au)]


def aus_of_region(roi: str) -> List[str]:
    return [au for au, _, _ in ROI_CANDIDATES.get(roi, [])]


def angular_distance(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def direction_fit(angle_deg: float, expected: Sequence[float]) -> str:
    if not expected:
        return "NO-FIT"
    best = min(angular_distance(angle_deg, ref) for ref in expected)
    if best <= 35.0:
        return "FIT"
    if best <= 70.0:
        return "PARTIAL"
    return "NO-FIT"


def direction_fit_score(angle_deg: float, expected: Sequence[float]) -> float:
    if not expected:
        return 0.0
    best = min(angular_distance(angle_deg, ref) for ref in expected)
    return max(0.0, 1.0 - best / 180.0)


def fit_au_at_region(au: str, roi: str, angle_deg: float) -> Optional[Dict[str, object]]:
    prior = AU_ROI_PRIOR.get((au, roi))
    if prior is None:
        return None
    significance, expected = prior
    verdict = direction_fit(angle_deg, expected)
    return {
        "au": au,
        "roi": roi,
        "significance": significance,
        "expected_deg": list(expected),
        "direction_fit": verdict,
        "fit_score": round(direction_fit_score(angle_deg, expected), 4),
        "weight": round(SIGNIFICANCE_WEIGHTS.get(significance, 0.5)
                        * FIT_WEIGHTS.get(verdict, 0.15), 4),
    }


def candidate_entries(roi: str, angle_deg: float) -> List[Dict[str, object]]:
    entries = []
    for au, significance, expected in ROI_CANDIDATES.get(roi, []):
        verdict = direction_fit(angle_deg, expected)
        entries.append({
            "au": au,
            "significance": significance,
            "direction_fit": verdict,
            "fit_score": round(direction_fit_score(angle_deg, expected), 4),
        })
    entries.sort(key=lambda e: (-float(e["fit_score"]),
                                -SIGNIFICANCE_WEIGHTS.get(str(e["significance"]), 0.0)))
    return entries


def is_ambiguous(au_a: str, au_b: str) -> bool:
    return frozenset((au_a, au_b)) in AMBIGUOUS_PAIRS


def is_antagonistic(au_a: str, au_b: str) -> bool:
    pair = frozenset((au_a, au_b))
    return pair in _ANTAGONIST_SET and pair not in AMBIGUOUS_PAIRS


def is_synergistic(au_a: str, au_b: str) -> bool:
    pair = frozenset((au_a, au_b))
    return pair in _SYNERGIST_SET and pair not in AMBIGUOUS_PAIRS


def prior_polarity(au_a: str, au_b: str) -> Optional[str]:
    if is_ambiguous(au_a, au_b):
        return None
    if frozenset((au_a, au_b)) in _SYNERGIST_SET:
        return "+"
    if frozenset((au_a, au_b)) in _ANTAGONIST_SET:
        return "-"
    return None


def hard_conflicts(active_aus: Sequence[str]) -> List[Tuple[str, str]]:
    active = sorted(set(active_aus))
    return [
        (a, b)
        for i, a in enumerate(active)
        for b in active[i + 1:]
        if is_antagonistic(a, b)
    ]


def au_label(au: str, lang: str = "en") -> str:
    table = AU_ANATOMY_ZH if lang == "zh" else AU_ANATOMY
    return table.get(au, au)


def roi_label(roi: str, lang: str = "en") -> str:
    if lang == "zh":
        return ROI_LABELS_ZH.get(roi, ROI_LABELS.get(roi, roi))
    return ROI_LABELS.get(roi, roi)


def describe_au(au: str, lang: str = "en") -> str:
    regions = ", ".join(roi_label(r, lang) for r in regions_of(au))
    return f"{au} ({au_label(au, lang)}) @ [{regions}]"


def knowledge_digest() -> Dict[str, object]:
    return {
        "version": KB_VERSION,
        "n_roi": N_ROI,
        "n_slots": K_SLOTS,
        "slots": SLOT_AUS,
        "au_regions": {au: [roi_label(r) for r in regions_of(au)] for au in SLOT_AUS},
        "antagonists": [list(p) for p in AU_ANTAGONISTS],
        "synergists": [list(p) for p in AU_SYNERGISTS],
    }


__all__ = [
    "KB_VERSION", "ROI_ORDER", "ROI_NAMES", "ROI_LABELS", "ROI_LABELS_ZH", "ROI_INDEX",
    "INDEX_TO_ROI", "N_ROI", "ROI_CANDIDATES", "AU_ANATOMY", "AU_ANATOMY_ZH",
    "SLOT_AUS", "K_SLOTS", "SLOT_INDEX", "ALL_ROI_AUS", "AU_REGIONS", "AU_ROI_PRIOR",
    "SYMMETRIC_PAIRS", "AU_ANTAGONISTS", "AU_SYNERGISTS", "AMBIGUOUS_PAIRS",
    "is_ambiguous", "SIGNIFICANCE_WEIGHTS",
    "FIT_WEIGHTS", "MAGNITUDE_WEIGHTS", "COHERENCE_WEIGHTS", "regions_of",
    "region_indices_of", "aus_of_region", "angular_distance", "direction_fit",
    "direction_fit_score", "fit_au_at_region", "candidate_entries", "is_antagonistic",
    "is_synergistic", "prior_polarity", "hard_conflicts", "au_label", "roi_label",
    "describe_au", "knowledge_digest",
]
