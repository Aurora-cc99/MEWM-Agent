"""Emotion prototype definitions: canonical AU patterns per emotion category."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .au_anatomy import SLOT_AUS, au_label

KB_VERSION = "K_E-1.0.0"


EMOTION_CODEBOOK: Dict[str, Dict[str, object]] = {
    "happiness": {
        "coarse": "positive",
        "aus": {"AU6": (0.74, 0.72), "AU12": (0.87, 0.64), "AU25": (0.26, 0.42), "AU14": (0.18, 0.80)},
    },
    "surprise": {
        "coarse": "surprise",
        "aus": {"AU1": (0.76, 0.52), "AU2": (0.66, 0.75), "AU5": (0.58, 0.82),
                "AU25": (0.38, 0.42), "AU26": (0.45, 0.60)},
    },
    "disgust": {
        "coarse": "negative",
        "aus": {"AU4": (0.80, 0.31), "AU9": (0.71, 0.78), "AU10": (0.55, 0.63),
                "AU17": (0.29, 0.40), "AU7": (0.23, 0.35)},
    },
    "anger": {
        "coarse": "negative",
        "aus": {"AU4": (0.78, 0.31), "AU7": (0.52, 0.35), "AU23": (0.45, 0.81),
                "AU24": (0.38, 0.55), "AU17": (0.33, 0.40)},
    },
    "fear": {
        "coarse": "negative",
        "aus": {"AU1": (0.74, 0.52), "AU2": (0.41, 0.75), "AU4": (0.65, 0.31),
                "AU5": (0.69, 0.82), "AU20": (0.43, 0.85), "AU25": (0.24, 0.42)},
    },
    "sadness": {
        "coarse": "negative",
        "aus": {"AU1": (0.71, 0.52), "AU4": (0.59, 0.31), "AU15": (0.44, 0.77),
                "AU17": (0.36, 0.40)},
    },
    "contempt": {
        "coarse": "negative",
        "aus": {"AU14": (0.72, 0.80), "AU12": (0.28, 0.64), "AU17": (0.17, 0.40)},
    },
    "repression": {
        "coarse": "negative",
        "aus": {"AU24": (0.61, 0.55), "AU23": (0.43, 0.81), "AU17": (0.24, 0.40),
                "AU7": (0.22, 0.35)},
    },
    "other": {
        "coarse": "other",
        "aus": {"AU4": (0.22, 0.18), "AU7": (0.20, 0.18), "AU12": (0.20, 0.18),
                "AU14": (0.22, 0.22), "AU17": (0.20, 0.18), "AU24": (0.22, 0.20),
                "AU25": (0.20, 0.18)},
    },
}

FINE_EMOTIONS: List[str] = list(EMOTION_CODEBOOK)
COARSE_EMOTIONS: List[str] = ["positive", "negative", "surprise", "other"]

FINE_TO_COARSE: Dict[str, str] = {
    name: str(meta["coarse"]) for name, meta in EMOTION_CODEBOOK.items()
}
COARSE_TO_FINE: Dict[str, List[str]] = {}
for _fine, _coarse in FINE_TO_COARSE.items():
    COARSE_TO_FINE.setdefault(_coarse, []).append(_fine)

EMOTION_ZH: Dict[str, str] = {
    "happiness": "高兴", "surprise": "惊讶", "disgust": "厌恶", "anger": "愤怒",
    "fear": "恐惧", "sadness": "悲伤", "contempt": "轻蔑", "repression": "压抑",
    "other": "其他",
}
COARSE_ZH: Dict[str, str] = {
    "positive": "积极", "negative": "消极", "surprise": "惊讶", "other": "其他",
}

CONTRADICTORY_AUS: Dict[str, List[str]] = {
    "happiness": ["AU4", "AU9", "AU15", "AU24"],
    "surprise": ["AU4", "AU7", "AU24", "AU23"],
    "disgust": ["AU12", "AU6", "AU5"],
    "anger": ["AU12", "AU6", "AU1"],
    "fear": ["AU12", "AU6", "AU9"],
    "sadness": ["AU12", "AU6", "AU5", "AU2"],
    "contempt": ["AU9", "AU5", "AU1"],
    "repression": ["AU12", "AU5", "AU26"],
    "other": [],
}

VALENCE_AROUSAL: Dict[str, Tuple[float, float]] = {
    "happiness": (0.85, 0.55), "surprise": (0.15, 0.85), "disgust": (-0.70, 0.45),
    "anger": (-0.75, 0.80), "fear": (-0.75, 0.85), "sadness": (-0.70, -0.35),
    "contempt": (-0.50, 0.20), "repression": (-0.35, -0.15), "other": (0.0, 0.0),
}

PHASE_SHAPE: Dict[str, Tuple[float, float]] = {
    "happiness": (0.40, 0.60), "surprise": (0.30, 0.70), "disgust": (0.35, 0.65),
    "anger": (0.40, 0.60), "fear": (0.28, 0.72), "sadness": (0.45, 0.55),
    "contempt": (0.40, 0.60), "repression": (0.50, 0.50), "other": (0.40, 0.60),
}

TEMPLATE_LENGTH = 32


@dataclass
class PrototypeTemplate:

    emotion: str
    variant: str
    curves: Dict[str, List[float]] = field(default_factory=dict)
    bands: Dict[str, List[float]] = field(default_factory=dict)
    n_samples: int = 0
    provenance: str = "analytic-default"

    def aus(self) -> List[str]:
        return sorted(self.curves, key=lambda a: int(a[2:]))

    def curve(self, au: str) -> List[float]:
        return self.curves.get(au, [0.0] * TEMPLATE_LENGTH)

    def band(self, au: str) -> List[float]:
        return self.bands.get(au, [0.15] * TEMPLATE_LENGTH)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _phase_curve(peak: float, length: int, rise: float, plateau: float = 0.12) -> List[float]:
    apex = max(1, int(round(length * rise)))
    hold = max(1, int(round(length * plateau)))
    curve: List[float] = []
    for i in range(length):
        if i < apex:
            x = i / max(1, apex)
            value = peak * (0.5 - 0.5 * math.cos(math.pi * x))
        elif i < apex + hold:
            value = peak
        else:
            x = (i - apex - hold) / max(1, length - apex - hold)
            value = peak * (0.5 + 0.5 * math.cos(math.pi * min(1.0, x)))
        curve.append(round(max(0.0, value), 5))
    return curve


def _analytic_template(emotion: str) -> PrototypeTemplate:
    meta = EMOTION_CODEBOOK[emotion]
    rise, _fall = PHASE_SHAPE.get(emotion, (0.4, 0.6))
    curves, bands = {}, {}
    ordered = sorted(meta["aus"].items(), key=lambda kv: -kv[1][0])
    for rank, (au, (weight, _spec)) in enumerate(ordered):
        shift = min(0.18, 0.05 * rank)
        curve = _phase_curve(float(weight), TEMPLATE_LENGTH, rise + shift)
        curves[au] = curve
        bands[au] = [round(0.10 + 0.12 * float(weight), 5)] * TEMPLATE_LENGTH
    return PrototypeTemplate(emotion, "full", curves, bands, provenance="analytic-default")


def _neutralised_from(full: PrototypeTemplate, cut: float = 0.35) -> PrototypeTemplate:
    ordered = sorted(full.curves.items(), key=lambda kv: -max(kv[1]))
    curves, bands = {}, {}
    for rank, (au, curve) in enumerate(ordered):
        scale = cut if rank == 0 else (0.55 if rank == 1 else 0.8)
        curves[au] = [round(v * scale, 5) for v in curve]
        bands[au] = list(full.band(au))
    return PrototypeTemplate(full.emotion, "neutralised", curves, bands,
                             n_samples=full.n_samples, provenance=full.provenance)


def _masked_from(full: PrototypeTemplate, emotion: str) -> PrototypeTemplate:
    curves = {au: list(c) for au, c in full.curves.items()}
    bands = {au: list(full.band(au)) for au in full.curves}
    contradictions = CONTRADICTORY_AUS.get(emotion, [])
    overlay = next((au for au in contradictions if au in SLOT_AUS), None)
    if overlay:
        curves[overlay] = [round(0.18 + 0.35 * (i / (TEMPLATE_LENGTH - 1)), 5)
                           for i in range(TEMPLATE_LENGTH)]
        bands[overlay] = [0.14] * TEMPLATE_LENGTH
    return PrototypeTemplate(full.emotion, "masked", curves, bands,
                             n_samples=full.n_samples, provenance=full.provenance)


class PrototypeLibrary:

    def __init__(self, templates: Optional[Dict[Tuple[str, str], PrototypeTemplate]] = None) -> None:
        self._templates: Dict[Tuple[str, str], PrototypeTemplate] = templates or {}
        if not self._templates:
            self._build_defaults()

    def _build_defaults(self) -> None:
        for emotion in FINE_EMOTIONS:
            full = _analytic_template(emotion)
            self._templates[(emotion, "full")] = full
            self._templates[(emotion, "neutralised")] = _neutralised_from(full)
            self._templates[(emotion, "masked")] = _masked_from(full, emotion)

    @property
    def provenance(self) -> str:
        sample = next(iter(self._templates.values()), None)
        return sample.provenance if sample else "empty"

    def get(self, emotion: str, variant: str = "full") -> PrototypeTemplate:
        key = (emotion, variant)
        if key not in self._templates:
            raise KeyError(f"no {variant} template for {emotion!r}")
        return self._templates[key]

    def emotions(self) -> List[str]:
        return sorted({e for e, _ in self._templates})

    def fit(
        self,
        samples: Iterable[Tuple[str, Dict[str, List[float]], Tuple[int, int, int]]],
        min_samples: int = 3,
    ) -> Dict[str, int]:
        buckets: Dict[str, List[Dict[str, List[float]]]] = {}
        for emotion, traj, anchors in samples:
            if emotion not in EMOTION_CODEBOOK:
                continue
            normalised = {
                au: _anchor_normalise(curve, anchors)
                for au, curve in traj.items() if curve
            }
            if normalised:
                buckets.setdefault(emotion, []).append(normalised)

        counts: Dict[str, int] = {}
        for emotion, entries in buckets.items():
            if len(entries) < min_samples:
                counts[emotion] = 0
                continue
            aus = sorted({au for e in entries for au in e}, key=lambda a: int(a[2:]))
            curves, bands = {}, {}
            for au in aus:
                stacked = [e[au] for e in entries if au in e]
                if not stacked:
                    continue
                mean = [sum(col) / len(col) for col in zip(*stacked)]
                var = [
                    math.sqrt(sum((v - m) ** 2 for v in col) / max(1, len(col) - 1))
                    for col, m in zip(zip(*stacked), mean)
                ]
                curves[au] = [round(v, 5) for v in mean]
                bands[au] = [round(max(0.02, v), 5) for v in var]
            full = PrototypeTemplate(emotion, "full", curves, bands,
                                     n_samples=len(entries), provenance="fitted")
            self._templates[(emotion, "full")] = full
            self._templates[(emotion, "neutralised")] = _neutralised_from(full)
            self._templates[(emotion, "masked")] = _masked_from(full, emotion)
            counts[emotion] = len(entries)
        return counts

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": KB_VERSION,
            "template_length": TEMPLATE_LENGTH,
            "templates": [t.to_dict() for t in self._templates.values()],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> "PrototypeLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        templates = {}
        for item in payload.get("templates", []):
            template = PrototypeTemplate(**item)
            templates[(template.emotion, template.variant)] = template
        return cls(templates)


def _anchor_normalise(
    curve: Sequence[float], anchors: Tuple[int, int, int], length: int = TEMPLATE_LENGTH
) -> List[float]:
    onset, apex, offset = anchors
    n = len(curve)
    if n == 0:
        return [0.0] * length
    if n == 1 or offset <= onset:
        return [float(curve[0])] * length
    apex_frac = min(0.9, max(0.1, (apex - onset) / max(1, offset - onset)))
    split = max(1, int(round(length * apex_frac)))

    def _resample(segment: Sequence[float], size: int) -> List[float]:
        if not segment:
            return [0.0] * size
        if len(segment) == 1:
            return [float(segment[0])] * size
        out = []
        for i in range(size):
            pos = i * (len(segment) - 1) / max(1, size - 1)
            lo = int(math.floor(pos))
            hi = min(len(segment) - 1, lo + 1)
            frac = pos - lo
            out.append(float(segment[lo]) * (1 - frac) + float(segment[hi]) * frac)
        return out

    apex_idx = min(n - 1, max(0, apex - onset))
    return [round(v, 5) for v in
            _resample(curve[: apex_idx + 1], split) + _resample(curve[apex_idx:], length - split)]


def core_aus(emotion: str) -> List[str]:
    return sorted(EMOTION_CODEBOOK.get(emotion, {}).get("aus", {}),
                  key=lambda a: int(a[2:]))


def au_weight(emotion: str, au: str) -> float:
    entry = EMOTION_CODEBOOK.get(emotion, {}).get("aus", {}).get(au)
    return float(entry[0]) if entry else 0.0


def au_specificity(emotion: str, au: str) -> float:
    entry = EMOTION_CODEBOOK.get(emotion, {}).get("aus", {}).get(au)
    return float(entry[1]) if entry else 0.0


def evidence_sufficiency(
    emotion: str,
    active_aus: Sequence[str],
    weak_aus: Sequence[str] = (),
    intensities: Optional[Dict[str, float]] = None,
) -> float:
    codebook = EMOTION_CODEBOOK.get(emotion, {}).get("aus", {})
    if not codebook:
        return 0.0
    active, weak = set(active_aus), set(weak_aus)
    intensities = intensities or {}
    total = sum(w * s for w, s in codebook.values())
    if total <= 0:
        return 0.0
    got = 0.0
    for au, (weight, spec) in codebook.items():
        strength = float(intensities.get(au, 1.0))
        if au in active:
            got += weight * spec * min(1.0, strength)
        elif au in weak:
            got += 0.5 * weight * spec * min(1.0, strength)
    penalty = sum(
        0.5 * au_specificity(emotion, au) or 0.25
        for au in CONTRADICTORY_AUS.get(emotion, []) if au in active
    )
    return round(max(0.0, min(1.0, (got - penalty) / total)), 4)


def normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: round((v - lo) / (hi - lo), 4) for k, v in scores.items()}


def emotion_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    va, aa = VALENCE_AROUSAL.get(a, (0.0, 0.0))
    vb, ab = VALENCE_AROUSAL.get(b, (0.0, 0.0))
    distance = math.hypot(va - vb, aa - ab)
    return round(max(0.0, 1.0 - distance / (2.0 * math.sqrt(2.0))), 4)


def prototype_completeness(emotion: str, active_aus: Sequence[str]) -> float:
    core = set(core_aus(emotion))
    active = set(active_aus)
    if not core and not active:
        return 0.0
    union = core | active
    return round(len(core & active) / len(union), 4) if union else 0.0


_UNDETERMINED_MARKERS = (
    "undetermined", "unknown", "none", "no_activation", "inconclusive", "n/a",
    "not determined", "unclear", "indeterminate", "no emotion", "neutral",
)


def canonical_fine_label(raw: str) -> Tuple[str, bool]:
    text = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in EMOTION_CODEBOOK:
        return text, True
    for name in EMOTION_CODEBOOK:
        if text.startswith(name) or name in text.split("_"):
            return name, True
    if any(marker in text for marker in _UNDETERMINED_MARKERS) or not text:
        return "other", False
    return "other", False


def coarse_of(fine: str) -> str:
    canonical, _known = canonical_fine_label(fine)
    return FINE_TO_COARSE.get(canonical, "other")


def labels_consistent(fine: str, coarse: str) -> bool:
    return FINE_TO_COARSE.get(fine, "other") == coarse


def competing_hypotheses(emotion: str, k: int = 3) -> List[str]:
    others = [e for e in FINE_EMOTIONS if e != emotion]
    others.sort(key=lambda e: -emotion_similarity(emotion, e))
    return others[:k]


def describe_emotion(emotion: str, lang: str = "en") -> str:
    aus = core_aus(emotion)
    rendered = ", ".join(f"{au}({au_label(au, lang)})" for au in aus)
    name = EMOTION_ZH.get(emotion, emotion) if lang == "zh" else emotion
    return f"{name}: core AUs = [{rendered}], coarse = {coarse_of(emotion)}"


def knowledge_digest(lang: str = "en") -> Dict[str, object]:
    return {
        "version": KB_VERSION,
        "coarse_classes": COARSE_EMOTIONS,
        "fine_classes": FINE_EMOTIONS,
        "codebook": {
            e: {
                "coarse": coarse_of(e),
                "core_aus": core_aus(e),
                "contradictory_aus": CONTRADICTORY_AUS.get(e, []),
            }
            for e in FINE_EMOTIONS
        },
        "descriptions": [describe_emotion(e, lang) for e in FINE_EMOTIONS],
    }


DEFAULT_LIBRARY = PrototypeLibrary()


__all__ = [
    "KB_VERSION", "EMOTION_CODEBOOK", "FINE_EMOTIONS", "COARSE_EMOTIONS",
    "FINE_TO_COARSE", "COARSE_TO_FINE", "EMOTION_ZH", "COARSE_ZH",
    "CONTRADICTORY_AUS", "VALENCE_AROUSAL", "PHASE_SHAPE", "TEMPLATE_LENGTH",
    "PrototypeTemplate", "PrototypeLibrary", "DEFAULT_LIBRARY", "core_aus",
    "au_weight", "au_specificity", "evidence_sufficiency", "normalise_scores",
    "emotion_similarity", "prototype_completeness", "coarse_of", "labels_consistent",
    "canonical_fine_label",
    "competing_hypotheses", "describe_emotion", "knowledge_digest",
]
