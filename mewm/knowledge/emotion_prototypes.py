"""``K_E`` -- the emotion prototype library (paper 3.5, appendix C.3).

* the emotion codebook (core AUs with weight and specificity, coarse mapping),
* prototype AU trajectory templates ``T_e``, plus the neutralised and masked variants
  the suppression / masquerade rules of appendix C.4 compare against,
* the emotion-wheel similarity used by ``R_emo`` to score near-misses.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .au_anatomy import SLOT_AUS, au_label

KB_VERSION = "K_E-1.0.0"

# ---------------------------------------------------------------------------
# Codebook: emotion -> {AU: (weight, specificity)}
# ---------------------------------------------------------------------------
# weight       -- how strongly the AU belongs to the prototype (drives ES)
# specificity  -- how exclusive the AU is to this emotion (down-weights shared AUs
#                 such as AU4, which is in disgust, anger, fear and sadness alike)

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
#: The four coarse classes of eq. (1).
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

# AUs whose presence argues *against* an emotion.  Used both by the ES penalty and by
# the masquerade rule of appendix C.4(i).
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

# Valence / arousal coordinates -- used for the continuous emotion-wheel similarity of
# R_emo and for the arousal rank correlation of the counterfactual structure metric.
VALENCE_AROUSAL: Dict[str, Tuple[float, float]] = {
    "happiness": (0.85, 0.55), "surprise": (0.15, 0.85), "disgust": (-0.70, 0.45),
    "anger": (-0.75, 0.80), "fear": (-0.75, 0.85), "sadness": (-0.70, -0.35),
    "contempt": (-0.50, 0.20), "repression": (-0.35, -0.15), "other": (0.0, 0.0),
}

# Onset -> apex -> offset shape family.  ``rise`` / ``fall`` are the fractions of the
# event spent before and after the apex; micro-expressions are onset-fast, offset-slow.
PHASE_SHAPE: Dict[str, Tuple[float, float]] = {
    "happiness": (0.40, 0.60), "surprise": (0.30, 0.70), "disgust": (0.35, 0.65),
    "anger": (0.40, 0.60), "fear": (0.28, 0.72), "sadness": (0.45, 0.55),
    "contempt": (0.40, 0.60), "repression": (0.50, 0.50), "other": (0.40, 0.60),
}

#: Resolution of the time-normalised prototype templates.
TEMPLATE_LENGTH = 32


# ---------------------------------------------------------------------------
# Prototype trajectory templates
# ---------------------------------------------------------------------------


@dataclass
class PrototypeTemplate:
    """``T_e = {sigma_bar^(e)_k(tau)}`` with the per-point variance band of C.3."""

    emotion: str
    variant: str                            # "full" | "neutralised" | "masked"
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
    """Asymmetric onset/apex/offset profile normalised to ``[0, 1]`` in time."""
    apex = max(1, int(round(length * rise)))
    hold = max(1, int(round(length * plateau)))
    curve: List[float] = []
    for i in range(length):
        if i < apex:
            # Smooth accelerating rise (cosine ease-in).
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
    """Default ``T_e`` built from the codebook weights and the phase-shape family."""
    meta = EMOTION_CODEBOOK[emotion]
    rise, _fall = PHASE_SHAPE.get(emotion, (0.4, 0.6))
    curves, bands = {}, {}
    # Higher-weight AUs lead slightly: the onset order is part of what DC scores.
    ordered = sorted(meta["aus"].items(), key=lambda kv: -kv[1][0])  # type: ignore[index]
    for rank, (au, (weight, _spec)) in enumerate(ordered):
        shift = min(0.18, 0.05 * rank)
        curve = _phase_curve(float(weight), TEMPLATE_LENGTH, rise + shift)
        curves[au] = curve
        bands[au] = [round(0.10 + 0.12 * float(weight), 5)] * TEMPLATE_LENGTH
    return PrototypeTemplate(emotion, "full", curves, bands, provenance="analytic-default")


def _neutralised_from(full: PrototypeTemplate, cut: float = 0.35) -> PrototypeTemplate:
    """Suppression: core AUs truncated to a residual trace, shapes otherwise intact.

    This is what a *neutralised* micro-expression looks like -- the intent reaches the
    face but is cut short, so the strongest AU keeps only a weak trace.
    """
    ordered = sorted(full.curves.items(), key=lambda kv: -max(kv[1]))
    curves, bands = {}, {}
    for rank, (au, curve) in enumerate(ordered):
        scale = cut if rank == 0 else (0.55 if rank == 1 else 0.8)
        curves[au] = [round(v * scale, 5) for v in curve]
        bands[au] = list(full.band(au))
    return PrototypeTemplate(full.emotion, "neutralised", curves, bands,
                             n_samples=full.n_samples, provenance=full.provenance)


def _masked_from(full: PrototypeTemplate, emotion: str) -> PrototypeTemplate:
    """Masquerade: the prototype plus a slowly ramping contradictory AU.

    The overlay rises across the whole window instead of spiking, which is exactly the
    social-smile signature rule C.4(iii) keys on (``kappa_rise`` below the transient
    threshold, plus an activation baseline that predates the proposal).
    """
    curves = {au: list(c) for au, c in full.curves.items()}
    bands = {au: list(full.band(au)) for au in full.curves}
    contradictions = CONTRADICTORY_AUS.get(emotion, [])
    overlay = next((au for au in contradictions if au in SLOT_AUS), None)
    if overlay:
        # Linear social ramp from an already non-zero baseline.
        curves[overlay] = [round(0.18 + 0.35 * (i / (TEMPLATE_LENGTH - 1)), 5)
                           for i in range(TEMPLATE_LENGTH)]
        bands[overlay] = [0.14] * TEMPLATE_LENGTH
    return PrototypeTemplate(full.emotion, "masked", curves, bands,
                             n_samples=full.n_samples, provenance=full.provenance)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class PrototypeLibrary:
    """Versioned, read-only store of ``T_e`` and its suppression variants."""

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

    # -- fitting from a training fold ---------------------------------------

    def fit(
        self,
        samples: Iterable[Tuple[str, Dict[str, List[float]], Tuple[int, int, int]]],
        min_samples: int = 3,
    ) -> Dict[str, int]:
        """Rebuild ``T_e`` by averaging real slot trajectories (appendix C.3).

        ``samples`` yields ``(emotion, {au: sigma_hat trajectory}, (onset, apex, offset))``
        with frame indices absolute.  Each trajectory is piecewise-linearly time
        normalised on the three anchors before averaging, so samples of different
        duration stay phase aligned.
        """
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

    # -- persistence --------------------------------------------------------

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
    """Piecewise-linear time normalisation on the onset/apex/offset anchors."""
    onset, apex, offset = anchors
    n = len(curve)
    if n == 0:
        return [0.0] * length
    if n == 1 or offset <= onset:
        return [float(curve[0])] * length
    # Fractional position of the apex within the event, clamped away from the edges.
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


# ---------------------------------------------------------------------------
# Scoring helpers used by the R-Agent and the reward
# ---------------------------------------------------------------------------


def core_aus(emotion: str) -> List[str]:
    """``A^+_e`` -- the prototype AU set."""
    return sorted(EMOTION_CODEBOOK.get(emotion, {}).get("aus", {}),  # type: ignore[arg-type]
                  key=lambda a: int(a[2:]))


def au_weight(emotion: str, au: str) -> float:
    entry = EMOTION_CODEBOOK.get(emotion, {}).get("aus", {}).get(au)  # type: ignore[union-attr]
    return float(entry[0]) if entry else 0.0


def au_specificity(emotion: str, au: str) -> float:
    entry = EMOTION_CODEBOOK.get(emotion, {}).get("aus", {}).get(au)  # type: ignore[union-attr]
    return float(entry[1]) if entry else 0.0


def evidence_sufficiency(
    emotion: str,
    active_aus: Sequence[str],
    weak_aus: Sequence[str] = (),
    intensities: Optional[Dict[str, float]] = None,
) -> float:
    """``ES(e)`` -- weighted share of the prototype's core mass that is present.

    Weak activations count at half weight, and each contradictory AU that *is* present
    removes its own specificity-scaled mass, so a hypothesis cannot score well merely by
    having some of its AUs present while carrying evidence that refutes it.
    """
    codebook = EMOTION_CODEBOOK.get(emotion, {}).get("aus", {})  # type: ignore[union-attr]
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
    """Min-max normalisation within the candidate set (appendix C.1) for DC."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: round((v - lo) / (hi - lo), 4) for k, v in scores.items()}


def emotion_similarity(a: str, b: str) -> float:
    """Continuous emotion-wheel similarity in ``[0, 1]``, for partial ``R_emo`` credit."""
    if a == b:
        return 1.0
    va, aa = VALENCE_AROUSAL.get(a, (0.0, 0.0))
    vb, ab = VALENCE_AROUSAL.get(b, (0.0, 0.0))
    distance = math.hypot(va - vb, aa - ab)
    return round(max(0.0, 1.0 - distance / (2.0 * math.sqrt(2.0))), 4)


def prototype_completeness(emotion: str, active_aus: Sequence[str]) -> float:
    """IoU of the activation set with the prototype core set (appendix H.6)."""
    core = set(core_aus(emotion))
    active = set(active_aus)
    if not core and not active:
        return 0.0
    union = core | active
    return round(len(core & active) / len(union), 4) if union else 0.0


#: Free-text answers models give when they decline to commit to an emotion. These are
#: legitimate positions, but they are not members of the label set, and letting one
#: through as ``fine_label`` puts an invented category into every downstream field --
#: the main path, the authenticity score, the AU chain of thought.
_UNDETERMINED_MARKERS = (
    "undetermined", "unknown", "none", "no_activation", "inconclusive", "n/a",
    "not determined", "unclear", "indeterminate", "no emotion", "neutral",
)


def canonical_fine_label(raw: str) -> Tuple[str, bool]:
    """Map a model-emitted fine label onto the vocabulary; ``(label, was_recognised)``.

    An unrecognised label is folded to ``other`` -- which is the label set's own way of
    saying "no determinate category" -- and flagged, so the answer can say the reading
    was indeterminate instead of inventing a class name and then scoring against it.
    """
    text = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in EMOTION_CODEBOOK:
        return text, True
    for name in EMOTION_CODEBOOK:
        # Tolerate "disgust (low intensity)" and similar decorations.
        if text.startswith(name) or name in text.split("_"):
            return name, True
    if any(marker in text for marker in _UNDETERMINED_MARKERS) or not text:
        return "other", False
    return "other", False


def coarse_of(fine: str) -> str:
    canonical, _known = canonical_fine_label(fine)
    return FINE_TO_COARSE.get(canonical, "other")


def labels_consistent(fine: str, coarse: str) -> bool:
    """Gate rule R5's mapping-consistency half."""
    return FINE_TO_COARSE.get(fine, "other") == coarse


def competing_hypotheses(emotion: str, k: int = 3) -> List[str]:
    """The ``k`` nearest emotions -- the candidate set the C-Agent must rule out."""
    others = [e for e in FINE_EMOTIONS if e != emotion]
    others.sort(key=lambda e: -emotion_similarity(emotion, e))
    return others[:k]


def describe_emotion(emotion: str, lang: str = "en") -> str:
    aus = core_aus(emotion)
    rendered = ", ".join(f"{au}({au_label(au, lang)})" for au in aus)
    name = EMOTION_ZH.get(emotion, emotion) if lang == "zh" else emotion
    return f"{name}: core AUs = [{rendered}], coarse = {coarse_of(emotion)}"


def knowledge_digest(lang: str = "en") -> Dict[str, object]:
    """Compact, promptable view of ``K_E``."""
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


#: Process-wide default library.
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
