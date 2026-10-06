"""Motion description utilities: converts flow tensors to natural-language summaries."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..knowledge.au_anatomy import candidate_entries
from ..schemas import ROIMeasurement

DEFAULT_TOP_K = 4


def describe_roi(measurement: ROIMeasurement, style: str = "full") -> str:
    entries = candidate_entries(measurement.roi_name, measurement.direction_deg)
    if style == "compact":
        aus = " ".join(
            f"{e['au']} {e['direction_fit']}" for e in entries[:2]
        )
        return (f"{measurement.roi_name} {measurement.direction_label} "
                f"{measurement.magnitude_px:.2f}px {measurement.magnitude_class} "
                f"coh {measurement.coherence_label}"
                + (f" {aus}" if aus else ""))

    aus = ", ".join(
        f"{e['au']} {e['direction_fit']}({e['significance']})" for e in entries
    )
    return (f"ROI {measurement.roi_index} {measurement.roi_name}: "
            f"direction {measurement.direction_deg:.1f}deg "
            f"({measurement.direction_label}), "
            f"magnitude {measurement.magnitude_px:.3f}px "
            f"{measurement.magnitude_class}, "
            f"coherence {measurement.coherence:.3f} {measurement.coherence_label}"
            + (f"; {aus}." if aus else "."))


def frame_motion_description(
    measurements: Sequence[ROIMeasurement],
    top_k: int = DEFAULT_TOP_K,
    salient_only: bool = True,
    style: str = "compact",
) -> str:
    pool = [m for m in measurements if m.salient] if salient_only else list(measurements)
    if not pool:
        pool = list(measurements)
    pool = sorted(pool, key=lambda m: (-m.magnitude_px, m.roi_index))[:top_k]
    if not pool:
        return "no facial motion observed"
    joiner = "; " if style == "compact" else "\n"
    return joiner.join(describe_roi(m, style=style) for m in pool)


def au_motion_description(
    activations: Dict[str, float],
    threshold: float = 0.35,
    edges: Optional[Sequence[Dict[str, object]]] = None,
    top_k: int = 4,
) -> str:
    ranked = sorted(activations.items(), key=lambda kv: (-kv[1], kv[0]))
    parts: List[str] = [
        f"{au} active (peak {value:.2f})"
        for au, value in ranked[:top_k] if value >= threshold
    ]
    if not parts:
        parts = ["no action unit above threshold"]
    for edge in list(edges or [])[:3]:
        source = edge.get("source", "")
        target = edge.get("target", "")
        weight = edge.get("weight", 0.0)
        if source and target:
            parts.append(f"{source}->{target} edge {float(weight):+.2f}")
    return "; ".join(parts) + "."


def phase_motion_description(
    measurements: Sequence[ROIMeasurement],
    phase: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    label = {"ap_on": "onset-to-apex", "on_off": "apex-to-offset"}.get(
        phase.lower(), phase)
    return f"{label} motion: " + frame_motion_description(
        measurements, top_k=top_k, style="compact")


__all__ = [
    "DEFAULT_TOP_K", "describe_roi", "frame_motion_description",
    "au_motion_description", "phase_motion_description",
]
