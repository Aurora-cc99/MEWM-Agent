"""P-Agent -- candidate localisation and motion evidence verification (paper 3.4.2).

* **scan** (segment-window granularity) -- read the ``S_t`` curve and its three-way
  decomposition, confirm the hysteresis boundaries, re-scan across segment joins, and
  route over-long intervals to the macro channel.
* **verify** (per proposal) -- transcribe the V1 measurements for every anatomical region
  into citable evidence entries, so an automatic measurement becomes something the rest
  of the chain can quote and the critic can challenge.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..knowledge.au_anatomy import roi_label
from ..orchestration.state import MEWMState, PHASE_P_SCAN, PHASE_P_VERIFY, Projection
from ..schemas import CandidateInterval, Evidence, ROIMeasurement
from .base import AgentResult, BaseAgent, as_dict

LOGGER = logging.getLogger(__name__)


class PerceptionAgent(BaseAgent):
    """Scan and verification phases of the perception role."""

    role_files = {
        PHASE_P_SCAN: "p_agent_scan",
        PHASE_P_VERIFY: "p_agent_verify",
    }

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__("P", model, **kwargs)

    def phases(self) -> Tuple[str, ...]:
        return (PHASE_P_SCAN, PHASE_P_VERIFY)

    # -- prompts ------------------------------------------------------------

    def build_user_prompt(self, phase: str, projection: Projection,
                          state: MEWMState, **kwargs: Any) -> str:
        if phase == PHASE_P_SCAN:
            return self._scan_prompt(projection, state, **kwargs)
        return self._verify_prompt(projection, state, **kwargs)

    def _scan_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        record = projection.get("error_record")
        proposals: Sequence[CandidateInterval] = kwargs.get("candidates") or state.proposals
        # This number is reference context, not a rule the detector itself enforces:
        # pipeline.py passes the uniform MICRO_CEILING_FRAMES (200) unless an explicit
        # max_micro_seconds override is configured -- the same ceiling the engine's own
        # micro/macro routing already applied, so P-Agent's channel call corroborates a
        # decision the detector made rather than being the only check left. Duration
        # alone must never force the verdict, or this reintroduces the blind-filter
        # regression documented in SpottingConfig. Weigh it alongside the energy
        # decomposition, physio overlap and attribution below, per p_agent_scan.md
        # rule 4. 0 means even that reference is unavailable; the line is then dropped
        # rather than defaulted to a number that does not exist.
        max_dur = int(kwargs.get("max_micro_frames", 0) or 0)
        lines = [
            f"Video: {state.video_id}  frames "
            f"[{state.video_meta.frame_lo if state.video_meta else 0}, "
            f"{state.video_meta.frame_hi if state.video_meta else 0}]  "
            f"fps {state.video_meta.fps if state.video_meta else 0}",
        ]
        lines.append(
            f"Micro-expression frame ceiling (reference only, not an automatic cutoff): "
            f"{max_dur} -- a candidate materially longer than this is evidence toward "
            f"the macro-expression channel, weighed together with its energy shape and "
            f"completeness; duration alone must not decide it either way."
            if max_dur > 0 else
            "Micro-expression frame ceiling: none -- do not reject a candidate for "
            "being long or short; judge it on the evidence in the window."
        )
        lines += [
            "",
            "Detection statistic and decomposition, sampled at the candidate intervals:",
        ]
        for candidate in proposals:
            window = record.window(candidate.t_on, candidate.t_off) if record else {}
            lines.append(
                f"- {candidate.cid} interval [{candidate.t_on}, {candidate.t_off}] "
                f"apex {candidate.apex} peak_S {candidate.peak_S} "
                f"duration {candidate.duration} "
                f"physio_overlap {candidate.physio_overlap} "
                f"channel {candidate.channel}"
            )
            if window:
                lines.append(
                    f"    S: {[round(v, 2) for v in window.get('S', [])[:12]]}"
                )
                lines.append(
                    f"    components  scene {sum(window.get('scene', [])):.4f}  "
                    f"physio {sum(window.get('physio', [])):.4f}  "
                    f"expr {sum(window.get('expr', [])):.4f}"
                )
            if candidate.attribution:
                lines.append(f"    error attribution: {candidate.attribution}")

        if record and record.physio_events:
            lines.append("")
            lines.append("Physiological events matched on this video:")
            for event in record.physio_events[:12]:
                lines.append(
                    f"- [{event.t_start}, {event.t_end}] {event.label} "
                    f"(match {event.match_energy})"
                )
        lines.append("")
        lines.append(
            "Confirm or reject each interval and return the JSON object. Keep each "
            "\"notes\" value to one short clause (about 12 words or fewer) -- the "
            "reason only, not a restatement of the statistics already listed above; "
            "a verbose response is more likely to be cut off before every interval "
            "in this batch is covered, which discards the whole response."
        )
        return "\n".join(lines)

    def _verify_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        measurements: Sequence[ROIMeasurement] = kwargs.get("measurements") or []
        proposal = state.proposal(projection.cid)
        summary = projection.get("error_summary") or {}
        lines = [
            f"Proposal {projection.cid}: interval "
            f"[{proposal.t_on if proposal else '?'}, {proposal.t_off if proposal else '?'}], "
            f"apex {proposal.apex if proposal else '?'}",
            f"Within-proposal error summary: {json.dumps(summary, ensure_ascii=False)}",
            "",
            "Measurements per anatomical region (the values you must verify and report):",
        ]
        for measurement in measurements:
            lines.append(
                f"- roi {measurement.roi_index} {measurement.roi_label}: "
                f"magnitude_px {measurement.magnitude_px:.4f}, "
                f"direction_deg {measurement.direction_deg:.2f} "
                f"({measurement.direction_label}), "
                f"coherence {measurement.coherence:.4f} "
                f"({measurement.coherence_label}), "
                f"salient {measurement.salient}"
            )
        lines.append("")
        lines.append(
            "Report every region. Remember that a sub-pixel magnitude with high "
            "coherence is salient."
        )
        return "\n".join(lines)

    # -- parsing ------------------------------------------------------------

    def parse(self, phase: str, payload: Dict[str, Any], projection: Projection,
              state: MEWMState, **kwargs: Any) -> AgentResult:
        if phase == PHASE_P_SCAN:
            return self._parse_scan(payload, projection, state, **kwargs)
        return self._parse_verify(payload, projection, state, **kwargs)

    def _parse_scan(self, payload: Dict[str, Any], projection: Projection,
                    state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_P_SCAN)
        candidates = {c.cid: c for c in (kwargs.get("candidates") or state.proposals)}
        confirmed: List[Dict[str, Any]] = []

        for item in payload.get("proposals", []) or []:
            cid = str(item.get("cid", ""))
            interval = item.get("interval") or []
            if len(interval) != 2:
                continue
            source = candidates.get(cid)
            # If the model omits "channel" outright, defer to the engine's own
            # pre-assigned channel for this candidate rather than blindly defaulting
            # to "micro" -- an omission should preserve whatever M2's length-based
            # routing (or a prior stage) already decided, not silently override it.
            default_channel = source.channel if source is not None else "micro"
            record = {
                "cid": cid,
                "interval": [int(interval[0]), int(interval[1])],
                "apex": int(item.get("apex", interval[0])),
                "peak_S": float(item.get("peak_S", 0.0)),
                "attribution": as_dict(item.get("attribution")),
                "physio_overlap": bool(item.get("physio_overlap", False)),
                "confirmed": bool(item.get("confirmed", True)),
                "channel": str(item.get("channel", default_channel)),
                "notes": str(item.get("notes", "")),
            }
            confirmed.append(record)
            if source is not None:
                source.confirmed = record["confirmed"]
                source.notes = record["notes"] or source.notes

            entry = self.emit(
                f"proposal {cid} confirmed over frames "
                f"[{record['interval'][0]}, {record['interval'][1]}]",
                payload={k: v for k, v in record.items() if k != "notes"},
                cid=cid,
            )
            result.entries.append(entry)

        result.product = {
            "proposals": confirmed,
            "curve_summary": str(payload.get("curve_summary", "")),
            "rejected": list(payload.get("rejected", []) or []),
        }
        return result

    def _parse_verify(self, payload: Dict[str, Any], projection: Projection,
                      state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_P_VERIFY)
        cid = projection.cid
        records: List[Dict[str, Any]] = []

        for item in payload.get("motion_evidence", []) or []:
            record = {
                "roi": str(item.get("roi", "")),
                "roi_index": int(item.get("roi_index", 0) or 0),
                "magnitude_px": float(item.get("magnitude_px", 0.0)),
                "direction_deg": float(item.get("direction_deg", 0.0)),
                "direction_label": str(item.get("direction_label", "")),
                "coherence": float(item.get("coherence", 0.0)),
                "salient": bool(item.get("salient", False)),
            }
            if item.get("note"):
                record["note"] = str(item["note"])
            records.append(record)
            result.entries.append(self.emit(
                f"region {record['roi']} moved {record['magnitude_px']:.3f}px at "
                f"{record['direction_deg']:.1f} deg with coherence "
                f"{record['coherence']:.3f}",
                payload=record, cid=cid,
            ))

        result.product = {
            "motion_evidence": records,
            "interval": list(payload.get("interval", []) or []),
            "summary": str(payload.get("summary", "")),
        }
        return result

    # -- deterministic fallback --------------------------------------------

    def fallback(self, phase: str, projection: Projection, state: MEWMState,
                 reason: str, **kwargs: Any) -> AgentResult:
        """Fall back on the engine's own output (paper D.3: "P falls back to V1")."""
        result = AgentResult(phase=phase, degraded=True, parsed=False)
        result.notes.append(f"degraded: {reason}; using the deterministic engine output")

        if phase == PHASE_P_SCAN:
            candidates = kwargs.get("candidates") or state.proposals
            records = []
            for candidate in candidates:
                candidate.confirmed = True
                records.append({
                    "cid": candidate.cid,
                    "interval": [candidate.t_on, candidate.t_off],
                    "apex": candidate.apex, "peak_S": candidate.peak_S,
                    "attribution": dict(candidate.attribution),
                    "physio_overlap": candidate.physio_overlap,
                    "confirmed": True, "channel": candidate.channel,
                    "notes": "auto-confirmed without model verification",
                })
                result.entries.append(self.emit(
                    f"proposal {candidate.cid} taken directly from the detection "
                    f"statistic over [{candidate.t_on}, {candidate.t_off}]",
                    payload=records[-1], confidence=0.6, cid=candidate.cid,
                ))
            result.product = {
                "proposals": records,
                "curve_summary": "engine output used verbatim (agent unavailable)",
                "rejected": [],
            }
            return result

        measurements: Sequence[ROIMeasurement] = kwargs.get("measurements") or []
        records = []
        for measurement in measurements:
            record = {
                "roi": measurement.roi_label, "roi_index": measurement.roi_index,
                "magnitude_px": round(measurement.magnitude_px, 4),
                "direction_deg": round(measurement.direction_deg, 2),
                "direction_label": measurement.direction_label,
                "coherence": round(measurement.coherence, 4),
                "salient": measurement.salient,
            }
            records.append(record)
            result.entries.append(self.emit(
                f"region {measurement.roi_label} moved {measurement.magnitude_px:.3f}px "
                f"at {measurement.direction_deg:.1f} deg with coherence "
                f"{measurement.coherence:.3f}",
                payload=record, confidence=0.6, cid=projection.cid,
            ))
        proposal = state.proposal(projection.cid)
        result.product = {
            "motion_evidence": records,
            "interval": [proposal.t_on, proposal.t_off] if proposal else [],
            "summary": "measurements taken directly from the V1 front end",
        }
        return result


__all__ = ["PerceptionAgent"]
