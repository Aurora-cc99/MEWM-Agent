"""Composes structured natural-language answers from agent evidence chains."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..knowledge.au_anatomy import AU_ANATOMY, au_label, roi_label
from ..knowledge.emotion_prototypes import EMOTION_ZH, coarse_of, labels_consistent
from ..schemas import AUDynGraph, CandidateInterval, CausalCoT, Verdict
from .answer_format import scrub, validate_answer
from .graph_analytics import (
    acyclicity_score, authenticity_score, betweenness_centrality, confidence_bands,
    gcn_propagation, main_path, strongest_links,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class AnswerSection:
    proposal_id: int
    cid: str
    onset: int
    offset: int
    apex: int
    coarse: str
    fine: str
    confidence: float
    text: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def _timestamp(frame: int, fps: float) -> float:
    return round(frame / fps, 2) if fps else 0.0


def _ordinal(index: int) -> str:
    return f"{index}-th"


def _format_aus(aus: Sequence[str]) -> str:
    return ", ".join(aus) if aus else "none"


def _fallback_static(graph: Optional[AUDynGraph], fine: str,
                     measurements: Sequence[Any] = ()) -> str:
    if graph is None or not graph.nodes:
        return ("No action unit reaches activation inside this interval, so no static "
                "facial configuration can be asserted from the observed motion.")
    active = graph.active_aus
    weak = graph.weak_aus
    parts = [
        f"The interval carries {len(active)} active action unit(s)"
        + (f" ({_format_aus(active)})" if active else "")
        + (f", with {_format_aus(weak)} present only weakly" if weak else "")
        + "."
    ]
    if active:
        leader = graph.onset_order()[0]
        parts.append(
            f"{leader} ({au_label(leader)}) leads at frame {graph.nodes[leader].t_on}, "
            f"peaking at {graph.nodes[leader].peak:.2f}."
        )
    salient = [m for m in measurements if getattr(m, "salient", False)]
    if salient:
        top = sorted(salient, key=lambda m: -m.magnitude_px)[:3]
        rendered = "; ".join(
            f"{m.roi_label} {m.magnitude_px:.3f}px at {m.direction_deg:.0f} deg "
            f"(coherence {m.coherence:.2f})" for m in top
        )
        parts.append(f"Strongest regional motion: {rendered}.")
    if fine:
        parts.append(f"The configuration is read as {fine}.")
    return " ".join(parts)


def _au_change_cot(graph: Optional[AUDynGraph], fine: str) -> str:
    if graph is None or not graph.nodes:
        return "No activation sequence could be resolved inside this interval."
    order = graph.onset_order()
    if not order:
        return "No activation sequence could be resolved inside this interval."

    clauses = []
    for position, au in enumerate(order[:4]):
        node = graph.nodes[au]
        connector = ("" if position == 0 else
                     ("then " if position < len(order[:4]) - 1 else "and finally "))
        clauses.append(
            f"{connector}{au} ({au_label(au)}) "
            f"{'strengthens' if position == 0 else 'starts'} at frame {node.t_on} "
            f"(apex {node.t_apex}, peak {node.peak:.2f})"
        )
    tail = (f", keeping the sequence most consistent with {fine}"
            if fine else "")
    return ", ".join(clauses) + tail + "."


class AnswerComposer:

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    def section(
        self,
        proposal_id: int,
        proposal: CandidateInterval,
        verdict: Optional[Verdict],
        graph: Optional[AUDynGraph],
        cot: Optional[CausalCoT],
        fps: float,
        cfi: Optional[Dict[str, float]] = None,
        annotated_aus: Sequence[str] = (),
        static_text: str = "",
        dynamic_text: str = "",
        measurements: Sequence[Any] = (),
        coherence_saturated: bool = False,
    ) -> AnswerSection:
        fine = (verdict.e_fine if verdict else "") or ""
        coarse = (verdict.e_coarse if verdict else "") or (coarse_of(fine) if fine else "")
        confidence = verdict.confidence if verdict else 0.0
        cfi = dict(cfi or {})

        chunks: List[str] = []

        chunks.append(
            f"{_ordinal(proposal_id)} micro-expression: frames {proposal.t_on}-"
            f"{proposal.t_off} (apex {proposal.apex}), "
            f"{_timestamp(proposal.t_on, fps)}s-{_timestamp(proposal.t_off, fps)}s "
            f"-- coarse-grained: {coarse or 'undetermined'}; "
            f"fine-grained: {fine or 'undetermined'}; "
            f"annotated action units: {_format_aus(annotated_aus)}."
        )

        chunks.append(scrub(static_text.strip())
                      or _fallback_static(graph, fine, measurements))
        if dynamic_text.strip():
            chunks.append(scrub(dynamic_text.strip()))

        metrics: Dict[str, Any] = {}

        if graph is not None and graph.nodes:
            links = strongest_links(graph, 3)
            active = graph.active_aus or list(graph.nodes)
            rendered = ", ".join(f"{a}->{b} ({w})" for a, b, w in links) or "none resolved"
            chunks.append(
                f"W-matrix summary: total_nodes={len(graph.nodes)}, "
                f"active_subset=[{', '.join(active)}], strongest links={rendered}. "
                "Only the observed active AU pairs carry the stronger cross-phase "
                "correlation values used in the causal reading."
            )
            metrics["strongest_links"] = links
            metrics["active_subset"] = active

            conflicts = [e for e in graph.edges if e.conflict]
            if conflicts:
                chunks.append(
                    "Unresolved AU pairings (expected and observed relations disagree, "
                    "so no relation is asserted): "
                    + ", ".join(f"{e.source}->{e.target}" for e in conflicts) + "."
                )

        if fine and coarse:
            consistent = labels_consistent(fine, coarse)
            chunks.append(
                f"Coarse/fine label consistency: {coarse}->{fine} "
                f"(coarse={'consistent' if consistent else 'inconsistent'}, "
                f"fine={'consistent' if consistent else 'inconsistent'})."
            )
            metrics["label_consistent"] = consistent

        if graph is not None and graph.nodes:
            path = main_path(graph, fine)
            chunks.append(f"Global main path: {'->'.join(path)}.")
            metrics["main_path"] = path

            acyclicity = acyclicity_score(graph)
            chunks.append(
                f"DAG validation: acyclicity score = {acyclicity:.4f} "
                f"({'acyclic' if acyclicity < 1e-3 else 'residual cyclicity present'})."
            )
            metrics["acyclicity"] = acyclicity

            centrality = betweenness_centrality(graph)
            if centrality:
                key_node = max(centrality, key=lambda k: centrality[k])
                chunks.append(
                    f"Key causal node: {key_node} betweenness centrality = "
                    f"{centrality[key_node]:.3f}."
                )
                metrics["betweenness"] = centrality
                metrics["key_node"] = key_node

        if cot is not None and fine:
            raw, normalised = authenticity_score(
                cot.es, cot.dc, fine,
                verdict.prototype_completeness if verdict else 0.0)
            chunks.append(
                f"Authenticity score: Score_auth({fine}) = {raw:.3f}; "
                f"normalized confidence = {normalised:.3f}; "
                f"calibrated verdict confidence = {confidence:.3f}."
            )
            metrics.update({"score_auth": raw, "normalised_confidence": normalised,
                            "verdict_confidence": confidence})

        if graph is not None and graph.nodes:
            propagation = gcn_propagation(graph)
            chunks.append(
                f"GCN-style validation: propagated emotion response = "
                f"{propagation['propagated_response']:.3f}, top message-passing nodes = "
                f"[{', '.join(propagation['top_nodes'])}], consistency gap = "
                f"{propagation['consistency_gap']:.3f}."
            )
            metrics["gcn"] = propagation

        if cfi:
            mean_cfi = float(np.mean(list(cfi.values())))
            rendered = ", ".join(f"{au} {value:.3f}"
                                 for au, value in sorted(cfi.items(),
                                                         key=lambda kv: -kv[1]))
            chunks.append(
                f"Counterfactual feature intervention (CFI): delta_y = {mean_cfi:.3f}. "
                f"CFI by AU: {rendered}. Each value is the shift in the emotion "
                "reading when that unit is withheld, so it measures how much the "
                "conclusion actually rests on that unit."
            )
            metrics["cfi"] = cfi
            metrics["cfi_mean"] = round(mean_cfi, 4)

            bands = confidence_bands(cfi)
            if bands["high"]:
                chunks.append(
                    f"High-confidence AUs under the graph-consistency check: "
                    f"[{', '.join(bands['high'])}]."
                )
            else:
                chunks.append(
                    "No AU reaches the high-confidence band under the current "
                    "graph-consistency check."
                )
            if bands["low"]:
                chunks.append(
                    f"Low-confidence AUs [{', '.join(bands['low'])}] provide auxiliary "
                    "support only: withholding them leaves the reading unchanged, so "
                    "they are named but not load-bearing."
                )
            metrics["confidence_bands"] = bands

        if cot is not None and fine and cot.fine_label and cot.fine_label != fine:
            reason = scrub(verdict.rationale) if (verdict and verdict.rationale) else ""
            if len(reason) > 300:
                reason = reason[:297].rsplit(" ", 1)[0] + "..."
            chunks.append(
                f"Label recheck: the free AU inference first favored {cot.fine_label}, "
                f"which was revised to {fine} on review"
                + (f" ({reason})" if reason else "")
                + ". The corrected causal path was rebuilt as "
                + f"{'->'.join(main_path(graph, fine)) if graph else fine}."
            )
        elif fine:
            chunks.append(
                f"Label recheck: the final label {fine} agrees with the causal reading "
                f"and maps consistently onto the coarse class {coarse}."
            )

        if verdict is not None and verdict.suppression != "none":
            descriptor = {
                "neutralised": "a neutralised display -- a core unit of the expression "
                               "is cut short but leaves a weak trace",
                "masked": "a masked display -- a contradictory unit rides over the leak "
                          "with a slow, socially timed onset",
            }.get(verdict.suppression, verdict.suppression)
            chunks.append(f"Suppression verdict: {descriptor}.")
            metrics["suppression"] = verdict.suppression

        if coherence_saturated:
            chunks.append(
                "Reliability note: every facial region in this interval shows a "
                "similarly uniform motion direction, so directional agreement does not "
                "separate one region from another here. The AU set was ranked on "
                "displacement and direction match alone and should be read as less "
                "certain than the individual scores suggest."
            )

        chunks.append("AU-change CoT: " + _au_change_cot(graph, fine))

        return AnswerSection(
            proposal_id=proposal_id, cid=proposal.cid, onset=proposal.t_on,
            offset=proposal.t_off, apex=proposal.apex, coarse=coarse, fine=fine,
            confidence=confidence, text=" ".join(c for c in chunks if c),
            metrics=metrics,
        )

    def compose(
        self,
        result: Any,
        annotated_events: Sequence[Any] = (),
        cfi_by_cid: Optional[Dict[str, Dict[str, float]]] = None,
        measurements_by_cid: Optional[Dict[str, Sequence[Any]]] = None,
        saturated_by_cid: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        state = result.state
        fps = state.video_meta.fps if state.video_meta else 30.0
        cfi_by_cid = cfi_by_cid or {}
        measurements_by_cid = measurements_by_cid or {}
        saturated_by_cid = saturated_by_cid or {}

        micro_events = [e for e in annotated_events if getattr(e, "is_micro", True)]
        n_annotated = len(annotated_events)
        n_detected = len(state.proposals)

        header = (
            f"This video contains {n_detected} micro-expression "
            f"event{'s' if n_detected != 1 else ''} detected by the framework"
        )
        if n_annotated:
            header += (f" out of {n_annotated} annotated expression events "
                       f"({len(micro_events)} annotated as micro-expressions)")
        header += "."

        if n_detected == 0:
            record = state.error_record
            evidence = ""
            if record is not None and record.s_curve:
                peak = max(record.s_curve)
                evidence = (f" The detection statistic stayed at baseline across the "
                            f"scanned span (peak S_t = {peak:.2f}), so no interval "
                            f"crossed the hysteresis trigger.")
            narrative = state.narrative.text if state.narrative else ""
            text = header + evidence + (" " + scrub(narrative) if narrative else "")
            return {
                "final_answer": text,
                "format_report": validate_answer(text).to_dict(),
                "sections": [], "n_detected": 0, "n_annotated": n_annotated,
            }

        sections: List[AnswerSection] = []
        narration = self._narration_lookup(state)
        for index, proposal in enumerate(state.proposals, start=1):
            matched = self._match_annotation(proposal, micro_events)
            static_text, dynamic_text = narration.get(proposal.cid, ("", ""))
            sections.append(self.section(
                proposal_id=index,
                proposal=proposal,
                verdict=state.verdicts.get(proposal.cid),
                graph=state.au_graphs.get(proposal.cid),
                cot=state.causal_cots.get(proposal.cid),
                fps=fps,
                cfi=cfi_by_cid.get(proposal.cid),
                annotated_aus=(matched.aus if matched else ()),
                static_text=static_text,
                dynamic_text=dynamic_text,
                measurements=measurements_by_cid.get(proposal.cid, ()),
                coherence_saturated=bool(saturated_by_cid.get(proposal.cid, False)),
            ))

        body = " ".join(s.text for s in sections)

        baseline = ""
        episodic = getattr(result, "episodic", None)
        if episodic is not None:
            spans = episodic.outside_proposal_summary()
            if spans:
                quiet = [s for s in spans if s.get("max", 0.0) < 1.0]
                mean = float(spans[0].get("mean", 0.0))
                verdict_word = ("quiet" if mean < 1.0
                                else "elevated" if mean < 5.0 else "noisy")
                baseline = (
                    f" Outside the detected intervals the error curve covers "
                    f"{len(spans)} segment(s), {len(quiet)} of them with no excursion "
                    f"above the release threshold; for example frames "
                    f"{spans[0]['interval'][0]}-{spans[0]['interval'][1]} average "
                    f"S_t = {mean:.2f} ({verdict_word})."
                )
                if mean >= 5.0:
                    baseline += (
                        " A baseline this high means the expressive residual is not "
                        "settling between events, so the proposals should not be read "
                        "as cleanly separated from background motion."
                    )

        links = ""
        if episodic is not None and getattr(episodic, "cross_links", None):
            rendered = "; ".join(f"{l.source}->{l.target}: {l.relation}"
                                 for l in episodic.cross_links[:4])
            links = f" Cross-proposal relations: {rendered}."

        narrative = state.narrative.text if state.narrative else ""
        if narrative:
            narrative = " " + scrub(narrative.strip())

        degradations = state.budget.degradations
        disclosure = ""
        if degradations:
            disclosure = (
                " Reliability: parts of this analysis could not be completed, so the "
                "reading above is provisional and the confidence figures are upper "
                "bounds."
            )

        final_text = header + " " + body + baseline + links + narrative + disclosure
        report = validate_answer(final_text)
        if not report.ok:
            LOGGER.warning("composed answer violates the output format: %s",
                           "; ".join(report.problems())[:400])

        return {
            "final_answer": final_text,
            "format_report": report.to_dict(),
            "sections": [
                {"proposal_id": s.proposal_id, "cid": s.cid,
                 "interval": [s.onset, s.offset], "apex": s.apex,
                 "coarse_label": s.coarse, "fine_label": s.fine,
                 "confidence": s.confidence, "text": s.text, "metrics": s.metrics}
                for s in sections
            ],
            "n_detected": n_detected,
            "n_annotated": n_annotated,
            "degraded": bool(degradations),
        }

    @staticmethod
    def _narration_lookup(state: Any) -> Dict[str, Tuple[str, str]]:
        out: Dict[str, Tuple[str, str]] = {}
        narrative = getattr(state, "narrative", None)
        if narrative is None:
            return out
        analysis = getattr(narrative, "part2_analysis", None) or []
        for index, item in enumerate(analysis):
            if not isinstance(item, dict):
                continue
            cid = str(item.get("cid") or "")
            if not cid and index < len(state.proposals):
                cid = state.proposals[index].cid
            out[cid] = (str(item.get("static_description", "")),
                        str(item.get("dynamic_description", "")))
        return out

    @staticmethod
    def _match_annotation(proposal: CandidateInterval, events: Sequence[Any]) -> Optional[Any]:
        best, best_iou = None, 0.0
        for event in events:
            overlap = proposal.iou(event.interval)
            if overlap > best_iou:
                best, best_iou = event, overlap
        return best


def compose_answer(
    result: Any,
    annotated_events: Sequence[Any] = (),
    cfi_by_cid: Optional[Dict[str, Dict[str, float]]] = None,
    measurements_by_cid: Optional[Dict[str, Sequence[Any]]] = None,
    saturated_by_cid: Optional[Dict[str, bool]] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    return AnswerComposer(lang).compose(result, annotated_events, cfi_by_cid,
                                        measurements_by_cid, saturated_by_cid)


__all__ = ["AnswerSection", "AnswerComposer", "compose_answer"]
