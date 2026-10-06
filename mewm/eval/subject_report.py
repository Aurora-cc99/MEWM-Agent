"""Per-subject diagnostic report generation for LOSO evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import RUNS_ROOT, load_config
from .megc_metrics import (
    au_scores, count_scores, megc_report, recognition_scores, spotting_scores,
    text_scores,
)


def qa_records_path(run_id: str, dataset: str, subject: str, video_id: str,
                    run_root: Optional[Path] = None) -> Path:
    base = Path(run_root) if run_root else RUNS_ROOT
    return base / run_id / dataset / subject / video_id / "qa_records.jsonl"


def write_qa_records(
    records: Sequence[Any],
    *,
    run_id: str,
    dataset: str,
    subject: str,
    video_id: str,
    run_root: Optional[Path] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Path:
    path = qa_records_path(run_id, dataset, subject, video_id, run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i, record in enumerate(records):
        row = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        row.setdefault("question_index", i)
        row["dataset"] = dataset
        row["subject"] = subject
        row["video_id"] = video_id
        if extra_fields:
            row.update(extra_fields)
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_qa_records(path: Path | str) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def iter_qa_record_files(run_id: str, run_root: Optional[Path] = None
                         ) -> List[Tuple[str, str, str, Path]]:
    base = (Path(run_root) if run_root else RUNS_ROOT) / run_id
    out: List[Tuple[str, str, str, Path]] = []
    if not base.is_dir():
        return out
    for path in base.glob("*/*/*/qa_records.jsonl"):
        video_id, subject, dataset = path.parent.name, path.parent.parent.name, \
            path.parent.parent.parent.name
        out.append((dataset, subject, video_id, path))
    return out


@dataclass
class EventMetricInputs:
    au_predicted: List[str] = field(default_factory=list)
    au_true: List[str] = field(default_factory=list)
    fine_true: str = ""
    fine_pred: str = ""
    text_candidate: str = ""
    text_reference: str = ""
    is_spotting_tp: bool = False


@dataclass
class VideoMetricInputs:
    dataset: str
    subject: str
    video_id: str
    proposals: List[Tuple[int, int]] = field(default_factory=list)
    proposal_types: List[str] = field(default_factory=list)
    proposal_labels: List[str] = field(default_factory=list)
    truth: List[Tuple[int, int]] = field(default_factory=list)
    truth_types: List[str] = field(default_factory=list)
    truth_labels: List[str] = field(default_factory=list)
    counts: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)
    events: List[EventMetricInputs] = field(default_factory=list)
    whole_text_candidate: str = ""
    whole_text_reference: str = ""


def _spotting_row(v: VideoMetricInputs) -> Dict[str, Any]:
    return {
        "video": v.video_id, "proposals": v.proposals,
        "proposal_types": v.proposal_types, "proposal_labels": v.proposal_labels,
        "truth": v.truth, "truth_types": v.truth_types, "truth_labels": v.truth_labels,
    }


def _group_report(videos: Sequence[VideoMetricInputs], iou_threshold: float = 0.5
                  ) -> Dict[str, Any]:
    spotting = spotting_scores([_spotting_row(v) for v in videos], iou_threshold)

    pairs_by_quantity: Dict[str, List[Tuple[int, int]]] = {}
    for v in videos:
        for quantity, (pred, true) in v.counts.items():
            if pred is None or true is None:
                continue
            pairs_by_quantity.setdefault(quantity, []).append((pred, true))
    counting = count_scores(pairs_by_quantity)

    all_events = [e for v in videos for e in v.events]
    tp_events = [e for e in all_events if e.is_spotting_tp]

    recognition_au = au_scores([e.au_predicted for e in all_events],
                               [e.au_true for e in all_events])
    recognition_emotion = recognition_scores([e.fine_true for e in all_events],
                                             [e.fine_pred for e in all_events])
    recognition_text = text_scores([e.text_candidate for e in all_events],
                                   [e.text_reference for e in all_events])

    strs_text = text_scores(
        [v.whole_text_candidate for v in videos],
        [v.whole_text_reference for v in videos])

    def _f1a(events: Sequence[EventMetricInputs]) -> Optional[float]:
        if not events:
            return None
        block = recognition_scores([e.fine_true for e in events],
                                   [e.fine_pred for e in events])
        fine_megc = block.get("fine", {}).get("megc", {})
        return fine_megc.get("reg_uf1") if fine_megc.get("status") == "ok" else None

    report = megc_report(
        spotting=spotting, counting=counting, recognition_au=recognition_au,
        recognition_emotion=recognition_emotion, recognition_text=recognition_text,
        strs_text=strs_text,
        f1_analysis_on_tp=_f1a(tp_events), f1_analysis_on_truth=_f1a(all_events),
    )
    report["n_videos"] = len(videos)
    report["n_subjects"] = len({v.subject for v in videos})
    report["n_events"] = len(all_events)
    return report


def collect_metric_inputs(
    runs: Sequence[Any],
    videos: Sequence[Any],
    qa_by_id: Optional[Dict[str, Sequence[Any]]] = None,
    config: Optional[Any] = None,
) -> List[VideoMetricInputs]:
    from .report import _canonical_or_empty, _paired_events, _qa_segment_reference

    config = config or load_config()
    by_id = {v.video_id: v for v in videos}
    inputs: List[VideoMetricInputs] = []
    for run in runs:
        video = by_id.get(run.video_id)
        if video is None:
            continue
        micro_events = list(video.micro_events())
        all_events = list(video.events)
        analyses = run.analyses()
        labels = [_canonical_or_empty(a.get("fine_label", "")) for a in analyses]
        item = VideoMetricInputs(
            dataset=str(getattr(video, "dataset", "casme_sq")),
            subject=str(getattr(video, "subject", "")),
            video_id=run.video_id,
            proposals=run.proposals(),
            proposal_types=["micro-expression"] * len(run.proposals()),
            proposal_labels=labels,
            truth=[e.interval for e in micro_events],
            truth_types=["micro-expression"] * len(micro_events),
            truth_labels=[str(e.fine_label or "") for e in micro_events],
            counts={
                "micro": (len(run.proposals()), len(micro_events)),
                "expression": (
                    int(run.answer.get("n_detected", len(run.proposals()))),
                    int(run.answer.get("n_annotated", len(all_events)))),
                "macro": (
                    int((run.summary.get("spotting") or {}).get("n_macro", 0)),
                    len(all_events) - len(micro_events)),
            },
        )

        qa_items: List[Any] = []
        if qa_by_id is not None:
            qa_items = list(qa_by_id.get(run.video_id) or [])
            whole = next((it for it in qa_items if it.qtype == "reason_full"), None)
            if whole is not None:
                item.whole_text_candidate = str(run.answer.get("answer", ""))
                item.whole_text_reference = whole.answer_text

        for analysis, event, match in _paired_events(run, micro_events, config):
            reference = ""
            if qa_items:
                interval = analysis.get("interval")
                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                    reference = _qa_segment_reference(
                        (int(interval[0]), int(interval[1])), qa_items) or ""
            item.events.append(EventMetricInputs(
                au_predicted=[str(a) for a in (analysis.get("active_aus") or [])],
                au_true=[str(a) for a in (event.aus or [])],
                fine_true=_canonical_or_empty(event.fine_label),
                fine_pred=_canonical_or_empty(analysis.get("fine_label", "")),
                text_candidate=str(analysis.get("au_cot") or ""),
                text_reference=reference,
                is_spotting_tp=bool(getattr(match, "is_tp_strict", False)),
            ))
        inputs.append(item)
    return inputs


def aggregate_final_metrics(
    videos: Sequence[VideoMetricInputs],
    *,
    run_id: str,
    protocol: str,
    mode: str,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    by_dataset: Dict[str, List[VideoMetricInputs]] = {}
    by_subject: Dict[str, List[VideoMetricInputs]] = {}
    for v in videos:
        by_dataset.setdefault(v.dataset, []).append(v)
        by_subject.setdefault(f"{v.dataset}/{v.subject}", []).append(v)

    summary: Dict[str, Any] = {
        "run_id": run_id, "protocol": protocol, "mode": mode,
        "iou_threshold": iou_threshold,
        "per_dataset": {name: _group_report(rows, iou_threshold)
                        for name, rows in sorted(by_dataset.items())},
        "per_subject": {name: _group_report(rows, iou_threshold)
                        for name, rows in sorted(by_subject.items())},
        "overall": _group_report(list(videos), iou_threshold) if videos else {
            "status": "unavailable", "reason": "no videos supplied to aggregate_final_metrics",
        },
        "notes": [
            "......",
        ],
    }
    return summary


def write_summary(summary: Dict[str, Any], *, run_id: str,
                  run_root: Optional[Path] = None) -> Tuple[Path, Path]:
    base = (Path(run_root) if run_root else RUNS_ROOT) / run_id
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "final_metrics_summary.json"
    md_path = base / "final_metrics_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    md_path.write_text(_format_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "n/a" if value is None else str(value)


def _format_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"......",
        "",
        f"- protocol: `{summary.get('protocol')}`　mode: `{summary.get('mode')}`　"
        f"iou_threshold: `{summary.get('iou_threshold')}`",
        "",
    ]

    def localisation_section(title: str, groups: Dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("......")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, report in groups.items():
            interval = report.get("spotting", {}).get("interval", {})
            strict = interval.get("strict_iou", {})
            unweighted = interval.get("unweighted_type", {})
            counting = report.get("spotting", {}).get("counting", {})
            mae = "/".join(_fmt(counting[q].get("mae")) if isinstance(counting.get(q), dict)
                           else "n/a" for q in ("expression", "micro", "macro"))
            rmse = "/".join(_fmt(counting[q].get("rmse")) if isinstance(counting.get(q), dict)
                            else "n/a" for q in ("expression", "micro", "macro"))
            lines.append(
                f"| {name} | {_fmt(strict.get('f1'))} | {_fmt(strict.get('precision'))} | "
                f"{_fmt(strict.get('recall'))} | {_fmt(strict.get('tp'))} | "
                f"{_fmt(unweighted.get('spot_uf1'))} | {_fmt(unweighted.get('spot_uar'))} | "
                f"{mae} | {rmse} | {report.get('n_videos')} |")
        lines.append("")

    def recognition_section(title: str, groups: Dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("......")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, report in groups.items():
            au = report.get("recognition", {}).get("action_units", {})
            emo = report.get("recognition", {}).get("emotion", {})
            fine = emo.get("fine", {}).get("megc", {})
            coarse = emo.get("coarse", {}).get("megc", {})
            text = report.get("recognition", {}).get("text", {})
            reg_uf1 = f"{_fmt(fine.get('reg_uf1'))}/{_fmt(coarse.get('reg_uf1'))}"
            reg_uar = f"{_fmt(fine.get('reg_uar'))}/{_fmt(coarse.get('reg_uar'))}"
            lines.append(
                f"| {name} | {_fmt(au.get('f1_au'))} | {_fmt(au.get('jaccard_au'))} | "
                f"{reg_uf1} | {reg_uar} | {_fmt(text.get('bleu'))} | "
                f"{_fmt(text.get('rouge_1'))} | {report.get('n_events')} |")
        lines.append("")

    def strs_section(title: str, groups: Dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("......")
        lines.append("|---|---|---|---|---|---|---|")
        for name, report in groups.items():
            strs_block = report.get("strs", {}).get("score", {})
            whole_text = report.get("strs", {}).get("text", {})
            lines.append(
                f"| {name} | {_fmt(strs_block.get('strs'))} | "
                f"{_fmt(strs_block.get('f1_spot'))} | {_fmt(strs_block.get('f1_analysis'))} | "
                f"{_fmt(whole_text.get('bleu'))} | {_fmt(whole_text.get('rouge_1'))} | "
                f"{report.get('n_videos')} |")
        lines.append("")

    localisation_section("...", summary.get("per_dataset", {}))
    recognition_section("...", summary.get("per_dataset", {}))
    strs_section("...", summary.get("per_dataset", {}))
    localisation_section("...", summary.get("per_subject", {}))
    recognition_section("...", summary.get("per_subject", {}))
    strs_section("...", summary.get("per_subject", {}))
    overall = summary.get("overall", {})
    if overall.get("status") != "unavailable":
        localisation_section("...", {"overall": overall})
        recognition_section("...", {"overall": overall})
        strs_section("...", {"overall": overall})
    lines.append("...")
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


__all__ = [
    "qa_records_path", "write_qa_records", "read_qa_records", "iter_qa_record_files",
    "EventMetricInputs", "VideoMetricInputs", "collect_metric_inputs",
    "aggregate_final_metrics", "write_summary",
]
