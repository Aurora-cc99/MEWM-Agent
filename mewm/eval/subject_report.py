"""按受试者的问答落盘 + 最终指标汇总（formwork.md 第 IV/V 条尾句，
``MEWM-Agent_完整执行方案.md`` 第 9 节）。

1. :func:`write_qa_records` —— 把 :class:`mewm.qa.interrogate.QARecord`（或等价的
   ``dict``）逐条写入 ``runs/<run_id>/<dataset>/<subject>/<video_id>/qa_records.jsonl``，
   asked 和 gated-skipped 的问题都在同一个文件里，互不覆盖（见 formwork.md IV）。
2. :func:`aggregate_final_metrics` + :func:`write_summary` —— 把若干视频的定位/识别/文本
   原始预测，按"每数据集""每受试者""全体"三个粒度分别喂给
   ``mewm.eval.megc_metrics.megc_report``（不重新实现任何一条指标公式），产出
   ``runs/<run_id>/final_metrics_summary.json`` 与同名 ``.md``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import RUNS_ROOT
from .megc_metrics import (
    au_scores, count_scores, megc_report, recognition_scores, spotting_scores,
    text_scores,
)

# ---------------------------------------------------------------------------
# 1. 每受试者·每视频·每问题记录
# ---------------------------------------------------------------------------


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
    """Append one ``qa_records.jsonl`` for a video.
    """
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
    """Every ``(dataset, subject, video_id, path)`` under one run's tree."""
    base = (Path(run_root) if run_root else RUNS_ROOT) / run_id
    out: List[Tuple[str, str, str, Path]] = []
    if not base.is_dir():
        return out
    for path in base.glob("*/*/*/qa_records.jsonl"):
        video_id, subject, dataset = path.parent.name, path.parent.parent.name, \
            path.parent.parent.parent.name
        out.append((dataset, subject, video_id, path))
    return out


# ---------------------------------------------------------------------------
# 2. 最终指标汇总
# ---------------------------------------------------------------------------


@dataclass
class EventMetricInputs:
    """One TP/candidate event's recognition material (第 8 节 F1AU/JaccardAU/RegUF1/...)."""

    au_predicted: List[str] = field(default_factory=list)
    au_true: List[str] = field(default_factory=list)
    fine_true: str = ""
    fine_pred: str = ""
    text_candidate: str = ""
    text_reference: str = ""
    #: Whether this event's proposal was a spotting true positive -- gates which events
    #: count toward ``f1_analysis_on_tp`` for STRS (MEGC2025 sec. 2.5).
    is_spotting_tp: bool = False


@dataclass
class VideoMetricInputs:
    """One video's spotting + counting + per-event recognition material."""

    dataset: str
    subject: str
    video_id: str
    proposals: List[Tuple[int, int]] = field(default_factory=list)
    proposal_types: List[str] = field(default_factory=list)
    proposal_labels: List[str] = field(default_factory=list)
    truth: List[Tuple[int, int]] = field(default_factory=list)
    truth_types: List[str] = field(default_factory=list)
    truth_labels: List[str] = field(default_factory=list)
    #: quantity ("expression"/"micro"/"macro") -> (predicted_count, true_count)
    counts: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)
    events: List[EventMetricInputs] = field(default_factory=list)


def _spotting_row(v: VideoMetricInputs) -> Dict[str, Any]:
    return {
        "video": v.video_id, "proposals": v.proposals,
        "proposal_types": v.proposal_types, "proposal_labels": v.proposal_labels,
        "truth": v.truth, "truth_types": v.truth_types, "truth_labels": v.truth_labels,
    }


def _group_report(videos: Sequence[VideoMetricInputs], iou_threshold: float = 0.5
                  ) -> Dict[str, Any]:
    """One :func:`megc_metrics.megc_report` computed over ``videos`` (a dataset,
    a subject, or the whole run) -- the exact same function every other grouping
    calls, so per-dataset/per-subject/overall numbers are guaranteed comparable.
    """
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

    # STRS's F1_a: MEGC fine-grained UF1 restricted to spotting-TP events (sec. 2.5),
    # with the ground-truth-interval version carried alongside for the visible gap
    # (megc_metrics.megc_report's own contract -- see its docstring point 3).
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
        strs_text=recognition_text,
        f1_analysis_on_tp=_f1a(tp_events), f1_analysis_on_truth=_f1a(all_events),
    )
    report["n_videos"] = len(videos)
    report["n_subjects"] = len({v.subject for v in videos})
    report["n_events"] = len(all_events)
    return report


def aggregate_final_metrics(
    videos: Sequence[VideoMetricInputs],
    *,
    run_id: str,
    protocol: str,
    mode: str,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """``per_dataset`` / ``per_subject`` / ``overall`` MEGC reports over every video.

    ``protocol`` is ``"loso"`` or ``"lodo"``; ``mode`` is ``"api"`` or ``"open_weight"``
    (第 6 节) -- both are carried through verbatim into the summary so a report can
    never be read out of context.
    """
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
            "MAE/RMSE 为逐视频事件计数误差（MEGC2026 附录 C 口径），不是 onset/offset 边界误差",
            "STRS 的识别侧 F1 只统计定位 TP 区间内的事件（MEGC2025 2.5 节），"
            "ground-truth 区间上的同一数值一并给出用于对比",
            "宏平均类指标 (RegUF1/RegUAR/SpotUF1/SpotUAR) 的 macro_divisor 字段标注了参与"
            "平均的类别数，空提案的分组会显式报告 status=unavailable 而不是 0",
        ],
    }
    return summary


def write_summary(summary: Dict[str, Any], *, run_id: str,
                  run_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """``final_metrics_summary.json`` + a human-readable ``.md`` beside it."""
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
        f"# MEWM-Agent 最终指标汇总 -- run `{summary.get('run_id')}`",
        "",
        f"- protocol: `{summary.get('protocol')}`　mode: `{summary.get('mode')}`　"
        f"iou_threshold: `{summary.get('iou_threshold')}`",
        "",
    ]

    def section(title: str, groups: Dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 分组 | SpotUF1 | SpotUAR | F1AU | JaccardAU | RegUF1(megc) | "
                     "RegUAR(megc) | BLEU | ROUGE-1 | STRS | n_videos | n_events |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, report in groups.items():
            spot = report.get("spotting", {}).get("interval", {}).get("unweighted_type", {})
            au = report.get("recognition", {}).get("action_units", {})
            emo = report.get("recognition", {}).get("emotion", {}).get("fine", {}).get("megc", {})
            text = report.get("recognition", {}).get("text", {})
            strs_block = report.get("strs", {}).get("score", {})
            lines.append(
                f"| {name} | {_fmt(spot.get('spot_uf1'))} | {_fmt(spot.get('spot_uar'))} | "
                f"{_fmt(au.get('f1_au'))} | {_fmt(au.get('jaccard_au'))} | "
                f"{_fmt(emo.get('reg_uf1'))} | {_fmt(emo.get('reg_uar'))} | "
                f"{_fmt(text.get('bleu'))} | {_fmt(text.get('rouge_1'))} | "
                f"{_fmt(strs_block.get('strs'))} | {report.get('n_videos')} | "
                f"{report.get('n_events')} |")
        lines.append("")

    section("按数据集 (per_dataset)", summary.get("per_dataset", {}))
    section("按受试者 (per_subject)", summary.get("per_subject", {}))
    overall = summary.get("overall", {})
    if overall.get("status") != "unavailable":
        section("全体 (overall)", {"overall": overall})
    lines.append("## 备注")
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


__all__ = [
    "qa_records_path", "write_qa_records", "read_qa_records", "iter_qa_record_files",
    "EventMetricInputs", "VideoMetricInputs", "aggregate_final_metrics", "write_summary",
]
