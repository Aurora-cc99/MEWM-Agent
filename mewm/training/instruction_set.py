"""Role-specific instruction sets injected into agent prompts during training."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..data.datasets import ExpressionEvent, LongVideo
from ..data.qa_loader import (
    ObservationRecord, QASet, TRIPLE_TASK_QUESTION_EN, TRIPLE_TASK_QUESTION_ZH,
)
from ..knowledge.au_anatomy import SLOT_INDEX, au_label, roi_label
from ..knowledge.emotion_prototypes import coarse_of, competing_hypotheses, core_aus
from ..schemas import AU_PATTERN, EMOTION_LEXICON, scan_forbidden_vocabulary

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentSupervision:

    role: str
    phase: str
    prompt: str
    target: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "prompt": self.prompt, "target": self.target}


@dataclass
class InstructionSample:

    sample_id: str
    dataset: str
    video: str
    video_path: str
    fps: float
    n_frames: int
    question: str
    gt_events: List[Dict[str, Any]] = field(default_factory=list)
    agent_supervision: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    answer: Dict[str, Any] = field(default_factory=dict)
    source: str = "annotation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id, "dataset": self.dataset, "video": self.video,
            "video_path": self.video_path, "fps": self.fps, "n_frames": self.n_frames,
            "question": self.question, "gt_events": self.gt_events,
            "agent_supervision": self.agent_supervision, "answer": self.answer,
            "source": self.source,
        }


ROLE_BANS: Dict[str, Dict[str, bool]] = {
    "P": {"ban_au": True, "ban_emotion": True},
    "A": {"ban_au": False, "ban_emotion": True},
    "R": {"ban_au": False, "ban_emotion": False},
    "C": {"ban_au": False, "ban_emotion": False},
}


def causal_mask_check(role: str, target: Dict[str, Any]) -> List[str]:
    bans = ROLE_BANS.get(role, {})
    if not (bans.get("ban_au") or bans.get("ban_emotion")):
        return []
    scannable = {k: v for k, v in target.items() if k != "attribution"}
    return scan_forbidden_vocabulary(
        scannable, ban_au=bans.get("ban_au", False),
        ban_emotion=bans.get("ban_emotion", False),
    )


class InstructionSetBuilder:

    def __init__(self, lang: str = "en", seed: int = 20260824) -> None:
        self.lang = lang
        self.rng = random.Random(seed)
        self.question = (TRIPLE_TASK_QUESTION_ZH if lang == "zh"
                         else TRIPLE_TASK_QUESTION_EN)
        self.violations: List[Tuple[str, str, List[str]]] = []


    def build_video_sample(
        self,
        video: LongVideo,
        qa: Optional[QASet] = None,
        observations: Optional[Sequence[ObservationRecord]] = None,
    ) -> Optional[InstructionSample]:
        events = video.micro_events()
        if not events:
            return None

        observations = list(observations or (
            qa.observation_targets(video.video_key) if qa else []
        ))
        by_span = {(o.onset, o.offset): o for o in observations}

        sample = InstructionSample(
            sample_id=f"melvqa_{video.dataset}_{video.video_key}",
            dataset=video.dataset, video=video.video_key,
            video_path=str(video.paths.frame_dir), fps=video.fps,
            n_frames=video.n_frames, question=self.question,
        )
        sample.gt_events = [
            {"onset": e.onset, "apex": e.apex, "offset": e.offset,
             "aus": e.aus, "fine": e.fine_label, "coarse": e.coarse_label}
            for e in events
        ]

        sample.agent_supervision = self._agent_supervision(video, events, by_span)
        sample.answer = self._answer(video, events, by_span, qa)
        return sample


    def _agent_supervision(
        self, video: LongVideo, events: Sequence[ExpressionEvent],
        by_span: Dict[Tuple[int, int], ObservationRecord],
    ) -> Dict[str, Dict[str, Any]]:
        primary = events[0]
        record = by_span.get((primary.onset, primary.offset))
        active = self._active_aus(primary, record)
        supervision: Dict[str, Dict[str, Any]] = {}

        motion_evidence = []
        if record is not None:
            for observation in record.salient_rois()[:6]:
                motion_evidence.append({
                    "roi": observation.region_name,
                    "roi_index": observation.roi,
                    "magnitude_px": observation.magnitude_px,
                    "direction_deg": observation.direction_deg,
                    "direction_label": observation.direction_label,
                    "coherence": observation.coherence,
                    "salient": True,
                })
        p_target = {
            "proposals": [[e.onset, e.offset] for e in events],
            "motion_evidence": motion_evidence,
        }
        supervision["P"] = AgentSupervision(
            "P", "P.verify", self._role_prompt("P"), p_target).to_dict()

        a_target = {
            "active_aus": active,
            "weak_aus": [au for au in (record.related_supplementary_aus if record else [])
                         if au not in active][:2],
            "au_graph": self._graph_target(record, primary),
        }
        supervision["A"] = AgentSupervision(
            "A", "A.graph", self._role_prompt("A"), a_target).to_dict()

        competitors = competing_hypotheses(primary.fine_label or "other", 2)
        r_target = {
            "cot_C_layer": self._c_layer_text(primary, active, competitors),
            "k_crit": active[:2],
            "fine": primary.fine_label, "coarse": primary.coarse_label or coarse_of(primary.fine_label),
        }
        supervision["R"] = AgentSupervision(
            "R", "R.reason", self._role_prompt("R"), r_target).to_dict()

        c_target = {
            "checks": ["likelihood_ratio", "masking_necessity", "template_comparison"],
            "k_crit": active[:2],
            "final": "rejected",
        }
        supervision["C"] = AgentSupervision(
            "C", "C.critic", self._role_prompt("C"), c_target).to_dict()

        for role, entry in supervision.items():
            problems = causal_mask_check(role, entry["target"])
            if problems:
                self.violations.append((video.video_key, role, problems))
                LOGGER.warning("causal-mask violation in %s target for %s: %s",
                               role, video.video_key, problems)
        return supervision

    def _active_aus(self, event: ExpressionEvent,
                    record: Optional[ObservationRecord]) -> List[str]:
        annotated = [au for au in event.aus if au in SLOT_INDEX]
        if annotated:
            return annotated
        if record is not None:
            return [au for au in record.active_aus if au in SLOT_INDEX]
        return [au for au in core_aus(event.fine_label or "other") if au in SLOT_INDEX][:3]

    def _graph_target(self, record: Optional[ObservationRecord],
                      event: ExpressionEvent) -> Dict[str, Any]:
        nodes: Dict[str, Any] = {}
        edges: List[List[Any]] = []
        active = self._active_aus(event, record)
        span = max(1, event.offset - event.onset)
        for index, au in enumerate(active):
            offset = int(index * span * 0.1)
            nodes[au] = {
                "phase": [event.onset + offset, event.apex, event.offset],
                "role": "onset leader" if index == 0 else "apex synergy",
            }
        if record is not None:
            for edge in record.w_matrix.strongest_edges(3):
                edges.append([edge.get("source"), edge.get("target"), "+",
                              round(float(edge.get("weight", 0.0)), 3)])
        return {"nodes": nodes, "edges": edges}

    def _c_layer_text(self, event: ExpressionEvent, active: Sequence[str],
                      competitors: Sequence[str]) -> str:
        fine = event.fine_label or "other"
        rendered = ", ".join(active)
        rival = ", ".join(competitors)
        if self.lang == "zh":
            return (f"The core unit present [{rendered}] supports {fine}; The competitive assumption [{rival}] is excluded as it lacks its core unit and its dynamics do not match.")
        return (f"Core units present [{rendered}] support {fine}; competing hypotheses "
                f"[{rival}] are excluded for missing their own core units and for "
                f"dynamics mismatch.")

    def _role_prompt(self, role: str) -> str:
        from ..agents.base import load_role
        mapping = {"P": "p_agent_verify", "A": "a_agent_graph",
                   "R": "r_agent_reason", "C": "c_agent_critic"}
        try:
            return load_role(mapping[role]).system_prompt(self.lang).strip()
        except Exception:
            return f"You are the {role} agent."


    def _answer(
        self, video: LongVideo, events: Sequence[ExpressionEvent],
        by_span: Dict[Tuple[int, int], ObservationRecord], qa: Optional[QASet],
    ) -> Dict[str, Any]:
        part1 = [
            {"proposal_id": i + 1, "onset": e.onset, "offset": e.offset}
            for i, e in enumerate(events)
        ]
        part2 = []
        for index, event in enumerate(events, start=1):
            record = by_span.get((event.onset, event.offset))
            active = self._active_aus(event, record)
            part2.append({
                "proposal_id": index,
                "static_description": (record.static_description if record else
                                       self._static_fallback(event, active)),
                "dynamic_description": (record.dynamic_description if record else
                                        self._dynamic_fallback(event, active)),
                "au_cot": self._au_cot(event, active, record),
                "coarse_label": event.coarse_label or coarse_of(event.fine_label),
                "fine_label": event.fine_label,
            })

        narrative = ""
        if qa is not None:
            item = qa.triple_task(video.video_key)
            if item is not None:
                narrative = item.answer_text
        if not narrative:
            narrative = self._narrative_fallback(video, events)

        return {"part1_proposals": part1, "part2_analysis": part2,
                "global_narrative": narrative}

    def _static_fallback(self, event: ExpressionEvent, active: Sequence[str]) -> str:
        rendered = ", ".join(f"{au} ({au_label(au)})" for au in active)
        return (f"Frames {event.onset}-{event.offset} hold a restrained configuration "
                f"consistent with {rendered}, read as {event.fine_label}.")

    def _dynamic_fallback(self, event: ExpressionEvent, active: Sequence[str]) -> str:
        if not active:
            return f"Frames {event.onset}-{event.offset} carry no resolvable unit motion."
        leader = active[0]
        followers = ", ".join(active[1:]) or "no secondary unit"
        return (f"onset to apex: {leader} ({au_label(leader)}) leads from frame "
                f"{event.onset}; {followers} join near the apex at {event.apex}. "
                f"apex to offset: all regions fall back together by {event.offset} with "
                f"no new activation.")

    def _au_cot(self, event: ExpressionEvent, active: Sequence[str],
                record: Optional[ObservationRecord]) -> str:
        if record is not None and record.w_matrix.main_path:
            return " -> ".join(str(p) for p in record.w_matrix.main_path)
        chain = " -> ".join(f"{au}({au_label(au)})" for au in active)
        return f"{chain} -> {event.fine_label} -> {event.coarse_label or coarse_of(event.fine_label)}"

    def _narrative_fallback(self, video: LongVideo,
                            events: Sequence[ExpressionEvent]) -> str:
        parts = [f"Across the {video.n_frames} frames of {video.video_key}, "
                 f"{len(events)} micro-expression event(s) were annotated."]
        for index, event in enumerate(events, start=1):
            parts.append(
                f"Event {index}: onset {event.onset}, apex {event.apex}, offset "
                f"{event.offset}, read as {event.fine_label} "
                f"({event.coarse_label or coarse_of(event.fine_label)})."
            )
        parts.append("Outside these intervals the face holds a neutral baseline.")
        return " ".join(parts)


    def build_dataset(
        self, videos: Sequence[LongVideo], qa: Optional[QASet] = None,
        limit: int = 0,
        include_subjects: Optional[Iterable[str]] = None,
        exclude_subjects: Optional[Iterable[str]] = None,
    ) -> List[InstructionSample]:
        keep = {str(s) for s in include_subjects} if include_subjects is not None else None
        drop = {str(s) for s in exclude_subjects} if exclude_subjects is not None else set()

        samples = []
        for video in videos:
            subject = str(video.subject)
            if keep is not None and subject not in keep:
                continue
            if subject in drop:
                continue
            sample = self.build_video_sample(video, qa)
            if sample is not None:
                samples.append(sample)
            if limit and len(samples) >= limit:
                break
        return samples

    def mix(
        self,
        agent_samples: Sequence[InstructionSample],
        qa_samples: Sequence[InstructionSample],
        self_produced: Sequence[InstructionSample] = (),
        ratio: Tuple[int, int, int] = (4, 4, 2),
    ) -> List[InstructionSample]:
        total = sum(ratio)
        budget = max(len(agent_samples), len(qa_samples), 1) * total // max(1, ratio[0])
        counts = [budget * r // total for r in ratio]
        pools = [list(agent_samples), list(qa_samples), list(self_produced)]
        mixed: List[InstructionSample] = []
        for pool, count in zip(pools, counts):
            if not pool:
                continue
            self.rng.shuffle(pool)
            mixed.extend(pool[:count] if count <= len(pool) else pool)
        self.rng.shuffle(mixed)
        return mixed

    def report(self) -> Dict[str, Any]:
        return {"causal_mask_violations": len(self.violations),
                "violations": self.violations[:10]}


def write_jsonl(samples: Iterable[InstructionSample], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: Path | str) -> List[InstructionSample]:
    samples = []
    with open(Path(path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(InstructionSample(
                sample_id=payload.get("id", ""), dataset=payload.get("dataset", ""),
                video=payload.get("video", ""), video_path=payload.get("video_path", ""),
                fps=float(payload.get("fps", 30.0)),
                n_frames=int(payload.get("n_frames", 0)),
                question=payload.get("question", ""),
                gt_events=payload.get("gt_events", []),
                agent_supervision=payload.get("agent_supervision", {}),
                answer=payload.get("answer", {}),
                source=payload.get("source", "annotation"),
            ))
    return samples


def to_chat_format(sample: InstructionSample, role: Optional[str] = None) -> Dict[str, Any]:
    if role:
        supervision = sample.agent_supervision.get(role)
        if not supervision:
            raise KeyError(f"sample {sample.sample_id} has no {role} supervision")
        return {
            "id": f"{sample.sample_id}#{role}",
            "messages": [
                {"role": "system", "content": supervision["prompt"]},
                {"role": "user", "content": json.dumps(
                    {"video": sample.video, "fps": sample.fps}, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(
                    supervision["target"], ensure_ascii=False)},
            ],
        }
    return {
        "id": sample.sample_id,
        "messages": [
            {"role": "user", "content": sample.question},
            {"role": "assistant", "content": json.dumps(sample.answer, ensure_ascii=False)},
        ],
    }


__all__ = [
    "AgentSupervision", "InstructionSample", "ROLE_BANS", "causal_mask_check",
    "InstructionSetBuilder", "write_jsonl", "read_jsonl", "to_chat_format",
]
