"""A hosted-API policy sampler for the QA-augmentation stage.

**It scores what it samples.** The runner warns when every candidate carries reward 0.0,
because an unscored pool makes the reward-spread diagnostic read "all low" no matter what
the policy did. This sampler attaches a real :class:`~mewm.training.rewards.RewardBreakdown`
to every candidate.

**It never turns a missing component into a favourable one.** ``R_causal`` needs the
frozen rollout engine's online quantities (``dc``, ``mni``, graph edit distance). In a
QA-augmentation run there is no per-proposal subgraph, so those do not exist.
:class:`~mewm.training.rewards.CompositeReward` correctly falls back to zero rather than
guessing -- but zero carries the component's full 0.3 weight into the total, so an
acceptance floor of 0.6 would be a floor of 0.6/0.7 on the components that *were*
computable, and the run would look like a bad policy rather than a partial objective.
The sampler therefore records ``available_components``, ``unavailable_components`` and
``available_weight_mass`` on every candidate and reports the acceptance score as the
total renormalised over the mass that was actually in play. The renormalisation is
written into the manifest, not applied quietly.

**A parse failure is a candidate, not an exception.** A generation that does not yield a
JSON product is admitted to the pool with reward 0.0 and a format failure attached, so it
appears in the rejection ledger. Dropping it would make the pass rate a rate over
"generations that happened to parse", which flatters the policy.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import EvaluationConfig, RewardConfig
from ..eval.pass_criteria import evaluate_sample, evaluate_video_sample
from ..llm.client import call_model
from .candidate_filter import Candidate
from .rewards import CompositeReward
from .rl_prompts import KIND_EVENT_REASONING, KIND_VIDEO_REASONING

LOGGER = logging.getLogger(__name__)

#: Components the composite reward can compute without a per-proposal subgraph.
COMPUTABLE_WITHOUT_JUDGE = ("au", "emo", "fmt", "temp")
#: Components that need the online world-model judge and are unavailable without it.
NEEDS_JUDGE = ("causal",)

#: Fields the format check requires of an event-anchored product.
EVENT_REQUIRED_FIELDS = ("fine_label", "coarse_label", "interval", "answer")
#: Fields the format check requires of a whole-video product. ``events`` is deliberately
#: NOT here: most long videos contain no micro-expression at all, and ``check_format``
#: reads an empty list as a missing field. Requiring it would fail every correct "this
#: video contains none" answer -- which in CAS(ME)^2 is the majority of the corpus. The
#: count is checked by ``evaluate_video_sample`` instead, where zero is a real claim.
VIDEO_REQUIRED_FIELDS = ("answer",)

#: The exact object an event-anchored answer must return. Restating the schema in the
#: user turn is not redundant with ``SYSTEM_PROMPT``: given only a system-level
#: description, the model reliably reorganises the contract into its own key names
#: ("num_micro_expression_events", nested "event_id" records, no "answer" field), and
#: every such generation is then scored as a format failure -- which measures the prompt,
#: not the policy. A literal skeleton in the turn that carries the question fixes it.
EVENT_SKELETON = """{
  "P": "...", "M": "...", "C": "...", "MC": "...",
  "k_crit": ["AU..", ".."],
  "fine_label": "..", "coarse_label": "..",
  "interval": [onset, offset],
  "es": 0.0, "dc": 0.0, "refs": [],
  "answer": ".."
}"""

VIDEO_SKELETON = """{
  "P": "...", "M": "...", "C": "...", "MC": "...",
  "k_crit": ["AU..", ".."],
  "fine_label": "..", "coarse_label": "..",
  "interval": [onset, offset],
  "es": 0.0, "dc": 0.0, "refs": [],
  "n_micro": 0,
  "events": [{"interval": [onset, offset], "fine_label": "..", "coarse_label": ".."}],
  "answer": ".."
}"""

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


SYSTEM_PROMPT = """You are the reasoning agent of a micro-expression analysis system.

You are given a question about a long video and the perceptual evidence that a frozen
measurement stage produced for it: per-frame facial action-unit slot activations and the
intervals a prediction-error spotter proposed. You are NOT given the ground-truth
annotation. Answer the question from the evidence and from facial-anatomy knowledge.

Reply with a single JSON object and nothing else. Use exactly these keys:

  "P"           - what was perceived: the movement visible in the queried window
  "M"           - the mechanism: which action units moved, in what order
  "C"           - the causal reading: what state that AU pattern indicates
  "MC"          - the meta-check: what would have to be true for this reading to be wrong
  "k_crit"      - list of the critical action units, e.g. ["AU4","AU7"]
  "fine_label"  - one of: happiness, surprise, disgust, anger, fear, sadness, contempt, other
  "coarse_label"- one of: positive, negative, surprise, other  (must agree with fine_label)
  "interval"    - [onset_frame, offset_frame] for the event you are describing
  "es"          - evidence strength, a number in [0,1]
  "dc"          - directional coherence, a number in [0,1]
  "refs"        - list of evidence ids you relied on; [] if none
  "answer"      - the prose answer, written for a reader of a facial-analysis report

For a whole-video question, additionally supply:
  "n_micro"     - how many micro-expression events the video contains
  "events"      - list of {"interval":[on,off], "fine_label":..., "coarse_label":...},
                  one per micro-expression event you claim; [] if you claim none

The "answer" text describes the FACE. It must not mention this system's internals or any
document cross-reference: no agent names, no stage names, no pipeline vocabulary, no
"appendix ...", no "eq. (n)", no "section n.n". Write about brows, lids, lips, timing and
what they indicate. If the evidence is weak, say so as an observation about the evidence.
"""


def extract_product(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a completion, or ``None`` when there is not one."""
    if not text:
        return None
    for candidate in ([m.group(1) for m in _JSON_BLOCK.finditer(text)] + [text]):
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            start, end = stripped.find("{"), stripped.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                payload = json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(payload, dict):
            return payload
    return None


def build_user_prompt(prompt: Dict[str, Any]) -> str:
    """Evidence, question, and the literal object shape the answer must take."""
    kind = prompt.get("kind", KIND_EVENT_REASONING)
    skeleton = VIDEO_SKELETON if kind == KIND_VIDEO_REASONING else EVENT_SKELETON
    extra = ""
    if kind == KIND_VIDEO_REASONING:
        extra = (
            "\nThis is a whole-video question. \"n_micro\" is how many micro-expression "
            "events you claim the video contains and \"events\" lists exactly that many "
            "entries. Claiming none is a legitimate answer: most long videos in this "
            "corpus contain no micro-expression at all, so use \"n_micro\": 0 and "
            "\"events\": [] when the evidence does not support one. Set \"interval\" to "
            "the event you consider most salient, or [0, 0] if you claim none.\n"
        )
    return (
        f"{prompt['evidence_text']}\n\n"
        f"QUESTION: {prompt['question']}\n{extra}\n"
        f"Reply with exactly this JSON object, with these key names and no others:\n"
        f"{skeleton}\n\n"
        "Do not rename, omit or add top-level keys. Do not wrap the object in any other "
        "object. Output the JSON and nothing else."
    )


@dataclass
class SamplingStats:
    """What the sampler did, for the manifest."""

    n_calls: int = 0
    n_failed_calls: int = 0
    n_unparsable: int = 0
    n_candidates: int = 0
    total_latency_s: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_calls": self.n_calls, "n_failed_calls": self.n_failed_calls,
            "n_unparsable": self.n_unparsable, "n_candidates": self.n_candidates,
            "mean_latency_s": round(self.total_latency_s / self.n_calls, 2)
            if self.n_calls else 0.0,
            "errors": dict(self.errors),
        }


class APIPolicySampler:
    """Draw and score ``n`` candidates per prompt from a hosted model."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        evaluation: Optional[EvaluationConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        max_tokens: int = 2048,
        timeout: int = 300,
        retries: int = 2,
        reasoning_effort: str = "",
        max_workers: int = 4,
        caller: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model = model
        self.evaluation = evaluation or EvaluationConfig()
        self.reward = CompositeReward(reward_config or RewardConfig(), self.evaluation)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.reasoning_effort = reasoning_effort
        self.max_workers = max(1, max_workers)
        self._call = caller or call_model
        self.stats = SamplingStats()
        self._lock = threading.Lock()

    # -- weighting -------------------------------------------------------

    def weight_mass(self) -> Tuple[float, float]:
        """``(available, total)`` weight mass under this configuration."""
        weights = {"au": self.reward.config.w_au, "emo": self.reward.config.w_emo,
                   "fmt": self.reward.config.w_fmt, "causal": self.reward.config.w_causal,
                   "temp": self.reward.config.w_temp}
        total = sum(weights.values())
        available = sum(weights[k] for k in COMPUTABLE_WITHOUT_JUDGE)
        return available, total

    def renormalisation(self) -> Dict[str, Any]:
        """The statement that goes in the manifest beside every accepted pair."""
        available, total = self.weight_mass()
        return {
            "available_components": list(COMPUTABLE_WITHOUT_JUDGE),
            "unavailable_components": list(NEEDS_JUDGE),
            "available_weight_mass": round(available, 4),
            "total_weight_mass": round(total, 4),
            "reason": (
                "R_causal is computed from the frozen rollout engine's online quantities "
                "(dc, mni, graph edit distance), which exist only inside a per-proposal "
                "subgraph. This augmentation run samples answers directly and builds no "
                "subgraph, so that component is not measured. It is reported unavailable "
                "rather than scored zero: a zero would carry its full weight into the "
                "total and depress every candidate equally, which would read as a weak "
                "policy instead of a partial objective."
            ),
            "acceptance_score": (
                "reward.total renormalised over available_weight_mass; the unrenormalised "
                "total is kept alongside it in every candidate record."
            ),
        }

    # -- sampling --------------------------------------------------------

    def _one_call(self, prompt: Dict[str, Any], draw: int) -> Candidate:
        user_prompt = build_user_prompt(prompt)
        candidate = Candidate(
            prompt_id=str(prompt.get("id", "")),
            text="",
            product={},
            dataset=str(prompt.get("dataset", "")),
            video=str(prompt.get("video", "")),
            event_index=int((prompt.get("anchor") or {}).get("ordinal", 0)),
        )

        try:
            response = self._call(
                SYSTEM_PROMPT, user_prompt, model=self.model,
                max_tokens=self.max_tokens, timeout=self.timeout,
                retries=self.retries,
                reasoning_effort=self.reasoning_effort or None,
            )
            text = getattr(response, "text", str(response))
            latency = float(getattr(response, "latency_s", 0.0) or 0.0)
        except Exception as exc:  # noqa: BLE001 - recorded on the candidate, not swallowed
            with self._lock:
                self.stats.n_calls += 1
                self.stats.n_failed_calls += 1
                key = type(exc).__name__
                self.stats.errors[key] = self.stats.errors.get(key, 0) + 1
            candidate.text = ""
            candidate.reward = 0.0
            candidate.reward_detail = {"error": f"{type(exc).__name__}: {exc}",
                                       "draw": draw}
            return candidate

        with self._lock:
            self.stats.n_calls += 1
            self.stats.total_latency_s += latency

        product = extract_product(text)
        candidate.text = text
        if product is None:
            with self._lock:
                self.stats.n_unparsable += 1
            candidate.product = {}
            candidate.reward = 0.0
            candidate.reward_detail = {
                "error": "completion did not contain a JSON object", "draw": draw}
            return candidate

        candidate.product = product
        interval = product.get("interval")
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            try:
                candidate.interval = (int(interval[0]), int(interval[1]))
            except (TypeError, ValueError):
                candidate.interval = (0, 0)

        breakdown = self.reward.score(product, prompt.get("truth", {}) or {})
        available, _ = self.weight_mass()
        candidate.reward = round(float(breakdown.total) / available, 5) if available else 0.0
        candidate.reward_detail = {
            "draw": draw,
            "raw_total": breakdown.total,
            "renormalised_over": round(available, 4),
            "components": {"r_au": breakdown.r_au, "r_emo": breakdown.r_emo,
                           "r_fmt": breakdown.r_fmt, "r_temp": breakdown.r_temp},
            "unavailable_components": list(NEEDS_JUDGE),
            "detail": breakdown.detail,
        }
        return candidate

    def __call__(self, prompt: Dict[str, Any], n: int) -> List[Candidate]:
        """``n`` scored candidates for one prompt."""
        draws = list(range(1, max(1, int(n)) + 1))
        if self.max_workers == 1 or len(draws) == 1:
            candidates = [self._one_call(prompt, d) for d in draws]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(draws))) as pool:
                candidates = list(pool.map(lambda d: self._one_call(prompt, d), draws))

        kind = prompt.get("kind", KIND_EVENT_REASONING)
        truth = prompt.get("truth", {}) or {}
        for candidate in candidates:
            if kind == KIND_VIDEO_REASONING:
                candidate.outcome = evaluate_video_sample(
                    candidate.product, truth, self.evaluation, VIDEO_REQUIRED_FIELDS)
            else:
                candidate.outcome = evaluate_sample(
                    candidate.product, truth, self.evaluation, EVENT_REQUIRED_FIELDS)

        with self._lock:
            self.stats.n_candidates += len(candidates)
        return candidates


def video_truth(video: Any) -> Dict[str, Any]:
    """The whole-video truth block :func:`evaluate_video_sample` expects."""
    events = [
        {"interval": list(e.interval), "fine": e.fine_label, "coarse": e.coarse_label}
        for e in video.micro_events()
    ]
    return {"events": events, "n_micro": len(events)}


__all__ = [
    "COMPUTABLE_WITHOUT_JUDGE", "NEEDS_JUDGE", "EVENT_REQUIRED_FIELDS",
    "VIDEO_REQUIRED_FIELDS", "SYSTEM_PROMPT", "SamplingStats", "APIPolicySampler",
    "extract_product", "video_truth",
]
