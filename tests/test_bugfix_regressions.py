"""Regression checks for defects found by end-to-end runs against live APIs.

Bugs 1-5 came from the first end-to-end run; bugs 6-10 from the gpt-5.6-sol high-effort
run on casme_sq. Each test reproduces the original failure condition and asserts the fix,
so a later change that reintroduces one fails here rather than silently degrading a run.

Run:  python tests/test_bugfix_regressions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[tuple[str, str]] = []
CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append((name, detail))


# ---------------------------------------------------------------------------
print("\nBug 1 -- agents returned Markdown instead of JSON")
# ---------------------------------------------------------------------------
from mewm.agents.base import (
    CONTRACT_SCHEMAS, JSON_ONLY_DIRECTIVE, load_role, schema_reminder,
)

ROLES = ["p_agent_scan", "p_agent_verify", "a_agent_encode", "a_agent_graph",
         "r_agent_reason", "r_agent_respond", "r_agent_adjudicate", "r_agent_narrate",
         "c_agent_critic"]

check("every role declares an output contract",
      all(load_role(name).output_contract for name in ROLES),
      str([name for name in ROLES if not load_role(name).output_contract]))
check("every declared contract has a schema",
      all(load_role(name).output_contract in CONTRACT_SCHEMAS for name in ROLES),
      str([load_role(n).output_contract for n in ROLES
           if load_role(n).output_contract not in CONTRACT_SCHEMAS]))
check("the schema reminder names the brace delimiters",
      all("{" in schema_reminder(load_role(n).output_contract) for n in ROLES))
check("the JSON directive forbids prose and fences",
      "No Markdown" in JSON_ONLY_DIRECTIVE and "code fences" in JSON_ONLY_DIRECTIVE)

# The reminder must survive token regulation -- it is an instruction, not evidence.
import inspect

from mewm.agents import base as base_module

source = inspect.getsource(base_module.BaseAgent.run)
check("the schema reminder is appended after token regulation",
      source.index("_regulate") < source.index("schema_reminder"),
      "regulation could otherwise drop the instruction")

# ---------------------------------------------------------------------------
print("\nBug 2 -- reference-graph open questions were discarded")
# ---------------------------------------------------------------------------
from mewm.pipeline import MEWMPipeline

source = inspect.getsource(MEWMPipeline._build_context)
check("the parser's open questions are registered into the state",
      "register_question" in source and "graph_questions" in source,
      "gate R4 fails at every later phase when they are dropped")
check("_build_context receives the state to register into",
      "state: Optional[MEWMState]" in inspect.getsource(MEWMPipeline._build_context)
      or "state," in source)

# End to end: a sign conflict must reach the state as an open question.
from mewm.agents.structure import build_reference_graph
from mewm.knowledge.au_anatomy import SLOT_AUS

# AU12 and AU15 are unambiguously antagonistic (unlike AU1/AU4, which the knowledge base
# lists both ways and therefore treats as carrying no prior sign). Making them co-vary
# *positively* forces the prior/observation sign conflict that must be registered. A
# sharp asymmetric pulse is used rather than a symmetric ramp so the surrogate null can
# actually reject.
frames = list(range(100, 140))
pulse = np.zeros(40)
pulse[10:20] = np.array([0.1, 0.35, 0.7, 0.9, 0.95, 0.8, 0.55, 0.3, 0.15, 0.05])
trajectories = {au: np.zeros(40) for au in SLOT_AUS}
trajectories["AU12"] = pulse
trajectories["AU15"] = np.roll(pulse, 2) * 0.9
graph, questions = build_reference_graph(trajectories, frames, cid="p01",
                                         n_permutations=60,
                                         active_aus=["AU12", "AU15"], weak_aus=[])
conflicts = [e for e in graph.edges if e.conflict]
check("an edge forms between co-varying units",
      bool(graph.edges), f"{len(graph.edges)} edges")
check("a prior/observation sign conflict is detected",
      bool(conflicts), f"{len(graph.edges)} edges, none conflicting")
check("each detected conflict yields an open question",
      len(questions) >= len(conflicts),
      f"{len(questions)} questions, {len(conflicts)} conflicts")
check("a conflicting edge carries an undefined weight",
      all(e.weight != e.weight for e in conflicts),
      "a conflicted edge must not be silently averaged")

# ---------------------------------------------------------------------------
print("\nBug 3 -- absolute threshold marked 13 of 16 AUs active")
# ---------------------------------------------------------------------------
from mewm.engines.v2_slots import (
    SlotReadout, coherence_is_saturated, select_active_slots,
)
from mewm.schemas import ROIMeasurement

# Reproduce the saturated-coherence condition that caused it.
saturated = [
    ROIMeasurement(i + 1, f"roi{i}", f"ROI {i}", 0.30, 200.0, 0.99, True)
    for i in range(29)
]
check("saturated coherence is detected", coherence_is_saturated(saturated))

varied = [
    ROIMeasurement(i + 1, f"roi{i}", f"ROI {i}", 0.30, 200.0, 0.1 + 0.03 * i, True)
    for i in range(29)
]
check("healthy coherence is not flagged as saturated",
      not coherence_is_saturated(varied))

# The original failure: many AUs just above an absolute cut.
crowded = {au: SlotReadout(au, value, value, 0.9, 0.9)
           for au, value in zip(SLOT_AUS, [0.63, 0.61, 0.50, 0.49, 0.43, 0.43, 0.42,
                                           0.38, 0.37, 0.37, 0.36, 0.33, 0.26, 0.23,
                                           0.21, 0.21])}
above_absolute = sum(1 for r in crowded.values() if r.activation >= 0.35)
active, weak = select_active_slots(crowded)
check("the absolute cut alone would over-select", above_absolute >= 11,
      f"{above_absolute} above 0.35")
check("competitive selection bounds the activation set",
      len(active) <= 5, f"{len(active)} active: {active}")
check("demoted units are kept as weak, not dropped", len(weak) > 0)
check("the strongest response is retained", SLOT_AUS[0] in active, str(active))

# The graph parser must not apply a second, independent threshold.
graph2, _q = build_reference_graph(
    {au: (pulse if au in {"AU4", "AU7", "AU24"} else np.zeros(pulse.size))
     for au in SLOT_AUS},
    frames, cid="p", n_permutations=20, active_aus=["AU4"], weak_aus=["AU7"])
check("the graph honours the upstream activation decision",
      set(graph2.active_aus) <= {"AU4"}, str(graph2.active_aus))
check("the graph excludes units outside the decision",
      "AU24" not in graph2.nodes, str(list(graph2.nodes)))

# ---------------------------------------------------------------------------
print("\nBug 4 -- narrative asserted a neutral baseline over a noisy curve")
# ---------------------------------------------------------------------------
from mewm.agents.reasoning import _template_narrative
from mewm.orchestration.state import MEWMState
from mewm.schemas import CandidateInterval, VideoMeta

state = MEWMState(video_meta=VideoMeta("v", "casme_sq", "/p", 30.0, 1000, "s15"))
state.proposals = [CandidateInterval("p01", 100, 110, 105, 6.0, {})]

noisy = _template_narrative(state, [{"interval": [8, 98], "mean": 14.37, "max": 80.0}])
check("a noisy baseline is not called neutral",
      "neutral" not in noisy.lower() or "too noisy to call neutral" in noisy.lower(),
      noisy[-160:])
check("a noisy baseline is described as such",
      "noisy" in noisy.lower(), noisy[-160:])

quiet = _template_narrative(state, [{"interval": [8, 98], "mean": 0.3, "max": 0.8}])
check("a genuinely quiet baseline is described as quiet",
      "quiet" in quiet.lower(), quiet[-160:])

# ---------------------------------------------------------------------------
print("\nBug 5 -- prose in typed fields crashed or corrupted the parse")
# ---------------------------------------------------------------------------
from mewm.agents.base import coerce_float, coerce_float_map
from mewm.schemas import SUPPRESSION_STATES, OpenQuestion, coerce_suppression

check("prose in a score map is dropped, numbers kept",
      coerce_float_map({"a": 0.8, "b": "high, coherence 0.85", "c": "n/a"})
      == {"a": 0.8, "b": 0.85})
check("a numeric field never raises on prose", coerce_float("about 0.47") == 0.47)
check("a numeric field never raises on None", coerce_float(None) == 0.0)
check("a numeric field never raises on a dict", coerce_float({"x": 1}) == 0.0)

for shape in ("bare string", {"kind": "k", "detail": "d"}, {"text": "t"}, 123):
    question = OpenQuestion.coerce(shape, "p01")
    check(f"an open question survives shape {type(shape).__name__}",
          bool(question.detail), repr(shape))

state_value, prose = coerce_suppression(
    "fine label 'surprise' suppressed to coarse 'other': prototype_completeness=0.0")
check("a suppression paragraph resolves to a valid state",
      state_value in SUPPRESSION_STATES, state_value)
check("a label downgrade is not recorded as a facial suppression finding",
      state_value == "none", state_value)
check("the suppression reasoning is preserved, not discarded", bool(prose))
check("a genuine masquerade description resolves to masked",
      coerce_suppression("a masquerading social smile over the leak")[0] == "masked")

# ---------------------------------------------------------------------------
print("\nBugs 6-10 -- defects found by the gpt-5.6-sol high-effort run on casme_sq")
# ---------------------------------------------------------------------------
import inspect

from mewm.agents.base import extract_json, json_reject_reason
from mewm.orchestration.gates import ConsistencyGate
from mewm.pipeline import MEWMPipeline
from mewm.schemas import scan_forbidden_vocabulary

# -- Bug 6: R2 named the banned token but not the field it sat in. The perception
# contract carries AU keys legitimately in its exempt `attribution` map, so a bare token
# list read as "delete the schema-mandated keys", and a temperature-0 agent answered all
# three gate retries identically. P.scan degraded on every real run.
product = {
    "proposals": [{
        "cid": "p01", "interval": [28, 40],
        "attribution": {"AU4": 0.16, "AU24": 0.11},             # exempt: engine provenance
        "notes": "AU4+AU24 co-activation, duration 13 frames",  # scanned: agent prose
    }],
    "curve_summary": "", "rejected": [],
}
hits = ConsistencyGate._r2_boundary(product, True, True)
check("R2 still fires on AU names written into agent prose", bool(hits))
check("the R2 hit names the offending field, not just the token",
      any("proposals[0].notes" in h for h in hits), str(hits))
check("the R2 hit does not blame the exempt attribution map",
      not any("attribution" in h for h in hits), str(hits))
check("AU keys confined to attribution still pass R2",
      ConsistencyGate._r2_boundary(
          {"proposals": [{"attribution": {"AU4": 0.1}, "notes": "expressive component"}]},
          True, True) == [])
check("a banned emotion word is located too",
      any("summary" in h for h in scan_forbidden_vocabulary(
          {"summary": "a disgust reaction"}, ban_au=False, ban_emotion=True)))

# -- Bug 7: the role file banned AU naming outright while the schema it printed two lines
# above demanded an AU-keyed attribution map. The model resolved the contradiction by
# keeping AU names everywhere, in notes as well as in the map.
role = load_role("p_agent_scan")
role_text = f"{role.body}\n{role.body_zh}"
check("the scan role states that attribution is the one legal place for AU names",
      "attribution" in role_text and "exempt" in role_text.lower())
check("the scan role names the prose fields the ban actually covers",
      all(field in role_text for field in ("notes", "curve_summary")))

# -- Bug 8: a literal newline inside a prose field made json.loads reject a structurally
# perfect multi-kilobyte object. A.graph degraded on 4-5 kB replies that opened with
# '{' and closed with '}'.
control_char = '{"summary": "line one\nline two", "open_questions": ["why?"]}'
check("a control character inside a string no longer discards the object",
      extract_json(control_char) is not None)
check("the repaired object keeps its content",
      (extract_json(control_char) or {}).get("summary") == "line one\nline two")
check("genuinely malformed JSON is still rejected",
      extract_json('{"a": {"b": 1, "c"') is None)
check("a JSON array is still not accepted as a product", extract_json("[1,2,3]") is None)
check("a fenced object still parses", extract_json('```json\n{"a": 1}\n```') is not None)

# -- Bug 9: "unparsable response" discarded the text, so a real failure could not be told
# apart from an empty reply, a truncation, or prose.
check("an empty reply is reported as such", json_reject_reason("") == "empty reply")
check("a truncated object reports the decoder's own offset",
      "char" in json_reject_reason('{"a": {"b": 1, "c"'))
check("a non-object reports what it decoded to",
      "list" in json_reject_reason("[1,2,3]"))

# -- Bug 10: the answer header reported the micro-only subset as the total annotated
# count, so "N annotated (M micro)" always printed N == M and understated ground truth.
source = inspect.getsource(MEWMPipeline.run)
check("the pipeline no longer seeds the annotated set from micro_events()",
      "annotated_events or video.micro_events()" not in source)
check("the P4 upper bound still substitutes micro intervals only",
      "enumerate(micro_events)" in source)

# -- Bug 11: the visibility matrix grants R.reason and R.adjudicate a measurement digest,
# but the orchestrator passed measurements only to P.verify and A.encode. A digest of
# None is None, and a *granted* field is never listed as withheld -- so the adjudicator
# was handed a null it had been told it could rely on, and returned confidence 0.0 for
# every proposal with the rationale "no supporting measurements were supplied".
from mewm.orchestration.orchestrator import Orchestrator
from mewm.orchestration.state import (
    DIGEST, PHASE_R_ADJUDICATE, PHASE_R_REASON, VISIBILITY_MATRIX, MEWMState, project,
)

subgraph = inspect.getsource(Orchestrator._proposal_subgraph)
for phase_const in ("PHASE_R_REASON", "PHASE_R_ADJUDICATE"):
    head = subgraph.split(phase_const, 1)[-1][:260]
    check(f"the orchestrator hands measurements to {phase_const}",
          "measurements=context.measurements" in head, head[:120])

check("both R phases are still granted a measurement digest",
      all(VISIBILITY_MATRIX[p]["v1_measurements"] == DIGEST
          for p in (PHASE_R_REASON, PHASE_R_ADJUDICATE)))

# A grant with nothing behind it is reported, not silently nulled.
starved = project(MEWMState(), PHASE_R_ADJUDICATE, cid="p01", measurements=None)
check("an unsupplied grant is reported as withheld",
      "v1_measurements" in starved.withheld, str(starved.withheld))
check("an unsupplied grant does not appear as a null field",
      "v1_measurements" not in starved.fields, str(sorted(starved.fields)))
check("a genuinely hidden field is still withheld",
      "other_proposals" in starved.withheld, str(starved.withheld))

# ---------------------------------------------------------------------------
print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
if FAILURES:
    print("\nfailures:")
    for name, detail in FAILURES:
        print(f"  {name}: {detail}")
raise SystemExit(1 if FAILURES else 0)
