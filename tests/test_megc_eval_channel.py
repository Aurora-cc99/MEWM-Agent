"""Oracle / degenerate stub test for the MEGC evaluation channel.

The channel's metrics are only trustworthy if a perfect answerer scores 1 and a useless
one scores 0. This drives the whole pipeline on the real casme_sq reference corpus with a
stub caller instead of a hosted model, so it costs nothing and is deterministic.

Run directly: ``python tests/test_megc_eval_channel.py``
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mewm.data.datasets import load_dataset
from mewm.eval import megc_metrics as mm
from mewm.eval.megc_questions import (
    GROUP_AU_SET, GROUP_COUNT_EXPRESSION, GROUP_COUNT_MACRO, GROUP_COUNT_MICRO,
    GROUP_EVENT_RECOGNITION, GROUP_SPOT_INTERVAL, GROUP_TYPE_BINARY, GROUP_VIDEO_STRS,
    normalise_au_set, parse_count, parse_expression_type, parse_intervals,
    route_question,
)
from mewm.training import qa_eval
from mewm.training.rl_prompts import VideoEvidence

REFERENCE = (Path(__file__).resolve().parents[2]
             / "Q-T-A" / "casme_sq" / "casme_sq_me_lvqa_gpt-5-6-sol.jsonl")

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


@dataclass
class StubResponse:
    text: str
    latency_s: float = 0.01


def synthetic_evidence(videos):
    """Evidence whose proposals are exactly the annotated events, split by type.

    The split matters: the engine's micro proposal set is scored against micro ground
    truth and its macro interval set against macro ground truth, so a type-correct oracle
    is what makes the perfect score assertable. Handing all events to the micro channel
    would (correctly) score badly, which is the bug this shape guards.
    """
    rng = np.random.default_rng(0)
    out = {}
    for video in videos:
        micro = {tuple(e.interval) for e in video.micro_events()}
        proposals = [
            (int(e.interval[0]), int(e.interval[1]),
             int((e.interval[0] + e.interval[1]) // 2), 0.8)
            for e in video.events if tuple(e.interval) in micro
        ]
        macro_intervals = [
            (int(e.interval[0]), int(e.interval[1]))
            for e in video.events if tuple(e.interval) not in micro
        ]
        out[video.video_key] = VideoEvidence(
            video=video.video_key, n_frames=500, fps=video.fps, frame_offset=0,
            slot_order=["AU1", "AU2", "AU4", "AU6", "AU12"],
            activations=rng.random((500, 5)).astype(np.float32),
            proposals=proposals, macro_intervals=macro_intervals,
            error_shares={"scene": 0.3}, scene_r2=0.5)
    return out


def oracle_caller(items):
    """A caller that answers every question exactly right, from the reference itself."""
    by_prompt = {qa_eval.build_user_prompt(i): i for i in items}

    def call(system, user, **kwargs):
        item = by_prompt[user]
        group = item.group
        reference = item.reference_answer

        if group in (GROUP_COUNT_EXPRESSION, GROUP_COUNT_MICRO, GROUP_COUNT_MACRO):
            payload = {"count": parse_count(reference), "answer": reference}
        elif group == GROUP_SPOT_INTERVAL:
            spans = parse_intervals(reference)
            payload = {"count": len(spans),
                       "events": [{"interval": list(s["interval"]), "type": s["type"]}
                                  for s in spans],
                       "answer": reference}
        elif group == GROUP_AU_SET:
            codes, _ = normalise_au_set(reference)
            payload = {"action_units": codes, "answer": reference}
        elif group == GROUP_TYPE_BINARY:
            payload = {"expression_type": parse_expression_type(reference),
                       "answer": reference}
        elif group == GROUP_EVENT_RECOGNITION:
            payload = {"fine_label": item.truth.get("fine_label", "other"),
                       "coarse_label": item.truth.get("coarse_label", "other"),
                       "action_units": [], "interval": item.truth.get("interval", [0, 0]),
                       "answer": reference}
        else:  # GROUP_VIDEO_STRS
            video = VIDEOS_BY_KEY[item.video]
            events = list(video.micro_events())
            payload = {"n_micro": len(events),
                       "events": [{"interval": list(e.interval),
                                   "fine_label": e.fine_label,
                                   "coarse_label": e.coarse_label} for e in events],
                       "answer": reference}
        return StubResponse(json.dumps(payload))

    return call


def degenerate_caller(items):
    """A caller that claims nothing and names the wrong emotion everywhere."""
    by_prompt = {qa_eval.build_user_prompt(i): i for i in items}

    def call(system, user, **kwargs):
        item = by_prompt[user]
        group = item.group
        if group in (GROUP_COUNT_EXPRESSION, GROUP_COUNT_MICRO, GROUP_COUNT_MACRO):
            payload = {"count": 99, "answer": "ninety-nine"}
        elif group == GROUP_SPOT_INTERVAL:
            payload = {"count": 0, "events": [], "answer": "nothing here"}
        elif group == GROUP_AU_SET:
            payload = {"action_units": ["AU99"], "answer": "no units"}
        elif group == GROUP_TYPE_BINARY:
            truth = parse_expression_type(item.reference_answer)
            flipped = ("macro-expression" if truth == "micro-expression"
                       else "micro-expression")
            payload = {"expression_type": flipped, "answer": flipped}
        elif group == GROUP_EVENT_RECOGNITION:
            payload = {"fine_label": "contempt", "coarse_label": "other",
                       "action_units": [], "interval": [0, 0],
                       "answer": "zzz qqq wwwww"}
        else:
            payload = {"n_micro": 0, "events": [], "answer": "zzz qqq wwwww"}
        return StubResponse(json.dumps(payload))

    return call


print("MEGC evaluation channel: oracle and degenerate stubs on real casme_sq reference")

# --------------------------------------------------------------- unit checks
print("\n[router]")
check("localisation beats the counting stem it shares",
      route_question("How many micro-expression events appear in this video? "
                     "Localize every micro-expression event in the video.")[0]
      == GROUP_SPOT_INTERVAL)
check("a bare count is a count",
      route_question("How many micro-expression events appear in this video?")[0]
      == GROUP_COUNT_MICRO)
check("the type question routes to the binary group",
      route_question("What is the expression type of the 2-th expression event in this "
                     "video?")[0] == GROUP_TYPE_BINARY)
check("an event-anchored question routes to recognition",
      route_question("In the 1-th expression event of this video (frames 10-40, apex "
                     "25): what does the movement indicate?")[0]
      == GROUP_EVENT_RECOGNITION)
check("an unknown template is unrouted, not silently absorbed",
      route_question("What colour is the wall behind the subject?")[0] == "unrouted")

print("\n[AU normalisation]")
codes, unmapped = normalise_au_set("brow lowerer, upper lip raiser, dimpler")
check("official FACS names map to codes", codes == ["AU4", "AU10", "AU14"], str(codes))
check("nothing unmapped there", unmapped == [], str(unmapped))
codes, unmapped = normalise_au_set(["AU4", "au 12", "AU-14", "brow draw-down"])
check("bare codes and codebase paraphrases both map",
      codes == ["AU4", "AU12", "AU14"], str(codes))
codes, unmapped = normalise_au_set("AU4 (brow lowerer), left eyebrow twitch")
check("a parenthesised gloss resolves to its code", "AU4" in codes, str(codes))
check("an unknown name is reported, not dropped",
      unmapped == ["left eyebrow twitch"], str(unmapped))

print("\n[answer parsing]")
check("a zero-event localisation answer parses as 0 spans",
      parse_intervals("0 micro-expression events") == [], "")
spans = parse_intervals(
    "2 expression events. 1-th macro-expression: frames 557-608 (apex 572), "
    "18.53s-20.23s; 2-th micro-expression: frames 2854-2871 (apex 2862)")
check("prose spans parse with their types",
      spans == [{"interval": (557, 608), "type": "macro-expression"},
                {"interval": (2854, 2871), "type": "micro-expression"}], str(spans))
check("a count of zero is not confused with no answer", parse_count("0") == 0)
check("an unanswerable string yields None, not 0", parse_count("no idea") is None)
check("micro is read before macro in a hedged answer",
      parse_expression_type("not a macro-expression but a micro-expression")
      == "micro-expression")

print("\n[perception cache]")
import tempfile

from mewm.config import load_config as _load_config
from mewm.training.perception_cache import PerceptionCache, fingerprint

_config = _load_config()
check("the fingerprint moves with the frame sampling",
      fingerprint(_config, 0, 1) != fingerprint(_config, 0, 2))
_root = Path(tempfile.mkdtemp())
_cache = PerceptionCache(_root, _config, 0, 1)
_ev = VideoEvidence(video="s01/clip", n_frames=7, fps=30.0, frame_offset=3,
                    slot_order=["AU4"], activations=np.arange(7, dtype=np.float32)[:, None],
                    proposals=[(1, 2, 2, 0.5)], macro_intervals=[(3, 6)],
                    error_shares={"scene": 0.25}, scene_r2=0.75)
check("a cold key misses", _cache.get("s01/clip") is None)
_cache.put(_ev)
_got = _cache.get("s01/clip")
check("the round trip preserves every field",
      (_got.n_frames, _got.frame_offset, _got.proposals, _got.macro_intervals,
       _got.scene_r2) == (7, 3, [(1, 2, 2, 0.5)], [(3, 6)], 0.75), str(_got))
check("activations survive", np.allclose(_got.activations, _ev.activations))
_cache.put(VideoEvidence(video="broken", status="unavailable", note="no usable frames"))
_broken = _cache.get("broken")
check("an unavailable entry caches its reason, not a silent empty",
      _broken.status == "unavailable" and _broken.note == "no usable frames"
      and not _broken.available, str(_broken))
check("a different fingerprint cannot see these entries",
      PerceptionCache(_root, _config, 0, 2).get("s01/clip") is None)

# ------------------------------------------------------------ integration
if not REFERENCE.exists():
    print(f"  SKIP reference corpus not found at {REFERENCE}")
    raise SystemExit(0)

index = load_dataset("casme_sq")
videos = list(index.videos)
VIDEOS_BY_KEY = {v.video_key: v for v in videos}
rows = [json.loads(line) for line in REFERENCE.open(encoding="utf-8") if line.strip()]
evidence = synthetic_evidence(videos)

print("\n[routing]")
items, routing = qa_eval.build_eval_items("casme_sq", videos, rows, evidence)
check("every reference row is routed", routing["n_evaluated"] == len(rows),
      f"{routing['n_evaluated']} != {len(rows)}")
check("nothing skipped", routing["n_skipped"] == 0, str(routing["n_skipped"]))
check("all eight groups populated", len(routing["by_group"]) == 8,
      str(routing["by_group"]))

# ---------------------------------------------------------------- oracle
print("\n[oracle: a perfect answerer]")
oracle_items, _ = qa_eval.build_eval_items("casme_sq", videos, rows, evidence)
stats = qa_eval.sample_answers(oracle_items, caller=oracle_caller(oracle_items),
                               max_workers=4)
check("no failed calls", stats.n_failed_calls == 0, str(stats.to_dict()))
check("nothing unparsable", stats.n_unparsable == 0, str(stats.to_dict()))

scored = qa_eval.score_items(oracle_items, videos, evidence)
loc, rec, whole = scored["localisation"], scored["recognition"], scored["whole_video"]

counting = loc["counting"]
for quantity in ("expression", "micro", "macro"):
    entry = counting[quantity]
    check(f"count MAE is 0 for {quantity}", entry.get("mae") == 0.0, str(entry))
    check(f"count RMSE is 0 for {quantity}", entry.get("rmse") == 0.0, str(entry))

engine = loc["interval"]["engine_proposals"]
for expression_type in ("micro_expression", "macro_expression", "pooled_both_types"):
    block = engine[expression_type]
    check(f"engine spotting F1 is 1.0 ({expression_type})",
          abs(block["headline_f1"] - 1.0) < 1e-9, str(block.get("strict_iou")))
check("the micro channel is the declared headline",
      engine["headline"] == "micro_expression", engine["headline"])
check("the micro channel is scored against micro truth only",
      engine["micro_expression"]["n_truth"] < engine["pooled_both_types"]["n_truth"],
      f"{engine['micro_expression']['n_truth']} vs "
      f"{engine['pooled_both_types']['n_truth']}")
for scope, entry in loc["interval"]["policy_localisation"].items():
    check(f"policy localisation F1 is 1.0 ({scope})",
          abs(entry["headline_f1"] - 1.0) < 1e-9, str(entry["strict_iou"]))

types = loc["expression_type_unweighted"]
check("SpotUF1 is 1.0", abs(types["spot_uf1"] - 1.0) < 1e-9, str(types))
check("SpotUAR is 1.0", abs(types["spot_uar"] - 1.0) < 1e-9, str(types))

au = rec["action_units"]
check("F1_AU is 1.0", abs(au["f1_au"] - 1.0) < 1e-9, str(au))
check("Jaccard_AU is 1.0", abs(au["jaccard_au"] - 1.0) < 1e-9, str(au))
check("no unmapped AU name in the reference",
      not au["unmapped_names"]["in_reference"], str(au["unmapped_names"]))

emotion = rec["event"]["emotion"]
repo_fine = emotion["fine"]["repo"]
check("RegUF1 is 1.0 under the repo vocabulary",
      abs(repo_fine.get("reg_uf1", 0) - 1.0) < 1e-9, str(repo_fine))
check("RegUAR is 1.0 under the repo vocabulary",
      abs(repo_fine.get("reg_uar", 0) - 1.0) < 1e-9, str(repo_fine))
# The divisor fix: casme_sq's event questions exhibit only 6 of the repo's 9 fine
# classes, and averaging over all 9 would cap a perfect answerer at 6/9.
check("the macro divisor counts only classes with support",
      repo_fine["macro_divisor"] == len(repo_fine["classes_with_support"])
      < len(repo_fine["classes"]), str(repo_fine.get("classes_without_support")))
check("the all-declared-classes average is reported alongside and is lower",
      repo_fine["reg_uf1_all_declared_classes"] < repo_fine["reg_uf1"],
      str(repo_fine["reg_uf1_all_declared_classes"]))

event_text = rec["event"]["text"]
check("event BLEU is 1.0", abs(event_text["bleu"] - 1.0) < 1e-9, str(event_text))
check("event ROUGE-1 is 1.0", abs(event_text["rouge_1"] - 1.0) < 1e-9, str(event_text))

check("whole-video BLEU is 1.0", abs(whole["text"]["bleu"] - 1.0) < 1e-9,
      str(whole["text"]))
check("whole-video ROUGE-1 is 1.0", abs(whole["text"]["rouge_1"] - 1.0) < 1e-9,
      str(whole["text"]))
strs_block = whole["strs"]
check("STRS F1_s is 1.0", abs(strs_block["f1_spotting"] - 1.0) < 1e-9, str(strs_block))
check("STRS F1_a is 1.0", abs(strs_block["f1_analysis"] - 1.0) < 1e-9, str(strs_block))
check("STRS is 1.0", abs(strs_block["score"] - 1.0) < 1e-9, str(strs_block))

# ------------------------------------------------------------ degenerate
print("\n[degenerate: claims nothing, labels wrongly]")
bad_items, _ = qa_eval.build_eval_items("casme_sq", videos, rows, evidence)
qa_eval.sample_answers(bad_items, caller=degenerate_caller(bad_items), max_workers=4)
bad = qa_eval.score_items(bad_items, videos, evidence)

bad_counting = bad["localisation"]["counting"]
check("count MAE is large when every count is 99",
      bad_counting["expression"]["mae"] > 90, str(bad_counting["expression"]))
for scope, entry in bad["localisation"]["interval"]["policy_localisation"].items():
    check(f"policy localisation F1 is 0 when nothing is claimed ({scope})",
          entry["headline_f1"] == 0.0, str(entry["strict_iou"]))
check("engine spotting is unaffected by the policy's silence",
      abs(bad["localisation"]["interval"]["engine_proposals"]["micro_expression"]
          ["headline_f1"] - 1.0) < 1e-9)
bad_types = bad["localisation"]["expression_type_unweighted"]
check("SpotUF1 is 0 when every type is flipped", bad_types["spot_uf1"] == 0.0,
      str(bad_types))
check("SpotUAR is 0 when every type is flipped", bad_types["spot_uar"] == 0.0,
      str(bad_types))
bad_au = bad["recognition"]["action_units"]
check("F1_AU is 0 for an invented AU", bad_au["f1_au"] == 0.0, str(bad_au))
check("whole-video STRS is 0", bad["whole_video"]["strs"]["score"] == 0.0,
      str(bad["whole_video"]["strs"]))
check("BLEU is 0 on disjoint text", bad["whole_video"]["text"]["bleu"] == 0.0,
      str(bad["whole_video"]["text"]))

# ------------------------------------------------- unparsable predictions
print("\n[an unparsable reply is scored, not dropped]")
silent_items, _ = qa_eval.build_eval_items("casme_sq", videos, rows, evidence)


def broken_caller(system, user, **kwargs):
    return StubResponse("I would rather not answer in JSON.")


silent_stats = qa_eval.sample_answers(silent_items, caller=broken_caller, max_workers=4)
check("every reply counted unparsable", silent_stats.n_unparsable == len(silent_items),
      f"{silent_stats.n_unparsable} != {len(silent_items)}")
silent = qa_eval.score_items(silent_items, videos, evidence)
n_type = silent["localisation"]["expression_type_unweighted"]["n_questions"]
check("the type group keeps its full denominator", n_type == 338, str(n_type))
check("unparsed predictions are reported",
      silent["localisation"]["expression_type_unweighted"]["unparsed_predictions"] == 338)
check("AU group keeps its denominator",
      silent["recognition"]["action_units"]["n_questions"] == 92)
check("counting reports unparsed rather than scoring 0 error",
      silent["localisation"]["counting"]["unparsed_predictions"]["expression"] == 92,
      str(silent["localisation"]["counting"]["unparsed_predictions"]))

# ------------------------------------------------------- report structure
print("\n[report]")
report = qa_eval.build_report("casme_sq", "stub-oracle", oracle_items, videos, evidence,
                             routing, {"n_videos": len(videos)}, stats.to_dict())
check("report is JSON-serialisable",
      isinstance(json.dumps(report, default=str), str))
check("per-subject section covers every subject",
      len(report["per_subject"]) == len(index.subjects()),
      f"{len(report['per_subject'])} vs {len(index.subjects())}")
check("per-subject entries carry the full metric table",
      all("localisation" in s["metrics"] and "recognition" in s["metrics"]
          and "whole_video" in s["metrics"] for s in report["per_subject"].values()))

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    raise SystemExit(1)
print("all checks passed")
