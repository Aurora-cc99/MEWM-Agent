"""Augmented QA pairs mined from the policy's own accepted outputs.

**Format fidelity.** The reference sets carry exactly four fields --
``video_id``, ``video``, ``question``, ``answer`` -- and ``answer`` is an ``int`` for the
counting questions and a ``str`` elsewhere. Adding a fifth field for provenance would make
the file a different format that merely looks compatible, so provenance goes into the
``video_id`` (which already encodes a split in the reference files:
``casme_sq_train_s15_15_0401girlcrashing_1``) and into a **sidecar manifest** that lives
next to the jsonl and is never loaded as training data.

**Fold isolation.** Every fold writes into its own directory, keyed by the held-out
subject. A shared output directory would let fold *A*'s augmented pairs -- derived from
videos that are in fold *B*'s **test** set -- be read while training fold *B*. That is a
leak with no symptom: training succeeds, and the reported number for fold *B* is simply
wrong. :func:`load_augmented` therefore takes the fold's permitted video keys and refuses
anything outside them rather than filtering quietly.

**Correct-answer bias.** Only accepted candidates are written. An augmented set that
contains the policy's mistakes would train it towards its own error distribution, which is
the opposite of the intent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ..data.paths import qa_dir

LOGGER = logging.getLogger(__name__)

#: Directory name under a dataset's QA root that holds machine-generated pairs.
AUGMENTED_SUBDIR = "augmented"

#: The four fields of the reference format, in order. Nothing else may be written.
QA_FIELDS = ("video_id", "video", "question", "answer")


class AugmentationError(RuntimeError):
    """Raised when an augmented set would violate format or fold isolation."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class AugmentedPair:
    """One generated QA pair plus the provenance that stays out of the jsonl."""

    video_id: str
    video: str
    question: str
    answer: Any

    # -- provenance: written to the manifest only ---------------------------
    dataset: str = ""
    fold: str = ""
    subject: str = ""
    event_index: int = 0
    interval: Sequence[int] = field(default_factory=tuple)
    reward: float = 0.0
    policy_model: str = ""
    accepted_because: List[str] = field(default_factory=list)
    #: The reference row's ``video_id`` the pair was sampled from -- the link back to
    #: the original QA pair, carried for the consolidated corpus-level file.
    source_video_id: str = ""

    def to_jsonl(self) -> Dict[str, Any]:
        """Exactly the four reference fields, in the reference order."""
        return {"video_id": self.video_id, "video": self.video,
                "question": self.question, "answer": self.answer}

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id, "video": self.video, "dataset": self.dataset,
            "fold": self.fold, "subject": self.subject,
            "event_index": self.event_index, "interval": list(self.interval),
            "reward": round(float(self.reward), 5), "policy_model": self.policy_model,
            "accepted_because": list(self.accepted_because),
            "source_video_id": self.source_video_id,
            "question": self.question,
        }


def validate_schema(payload: Dict[str, Any]) -> None:
    """Reject anything that is not the reference format."""
    keys = tuple(payload.keys())
    if keys != QA_FIELDS:
        raise AugmentationError(
            f"augmented QA record must carry exactly {QA_FIELDS} in order; got {keys}")
    if not isinstance(payload["video_id"], str) or not payload["video_id"]:
        raise AugmentationError("video_id must be a non-empty string")
    if not isinstance(payload["video"], str) or not payload["video"]:
        raise AugmentationError("video must be a non-empty string")
    if not isinstance(payload["question"], str) or not payload["question"]:
        raise AugmentationError("question must be a non-empty string")
    if not isinstance(payload["answer"], (str, int, float)):
        raise AugmentationError(
            f"answer must be a string or a number, got {type(payload['answer']).__name__}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def augmented_dir(dataset: str, fold: str) -> Path:
    """``Q-T-A/<dataset>/augmented/fold_<held-out subject>/``."""
    return qa_dir(dataset) / AUGMENTED_SUBDIR / f"fold_{fold}"


def augmented_jsonl(dataset: str, fold: str) -> Path:
    return augmented_dir(dataset, fold) / f"{dataset}_me_lvqa_aug.jsonl"


def augmented_manifest(dataset: str, fold: str) -> Path:
    return augmented_dir(dataset, fold) / f"{dataset}_me_lvqa_aug_manifest.json"


def consolidated_path(dataset: str, policy_model: str) -> Path:
    """The corpus-level JSON beside the per-fold directories.

    ``Q-T-A/<dataset>/augmented/<dataset>_augmented_full_<model>.json`` -- every
    accepted pair exactly once, beside the reference QA pair it was sampled from.
    """
    safe = str(policy_model).replace(".", "-")
    return qa_dir(dataset) / AUGMENTED_SUBDIR / f"{dataset}_augmented_full_{safe}.json"


def sampling_checkpoint_path(dataset: str, policy_model: str) -> Path:
    """The scratch jsonl a long sweep writes as it goes, so ``--resume`` can pick it up.

    Underscore-prefixed on purpose: it is not a QA artefact and is deleted once the
    sweep has written its folds and the consolidated file.
    """
    safe = str(policy_model).replace(".", "-")
    return (qa_dir(dataset) / AUGMENTED_SUBDIR
            / f"_{dataset}_sampling_checkpoint_{safe}.jsonl")


def make_video_id(dataset: str, fold: str, video: str, index: int) -> str:
    """Mirror the reference id shape with ``aug<fold>`` in the split position."""
    return f"{dataset}_aug{fold}_{video}_{index}"


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_pairs(
    accepted: Sequence[Any],
    dataset: str,
    fold: str,
    question_by_prompt: Dict[str, str],
    subject_by_video: Optional[Dict[str, str]] = None,
    policy_model: str = "",
    tp_iou_threshold: float = 0.5,
    require_strict_tp: bool = True,
) -> List[AugmentedPair]:
    """Turn accepted candidates into augmented pairs.

    **Only true positives are augmented.** With ``require_strict_tp`` the candidate must
    have localised its event at ``IoU >= tp_iou_threshold`` -- the MEGC2025 §2.3 spotting
    criterion -- and must not have been *rescued*, which is the label-only pass granted to
    a proposal that missed the interval. The direction matters: augmented data is a
    training target, so a pair built on a mislocalised interval teaches the policy to
    describe a window that does not contain the event. Reward alone does not protect
    against this, because the description reward can be high for a fluent answer about the
    wrong frames. Filtering here rather than at write time keeps the manifest's
    ``accepted_because`` honest about what admitted each pair.
    """
    subject_by_video = subject_by_video or {}
    pairs: List[AugmentedPair] = []
    per_video_counter: Dict[str, int] = {}
    dropped: Dict[str, int] = {}

    for candidate in accepted:
        video = getattr(candidate, "video", "") or ""
        if not video:
            LOGGER.warning("skipping accepted candidate with no video key (prompt %s)",
                           getattr(candidate, "prompt_id", "?"))
            continue

        outcome = getattr(candidate, "outcome", None)
        if require_strict_tp:
            # No outcome means the localisation was never scored, which is not evidence
            # of a true positive -- treat it as a drop rather than assume it passed.
            if outcome is None:
                dropped["no_outcome"] = dropped.get("no_outcome", 0) + 1
                continue
            if getattr(outcome, "rescued", False):
                dropped["rescued"] = dropped.get("rescued", 0) + 1
                continue
            iou = float(getattr(outcome, "iou", 0.0) or 0.0)
            if iou < tp_iou_threshold:
                dropped["iou_below_threshold"] = dropped.get("iou_below_threshold", 0) + 1
                continue

        question = question_by_prompt.get(getattr(candidate, "prompt_id", ""), "")
        if not question:
            LOGGER.warning("skipping %s: no question text for prompt %s",
                           video, getattr(candidate, "prompt_id", "?"))
            continue

        product = getattr(candidate, "product", {}) or {}
        answer = product.get("answer") or product.get("global_narrative") or ""
        if not isinstance(answer, str) or not answer.strip():
            LOGGER.warning("skipping %s: accepted candidate carries no answer text", video)
            continue

        index = per_video_counter.get(video, 0) + 1
        per_video_counter[video] = index

        pairs.append(AugmentedPair(
            video_id=make_video_id(dataset, fold, video, index),
            video=video,
            question=question,
            answer=answer.strip(),
            dataset=dataset, fold=fold,
            subject=subject_by_video.get(video, ""),
            event_index=int(getattr(candidate, "event_index", 0)),
            interval=tuple(getattr(candidate, "interval", ()) or ()),
            reward=float(getattr(candidate, "reward", 0.0)),
            policy_model=policy_model,
            accepted_because=[
                f"passed the shared criterion (IoU {outcome.iou:.3f}"
                f"{', rescued' if outcome.rescued else ''})" if outcome else "passed",
                f"reward {float(getattr(candidate, 'reward', 0.0)):.3f}",
            ] + ([f"true positive at IoU >= {tp_iou_threshold:.2f}"]
                 if require_strict_tp else []),
            source_video_id=str(getattr(candidate, "prompt_id", "")),
        ))

    if dropped:
        LOGGER.info(
            "fold %s: TP gate dropped %d accepted candidate(s) before augmentation (%s)",
            fold, sum(dropped.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())))
    return pairs


def write_augmented(
    pairs: Sequence[AugmentedPair], dataset: str, fold: str,
    allowed_videos: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Write the jsonl and its manifest; refuse on a fold violation.

    ``allowed_videos`` is the fold's training-pool video keys. A pair outside that set
    would mean the policy was sampled on material it is about to be evaluated on, so this
    raises rather than dropping the offending row -- a silent drop would leave a smaller
    file and no indication that the sampling stage was misconfigured.
    """
    if allowed_videos is not None:
        stray = sorted({p.video for p in pairs if p.video not in allowed_videos})
        if stray:
            raise AugmentationError(
                f"fold {fold}: {len(stray)} augmented pair(s) come from videos outside "
                f"the training pool: {stray[:5]}. Refusing to write -- the sampling "
                f"stage was given material that is not in this fold's training set."
            )

    target_dir = augmented_dir(dataset, fold)
    target_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = augmented_jsonl(dataset, fold)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            payload = pair.to_jsonl()
            validate_schema(payload)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": dataset,
        "fold": fold,
        "held_out_subject": fold,
        "n_pairs": len(pairs),
        "n_videos": len({p.video for p in pairs}),
        "policy_model": pairs[0].policy_model if pairs else "",
        "format": list(QA_FIELDS),
        "note": (
            "Machine-generated pairs, accepted by the shared pass criterion and the "
            "reward floor. Provenance lives here rather than in the jsonl because the "
            "jsonl must stay byte-compatible with the reference format. These pairs are "
            "valid supervision only for the fold named above: every video listed is in "
            "that fold's training pool and none is from its held-out subject."
        ),
        "pairs": [pair.to_manifest() for pair in pairs],
    }
    manifest_path = augmented_manifest(dataset, fold)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    LOGGER.info("fold %s: wrote %d augmented QA pair(s) to %s",
                fold, len(pairs), jsonl_path)
    return {"jsonl": str(jsonl_path), "manifest": str(manifest_path),
            "n_pairs": len(pairs), "n_videos": manifest["n_videos"]}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_augmented(
    dataset: str, fold: str, allowed_videos: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Read one fold's augmented pairs, enforcing fold isolation on the way in.

    Validation is on *video keys* rather than parsed subject ids on purpose: the key is
    exact, while deriving a subject from a video string differs per corpus (SAMM's long
    videos are flat) and a parsing mistake would turn the leak check into a no-op.
    """
    path = augmented_jsonl(dataset, fold)
    if not path.is_file():
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                validate_schema(payload)
            except AugmentationError as exc:
                raise AugmentationError(f"{path}:{line_number}: {exc}") from exc
            if allowed_videos is not None and payload["video"] not in allowed_videos:
                raise AugmentationError(
                    f"{path}:{line_number}: video {payload['video']!r} is not in fold "
                    f"{fold}'s training pool. This file was built for a different fold; "
                    f"loading it here would train on held-out material."
                )
            records.append(payload)

    LOGGER.info("fold %s: loaded %d augmented QA pair(s)", fold, len(records))
    return records


def load_consolidated(
    dataset: str, policy_model: str, allowed_videos: Optional[Set[str]] = None,
    fold: str = "",
) -> List[Dict[str, Any]]:
    """Read the corpus-level sweep and project it onto one fold's training pool.
    """
    path = consolidated_path(dataset, policy_model)
    if not path.is_file():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    disputed = 0
    for position, entry in enumerate(payload.get("pairs", []), start=1):
        row = {
            "video_id": entry.get("video_id", ""),
            "video": entry.get("video", ""),
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
        }
        try:
            validate_schema(row)
        except AugmentationError as exc:
            raise AugmentationError(f"{path} pair {position}: {exc}") from exc
        if allowed_videos is not None and row["video"] not in allowed_videos:
            continue
        if fold:
            claimed = entry.get("provenance", {}).get("folds") or []
            if claimed and fold not in claimed:
                disputed += 1
        records.append(row)

    if disputed:
        LOGGER.warning(
            "%s: %d pair(s) are in fold %s's pool by video key but do not list that "
            "fold in provenance.folds. Keeping them -- the video key is authoritative "
            "-- but the sweep's fold arithmetic and the runner's disagree, which means "
            "one of the two saw a different subject list.", path.name, disputed, fold)

    LOGGER.info("fold %s: %d consolidated augmented pair(s) admitted from %s",
                fold or "-", len(records), path.name)
    return records


def discover_folds(dataset: str) -> List[str]:
    """Folds that already have an augmented set on disk."""
    root = qa_dir(dataset) / AUGMENTED_SUBDIR
    if not root.is_dir():
        return []
    return sorted(p.name[len("fold_"):] for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("fold_")
                  and (p / f"{dataset}_me_lvqa_aug.jsonl").is_file())


__all__ = [
    "AUGMENTED_SUBDIR", "QA_FIELDS", "AugmentationError", "AugmentedPair",
    "validate_schema", "augmented_dir", "augmented_jsonl", "augmented_manifest",
    "consolidated_path", "sampling_checkpoint_path",
    "make_video_id", "build_pairs", "write_augmented", "load_augmented",
    "load_consolidated", "discover_folds",
]
