"""Paths, thresholds and runtime configuration for MEWM-Agent.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Filesystem anchors
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
#: Kept as separate names for drop-in compatibility with code copied from the
#: reference implementation (which had a real intermediate pipeline directory);
#: in this repository's flat layout all three point at the same directory.
PIPELINE_ROOT = PACKAGE_ROOT
PROJECT_ROOT = PACKAGE_ROOT


def _resolve_dataset_root() -> Path:
    """Where the four raw datasets (casme_sq/samm/casme3/4dme) actually live.

    1. ``MEWM_DATASET_ROOT`` environment variable (or ``MEWM_DATASET__ROOT`` via the
       normal ``_overlay_env`` mechanism -- both are honoured, see ``load_config``).
    2. ``<PROJECT_ROOT>/dataset`` if it exists (a local copy was placed inside this
       repository).
    3. ``E:/code/MEWM-Agent/dataset`` -- the exact path formwork.md 第I条 gives for
       every one of the four datasets' frame roots and annotation workbooks. This is
       an explicit, documented fallback (not a silent guess): if none of the three
       candidates exists on disk, the *third* candidate is still returned so that
       downstream error messages point at a concrete, correct path instead of an
       empty ``PROJECT_ROOT/dataset`` that was never going to exist.
    """
    env = os.environ.get("MEWM_DATASET_ROOT")
    if env:
        return Path(env)
    local = PROJECT_ROOT / "dataset"
    if local.is_dir():
        return local
    sibling = PROJECT_ROOT.parent / "MEWM-Agent" / "dataset"
    if sibling.is_dir():
        return sibling
    # formwork.md 第I条 literal path; kept even when absent so error messages are
    # actionable ("set MEWM_DATASET_ROOT" instead of a bare FileNotFoundError).
    return Path(r"E:/code/MEWM-Agent/dataset")


DATASET_ROOT = _resolve_dataset_root()
FLOW_ROOT = PIPELINE_ROOT / "pre_datasets"
FLOW_CROP_ROOT = PIPELINE_ROOT / "pre_datasets_crop"
#: Face-aligned-crop tree used as the flow front end's *input* for CASME3/4DME only
#: (formwork.md 第I条 第9/11款). See ``mewm.data.paths.face_crop_root``.
FACE_CROP_ROOT = PIPELINE_ROOT / "pre_datasets" / "face_crop"
QTA_ROOT = PIPELINE_ROOT / "Q-T-A"
PRE_PROCESS_DIR = PIPELINE_ROOT / "pre_process"

CONFIG_DIR = PACKAGE_ROOT / "configs"
SKILLS_DIR = PACKAGE_ROOT / "skills"
RUNS_ROOT = PACKAGE_ROOT / "runs"
CACHE_ROOT = PACKAGE_ROOT / "cache"
CHECKPOINT_ROOT = RUNS_ROOT / "checkpoints"

# The engine weights produced by stage 0 (appendix F.1).  Frozen and versioned: the
# hash of this directory is what ``RolloutRecord.model_version`` records.
ENGINE_CKPT_ROOT = PACKAGE_ROOT / "engine_ckpt"


# ---------------------------------------------------------------------------
# .env credential loading (2026-09-03)
# ---------------------------------------------------------------------------
#
# Nothing in ``mewm.llm.registry`` / ``mewm.llm.client`` ever called a dotenv loader --
# every provider key is read with a bare ``os.environ.get(...)`` (see
# ``ProviderSpec.base_url`` / ``.api_key``), so a ``.env`` file only ever took effect
# if the shell (or an IDE run configuration) sourced it *before* the process started.
# That is a real, separate step the operator has to remember on every fresh shell, and
# forgetting it fails "no credentials for: gemini-3-pro" late, from inside a CLI
# command, without saying where a ``.env`` was expected. This loader makes the .env
# file itself the single source of truth: it is read once, here, at import time --
# before any CLI command or module reads ``os.environ`` for a provider key -- and only
# fills in variables the environment does not already define, so a real shell-exported
# key always wins over the file (the file is a fallback, not an override).


def _find_env_file() -> Optional[Path]:
    """Search order, highest priority first.

    1. ``MEWM_ENV_FILE`` -- an explicit path, for a operator who keeps credentials
       somewhere else entirely.
    2. ``<PACKAGE_ROOT>/.env`` -- a local file placed directly in this repository.
    3. The reference implementation's ``.env`` at
       ``E:\\code\\MEWM-Agent\\MEWM-Agent_pinpline\\main_tree\\.env`` -- the credential
       file this repository's own README points operators at; kept as an explicit,
       documented fallback rather than requiring every operator to copy secrets into a
       second location.
    """
    override = os.environ.get("MEWM_ENV_FILE")
    if override:
        return Path(override)
    local = PACKAGE_ROOT / ".env"
    if local.is_file():
        return local
    reference = (PACKAGE_ROOT.parent / "MEWM-Agent" / "MEWM-Agent_pinpline"
                / "main_tree" / ".env")
    if reference.is_file():
        return reference
    return None


def _load_dotenv() -> Optional[Path]:
    """Parse ``KEY=VALUE`` lines from the resolved ``.env`` file into ``os.environ``.
    """
    path = _find_env_file()
    if path is None:
        return None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return path


#: The ``.env`` file this process actually loaded, or ``None`` if none was found --
#: read by ``mewm doctor`` to report exactly which credential source is in effect.
LOADED_ENV_FILE: Optional[Path] = _load_dotenv()


def ensure_pre_process_importable() -> None:
    """Put ``pre_process`` on ``sys.path``.

    The dataset loaders, the MEFlowNet runner and the 29-ROI box builder already exist
    there and are the reference implementation for this project's front end; importing
    them keeps a single source of truth for frame naming and ROI geometry.
    """

    path = str(PRE_PROCESS_DIR)
    if PRE_PROCESS_DIR.is_dir() and path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Threshold / hyper-parameter blocks
# ---------------------------------------------------------------------------


@dataclass
class MotionConfig:
    """V1 - deterministic motion quantisation front end (paper 3.2.1)."""

    n_roi: int = 29
    # Disjunctive salience test: magnitude OR coherence (never magnitude alone --
    # a lone magnitude threshold filters the sub-pixel micro-expression signal out).
    m_min: float = 0.15
    c_min: float = 0.60
    # Frame gap used to build the flow pair (t-k, t).  Overridden per dataset by the
    # value the flow front end actually used.
    default_flow_gap: int = 7


@dataclass
class RepresentationConfig:
    """V2/V3 - slot encoder and two-timescale latent state (paper 3.2.2/3.2.3, 4.3)."""

    n_slots: int = 16              # K = 16 candidate AUs
    slot_dim: int = 128            # d_a
    slow_dim: int = 256            # z^s
    fast_dim: int = 128            # z^m
    belief_dim: int = 64           # z^e
    appearance_dim: int = 64       # phi_{k,t}
    slot_dropout: float = 0.15     # keeps mask() inside the training distribution
    # Activation selection. The absolute floor alone is unsafe when the coherence
    # channel saturates (rendered flow) or magnitudes arrive frame-relative: a dozen AUs
    # clear it at once. Selection is therefore also competitive -- see
    # v2_slots.select_active_slots.
    activation_threshold: float = 0.35
    weak_threshold: float = 0.20
    max_active_aus: int = 3        # a micro-expression involves a handful of units
    activation_relative_margin: float = 0.75
    # Slow variable random-walk prior: z^s_{t+1} = z^s_t + eps,  eps ~ N(0, Q)
    #
    # Q must stay well below the observation noise. The steady-state Kalman gain is
    # (sigma + Q) / (sigma + Q + theta^2/|S|), so Q sets how much of a step the slow
    # term can absorb -- the ``h`` of proposition B.3. At Q = 1e-4 the gain converges to
    # about 0.42, which is emphatically not ``h << 1``: the slow variable would swallow
    # a large share of every micro-expression transition, and the expressive residual
    # that spike is supposed to produce would go with it. At 1e-5 the gain settles near
    # 0.16, keeping the slow variable's time constant far longer than an event.
    slow_process_noise: float = 1e-5
    slow_obs_noise: float = 1e-2
    changepoint_frames: int = 15   # L_cp
    changepoint_nll: float = 12.0


@dataclass
class DynamicsConfig:
    """M1 - emotion-conditioned AU object dynamics (paper 3.3.1, 4.3)."""

    n_gat_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 256
    n_mixture: int = 5             # mixture-density head components
    rollout_steps: int = 3         # S
    future_queries: int = 8        # N_s
    kappa_low: float = 0.7         # temporal split ratio kappa ~ U[0.7, 1.0)
    lambda_imagine: float = 1.0
    lambda_flow: float = 0.5
    lambda_slow: float = 0.1


@dataclass
class SpottingConfig:
    """M2 - error decomposition and proposal generation (paper 3.3.2, eq. 6/7)."""

    window: int = 300              # W, robust-normalisation window
    tau_hi: float = 3.5            # hysteresis trigger
    tau_lo: float = 1.5            # hysteresis release
    # --- Proposal extent: UNBOUNDED by a flat literature constant, by default -----
    # The boundaries come off the frame-level S curve (hysteresis) and off the
    # matched filter's own argmax over the duration bank; a preset length window
    # only overrides that decision with a prior, and on casme_sq a flat, dataset-
    # blind prior was wrong in both directions -- a 0.5 s ceiling routed 8 of 52
    # annotated micro events (15.4%) to the macro channel where micro ground truth
    # can never score them, and a 2-frame floor is arbitrary at any capture rate
    # other than the one it was picked at.
    #
    # min_duration_frames and min_micro_seconds are OPT-IN: <= 0 means "no bound",
    # which is the default. Set a positive value only for a corpus whose event-
    # length range is genuinely known, and re-run `diagnose-localisation` after
    # doing so.
    #
    # The micro/macro routing ceiling is a UNIFORM 200 frames for every dataset
    # (mewm.data.paths.MICRO_CEILING_FRAMES, user directive 2026-08-31: no
    # per-dataset empirical ceilings). The per-dataset "longest annotated micro"
    # values that used to sit in MICRO_CEILING_SECONDS misfired structurally: a
    # hysteresis span around a true event is wider than the annotation, so on
    # casme_sq 13 of 52 truths were covered by a span at IoU > 0.5 yet discarded
    # by the 17-frame ceiling (2026-08-31 full-dataset measurement, 0 TP run).
    # 200 frames is far beyond any genuine micro-expression, so it never
    # re-routes a real event; it only keeps long scene excursions in the macro
    # channel. A positive ``max_micro_seconds`` still overrides the uniform
    # ceiling with a physical duration of your own.
    min_duration_frames: int = 0   # <= 0: keep every hysteresis span
    min_micro_seconds: float = 0.0 # <= 0: bank starts at the shortest decodable extent
    max_micro_seconds: float = 0.0 # <= 0: uniform MICRO_CEILING_FRAMES (200) for all datasets
    # Physiological template dictionary (blink / swallow / speech), matching-pursuit.
    n_physio_templates: int = 12
    physio_match_thresh: float = 0.55
    merge_gap_frames: int = 3

    # --- Proposal refinement: raw-energy ranking + energy-contour trim ---------
    # The normalised S curve is dominated by sharp noise spikes: on a flat stretch
    # the trailing MAD collapses, so a tiny wobble divides by an ~eps scale and
    # saturates the 1000 clip while a genuine micro-expression bump -- sitting on a
    # busier stretch -- scores orders of magnitude lower (measured 2026-08-31 on the
    # ceiling-200 run: s30's truth span ranked 33/49 by peak_S yet 1/53 by RAW
    # expressive peak; s27's truth span ranked 22/37 by peak_S yet 5/39 raw). The
    # CLIP logit curve itself is discriminative (fold frame AUC), so the raw
    # expressive-peak energy is the honest rank key. The contour trim then cuts each
    # kept span to the frames whose energy stays above ``proposal_trim_fraction`` of
    # the span's own peak, tightening the wide hysteresis spans to the event's core:
    # on the four analysed videos truth IoU moved 0.53->0.64, 0.71->0.75, 0.50->0.62,
    # 0.52->0.63 (fraction 0.3). Together they cap the per-video proposal count at
    # ``proposal_max_per_video``, fixing the 47-54 false-positive fragmentation the
    # uniform 200-frame ceiling exposed.
    proposal_max_per_video: int = 10    # top-K hysteresis spans by raw peak; <=0 keeps all
    proposal_trim_fraction: float = 0.3 # energy-contour trim; <=0 disables the trim
    # 2026-09-02 layer-attribution follow-ups (offline ablation, 5 videos):
    #   raw-curve spans      -- hysteresis on the RAW delta_expr (no normalisation) at
    #                           ``frac x max(delta_expr)``, merged into the candidate
    #                           pool. Recovers truths whose peak the running-window
    #                           normalisation dilutes (attribution bucket B, 7/52) and
    #                           tightens IoU where the raw curve hugs the truth
    #                           (s24_0401 0.75->0.93, s30_0101 0.63->0.67).
    #   soft top-K margin    -- keep spans ranked below K whose raw peak stays within
    #                           ``margin x K-th peak``, so a truth just outside the
    #                           top-10 is not discarded outright (bucket C, 3/52).
    proposal_raw_curve_frac: float = 0.4 # <=0 disables the raw-curve pass
    # 2026-09-03: soft margin set to 0 -- the extfix experiment showed it lets
    # noise spans through the top-K (s23: 10 -> 46 confirmed proposals with no
    # recall gain). The raw-curve pass stays: it tightened s24's hit 0.75 -> 0.93.
    proposal_soft_k_margin: float = 0.0  # <=0 disables the soft margin

    # --- M2b, the double-burst extent decoder (mewm/engines/m2_localiser.py) ---
    # S is a prediction error, so it peaks on the onset ramp and again on the
    # offset relaxation and dips at the apex. Thresholding it returns the two
    # flanks as separate short spans; these settings govern the matched filter
    # that recovers the onset-to-offset extent instead.
    localiser_enabled: bool = True
    localiser_bank_size: int = 8         # candidate durations searched
    # 0.25, not 0.30. Swept on all 31 casme_sq videos carrying micro-expression
    # annotations, against the undecoded baseline (TP 3, FP 2196, F1 0.0027):
    #   flank 0.25 -> TP 3, FP 369, F1 0.0142   (keeps every baseline TP)
    #   flank 0.30 -> TP 1, FP 371, F1 0.0047   (higher F1 than baseline, but by
    #                                            losing 2 of the 3 true hits)
    # The narrower flank wins on F1 *and* on recall, so it is not a precision
    # trade. Verify with `diagnose-localisation` after touching this.
    localiser_flank_fraction: float = 0.25
    localiser_trough_weight: float = 1.0
    localiser_max_per_video: int = 12
    localiser_min_score: float = 0.0
    localiser_nms_iou: float = 0.20
    localiser_min_separation_seconds: float = 0.15

    # --- Attribution sharpening (opt-in; OFF by default to preserve pi_{k,j}) ---
    # ``_attribute`` reports pi_{k,j} as a linear (L1) share of expressive-error
    # mass per AU -- the paper-documented quantity, and what every downstream
    # consumer (structure/perception prompts, memory, training bans) treats as
    # pass-through metadata, not an operand. Setting this to a value in (0, 1]
    # applies a temperature-scaled softmax on top of that share vector, before
    # the 2% cutoff -- sharpen(pi)_k = softmax(pi_k / T) -- which amplifies
    # high-activation AUs and suppresses marginal ones for display or
    # downstream-prompt salience. T < 1 sharpens (smaller T = sharper); T == 1
    # is a near no-op up to renormalisation. Deliberately applied to the
    # already-normalised shares, not the raw pre-normalisation mass: those
    # values are small and video-scale-dependent, so a raw softmax over them
    # saturates toward uniform instead of sharpening. None or <= 0: unchanged
    # linear pi_{k,j} (default; matches every prior run and the paper).
    attribution_sharpen_temperature: Optional[float] = None


@dataclass
class CriticConfig:
    """C-Agent decision thresholds (paper 3.4.5, appendix C.4)."""

    eta_lambda: float = 2.0        # likelihood-ratio floor
    eta_cfs: float = 0.10          # CFS margin floor
    eta_suppression: float = 0.15  # neutralised-template distance margin
    kappa_micro: float = 0.04      # transient rise-slope discriminator (per frame)
    mni_hallucination: float = 0.02
    max_challenges_per_round: int = 3
    max_challenge_rounds: int = 3


@dataclass
class SchedulerConfig:
    """M4 - fast / standard / deep routing (appendix F.4)."""

    theta_fast: float = 0.20       # static evidence margin Delta
    eta_fast: float = 3.0          # likelihood ratio Lambda
    theta_deep: float = 0.55       # confidence floor
    theta_var: float = 0.35        # belief variance ceiling


@dataclass
class TokenRegulatorConfig:
    """V4 - evidence token regulation (paper 3.2.4, appendix E.3)."""

    enabled: bool = True
    n_r: int = 9                   # trigger-layer interval
    delta: float = 4.0             # intervention strength
    p_v: float = 0.03              # visual admission ratio
    p_t: float = 0.20              # textual admission ratio
    gamma_anchor_min: float = 1.0  # anchor clamp
    keep_instruction_tokens: bool = True


@dataclass
class MemoryConfig:
    """Three-layer memory (paper 3.5, appendix E)."""

    stationary_run_length: int = 30   # L, frames folded into one segment summary
    n_query_tokens_multi_au: int = 8  # N_q
    n_query_tokens_single_au: int = 4
    piecewise_linear_tol: float = 0.05  # eps_pl for the S_t index
    retrieval_jaccard: float = 0.4      # theta_j
    retrieval_max_per_level: int = 2
    retrieval_max_tokens: int = 800
    retrieval_enabled: bool = True


@dataclass
class OrchestratorConfig:
    """Deterministic state machine budgets (paper 4.3, appendix D.3)."""

    max_gate_retries: int = 2
    max_challenge_rounds: int = 3
    max_cascade_rollbacks: int = 1
    max_narrative_revisions: int = 1
    subgraph_parallelism: int = 8
    # 2500, not the old 400: after the 2026-08-31 localisation fixes (uniform 200-frame
    # micro ceiling + zero scene baseline) P.scan confirms nearly every engine
    # candidate, so a candidate-rich video runs its full deep-path subgraph per
    # candidate. Measured on the ceiling-200 test runs: s23 (23 candidates) took 173
    # calls, s19 (91 candidates) was on pace for ~800 -- under the old 400 cap every
    # such video degraded mid-subgraph with verdicts missing for later candidates.
    max_llm_calls_per_video: int = 2500
    checkpoint_backend: str = "sqlite"

    # P.scan reviews every duration-eligible candidate for one video in a single
    # response (one JSON array entry per candidate, each carrying a "notes" string).
    # On a video with many candidates that response can run long enough to be cut off
    # mid-string -- observed: gemini-3-flash truncated mid "notes" value on a 29-
    # candidate call, which invalidated the whole batch's JSON and fell back to
    # accept-everything (no candidate actually rejected). Chunking the candidate list
    # keeps each call's expected output short regardless of how long the video's
    # candidate list grows; each chunk is independent, so one chunk truncating only
    # falls back for the candidates in that chunk, not the whole video.
    p_scan_batch_size: int = 10


@dataclass
class RewardConfig:
    """Five-dimensional composite reward (paper eq. 11, appendix F.3)."""

    w_au: float = 0.20
    w_emo: float = 0.20
    w_fmt: float = 0.10
    w_causal: float = 0.30
    w_temp: float = 0.20
    lambda_graph: float = 0.35     # lambda_g in exp(-lambda_g * Edit)
    hallucination_penalty: float = 0.5

    def as_vector(self) -> Dict[str, float]:
        return {
            "R_AU": self.w_au,
            "R_emo": self.w_emo,
            "R_fmt": self.w_fmt,
            "R_causal": self.w_causal,
            "R_temp": self.w_temp,
        }


@dataclass
class EvaluationConfig:
    """Proposal-evaluation criterion (paper eq. 2, protocol P1).
    """

    #: Temporal IoU above which a proposal is a true positive outright.
    iou_threshold: float = 0.5
    #: Enable the second clause of eq. (2): a low-IoU proposal whose fine label matches
    #: the event it overlaps still counts. Turn off to measure pure localisation.
    affective_rescue: bool = True
    #: A rescued proposal must still overlap the event by at least this much. 0.0
    #: reproduces the paper exactly; raising it stops a barely-touching proposal being
    #: rescued on its label alone.
    rescue_min_iou: float = 0.0
    #: Sweep grid for the sensitivity study of paper 4.6.
    iou_sweep: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    #: Credit given to a rescued proposal by ``R_temp`` relative to a genuine hit.
    rescue_reward_scale: float = 0.5


@dataclass
class TrainingConfig:
    """The LOSO training protocol: stage-1 SFT, stage-2 RFT, stage-3 RL.
    """

    # -- fold construction --------------------------------------------------
    #: Subjects drawn out of the N-1 non-test pool for inner validation.
    n_val_subjects: int = 0
    fold_seed: int = 20260824
    #: Trainable policy; must resolve to an ``open_weights`` entry of the registry.
    policy_model: str = "Qwen3-VL-8B"

    # -- stage 0: the CLIP engine -------------------------------------------
    #: Fine-tune the dual-tower CLIP localiser on the fold's pool *before* stage 1,
    #: inside the same run. On by default: every downstream stage consumes the
    #: representation this engine produces, so a stock CLIP would make the SFT/RFT/GRPO
    #: numbers describe a different engine than the one the report names. The
    #: checkpoint records the subjects it saw and inference asserts the held-out
    #: subject is absent, so the fold discipline survives the merge.
    clip_finetune: bool = True
    #: Reuse ``<clip.checkpoint_root>/<dataset>/fold_<subject>/clip_localiser.pt`` when
    #: it already exists instead of re-fitting. Set False to force a retrain -- the only
    #: reason to, since the checkpoint is fold-scoped and cannot be stale across folds.
    clip_reuse_checkpoint: bool = True
    #: Refuse to continue a fold whose CLIP engine could not be fitted or loaded. A
    #: silent fall-back to the analytic front end would publish a "fine-tuned CLIP"
    #: number that no fine-tuned CLIP ever produced.
    clip_required: bool = True

    # -- stage 1: SFT -------------------------------------------------------
    #: A ceiling, not a target: training stops when the gate below is satisfied.
    sft_max_epochs: int = 100
    sft_batch_size: int = 32
    sft_learning_rate: float = 1e-5
    sft_weight_decay: float = 0.01
    sft_warmup_ratio: float = 0.03
    sft_grad_clip: float = 1.0
    sft_early_stop: bool = True
    sft_patience: int = 8
    sft_min_delta: float = 1e-4
    #: 方案 §3 -- lambda_prop. Extra cross-entropy weight on the onset/offset/apex
    #: number tokens of ``part1_proposals`` in the assistant target, so the localisation
    #: tokens and the analysis tokens are supervised in the same backward pass.
    #: 0 disables the term; it only applies to gradient-capable (local) backends.
    sft_prop_weight: float = 1.0
    #: How many times the gate may send training back for more SFT before giving up.
    max_sft_rounds: int = 3

    # -- sufficiency gate ---------------------------------------------------
    #: (1a) JSON-parseable and section-complete share of sampled outputs.
    format_pass_rate: float = 0.95
    #: (1b) The plateau is judged on the trailing fraction of the loss curve.
    plateau_window: float = 0.2
    plateau_max_slope: float = 1e-3
    plateau_max_cv: float = 0.05
    #: (2a) pass@k headroom. ``pass@k >= pass@1`` holds by construction, so the gate is
    #: the *gap* plus a floor on the achievable rate, never the inequality itself.
    pass_k: int = 8
    pass_at_k_samples: int = 16
    pass_gap_min: float = 0.15
    pass_at_k_floor: float = 0.50
    #: (2b) Reward spread. A group with no spread carries no preference information --
    #: high or low alike -- because the group-relative advantage is then identically zero.
    reward_std_min: float = 0.05
    reward_mean_low: float = 0.20
    reward_mean_high: float = 0.85

    # -- stage 2: RFT and QA augmentation -----------------------------------
    #: Stage 2 draws no samples of its own: it ranks and filters the ``pass_at_k_samples``
    #: candidates the gate already drew for the pass@k diagnostic. (There was a
    #: ``rft_samples_per_prompt`` here that nothing ever read -- it promised an
    #: independent stage-2 sampling budget that does not exist, so it is gone rather
    #: than left to mislead whoever tunes the config.)
    rft_accept_top_k: int = 2
    rft_min_reward: float = 0.6
    write_augmented_qa: bool = True
    use_augmented_qa: bool = True
    #: Also read the corpus-level augmented pool written by ``mewm augment-qa``
    #: (``Q-T-A/<ds>/augmented/<ds>_augmented_full_<model>.json``) and keep only the
    #: rows whose video is in this fold's pool. The sweep is expensive and its policy
    #: is frozen and fold-agnostic, so drawing it once and projecting it per fold is
    #: statistically identical to redrawing per fold -- and it is what stops
    #: augmentation from being a separate pipeline bolted on beside this one.
    #: Isolation is enforced on the *video key*, which is exact, not on the
    #: ``provenance.folds`` list, which is only cross-checked.
    use_consolidated_augmented: bool = True
    #: Which sweep to read. Must name the model the sweep actually ran under: two
    #: annotators produce two answer distributions and mixing them silently is the
    #: failure this field exists to prevent.
    consolidated_augmented_model: str = "claude-sonnet-5"

    # -- stage 3: RL --------------------------------------------------------
    rl_total_steps: int = 1000
    rl_group_size: int = 8
    rl_prompts_per_step: int = 4
    rl_eval_every: int = 50
    rl_temperature: float = 0.7
    #: Drop prompts whose reward spread leaves the policy-gradient update a no-op.
    rl_admit_prompts: bool = True

    # -- stage 4: the held-out test ------------------------------------------
    #: Test-time augmentation on the held-out subject: draw ``tta_samples`` answers per
    #: test prompt and aggregate, with **no gradient step of any kind**. This is an
    #: inference strategy, not training -- the held-out subject never reaches a loss
    #: term, an optimiser, or an augmented-QA file. That distinction is the whole reason
    #: it may touch the test set at all, so every report carries ``tta`` with the draw
    #: count and pass@1 is reported beside the aggregated number, never replaced by it.
    test_time_augmentation: bool = True
    #: Draws per test prompt. 1 is plain pass@1 and disables aggregation.
    tta_samples: int = 5


@dataclass
class ClipConfig:
    """CLIP dual-tower motion representation engine (修改方案 §1 / §2).
    """

    enabled: bool = True
    #: 2026-09-03: retargeted to this repo's own local checkout (verified present:
    #: config.json + pytorch_model.bin under this exact path) instead of the
    #: reference implementation's ``Mamba_CLIP`` location.
    weights_path: str = r"E:/code/MEWM-Agent-main/Weights/clip/clip-vit-base-patch16"
    vision_unfreeze_layers: int = 6
    text_unfreeze_layers: int = 2
    temperature: float = 0.07

    # -- loss weights (方案 §2.3) ---------------------------------------------
    lambda_align: float = 0.5
    lambda_loc: float = 1.0
    lambda_cont: float = 0.3
    lambda_prop: float = 0.5
    lambda_distill: float = 0.25
    contrastive_margin: float = 0.2
    #: Head-motion negatives: frames above this per-video speed percentile that fall
    #: outside every annotated event (方案 §2.2 -- the explicit negative class).
    head_speed_percentile: float = 75.0

    # -- localisation head -----------------------------------------------------
    head_channels: int = 64
    head_dropout: float = 0.1

    # -- local cross-attention fusion (formwork.md 第 III 条, 完整执行方案 第 2.4 节,
    # 2026-09-03 新增) -- a learnable cross-attention layer on top of the existing
    # gated fusion: each frame's visual embedding attends over the motion-description
    # embeddings of a local temporal window around it (radius in frames), so the fused
    # representation captures short-range temporal context of the flow description
    # rather than only the current frame's. Additive to the gated fusion output
    # (u' = pool(Attn) + u), so disabling it exactly reproduces the pre-2026-09-03
    # fusion.
    use_cross_attention: bool = True
    cross_attention_heads: int = 4
    cross_attention_window_radius: int = 1
    cross_attention_dropout: float = 0.1

    # -- training schedule -------------------------------------------------------
    epochs: int = 30
    window: int = 64                 # frames per training window (CLIP forward cost)
    batch_frames: int = 32           # CLIP image batch inside a window
    uniform_windows_per_video: int = 4
    learning_rate_head: float = 3e-4
    learning_rate_towers: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 4
    pos_weight_cap: float = 20.0
    focal_gamma: float = 1.0
    val_subject_fraction: float = 0.25
    min_val_subjects: int = 1
    device: str = "cuda"
    autocast_bf16: bool = True
    seed: int = 20260829
    #: Where per-fold checkpoints land: <checkpoint_root>/<dataset>/fold_<subject>/.
    checkpoint_root: str = "runs/clip_localiser"
    #: At inference, replace the analytic slot activations with the transition head's
    #: (方案 §1.4 替换关系). The (T, K) shape and AU ordering are unchanged.
    replace_activations: bool = True


@dataclass
class BackendsConfig:
    """Hosted / local double-backend runtime (修改方案 §5).

    ``fallback_to_api`` is the *only* sanctioned degradation: with it off (default),
    missing local weights are a hard error; with it on, each fallback is WARNING-logged
    and written to the run's ``backend_manifest`` so the report can mark the video as
    partially API-served.
    """

    local_dtype: str = "bfloat16"
    local_quantization: str = "4bit"     # none | 4bit (nf4 double quantisation)
    fallback_to_api: bool = False
    #: Model ids to ensure_weights() before a run starts (断点续传, 方案 §4.2).
    prefetch: tuple = ()


@dataclass
class LLMConfig:
    """Per-role model binding.
    """

    reasoning_model: str = "gpt-5.6-sol"
    perception_model: str = "gpt-5.6-sol"
    structure_model: str = "gpt-5.6-sol"
    critic_model: str = "claude-sonnet-5"  # heterogeneous w.r.t. the R-Agent base
    orchestrator_model: str = "gpt-5.6-sol"

    # -- per-role backend (方案 §5.6): 'hosted' | 'local', explicit, never guessed.
    # A hosted backend with an open-weight id (or vice versa) is a BackendMismatch
    # error at startup, not a silent re-route.
    reasoning_backend: str = "hosted"
    perception_backend: str = "hosted"
    structure_backend: str = "hosted"
    critic_backend: str = "hosted"

    #: Per-role reasoning tier; "" uses each model's registered default.
    reasoning_effort: str = ""
    perception_effort: str = ""
    structure_effort: str = ""
    critic_effort: str = ""

    temperature_p_scan: float = 0.0
    temperature_p_verify: float = 0.2
    temperature_a: float = 0.25
    temperature_r_infer: float = 0.3
    temperature_r_train: float = 0.7
    temperature_c: float = 0.5

    max_tokens: int = 8192
    # P.scan's per-call output is one JSON array entry (with a prose "notes" field)
    # per candidate in its batch (see OrchestratorConfig.p_scan_batch_size). Batching
    # bounds candidates-per-call, but a wordy model can still run long on a full
    # batch, so this phase gets its own, larger ceiling on top of that rather than
    # raising max_tokens globally for every other, much shorter, call.
    p_scan_max_tokens: int = 16384
    timeout: int = 600
    retries: int = 3

    # -- open-weight runtime knobs
    local_dtype: str = "bfloat16"
    local_device_map: str = "auto"
    local_max_new_tokens: int = 2048

    def effort_for(self, role: str) -> str:
        return {
            "R": self.reasoning_effort, "P": self.perception_effort,
            "A": self.structure_effort, "C": self.critic_effort,
        }.get(role, "")

    def max_tokens_for(self, phase: str) -> int:
        """Per-phase output-token ceiling; every phase but P.scan uses ``max_tokens``.
        """
        return self.p_scan_max_tokens if phase == "P.scan" else self.max_tokens

    def model_for(self, role: str) -> str:
        return {
            "R": self.reasoning_model, "P": self.perception_model,
            "A": self.structure_model, "C": self.critic_model,
        }.get(role, self.reasoning_model)

    def backend_for_role(self, role: str) -> str:
        return {
            "R": self.reasoning_backend, "P": self.perception_backend,
            "A": self.structure_backend, "C": self.critic_backend,
        }.get(role, self.reasoning_backend)


@dataclass
class MEWMConfig:
    """Root configuration object."""

    motion: MotionConfig = field(default_factory=MotionConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    spotting: SpottingConfig = field(default_factory=SpottingConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    token_regulator: TokenRegulatorConfig = field(default_factory=TokenRegulatorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)

    # ES / DC convex combination weight alpha (paper 3.4.4).
    es_dc_alpha: float = 0.5
    # Confidence fusion weights lambda_1..lambda_4 (appendix H.6).
    confidence_weights: tuple = (0.35, 0.25, 0.20, 0.20)
    seed: int = 20260824

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading / overlaying
# ---------------------------------------------------------------------------

_ENV_PREFIX = "MEWM_"


def _coerce(current: Any, raw: Any) -> Any:
    """Coerce ``raw`` to the type of the existing default."""
    if isinstance(current, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(raw)
    return raw


def _overlay(target: Any, values: Dict[str, Any]) -> None:
    for key, value in (values or {}).items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _overlay(current, value)
        else:
            setattr(target, key, _coerce(current, value))


def _overlay_env(config: MEWMConfig) -> None:
    """Apply ``MEWM_<SECTION>_<FIELD>`` environment overrides."""
    for name, value in os.environ.items():
        if not name.startswith(_ENV_PREFIX):
            continue
        parts = name[len(_ENV_PREFIX):].lower().split("__")
        if len(parts) != 2:
            continue
        section, field_name = parts
        block = getattr(config, section, None)
        if block is None or not hasattr(block, field_name):
            continue
        setattr(block, field_name, _coerce(getattr(block, field_name), value))


def load_config(path: Optional[Path | str] = None) -> MEWMConfig:
    """Build the runtime config: defaults <- YAML <- environment."""

    config = MEWMConfig()
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend([CONFIG_DIR / "mewm_agent.yaml", CONFIG_DIR / "thresholds.yaml"])

    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            import yaml
        except ImportError:  # pragma: no cover - yaml is a soft dependency
            break
        with open(candidate, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        _overlay(config, payload)

    _overlay_env(config)
    return config


def run_dir(video_id: str, root: Optional[Path] = None) -> Path:
    """Per-video output directory; created on demand."""
    base = Path(root) if root else RUNS_ROOT
    target = base / video_id
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "PACKAGE_ROOT", "PIPELINE_ROOT", "PROJECT_ROOT", "DATASET_ROOT", "FLOW_ROOT",
    "FLOW_CROP_ROOT", "FACE_CROP_ROOT", "QTA_ROOT", "PRE_PROCESS_DIR", "CONFIG_DIR", "SKILLS_DIR",
    "RUNS_ROOT", "CACHE_ROOT", "CHECKPOINT_ROOT", "ENGINE_CKPT_ROOT", "LOADED_ENV_FILE",
    "ensure_pre_process_importable", "MotionConfig", "RepresentationConfig",
    "DynamicsConfig", "SpottingConfig", "CriticConfig", "SchedulerConfig",
    "TokenRegulatorConfig", "MemoryConfig", "OrchestratorConfig", "RewardConfig",
    "EvaluationConfig", "TrainingConfig", "ClipConfig", "BackendsConfig",
    "LLMConfig", "MEWMConfig", "load_config", "run_dir",
]
