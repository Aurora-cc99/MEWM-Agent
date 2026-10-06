"""All dataclass configs and load_config() entry point for MEWM-Agent."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = PACKAGE_ROOT
PROJECT_ROOT = PACKAGE_ROOT


def _resolve_dataset_root() -> Path:
    env = os.environ.get("MEWM_DATASET_ROOT")
    if env:
        return Path(env)
    return PROJECT_ROOT / "dataset"


DATASET_ROOT = _resolve_dataset_root()
FLOW_ROOT = PIPELINE_ROOT / "pre_datasets"
FLOW_CROP_ROOT = PIPELINE_ROOT / "pre_datasets_crop"
FACE_CROP_ROOT = PIPELINE_ROOT / "pre_datasets" / "face_crop"
QTA_ROOT = PIPELINE_ROOT / "Q-T-A"
PRE_PROCESS_DIR = PIPELINE_ROOT / "pre_process"

CONFIG_DIR = PACKAGE_ROOT / "configs"
SKILLS_DIR = PACKAGE_ROOT / "skills"
RUNS_ROOT = Path(os.environ.get("MEWM_RUNS_ROOT", str(PACKAGE_ROOT / "runs")))
CACHE_ROOT = PACKAGE_ROOT / "cache"
CHECKPOINT_ROOT = RUNS_ROOT / "checkpoints"

ENGINE_CKPT_ROOT = PACKAGE_ROOT / "engine_ckpt"


def _find_env_file() -> Optional[Path]:
    override = os.environ.get("MEWM_ENV_FILE")
    if override:
        return Path(override)
    local = PACKAGE_ROOT / ".env"
    if local.is_file():
        return local
    return None


def _load_dotenv() -> Optional[Path]:
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


LOADED_ENV_FILE: Optional[Path] = _load_dotenv()


def ensure_pre_process_importable() -> None:
    path = str(PRE_PROCESS_DIR)
    if PRE_PROCESS_DIR.is_dir() and path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class MotionConfig:
    n_roi: int = 29
    m_min: float = 0.15
    c_min: float = 0.60
    default_flow_gap: int = 7


@dataclass
class RepresentationConfig:
    n_slots: int = 16
    slot_dim: int = 128
    slow_dim: int = 256
    fast_dim: int = 128
    belief_dim: int = 64
    appearance_dim: int = 64
    slot_dropout: float = 0.15
    activation_threshold: float = 0.35
    weak_threshold: float = 0.20
    max_active_aus: int = 3
    activation_relative_margin: float = 0.75
    slow_process_noise: float = 1e-5
    slow_obs_noise: float = 1e-2
    changepoint_frames: int = 15
    changepoint_nll: float = 12.0


@dataclass
class DynamicsConfig:
    n_gat_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 256
    n_mixture: int = 5
    rollout_steps: int = 3
    future_queries: int = 8
    kappa_low: float = 0.7
    lambda_imagine: float = 1.0
    lambda_flow: float = 0.5
    lambda_slow: float = 0.1


@dataclass
class SpottingConfig:
    window: int = 300
    tau_hi: float = 3.5
    tau_lo: float = 1.5
    min_duration_frames: int = 0
    min_micro_seconds: float = 0.0
    max_micro_seconds: float = 0.0
    n_physio_templates: int = 12
    physio_match_thresh: float = 0.55
    merge_gap_frames: int = 3

    proposal_max_per_video: int = 10
    proposal_trim_fraction: float = 0.3
    proposal_raw_curve_frac: float = 0.4
    proposal_soft_k_margin: float = 0.0

    localiser_enabled: bool = True
    localiser_bank_size: int = 8
    localiser_flank_fraction: float = 0.25
    localiser_trough_weight: float = 1.0
    localiser_max_per_video: int = 12
    localiser_min_score: float = 0.0
    localiser_nms_iou: float = 0.20
    localiser_min_separation_seconds: float = 0.15

    attribution_sharpen_temperature: Optional[float] = None


@dataclass
class CriticConfig:
    eta_lambda: float = 2.0
    eta_cfs: float = 0.10
    eta_suppression: float = 0.15
    kappa_micro: float = 0.04
    mni_hallucination: float = 0.02
    max_challenges_per_round: int = 3
    max_challenge_rounds: int = 3


@dataclass
class SchedulerConfig:
    theta_fast: float = 0.20
    eta_fast: float = 3.0
    theta_deep: float = 0.55
    theta_var: float = 0.35


@dataclass
class TokenRegulatorConfig:
    enabled: bool = True
    n_r: int = 9
    delta: float = 4.0
    p_v: float = 0.03
    p_t: float = 0.20
    gamma_anchor_min: float = 1.0
    keep_instruction_tokens: bool = True


@dataclass
class MemoryConfig:
    stationary_run_length: int = 30
    n_query_tokens_multi_au: int = 8
    n_query_tokens_single_au: int = 4
    piecewise_linear_tol: float = 0.05
    retrieval_jaccard: float = 0.4
    retrieval_max_per_level: int = 2
    retrieval_max_tokens: int = 800
    retrieval_enabled: bool = True


@dataclass
class OrchestratorConfig:
    max_gate_retries: int = 2
    max_challenge_rounds: int = 3
    max_cascade_rollbacks: int = 1
    max_narrative_revisions: int = 1
    subgraph_parallelism: int = 8
    max_llm_calls_per_video: int = 2500
    checkpoint_backend: str = "sqlite"

    p_scan_batch_size: int = 10


@dataclass
class RewardConfig:
    w_au: float = 0.20
    w_emo: float = 0.20
    w_fmt: float = 0.10
    w_causal: float = 0.30
    w_temp: float = 0.20
    lambda_graph: float = 0.35
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
    iou_threshold: float = 0.5
    affective_rescue: bool = True
    rescue_min_iou: float = 0.0
    iou_sweep: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    rescue_reward_scale: float = 0.5


@dataclass
class TrainingConfig:
    n_val_subjects: int = 0
    fold_seed: int = 20260824
    policy_model: str = ""

    clip_finetune: bool = True
    clip_reuse_checkpoint: bool = True
    clip_required: bool = True

    softnet_finetune: bool = True
    softnet_reuse_checkpoint: bool = True
    softnet_required: bool = True
    softnet_epochs: int = 300

    sft_max_epochs: int = 100
    sft_batch_size: int = 32
    sft_learning_rate: float = 1e-5
    sft_weight_decay: float = 0.01
    sft_warmup_ratio: float = 0.03
    sft_grad_clip: float = 1.0
    sft_early_stop: bool = True
    sft_patience: int = 5
    sft_min_delta: float = 1e-4
    sft_min_delta_relative: float = 0.05
    sft_prop_weight: float = 1.0
    max_sft_rounds: int = 3

    format_pass_rate: float = 0.95
    plateau_window: float = 0.2
    plateau_max_slope: float = 1e-3
    plateau_max_cv: float = 0.05
    pass_k: int = 8
    pass_at_k_samples: int = 24
    pass_gap_min: float = 0.15
    pass_at_k_floor: float = 0.50
    reward_std_min: float = 0.05
    reward_mean_low: float = 0.20
    reward_mean_high: float = 0.85

    # stage-2 ranks/filters the pass@k candidates; no independent sampling budget
    rft_accept_top_k: int = 2
    rft_min_reward: float = 0.6
    write_augmented_qa: bool = True
    use_augmented_qa: bool = True
    use_consolidated_augmented: bool = True
    consolidated_augmented_model: str = ""

    rl_total_steps: int = 1000
    rl_group_size: int = 8
    rl_prompts_per_step: int = 6
    rl_eval_every: int = 50
    rl_temperature: float = 0.7
    rl_admit_prompts: bool = True

    test_time_augmentation: bool = True
    tta_samples: int = 5


@dataclass
class ClipConfig:
    enabled: bool = True
    weights_path: str = ""
    vision_unfreeze_layers: int = 6
    text_unfreeze_layers: int = 6
    temperature: float = 0.07

    lambda_align: float = 0.5
    lambda_loc: float = 1.0
    lambda_cont: float = 0.3
    lambda_prop: float = 0.5
    lambda_distill: float = 0.25
    contrastive_margin: float = 0.2
    head_speed_percentile: float = 75.0

    head_channels: int = 64
    head_dropout: float = 0.1

    use_cross_attention: bool = True
    cross_attention_heads: int = 4
    cross_attention_window_radius: int = 1
    cross_attention_dropout: float = 0.1

    epochs: int = 50
    window: int = 64
    batch_frames: int = 32
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
    checkpoint_root: str = field(default_factory=lambda: str(RUNS_ROOT / "clip_localiser"))
    replace_activations: bool = True


@dataclass
class BackendsConfig:
    # fallback_to_api=True logs a warning and records partial API serving in the manifest
    local_dtype: str = "bfloat16"
    local_quantization: str = "4bit"
    fallback_to_api: bool = False
    prefetch: tuple = ()


@dataclass
class SFTLoraConfig:
    policy: str = ""
    quantization: str = "4bit"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"
    batch_size: int = 2
    max_length: int = 4096
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    epochs: int = 6
    held_out_subject: str = ""
    evaluate_videos: tuple = ()
    merge_output: str = ""
    test_runs_output: str = ""

    _PROFILES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "Qwen2.5-VL-7B": dict(
            policy="Qwen2.5-VL-7B", quantization="none",
            r=32, alpha=64, dropout=0.05, target_modules="all-linear",
            batch_size=8, max_length=4096,
            learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.03, grad_clip=1.0,
            epochs=6,
        ),
        "Qwen3-VL-8B": dict(
            policy="Qwen3-VL-8B", quantization="none",
            r=32, alpha=64, dropout=0.05, target_modules="all-linear",
            batch_size=6, max_length=4096,
            learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.03, grad_clip=1.0,
            epochs=6,
        ),
        "GLM-4.1V-9B-Thinking": dict(
            policy="GLM-4.1V-9B-Thinking", quantization="none",
            r=32, alpha=64, dropout=0.05, target_modules="all-linear",
            batch_size=5, max_length=4096,
            learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.03, grad_clip=1.0,
            epochs=6,
        ),
        "Qwen2.5-VL-32B": dict(
            policy="Qwen2.5-VL-32B", quantization="4bit",
            r=32, alpha=64, dropout=0.05, target_modules="all-linear",
            batch_size=2, max_length=4096,
            learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.03, grad_clip=1.0,
            epochs=6,
        ),
        "Qwen2.5-Omni-7B": dict(
            policy="Qwen2.5-Omni-7B", quantization="none",
            r=32, alpha=64, dropout=0.05, target_modules="all-linear",
            batch_size=6, max_length=4096,
            learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.03, grad_clip=1.0,
            epochs=6,
        ),
        "Qwen3-VL-30B": dict(
            policy="Qwen3-VL-30B", quantization="4bit",
            r=32, alpha=64, dropout=0.05, target_modules="all-linear",
            batch_size=2, max_length=4096,
            learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.03, grad_clip=1.0,
            epochs=6,
        ),
    }

    @classmethod
    def for_policy(cls, policy: str, dataset: str = "casme_sq",
                    held_out_subject: str = "s23",
                    evaluate_videos: tuple = ("s23_23_0102eatingworms",)
                    ) -> "SFTLoraConfig":
        import logging
        overrides = cls._PROFILES.get(policy)
        if overrides is None:
            logging.getLogger("mewm").warning(
                "no 32GB LoRA profile for policy %r; using the 8GB-card dataclass "
                "defaults (r=%d, 4bit) -- add an entry to SFTLoraConfig._PROFILES "
                "to size this policy for the deployment card.", policy, cls.r)
            overrides = {"policy": policy}
        merge_output = f"runs/sft/{dataset}/merged"
        test_runs_output = f"runs/sft/{dataset}/test_runs"
        return cls(held_out_subject=held_out_subject, evaluate_videos=evaluate_videos,
                    merge_output=merge_output, test_runs_output=test_runs_output,
                    **overrides)


@dataclass
class LLMConfig:
    reasoning_model: str = ""
    perception_model: str = ""
    structure_model: str = ""
    critic_model: str = ""
    orchestrator_model: str = ""

    reasoning_backend: str = "hosted"
    perception_backend: str = "hosted"
    structure_backend: str = "hosted"
    critic_backend: str = "hosted"

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
    p_scan_max_tokens: int = 16384
    timeout: int = 600
    retries: int = 3

    local_dtype: str = "bfloat16"
    local_device_map: str = "auto"
    local_max_new_tokens: int = 2048

    def effort_for(self, role: str) -> str:
        return {
            "R": self.reasoning_effort, "P": self.perception_effort,
            "A": self.structure_effort, "C": self.critic_effort,
        }.get(role, "")

    def max_tokens_for(self, phase: str) -> int:
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
    sft_lora: SFTLoraConfig = field(default_factory=SFTLoraConfig)

    es_dc_alpha: float = 0.5
    confidence_weights: tuple = (0.35, 0.25, 0.20, 0.20)
    seed: int = 20260824

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_ENV_PREFIX = "MEWM_"


def _coerce(current: Any, raw: Any) -> Any:
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
        except ImportError:
            break
        with open(candidate, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        _overlay(config, payload)

    _overlay_env(config)
    return config


def run_dir(video_id: str, root: Optional[Path] = None) -> Path:
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

