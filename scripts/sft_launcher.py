"""SFT stage launcher: runs supervised fine-tuning for all four agents."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("sft_launcher")


def _resolve_qa(dataset: str):
    from mewm.data.qa_loader import load_qa_set
    try:
        return load_qa_set(dataset)
    except Exception as exc:
        LOGGER.warning("no QA set loaded for %s (%s); training on annotation "
                       "instruction samples only", dataset, exc)
        return None


def _apply_lora(model: Any, r: int, alpha: int) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("LoRA applied (r=%d, alpha=%d): %d trainable parameter(s)", r, alpha,
                trainable)
    return model


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["casme_sq", "samm",
                                                             "casme3", "4dme"])
    parser.add_argument("--policy", default="Qwen3-VL-8B",
                        help="open-weight model id (registry name)")
    parser.add_argument("--exclude-subject", default="",
                        help="held-out subject (LOSO fold); empty = train on all")
    parser.add_argument("--include-subjects", default="",
                        help="comma-separated whitelist; overrides --exclude-subject")
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument("--epochs", type=int, default=0,
                        help="override training.sft_max_epochs (0 = config value)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="override training.sft_batch_size (0 = config value)")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--quantization", default="4bit",
                        choices=["4bit", "none"],
                        help="'4bit' keeps the 8 GB path; 'none' loads bf16")
    parser.add_argument("--output", required=True,
                        help="checkpoint directory for the run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from mewm.config import load_config
    from mewm.data.datasets import load_dataset
    from mewm.llm.local_models import load_local_model
    from mewm.llm.registry import resolve
    from mewm.training.instruction_set import InstructionSetBuilder
    from mewm.training.loso import require_open_weight
    from mewm.training.sft import LoRASFTBackend, SFTTrainer, build_sft_samples

    config = load_config()
    training = config.training
    if args.epochs:
        training.sft_max_epochs = args.epochs
    if args.batch_size:
        training.sft_batch_size = args.batch_size
    require_open_weight(args.policy)

    index = load_dataset(args.dataset)
    videos = list(index.videos)
    include = ([s.strip() for s in args.include_subjects.split(",") if s.strip()]
               if args.include_subjects else None)
    exclude = ([args.exclude_subject] if args.exclude_subject else None)
    if include is not None:
        videos = [v for v in videos if str(v.subject) in set(include)]
    elif exclude is not None:
        videos = [v for v in videos if str(v.subject) not in set(exclude)]
    if not videos:
        print("no videos selected", file=__import__("sys").stderr)
        return 1

    qa = _resolve_qa(args.dataset)
    builder = InstructionSetBuilder(lang=args.lang, seed=training.fold_seed)
    instruction_samples = builder.build_dataset(videos, qa)
    samples = build_sft_samples(instruction_samples)
    if not samples:
        print("no SFT samples built (dataset carries no annotated micro-expressions?)",
              file=__import__("sys").stderr)
        return 1

    spec = resolve(args.policy)
    loaded = load_local_model(spec, quantization=args.quantization)
    model = _apply_lora(loaded.model, args.lora_r, args.lora_alpha)
    tokenizer = loaded.processor.tokenizer

    backend = LoRASFTBackend(model=model, tokenizer=tokenizer,
                             max_length=args.max_length,
                             prop_weight=training.sft_prop_weight)
    trainer = SFTTrainer(backend, training, config, seed=training.fold_seed)
    output = Path(args.output)
    outcome = trainer.fit(samples, output_dir=output, in_sample_eval=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output / "lora_adapter")
        print(f"adapter -> {output / 'lora_adapter'}")
    (output / "sft_outcome.json").write_text(
        json.dumps({"policy": args.policy, "n_chats": len(samples),
                    "exclude_subject": args.exclude_subject, **outcome.to_dict()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"outcome -> {output / 'sft_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
