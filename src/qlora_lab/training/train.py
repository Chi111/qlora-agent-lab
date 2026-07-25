from __future__ import annotations

import argparse
import json
from pathlib import Path

from qlora_lab.common.logging import configure_logging
from qlora_lab.training.config import TrainConfig
from qlora_lab.training.dataset import load_training_dataset
from qlora_lab.training.validate_data import validate_dataset_pair


def train(config: TrainConfig) -> None:
    validate_dataset_pair(config.train_file, config.eval_file)
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[ml]"') from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA training profile.")

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    args = SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config.max_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=use_bf16,
        fp16=not use_bf16,
        packing=False,
        assistant_only_loss=config.assistant_only_loss,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=load_training_dataset(config.train_file),
        eval_dataset=load_training_dataset(config.eval_file),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    (config.output_dir / "training_config.json").write_text(
        json.dumps(config.as_serializable_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_8gb.yaml"),
        help="Path to a YAML training profile.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    train(TrainConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
