"""Train GRPO or SFT on tool use from a resolved params file.

The orchestrator wrapper owns the W&B run (``WANDB_*`` env vars). This script
does not set ``run_name``, project, entity, or id. Unknown params fail when
the trainer config is constructed. The final model is saved; there is no
checkpoint selection, matching the paper's sweep-then-evaluate protocol.
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from orch.yamlio import load_path
from rls_razor.tooluse import format_for_grpo, format_for_sft, tooluse_reward

POPPED = ("method", "model_name", "dtype", "train_data")


def build_config(params: dict, run_dir: str):
    """Return ``(method, model_name, train_data, training_args)``."""
    cfg = dict(params)
    missing = [key for key in (*POPPED, "seed") if key not in cfg]
    if missing:
        raise SystemExit(f"config is missing {', '.join(missing)}")
    method = cfg.pop("method")
    model_name = cfg.pop("model_name")
    dtype = cfg.pop("dtype")
    train_data = cfg.pop("train_data")
    common = {
        "output_dir": os.path.join(run_dir, "checkpoints"),
        "report_to": "wandb",
        "eval_strategy": "no",
        "save_strategy": "no",
        "model_init_kwargs": {"device_map": {"": 0}, "dtype": dtype},
    }
    if method == "grpo":
        from trl import GRPOConfig

        training_args = GRPOConfig(**cfg, **common)
    elif method == "sft":
        from trl import SFTConfig

        training_args = SFTConfig(**cfg, **common)
    else:
        raise SystemExit(f"unknown method {method!r}")
    return method, model_name, train_data, training_args


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train GRPO or SFT from a params file")
    parser.add_argument("--config", type=Path, required=True, help="Resolved params.yaml")
    args = parser.parse_args()
    loaded = load_path(args.config)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{args.config} is not a mapping")
    run_dir = os.environ.get("ORCH_RUN_DIR")
    model_dir = os.environ.get("ORCH_MODEL_DIR")
    if not run_dir or not model_dir:
        raise SystemExit("ORCH_RUN_DIR and ORCH_MODEL_DIR must be set")

    method, model_name, train_data, training_args = build_config(loaded, run_dir)
    raw_dataset = load_from_disk(train_data)
    if not isinstance(raw_dataset, Dataset):
        raise SystemExit(f"{train_data} is not a datasets.Dataset")
    format_row = format_for_grpo if method == "grpo" else format_for_sft
    train_dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if method == "grpo":
        from trl import GRPOTrainer

        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=tooluse_reward,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    else:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    trainer.train()
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
