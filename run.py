from __future__ import annotations

import inspect
import os
import random
import socket
import contextlib
from functools import partial
from pathlib import Path
from typing import *

import click
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils.argument import ScriptArguments
from utils.data import (
    RewardDataCollatorWithPadding,
    build_dataset,
    post_filter_by_ratio,
)
from utils.trainer import (
    BTTRewardTrainer,
    RewardTrainer,
    RewardTrainerWithOracleCE,
    RewardTrainerWithRingeMargin,
    compute_CE_oracle,
    compute_ML_oracle,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set for all GPUs
    set_seed(seed)  # Hugging Face's Trainer consistency

    # Ensure deterministic behavior across multi-GPU environments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_free_port() -> str:
    """Pick an available port on localhost for single-process runs."""
    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def ensure_single_process_dist_defaults():
    """
    DeepSpeed tries to discover MPI settings when rank/world info is missing,
    which pulls in mpi4py/libmpi. On single-device setups we can short-circuit
    that path by pre-populating the environment variables.
    """
    # Use a dynamic port to avoid collisions when multiple single-process jobs
    # are launched on the same machine.
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": _find_free_port(),
    }
    missing = {k: v for k, v in defaults.items() if k not in os.environ}
    if not missing:
        return

    for key, value in missing.items():
        os.environ[key] = value
    os.environ.setdefault("DEEPSPEED_DISABLE_MPI", "1")
    logger.info(
        "Distributed env vars were missing; defaulting to single-process values to avoid MPI dependency."
    )


@click.command()
@click.argument("script_config_path", type=str)
@click.option("--seed", type=int, default=None, help="Random seed")
@click.option("--lr", type=float, default=None, help="Learning rate")
def main(
    script_config_path: str,
    seed: Optional[int] = None,
    lr: Optional[float] = None,
):
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_json_file(json_file=script_config_path)[0]

    if seed is not None:
        seed = int(seed)
    else:
        seed = script_args.seed

    if lr is not None:
        lr = float(lr)
    else:
        lr = script_args.learning_rate

    set_random_seed(seed)
    ensure_single_process_dist_defaults()

    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    # handle those with no official pad token
    no_predefined_pad_flag = False
    if tokenizer.pad_token is None:
        no_predefined_pad_flag = True
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length

    train_path = script_args.train_set_path
    output_name = f"{script_args.model_name.split('/')[-1]}_lr{lr}_trainbs{script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps}_trainer{script_args.trainer_type}_label{script_args.label_type}_seed{seed}"

    if script_args.selected_pos_ratio is not None:
        output_name += f"_pos{script_args.selected_pos_ratio}"
        output_name += f"_met{script_args.select_method}"
    if script_args.trainer_type == "bttr":
        output_name += f"_bttrtheta{script_args.margin_delta}"
    if script_args.trainer_type == "ringemargin":
        output_name += f"_margin{script_args.margin_delta}"
    if script_args.use_lora:
        output_name += "_LoRA"

    output = str(Path(script_args.output_path) / output_name)
    logger.info(f"save to: {output}")

    train_dataset, eval_dataset = build_dataset(
        tokenizer,
        train_path,
        max_train_size=script_args.max_train_size,
        label_type=script_args.label_type,
        diff_scaling_factor=script_args.diff_rescaling_factor,
        seed=seed,
    )

    # Allow a different evaluation split if provided in config.
    eval_path = getattr(script_args, "eval_set_path", None) or train_path
    if eval_path != train_path:
        _, eval_dataset = build_dataset(
            tokenizer,
            eval_path,
            max_train_size=script_args.max_train_size,
            label_type=script_args.label_type,
            diff_scaling_factor=script_args.diff_rescaling_factor,
            seed=seed,
        )

    n_samples=4* 1024
    if script_args.selected_pos_ratio is not None:
        train_dataset = post_filter_by_ratio(
            train_dataset,
            # ! 32* 1024 data samples for training 16*1024,12*1024, 8*1024, 4*1024, 
            n_samples=n_samples,
            positive_ratio=script_args.selected_pos_ratio,
            seed=seed,
            select=script_args.select_method,
        )
    
    output_name += "train_size" + str(n_samples/1024)
    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"eval_dataset size: {len(eval_dataset)}")

    training_args_kwargs = {
        "output_dir": output,
        "learning_rate": lr,
        "per_device_train_batch_size": script_args.per_device_train_batch_size,
        "per_device_eval_batch_size": script_args.per_device_eval_batch_size,
        "num_train_epochs": script_args.num_train_epochs,
        "weight_decay": script_args.weight_decay,
        "adam_beta1": script_args.adam_beta1,
        "evaluation_strategy": "steps",
        "eval_steps": script_args.eval_every_steps,
        "save_strategy": "no",  # ! not saving during training
        # "save_steps": script_args.save_every_steps,
        "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
        "gradient_checkpointing": script_args.gradient_checkpointing,
        "deepspeed": script_args.deepspeed,
        "local_rank": script_args.local_rank,
        "remove_unused_columns": False,
        "label_names": ["labels_j", "labels_k"],
        "bf16": script_args.bf16,
        "logging_strategy": "steps",
        "logging_steps": 16,
        "optim": script_args.optim,
        "lr_scheduler_type": script_args.lr_scheduler_type,
        "warmup_ratio": script_args.warmup_ratio,
        "report_to": "wandb",
        "include_inputs_for_metrics": True,
        "seed": seed,
    }

    ta_signature = inspect.signature(TrainingArguments.__init__)
    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in ta_signature.parameters.values()
    )
    if has_var_kwargs:
        filtered_training_args_kwargs = training_args_kwargs
    else:
        supported_keys = set(ta_signature.parameters.keys())
        supported_keys.discard("self")
        # HF renamed `evaluation_strategy` -> `eval_strategy` in newer releases.
        if (
            "evaluation_strategy" in training_args_kwargs
            and "evaluation_strategy" not in supported_keys
            and "eval_strategy" in supported_keys
        ):
            training_args_kwargs["eval_strategy"] = training_args_kwargs.pop(
                "evaluation_strategy"
            )
        filtered_training_args_kwargs = {
            key: value
            for key, value in training_args_kwargs.items()
            if key in supported_keys
        }
        dropped = sorted(
            key
            for key in training_args_kwargs
            if key not in filtered_training_args_kwargs
        )
        if dropped:
            logger.warning(
                f"TrainingArguments dropped unsupported kwargs: {dropped}"
            )

    training_args = TrainingArguments(**filtered_training_args_kwargs)

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # ! either "flash_attention_2" or "eager"
    )

    if no_predefined_pad_flag:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
        ],
    )  # ! it is not well tuned, modify it if needed

    if script_args.use_lora:
        model = get_peft_model(model, peft_config)
        logger.info("Using LORA model")
        model.print_trainable_parameters()

    logger.info(model)

    model.config.use_cache = not script_args.gradient_checkpointing

    if script_args.trainer_type == "vanilla":
        trainer_cls = RewardTrainer
    elif script_args.trainer_type == "oraclece":
        trainer_cls = RewardTrainerWithOracleCE
    elif script_args.trainer_type == "bttr":
        trainer_cls = BTTRewardTrainer
    elif script_args.trainer_type == "ringemargin":
        trainer_cls = RewardTrainerWithRingeMargin
    else:
        raise ValueError(
            f"Unsupported trainer type: {script_args.trainer_type}"
        )

    # ! a little nasty, but it works
    if script_args.trainer_type in (
        "oraclece",
        "bttr",
        "vanilla",
    ):
        metric_func = compute_CE_oracle  # type: ignore
    elif script_args.trainer_type in ("ringemargin",):
        metric_func = partial(
            compute_ML_oracle, delta=script_args.margin_delta
        )  # type: ignore

    trainer = trainer_cls(
        delta=script_args.margin_delta,
        tie_thrsd=0.15,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_func,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length
        ),
    )

    # ! make sure this part is behind the trainer initialization
    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(
            project="AccessRM",
            name=output_name,
        )

    trainer.train()

    logger.info("Finished training, saving the model...")
    trainer.save_model(str(Path(output) / "last_ckpt"))
    tokenizer.save_pretrained(str(Path(output) / "last_ckpt"))
    if script_args.use_lora:
        model.save_pretrained(str(Path(output) / "last_ckpt"))


if __name__ == "__main__":
    main()
