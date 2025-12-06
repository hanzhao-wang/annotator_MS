from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import *

import os
import matplotlib.pyplot as plt
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


    ####plot histogram
    # Suppose 'preference_score' is your data from train_data['label']
    preference_score = train_dataset['label']


    plt.figure(figsize=(8, 6))  
    plt.hist(preference_score, bins=30, edgecolor='black')  # Adjust 'bins' as needed

    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    os.makedirs("fig", exist_ok=True)
    plt.savefig("fig/histogram.png", dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure

    plt.show()
if __name__ == "__main__":
    main()