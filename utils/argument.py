from __future__ import annotations

from dataclasses import dataclass, field
from typing import *

from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    adam_beta1: Optional[float] = field(default=0.9)
    model_name: Optional[str] = field(
        default="google/gemma-2b-it",  # "mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of training epochs for the reward model."
        },
    )
    train_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="./bt_models/gemma2b_rm",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    use_lora: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use LORA or not"}
    )
    max_train_size: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of training data to use"},
    )
    trainer_type: Literal[
        "vanilla", "marginonly", "marginandtie", "multiplemargin"
    ] = field(
        default="vanilla",
        metadata={"help": "The type of trainer to use. "},
    )
    margin_delta: Optional[float] = field(
        default=1.0,
        metadata={"help": "The margin delta for the margin loss"},
    )
    label_type: Optional[Literal["oracle", "sampled", "interpolated"]] = field(
        default=None,
        metadata={
            "help": "The type of label to use for the logsitic validaton"
        },
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "The random seed to use"}
    )
    diff_rescaling_factor: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The factor to rescale the difference between the two rank scores"
        },
    )
    selected_pos_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "The ratio of the selected positive samples to the total number of samples"
        },
    )
    select_method: Optional[str] = field(
        default='uniform',
        metadata={
            "help": "The selection method"
        },
    )
    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": "The warmup ratio for the learning rate scheduler"},
    )
