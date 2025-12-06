from __future__ import annotations

from dataclasses import dataclass
from typing import *
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
import pandas as pd

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy
from utils.argument import ScriptArguments
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
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

def token_counter(
    tokenizer: AutoTokenizer,
    ds: Union[Dataset, str],
    label_type: Optional[Literal["oracle", "sampled", "interpolated", "fineinterpolated"]] = None,
    seed: int = 42,
    num_proc: int = 16,
    num_data: Optional[int] = None,
) -> Tuple[List[int],List[float]]:
    """
    Count the number of tokens of k and j samples in the dataset
    """
    def tokenize(sample):
        """ """
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        sample["score_j"] = float(sample["chosen_score"])
        sample["score_k"] = float(sample["rejected_score"])
        # add discrete label
        if label_type is None:
            pass
        elif label_type=="original":
            sample["label"] = 1.0
            sample["score_gap"] = sample["score_j"] - sample["score_k"]
            sample["preference_score"] = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            ).item()
      
        elif label_type == "oracle":
            sample["label"] = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            ).item()  # remember to convert to regular float
        elif label_type == "sampled":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            sampled_p = torch.bernoulli(true_p)
            sample["label"] = sampled_p.item()
        else:
            raise ValueError("Invalid label_type")

        return sample

    if isinstance(ds, str):
        try:
            ds = load_dataset(ds, split="train")
        except:
            ds = load_from_disk(ds)
    ds = ds.map(lambda x, i: {"__ID__": i}, with_indices=True)
    ds = ds.shuffle(seed=seed)
    if num_data is not None:
        ds = ds.select(range(num_data))

    ds = ds.map(tokenize, num_proc=num_proc)
    token_num_gap = []
    token_score_gap = []

    for i in range(len(ds)):
        token_num_gap.append(len(ds[i]['input_ids_j'])-len(ds[i]['input_ids_k']))
        token_score_gap.append(ds[i]['score_j']-ds[i]['score_k'])
    return token_num_gap,token_score_gap



  

def build_dataset(
    tokenizer: AutoTokenizer,
    ds: Union[Dataset, str],
    label_type: Optional[
        Literal["oracle", "sampled", "interpolated", "fineinterpolated"]
    ] = None,
    diff_scaling_factor: float = 1.0,
    seed: int = 42,
    id_eval_size: int = 1024,
    num_proc: int = 16,
    max_train_size: Optional[int] = None,
    clean_cache: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Perform tokenization and annotation from preference dataset.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Any tokenizer from Huggingface transformers.
    ds : Union[Dataset, str]
        A dataset or a path to a dataset (online or local).
        The dataset should at least have the following columns:
            - chosen
            - rejected
        To add labels, also make sure the dataset has the following columns:
            - chosen_score
            - rejected_score
    label_type : Literal[oracle, sampled, interpolated, fineinterpolated], optional
        The ordinal feedback label type, by default None
        - oracle: as is
        - sampled: 2-level
        - interpolated: 3-level
        - fineinterpolated: 5-level
    diff_scaling_factor : float, optional
        the scaling factor (to be multiplied with the raw score), by default 1.0
    seed : int, optional
        random seed for reproducibility, by default 42
    id_eval_size : int, optional
        eval set size sampled from the training set, by default 1024
    num_proc : int, optional
        process number for tokenization, by default 16
    max_train_size : Optional[int], optional
        maximum size of the training set, by default None
    clean_cache : bool, optional
        clean cache files after loading the dataset, by default False

    Returns
    -------
    Tuple[Dataset, Dataset]
        train_dataset, eval_dataset
    """

    def tokenize(sample):
        """ """
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        sample["score_j"] = float(sample["chosen_score"])
        sample["score_k"] = float(sample["rejected_score"])
        '''
            # ! we make sure chosen/rejected are determined by score_j and score_k
            if sample["score_j"] - sample["score_k"] < 0:
                sample["input_ids_j"] = tokenized_neg["input_ids"]
                sample["attention_mask_j"] = tokenized_neg["attention_mask"]
                sample["input_ids_k"] = tokenized_pos["input_ids"]
                sample["attention_mask_k"] = tokenized_pos["attention_mask"]
                sample["score_j"] = float(sample["rejected_score"])
                sample["score_k"] = float(sample["chosen_score"])
        '''
        # ! we scale score_j and score_k with diff_scaling_factor here
        # ! instead of keeping the raw score in the dataitem,
        # ! as we might use them for evaluation loss computation later.
        sample["score_j"] *= diff_scaling_factor
        sample["score_k"] *= diff_scaling_factor
        # add discrete label
        if label_type is None:
            pass
        elif label_type=="original":
            sample["label"] = 1.0
            sample["score_gap"] = sample["score_j"] - sample["score_k"]
            sample["preference_score"] = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            ).item()
      
        elif label_type == "oracle":
            sample["label"] = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            ).item()  # remember to convert to regular float
        elif label_type == "sampled":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            sampled_p = torch.bernoulli(true_p)
            sample["label"] = sampled_p.item()
        elif label_type == "interpolated":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            if true_p.item() < 0.5:
                sampled_p = torch.bernoulli(true_p * 2)
                sample["label"] = sampled_p.item() * 0.5
            elif true_p.item() > 0.5:
                sampled_p = torch.bernoulli(true_p * 2 - 1)
                sample["label"] = sampled_p.item() * 0.5 + 0.5
            else:
                sample["label"] = 0.5
        elif label_type == "fineinterpolated":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            # we have 5 interpolation points: [0, 0.2, 0.5, 0.8, 1]
            p_intervals = {
                "significantly_worse": (0, 0.2),
                "slightly_worse": (0.2, 0.5),
                "slightly_better": (0.5, 0.8),
                "significantly_better": (0.8, 1),
            }
            for k, v in p_intervals.items():
                if v[0] <= true_p.item() <= v[1]:
                    p_case = k
                    break
            lower, upper = p_intervals[p_case]
            # Bernoulli sampling with
            # (p - p_lower) / (p_upper - p_lower) -> p_upper
            # (p_upper - p) / (p_upper - p_lower) -> p_lower
            sampled_p = (
                torch.bernoulli((true_p - lower) / (upper - lower))
                * (upper - lower)
                + lower
            )
            sample["label"] = sampled_p.item()
        else:
            raise ValueError("Invalid label_type")

        return sample

    if isinstance(ds, str):
        try:
            ds = load_dataset(ds, split="train")
        except:
            ds = load_from_disk(ds)

    if clean_cache:
        ds.cleanup_cache_files()

    ds = ds.map(lambda x, i: {"__ID__": i}, with_indices=True)
    ds = ds.shuffle(seed=seed)
    ds = ds.map(tokenize, num_proc=num_proc)
    if id_eval_size==0:
        id_eval_size=int(len(ds)/2)
    eval_dataset = ds.select(range(id_eval_size))

    if max_train_size is None:
        train_dataset = ds.select(range(id_eval_size, len(ds)))
    else:
        train_dataset = ds.select(
            range(id_eval_size, min(max_train_size + id_eval_size, len(ds)))
        )

    return train_dataset, eval_dataset


def get_data(script_config_path):
    seed=42
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_json_file(json_file=script_config_path)[0]

    if seed is not None:
        seed = int(seed)
    else:
        seed = script_args.seed
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

    train_dataset, eval_dataset = build_dataset(
        tokenizer,
        train_path,
        max_train_size=script_args.max_train_size,
        label_type=script_args.label_type,
        diff_scaling_factor=script_args.diff_rescaling_factor,
        seed=seed,
        clean_cache=True,
        id_eval_size=0,
    )
    size=len(train_dataset['preference_score'])
    random_flips = torch.bernoulli(torch.tensor([0.5]*size)).int().tolist()
    train_dataset = train_dataset.map(lambda x, idx: {"label": 1-random_flips[idx]}, with_indices=True)
    train_dataset = train_dataset.map(lambda x, idx: {"preference_score": abs(random_flips[idx]-x["preference_score"])}, with_indices=True)
    size=len(eval_dataset['preference_score'])
    random_flips = torch.bernoulli(torch.tensor([0.5]*size)).int().tolist()
    eval_dataset = eval_dataset.map(lambda x, idx: {"label": 1-random_flips[idx]}, with_indices=True)
    eval_dataset = eval_dataset.map(lambda x, idx: {"preference_score": abs(random_flips[idx]-x["preference_score"])}, with_indices=True)

    return train_dataset,eval_dataset
def post_filter_by_ratio(
    ds: Dataset,
    n_samples: int = 5000,
    positive_ratio: float = 1.0,
    seed: int = 42,
    select: str = 'uniform'
):
    """Filter the dataset by ratio of binary (positive) and randomly flipped samples.

    Parameters
    ----------
    ds : Dataset
        Dataset from `build_dataset` function.
    n_samples : int, optional
        the number of samples to return, by default 5000
    positive_ratio : float, optional
        the binary sample ratio, by default 1.0
    seed : int, optional
        seed for reproducibility, by default 42
    select :  optional
        uniformly flip (uniform) or only the most positive_ratio% difficult part (most)
    Returns
    -------
    Dataset
        dataset with 1-positive ratio X n_samples randomly flipped labels; When X>0 need the input dataset to be oracle
    """
    n_samples=min(n_samples,len(ds))
    n_pos = int(n_samples * positive_ratio)
    n_flipped = n_samples - n_pos

    ds = ds.shuffle(seed=seed)
    if select=='fixed':
            #fix the positive lable data as half of the total data, and then use 1-positve part of the rest data as the randomly flipped data
            ds_subset = ds.select(range(n_samples))

            ds_pos = ds_subset.select(range(int(1/2*n_samples)))
            n_flipped=int(1/2*n_samples*(1-positive_ratio))
            ds_flipped = ds_subset.select(range(int(1/2*n_samples),int(1/2*n_samples)+n_flipped))
            random_flips = torch.bernoulli(torch.tensor([0.5]*n_flipped)).int().tolist()
            ds_flipped = ds_flipped.map(lambda x, idx: {"label": random_flips[idx]}, with_indices=True)
            return concatenate_datasets([ds_pos, ds_flipped])
    elif positive_ratio == 1.0:
        # All positive, just select the first n_samples
        ds_pos = ds.select(range(n_samples))
        return ds_pos

    elif positive_ratio == 0.0:
        # All randomly flipped
        ds_flipped = ds.select(range(n_samples))
        random_flips = torch.bernoulli(torch.tensor([0.5]*n_samples)).int().tolist()
        ds_flipped = ds_flipped.map(lambda x, idx: {"label": random_flips[idx]}, with_indices=True)
        return ds_flipped

    else:
        # Partial ratio
        if select=='uniform':
            ds_pos = ds.select(range(n_pos))
            ds_flipped = ds.select(range(n_pos, n_pos + n_flipped))

            random_flips = torch.bernoulli(torch.tensor([0.5]*n_flipped)).int().tolist()
            ds_flipped = ds_flipped.map(lambda x, idx: {"label": random_flips[idx]}, with_indices=True)
        elif select=='diff':
            # "most": pick the top n_flipped samples closest to 0.5 and flip them
            # First select the total n_samples
            ds_subset = ds.select(range(n_samples))

            # Compute difficulty as the absolute difference from 0.5
            ds_subset = ds_subset.map(lambda x: {"distance": abs(x["score_gap"])})

            # Sort by 'distance' to get most difficult samples at the top
            ds_subset = ds_subset.sort("distance")

            # ds_flipped: first n_flipped (most difficult)
            ds_flipped = ds_subset.select(range(n_flipped))
            # ds_pos: next n_pos
            ds_pos = ds_subset.select(range(n_flipped, n_flipped + n_pos))

            random_flips = torch.bernoulli(torch.tensor([0.5]*n_flipped)).int().tolist()
            ds_flipped = ds_flipped.map(lambda x, idx: {"label": random_flips[idx]}, with_indices=True)

        return concatenate_datasets([ds_pos, ds_flipped])


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = []
        merged_features = []
        scores_j = []
        scores_k = []
        ids = []
        if "label" in features[0]:
            extra_labels = []
        for feature in features:
            # by sequentially appending j and k samples,
            # we make sure positive and negative samples are paired; then we apply padding.
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            scores.append(feature["score_j"])
            scores_j.append(feature["score_j"])
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
            scores.append(feature["score_k"])
            scores_k.append(feature["score_k"])
            if "label" in feature:
                extra_labels.append(feature["label"])
            ids.append(feature["__ID__"])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": torch.tensor(scores, dtype=torch.float),
            "labels_j": torch.tensor(scores_j, dtype=torch.float),
            "labels_k": torch.tensor(scores_k, dtype=torch.float),
            "ids": torch.tensor(ids, dtype=torch.long),
            "return_loss": True,
        }
        if "label" in features[0]:  # used for loss computation
            batch["labels_extra"] = torch.tensor(
                extra_labels, dtype=torch.float
            )

        return batch