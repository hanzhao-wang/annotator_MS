from __future__ import annotations

import os
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import *

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from accelerate import Accelerator
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import Trainer, TrainerCallback
from transformers.optimization import get_scheduler


def softmax_np(x: List[float], temperature: float = 1.0):
    x_np = np.array(x)
    scaled_x = x_np / temperature
    exp_x = np.exp(scaled_x - np.max(scaled_x))
    return exp_x / np.sum(exp_x)


def compute_metrics(eval_pred):
    result = {}

    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]

    result["accuracy"] = np.sum(
        pos_predictions_scores > neg_predictions_scores
    ) / len(pos_predictions_scores)
    return result


# ! the nan values are handled for BTT model.


def _extract_pairwise_logits_and_labels(eval_pred):
    """
    Trainer returns flattened logits for j/k pairs; regroup them and align with labels.
    Works with either tuple(predictions, ...) or raw ndarray predictions.
    """
    logits = eval_pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = np.asarray(logits, dtype=float).squeeze()
    if logits.ndim > 1:
        logits = logits.reshape(-1)
    if logits.size % 2 != 0:
        # Drop the last stray logit to keep pairs aligned instead of failing hard.
        logits = logits[:-1]

    labels = eval_pred.label_ids
    if isinstance(labels, (tuple, list)) and len(labels) >= 2:
        labels_j = np.asarray(labels[0], dtype=float).reshape(-1)
        labels_k = np.asarray(labels[1], dtype=float).reshape(-1)
    else:
        labels_arr = np.asarray(labels, dtype=float)
        if labels_arr.ndim == 2 and labels_arr.shape[1] >= 2:
            labels_j, labels_k = labels_arr[:, 0], labels_arr[:, 1]
        else:
            labels_j = labels_arr[0::2]
            labels_k = labels_arr[1::2]

    pair_len = min(len(labels_j), len(labels_k), logits.size // 2)
    rewards_j = logits[0 : 2 * pair_len : 2]
    rewards_k = logits[1 : 2 * pair_len : 2]
    labels_j = labels_j[:pair_len]
    labels_k = labels_k[:pair_len]

    # mask out nans to keep metrics finite
    rewards_j = np.nan_to_num(rewards_j)
    rewards_k = np.nan_to_num(rewards_k)
    labels_j = np.nan_to_num(labels_j)
    labels_k = np.nan_to_num(labels_k)
    return rewards_j, rewards_k, labels_j, labels_k


def compute_CE_oracle(eval_pred):
    result = {}
    rewards_j, rewards_k, scores_j, scores_k = _extract_pairwise_logits_and_labels(
        eval_pred
    )
    score_diff = torch.tensor(scores_j - scores_k, dtype=torch.float32)
    pred_diff = torch.tensor(rewards_j - rewards_k, dtype=torch.float32)

    target_prob = nn.functional.sigmoid(score_diff)
    oracle_CE_loss = (
        -nn.functional.logsigmoid(pred_diff) * target_prob
        - nn.functional.logsigmoid(-pred_diff) * (1 - target_prob)
    ).mean()

    accuracy = float(np.mean(rewards_j > rewards_k))

    result["oracle_CE_loss"] = oracle_CE_loss.item()
    result["binary_accuracy"] = accuracy

    return result


def compute_ML_oracle(eval_pred, delta=1):
    result = {}
    rewards_j, rewards_k, scores_j, scores_k = _extract_pairwise_logits_and_labels(
        eval_pred
    )

    true_prob = nn.functional.sigmoid(
        torch.tensor(scores_j - scores_k, dtype=torch.float32)
    )
    score_gap = torch.tensor(rewards_j - rewards_k, dtype=torch.float32)

    oracle_ML_loss = (
        nn.functional.relu(delta / 2 - score_gap) * true_prob
        + nn.functional.relu(delta / 2 + score_gap) * (1 - true_prob)
    ).mean()

    accuracy = float(np.mean(rewards_j > rewards_k))

    result["oracle_ML_loss"] = oracle_ML_loss.item()
    result["binary_accuracy"] = accuracy

    return result


class RewardTrainer(Trainer):
    def __init__(self, *args, delta=1.0, tie_thrsd=0.5, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: int | None = None,  # HF passes this in newer versions
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid((rewards_j - rewards_k)).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class RewardTrainerWithOracleCE(Trainer):
    def __init__(self, *args, delta=1.0, tie_thrsd=0.5, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: int | None = None,
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        scores_j = inputs["labels"][jidx]
        scores_k = inputs["labels"][kidx]
        scores_diff = scores_j - scores_k
        labels_p = inputs["labels_extra"]
        loss = (
            -nn.functional.logsigmoid((rewards_j - rewards_k)) * labels_p
            - nn.functional.logsigmoid((rewards_k - rewards_j)) * (1 - labels_p)
        ).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class RewardTrainerWithRingeMargin(Trainer):
    def __init__(self, *args, delta=1.0, tie_thrsd=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: int | None = None,
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        scores_j = inputs["labels"][jidx]
        scores_k = inputs["labels"][kidx]
        scores_diff = scores_j - scores_k
        labels_p = inputs["labels_extra"]

        # we consider the sum of two hinge loss
        # the first punishes the negative predictions < -delta / 2 with weight labels_p
        # the second punishes the positive predictions > delta / 2 with weight (1 - labels_p)

        loss = (
            nn.functional.relu(self.delta / 2 - (rewards_j - rewards_k))
            * labels_p
            + nn.functional.relu(self.delta / 2 + (rewards_j - rewards_k))
            * (1 - labels_p)
        ).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class BTTRewardTrainer(Trainer):
    def __init__(self, *args, delta=2.0, tie_thrsd=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = delta  # ! note here we use `delta` as the dummy encoding parameter for theta in BTT

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: int | None = None,
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        scores_j = inputs["labels"][jidx]
        scores_k = inputs["labels"][kidx]
        # we find preference / tied pairs based on scores_j and scores_k
        scores_diff = scores_j - scores_k
        # ! we only use interpolated setting, so the tied pairs should have the same scores
        # ! remember to convert the boolean mask to float
        tied_mask = (scores_diff == 0).view(-1).float()
        j_greater_mask = (scores_diff > 0).view(-1).float()
        k_greater_mask = (scores_diff < 0).view(-1).float()

        # ! print warning if there are nan in rewards
        if torch.isnan(rewards_j).any() or torch.isnan(rewards_k).any():
            print("[Train] Warning: nan in rewards")

        # ! we make an extra masking step to mask all the possbile nan values in rewards
        rewards_j = torch.nan_to_num(rewards_j)
        rewards_k = torch.nan_to_num(rewards_k)

        # ! we also clamp the range of rewards to avoid overflow
        rewards_j = torch.clamp(rewards_j, min=-30, max=30)
        rewards_k = torch.clamp(rewards_k, min=-30, max=30)

        exp_rewards_j = torch.exp(rewards_j)
        exp_rewards_k = torch.exp(rewards_k)
        prob_j_greater = exp_rewards_j / (
            exp_rewards_j + self.theta * exp_rewards_k
        )
        prob_k_greater = exp_rewards_k / (
            self.theta * exp_rewards_j + exp_rewards_k
        )
        prob_tied = (
            (self.theta**2 - 1)
            * (exp_rewards_j * exp_rewards_k)
            / (
                (exp_rewards_j + self.theta * exp_rewards_k)
                * (self.theta * exp_rewards_j + exp_rewards_k)
            )
        )
        # now we use mask to select the corresponding pairs
        eps = 1e-7
        loss = -(
            (tied_mask * torch.log(prob_tied + eps)).sum()
            + (j_greater_mask * torch.log(prob_j_greater + eps)).sum()
            + (k_greater_mask * torch.log(prob_k_greater + eps)).sum()
        ) / (bsz / 2)

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
