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


def compute_CE_oracle(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    pos_true_scores = eval_pred.label_ids[0]
    neg_true_scores = eval_pred.label_ids[1]

    # ! if containing nan in predictions_scores, print warning
    if (
        np.isnan(pos_predictions_scores).any()
        or np.isnan(neg_predictions_scores).any()
    ):
        print("[Eval] Warning: nan in predictions_scores")

    # ! we make an extra masking step to mask all the possbile nan values in predictions_scores
    pos_predictions_scores = np.nan_to_num(pos_predictions_scores)
    neg_predictions_scores = np.nan_to_num(neg_predictions_scores)

    oracle_CE_loss = (
        -nn.functional.logsigmoid(
            torch.tensor(
                pos_predictions_scores - neg_predictions_scores,
                dtype=torch.float32,
            )
        )
        * nn.functional.sigmoid(
            torch.tensor(pos_true_scores - neg_true_scores, dtype=torch.float32)
        )
        - nn.functional.logsigmoid(
            torch.tensor(
                neg_predictions_scores - pos_predictions_scores,
                dtype=torch.float32,
            )
        )
        * (
            1
            - nn.functional.sigmoid(
                torch.tensor(
                    pos_true_scores - neg_true_scores, dtype=torch.float32
                )
            )
        )
    ).mean()

    accuracy = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )

    result["oracle_CE_loss"] = oracle_CE_loss.item()
    result["binary_accuracy"] = accuracy

    return result


def compute_ML_oracle(eval_pred, delta=1):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    pos_true_scores = eval_pred.label_ids[0]
    neg_true_scores = eval_pred.label_ids[1]

    # ! if containing nan in predictions_scores, print warning
    if (
        np.isnan(pos_predictions_scores).any()
        or np.isnan(neg_predictions_scores).any()
    ):
        print("[Eval] Warning: nan in predictions_scores")

    # ! we make an extra masking step to mask all the possbile nan values in predictions_scores
    pos_predictions_scores = np.nan_to_num(pos_predictions_scores)
    neg_predictions_scores = np.nan_to_num(neg_predictions_scores)

    true_prob = nn.functional.sigmoid(
        torch.tensor(pos_true_scores - neg_true_scores, dtype=torch.float32)
    )
    oracle_ML_loss = (
        nn.functional.relu(
            torch.tensor(
                delta / 2 - (pos_predictions_scores - neg_predictions_scores),
                dtype=torch.float32,
            )
        )
        * true_prob
        + nn.functional.relu(
            torch.tensor(
                delta / 2 + (pos_predictions_scores - neg_predictions_scores),
                dtype=torch.float32,
            )
        )
        * (1 - true_prob)
    ).mean()

    accuracy = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )

    result["oracle_ML_loss"] = oracle_ML_loss.item()
    result["binary_accuracy"] = accuracy

    return result


class RewardTrainer(Trainer):
    def __init__(self, *args, delta=1.0, tie_thrsd=0.5, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
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

    def compute_loss(self, model, inputs, return_outputs=False):
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

    def compute_loss(self, model, inputs, return_outputs=False):
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

    def compute_loss(self, model, inputs, return_outputs=False):
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
