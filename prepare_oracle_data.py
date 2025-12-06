from __future__ import annotations

from pathlib import Path
from typing import *

import click
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def infer_rm_score_formatted(
    ds: Dataset,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B",
):
    ds_processed = []

    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # specified by official
        num_labels=1,
    )

    dummy_device = "cuda:0"

    def _list_to_plain_text(messages):
        if isinstance(messages, str):
            return messages
        if isinstance(messages, dict):
            role = messages.get("role")
            content = messages.get("content")
            if isinstance(content, (list, dict)):
                content = _list_to_plain_text(content)
            if role:
                return f"{role}: {content}"
            return str(content)
        if isinstance(messages, list):
            buffer = []
            for msg in messages:
                buffer.append(_list_to_plain_text(msg))
            return "\n".join(buffer)
        return str(messages)

    use_chat_template = bool(getattr(rm_tokenizer, "chat_template", None))

    max_length = getattr(rm_tokenizer, "model_max_length", 4096)
    if (
        max_length is None
        or not isinstance(max_length, int)
        or max_length <= 0
        or max_length > 32768
    ):
        max_length = 4096

    def _format_inputs(sample):
        if use_chat_template:
            plain_text = rm_tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            plain_text = _list_to_plain_text(sample)
        return rm_tokenizer(
            plain_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(dummy_device)

    for i in tqdm(range(len(ds)), desc=f"RM inference with {model_name}"):
        pos_input = _format_inputs(ds[i]["chosen"])
        neg_input = _format_inputs(ds[i]["rejected"])

        with torch.no_grad():
            pos_out = rm(**pos_input).logits[0][0].item()
            neg_out = rm(**neg_input).logits[0][0].item()
            preference_score = torch.sigmoid(
                torch.tensor(pos_out - neg_out, dtype=torch.float32)
            ).item()

        sample = ds[i]
        sample.update(
            {
                "chosen_score": pos_out,
                "rejected_score": neg_out,
                "preference_score": preference_score,
            }
        )
        ds_processed.append(sample)

    ds_rslt = Dataset.from_list(ds_processed)
    return ds_rslt

def build(
    built_from: Literal["local", "hub"],
    data_path: str = "RLHFlow/UltraFeedback-preference-standard",
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B",
    save_as_path: str | None = None,
):
    target_path = save_as_path or data_path
    if built_from == "local":
        ds = load_dataset(data_path, name="default", split="train")
        if model_name:
            ds = infer_rm_score_formatted(
                ds,
                model_name=model_name,
            )
        target_dir = Path("statdata") / target_path
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(target_dir.as_posix())

    elif built_from == "hub":
        if data_path == "Skywork-Reward-Preference-80K-v0.2":
            ds = load_dataset(
                "BigCatc/Skywork-Reward-Preference-80K-v0.2-ordinal", split="train"
            )
        else:
            ds = load_dataset(
                data_path, split="train"
            )
        target_dir = Path("statdata") / target_path
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(target_dir.as_posix())

def main():
    data_names=[
        #'RLHFlow/UltraFeedback-preference-standard',
        "RLHFlow/Helpsteer-preference-standard",
        #"Skywork-Reward-Preference-80K-v0.2",
        "RLHFlow/PKU-SafeRLHF-30K-standard",
    ]
    model_names=[#'Skywork/Skywork-Reward-Llama-3.1-8B',
                 'Skywork/Skywork-Reward-Llama-3.1-8B',
                 #'Skywork-Reward-Gemma-2-27B-v0.2',
                 'PKU-Alignment/beaver-7b-v2.0-reward']
    build_froms=[#'local',
                 'local',
                 #'hub',
                 'local']
    save_as_paths=[
        #'RLHFlow/UltraFeedback-preference-standard',
        'RLHFlow/Helpsteer-preference-standard',
        #'prefer_skywork_Skywork/Skywork-Reward-Gemma-2-27B-v0.2',
        'RLHFlow/PKU-preference-standard',
    ]
    for data_name, model_name, built_from, save_as_path in zip(data_names, model_names, build_froms, save_as_paths):
        build(built_from=built_from,
              data_path=data_name,
              model_name=model_name,
              save_as_path=save_as_path)


if __name__ == "__main__":
    main()
