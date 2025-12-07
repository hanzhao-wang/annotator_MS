from __future__ import annotations

"""
Downstream experiment that links contract design -> annotator quality (eta) -> reward-model training data.

Workflow:
1) For each dataset (PKU / HelpSteer), compute the optimal annotation quality under
   - monitoring_type: self-consistency vs expert-based (same monitoring budget n_monitor for fairness)
   - contract_type: linear vs binary
   using a simple principal-agent solver with convex effort cost and exponential utility.
2) Simulate annotated data by corrupting the oracle/preference scores according to the chosen eta.
3) Emit a training config that points run.py to the simulated data. Optionally trigger training.

The script is intentionally dependency-light: only numpy, click, datasets are required.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import click
import numpy as np
from datasets import Dataset, load_from_disk
from numpy.random import default_rng
from scipy.stats import binom


@dataclass
class DatasetConfig:
    name: str
    base_config: Path
    cost_scale: float
    mu_scale: float
    delta: float = 0.0  # self-monitoring penalty on effort -> mean accuracy


# Default setup for the two datasets 
DATASETS: Dict[str, DatasetConfig] = {
    "helpsteer": DatasetConfig(
        name="Helpsteer",
        base_config=Path("paper_experiment_configs/llama-Helpsteer.json"),
        cost_scale=0.18,
        mu_scale=1.0,
        delta=0.02,
    ),
    "pku": DatasetConfig(
        name="PKU",
        base_config=Path("paper_experiment_configs/llama-PKU.json"),
        cost_scale=0.18,
        mu_scale=1,
        delta=0.02,
    ),
}


def risk_adjusted_payment(x: np.ndarray, exp_para: float) -> np.ndarray:
    return exp_para-exp_para*np.exp(-exp_para*x)


def binary_expected_utility(
    pass_prob: np.ndarray, success_payment: float, failure_payment: float, exp_para: float
) -> np.ndarray:
    """
    Compute the expected CARA utility for a binary reward that pays
    `success_payment` with probability `pass_prob` and `failure_payment` otherwise.
    The probability is clamped to [0, 1] so downstream contracts remain valid
    even when floating-point noise pushes the survival function slightly outside
    the unit interval.
    """
    prob_array = np.clip(np.asarray(pass_prob, dtype=float), 0.0, 1.0)
    success_utility = risk_adjusted_payment(
        np.full_like(prob_array, success_payment, dtype=float), exp_para
    )
    failure_utility = risk_adjusted_payment(
        np.full_like(prob_array, failure_payment, dtype=float), exp_para
    )
    return prob_array * success_utility + (1 - prob_array) * failure_utility


def mean_annotation_quality(
    effort: np.ndarray,
    monitor_type: str,
    pref_mean: float,
    delta: float,
) -> np.ndarray:
    """
    Map annotator effort r -> expected accuracy/quality eta under the monitoring method.
    Self-monitoring rewards effort directly; expert monitoring depends on underlying task difficulty (pref_mean).
    """
    if monitor_type == "self":
        return (1 + effort * (1 - delta)) / 2
    if monitor_type == "expert":
        return effort * (pref_mean - 0.5) + 0.5
    raise ValueError(f"Unknown monitor_type: {monitor_type}")


def solve_contract(
    preference_scores: np.ndarray,
    monitor_type: str,
    contract_type: str,
    effort_space: Iterable[float],
    c0_grid: Iterable[float],
    c1_grid: Iterable[float],
    c2_grid: Iterable[float],
    cost_scale: float,
    mu_scale: float,
    n_monitor: int,
    delta: float,
    reservation: float = 0.0,
    exp_para: float = 1,
) -> Tuple[float, float, Tuple[float, float, float], float, float]:
    """
    Simple principal-agent search:
    - Principal proposes (c0, c1, c2)
    - Agent best-responds with effort r in effort_space
    - Principal picks contract maximizing utility subject to agent participation.

    Returns:
        eta_star (mean annotation quality), effort_star, (c0, c1, c2),
        agent_utility, principal_utility
    """
    pref_mean = float(np.mean(np.where(preference_scores < 0.5, 1 - preference_scores, preference_scores)))

    # (eta, effort, contract, agent_u, principal_u)
    best_tuple = (0.0, 0.0, (0.0, 0.0, 0.0), float("-inf"), float("-inf"))

    efforts = np.array(list(effort_space))
    mean_values = mean_annotation_quality(efforts, monitor_type, pref_mean, delta)
    cost = cost_scale * efforts ** 2
    mu_values = 0.5* efforts**0.8  # downstream benefit 

    for c0 in c0_grid:
        for c1 in c1_grid:
            for c2 in c2_grid:
                if contract_type == "linear":
                    payments = mean_values * c1 + c2
                    g_pay = risk_adjusted_payment(payments, exp_para)
                    expected_payment = payments
                elif contract_type == "binary":
                    # probability that empirical mean over n_monitor exceeds threshold c0
                    pass_prob = binom.sf(c0 * n_monitor - 1, n_monitor, mean_values)
                    pass_prob = np.clip(pass_prob, 0.0, 1.0)
                    payments = pass_prob * c1 + c2
                    g_pay = binary_expected_utility(pass_prob, c1 + c2, c2, exp_para)
                    expected_payment = payments
                else:
                    raise ValueError(f"Unknown contract_type: {contract_type}")

                agent_u = g_pay - cost
                best_effort_idx = int(np.argmax(agent_u))
                best_effort = efforts[best_effort_idx]
                best_agent_u = agent_u[best_effort_idx]

                if best_agent_u < reservation:
                    continue

                principal_u = mu_values[best_effort_idx] - expected_payment[best_effort_idx]
                if principal_u > best_tuple[4]:
                    effort_star = float(best_effort)
                    eta_star = float(mean_values[best_effort_idx])  # actual pass probability/quality
                    best_tuple = (
                        eta_star,
                        effort_star,
                        (c0, c1, c2),
                        float(best_agent_u),
                        float(principal_u),
                    )

    if best_tuple[4] == float("-inf"):
        raise RuntimeError("No feasible contract found for the given grids and reservation utility.")
    return best_tuple


def corrupt_dataset(ds: Dataset, eta: float, seed: int) -> Dataset:
    """
    Simulate noisy annotations: preference_score is pulled towards 0.5 and labels may flip with prob (1-eta).
    """
    rng = default_rng(seed)

    def _mutate(sample):
        score = float(sample["preference_score"])
        sample["preference_score"] = eta * score + (1 - eta) * 0.5
        if "label" in sample:
            if rng.random() > eta:
                sample["label"] = int(1 - int(sample["label"]))
        return sample

    return ds.map(_mutate, desc=f"Simulating annotations (eta={eta:.3f})")


def save_config(base_config: Path, train_dir: Path, suffix: str, eval_dir: Path | None = None) -> Path:
    cfg = json.loads(base_config.read_text())
    cfg["train_set_path"] = str(train_dir)
    if eval_dir is not None:
        cfg["eval_set_path"] = str(eval_dir)
    elif "eval_set_path" not in cfg:
        cfg["eval_set_path"] = str(train_dir)
    output_root = Path(cfg.get("output_path", "./bt_models"))
    cfg["output_path"] = str(output_root / suffix)

    out_path = base_config.with_name(f"{base_config.stem}-{suffix}.json")
    out_path.write_text(json.dumps(cfg, indent=2))
    return out_path


def run_training(config_path: Path):
    subprocess.run(["python", "run.py", str(config_path)], check=True)


@click.command()
@click.option(
    "--dataset",
    "datasets_",
    multiple=True,
    type=click.Choice(list(DATASETS.keys()), case_sensitive=False),
    default=["helpsteer", "pku"],
    help="Datasets to run (helpsteer, pku).",
)
@click.option(
    "--monitor",
    "monitor_types",
    multiple=True,
    type=click.Choice(["self", "expert"], case_sensitive=False),
    default=["self", "expert"],
    help="Monitoring modes to compare.",
)
@click.option(
    "--contract",
    "contract_types",
    multiple=True,
    type=click.Choice(["linear", "binary"], case_sensitive=False),
    default=["linear", "binary"],
    help="Contract families to compare.",
)
@click.option(
    "--n-monitor",
    default=None,
    type=int,
    help="Number of self-check/expert audited samples (fairly matched). Defaults to full dataset size.",
)
@click.option("--seed", default=42, show_default=True, help="Random seed for simulation noise.")
@click.option("--train", "do_train", is_flag=True, help="If set, launch run.py for each generated config.")
def main(
    datasets_: Tuple[str, ...],
    monitor_types: Tuple[str, ...],
    contract_types: Tuple[str, ...],
    n_monitor: int,
    seed: int,
    do_train: bool,
):
    effort_space = np.linspace(0.1, 0.99, 50)


    # First pass: solve contracts and collect summaries
    summaries = []
    original_cache: Dict[str, Dataset] = {}
    for ds_key in datasets_:
        cfg = DATASETS[ds_key.lower()]
        if not cfg.base_config.exists():
            raise FileNotFoundError(f"Base config not found: {cfg.base_config}")

        original_path = Path("statdata") / f"RLHFlow/{cfg.name}-preference-standard"
        if not original_path.exists():
            raise FileNotFoundError(f"Oracle data missing at {original_path}. Run prepare_oracle_data.py first.")
        original_ds = load_from_disk(original_path)
        original_cache[ds_key.lower()] = original_ds
        local_n_monitor = n_monitor or len(original_ds)
        preference_scores = np.array(original_ds["preference_score"], dtype=float)

        for monitor in monitor_types:
            for contract in contract_types:
                if contract=='linear':
                        c0_grid=np.arange(0,1,1)
                        c1_grid=np.arange(0,10,0.05)
                        c2_grid=np.arange(-10,10,0.05)
                       
                elif contract=='binary':
                        c0_grid=np.arange(0,1.02,0.02)
                        c1_grid=np.arange(0,10,0.05)
                        c2_grid=np.arange(-10,10,0.05)
                
                eta, effort, contract_tuple, agent_u, principal_u = solve_contract(
                    preference_scores=preference_scores,
                    monitor_type=monitor,
                    contract_type=contract,
                    effort_space=effort_space,
                    c0_grid=c0_grid,
                    c1_grid=c1_grid,
                    c2_grid=c2_grid,
                    cost_scale=cfg.cost_scale,
                    mu_scale=cfg.mu_scale,
                    n_monitor=local_n_monitor,
                    delta=cfg.delta,
                )
                summaries.append(
                    {
                        "ds_key": ds_key.lower(),
                        "ds_name": cfg.name,
                        "monitor": monitor,
                        "contract": contract,
                        "eta": eta,
                        "effort": effort,
                        "agent_u": agent_u,
                        "principal_u": principal_u,
                        "contract_tuple": contract_tuple,
                        "cfg": cfg,
                    }
                )

    # Print summary of optimal etas before any corruption/training
    print("\n=== Optimal contracts (used for corruption) ===")
    for entry in summaries:
        print(
            f"[{entry['ds_name']}] monitor={entry['monitor']:<6} "
            f"contract={entry['contract']:<6} eta={entry['eta']:.3f} "
            f"effort={entry['effort']:.3f} agent_u={entry['agent_u']:.3f} "
            f"principal_u={entry['principal_u']:.3f} contract={entry['contract_tuple']}"
        )
    print("=== End summary ===\n")

    # Second pass: corrupt data, save configs, and optionally train
    for entry in summaries:
        cfg: DatasetConfig = entry["cfg"]
        original_ds = original_cache[entry["ds_key"]]
        eta = entry["eta"]
        monitor = entry["monitor"]
        contract = entry["contract"]

        split_ds = original_ds.train_test_split(test_size=0.2, seed=seed)
        train_ds = split_ds["train"]
        eval_ds = split_ds["test"]

        sim_train_ds = corrupt_dataset(train_ds, eta=eta, seed=seed)

        sim_base = Path("statdata") / "simulated" / f"{cfg.name}-{monitor}-{contract}-eta{eta:.2f}"
        sim_train_dir = sim_base / "train"
        sim_eval_dir = sim_base / "eval"
        sim_train_dir.parent.mkdir(parents=True, exist_ok=True)
        sim_train_ds.save_to_disk(sim_train_dir.as_posix())
        eval_ds.save_to_disk(sim_eval_dir.as_posix())

        suffix = f"{cfg.name}-{monitor}-{contract}-eta{eta:.2f}"
        config_path = save_config(cfg.base_config, sim_train_dir, suffix, eval_dir=sim_eval_dir)

        print(
            f"[{cfg.name}] monitor={monitor} contract={contract} -> eta={eta:.3f} (effort={entry['effort']:.3f}), "
            f"agent_u={entry['agent_u']:.3f}, principal_u={entry['principal_u']:.3f}, "
            f"contract={entry['contract_tuple']}; config={config_path}"
        )

        if do_train:
            run_training(config_path)


if __name__ == "__main__":
    main()
