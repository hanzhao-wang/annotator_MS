from __future__ import annotations

"""
Delta-sweep experiments for Section 4.

This script extends the contract-vs-monitoring simulation to a wider range of
disagreement probabilities (delta) so we can show how sensitive the principal's
utility gap is to the choice of delta, addressing the reviewer concern that the
original Figure 3 used a single small value.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from Section4 import calibrate_data, plot_func, set_random_seed, solver
from utils.data import get_data


@dataclass
class DatasetSpec:
    name: str
    config_path: Path
    calibration_bins: int | None = None
    n_override: Sequence[int] | None = None


DATASETS: Dict[str, DatasetSpec] = {
    "pku": DatasetSpec(
        name="PKU",
        config_path=Path("paper_experiment_configs/llama-PKU.json"),
        calibration_bins=30,
        n_override=np.arange(1, 202, 10).tolist(),
    ),
    "sky": DatasetSpec(
        name="sky",
        config_path=Path("paper_experiment_configs/llama-sky.json"),
        calibration_bins=30,
    ),
    "helpsteer": DatasetSpec(
        name="Helpsteer",
        config_path=Path("paper_experiment_configs/llama-Helpsteer.json"),
    ),
    "ultra": DatasetSpec(
        name="Ultra",
        config_path=Path("paper_experiment_configs/llama-Ultra.json"),
    ),
}

DEFAULT_DELTAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def _contract_space(contract_type: str) -> List[List[float]]:
    if contract_type == "linear":
        c0 = np.arange(0, 1, 1).tolist()
        c1 = np.arange(0, 10, 0.05).tolist()
        c2 = np.arange(-10, 10, 0.05).tolist()
        return [c0, c1, c2]
    if contract_type == "binary":
        c0 = np.arange(0, 1.02, 0.02).tolist()
        c1 = np.arange(0, 10, 0.05).tolist()
        c2 = np.arange(-10, 10, 0.05).tolist()
        return [c0, c1, c2]
    raise ValueError(f"Unsupported contract type: {contract_type}")


def _load_preference_scores(spec: DatasetSpec) -> np.ndarray:
    train_dataset, eval_dataset = get_data(script_config_path=str(spec.config_path))
    if spec.calibration_bins:
        return calibrate_data(train_dataset, eval_dataset, n_bins=spec.calibration_bins)
    return np.array(eval_dataset["preference_score"])


def _run_single_dataset(
    spec: DatasetSpec,
    delta: float,
    monitor_types: Sequence[str],
    contract_types: Sequence[str],
    n_list: Sequence[int],
    effort_space: Sequence[float],
    G_func,
    E_func,
    mu_func,
    exp_para: float,
    U_0: float,
) -> Tuple[Dict[Tuple, Dict[str, float]], List[Dict[str, float]]]:
    preference_scores = _load_preference_scores(spec)
    results: Dict[Tuple, Dict[str, float]] = {}
    records: List[Dict[str, float]] = []

    for monitor_type in monitor_types:
        for contract_type in contract_types:
            contract_space = _contract_space(contract_type)
            for n in n_list:
                solver_instance = solver(
                    preference_scores=preference_scores,
                    effort_space=effort_space,
                    contract_space=contract_space,
                    G_func=G_func,
                    E_func=E_func,
                    mu_func=mu_func,
                    n=n,
                    U_0=U_0,
                    delta=delta,
                    contract_type=contract_type,
                    monitor_type=monitor_type,
                )
                solver_instance.Util_compute_exact(exp_para)

                fb_contract, fb_principal, fb_effort, fb_agent = solver_instance.FB_solve()
                fb_c0, fb_c1, fb_c2 = fb_contract if isinstance(fb_contract, tuple) else (None, None, None)
                results[(spec.name, monitor_type, contract_type, n, "FB")] = {
                    "best_c0": fb_c0,
                    "best_c1": fb_c1,
                    "best_c2": fb_c2,
                    "best_principal_util": fb_principal,
                    "best_effort": fb_effort,
                    "agent_util": fb_agent,
                }
                records.append(
                    {
                        "dataset": spec.name,
                        "delta": delta,
                        "monitor": monitor_type,
                        "contract": contract_type,
                        "n": n,
                        "regime": "FB",
                        "principal_utility": fb_principal,
                        "agent_utility": fb_agent,
                        "effort": fb_effort,
                        "c0": fb_c0,
                        "c1": fb_c1,
                        "c2": fb_c2,
                        "exp_para": exp_para,
                        "U_0": U_0,
                    }
                )

                e_star = fb_effort
                sbt_contract, sbt_principal, sbt_effort, sbt_agent = solver_instance.SB_solve_tilde(e_star)
                sbt_c0, sbt_c1, sbt_c2 = sbt_contract if isinstance(sbt_contract, tuple) else (None, None, None)
                results[(spec.name, monitor_type, contract_type, n, "SB_tilde")] = {
                    "best_c0": sbt_c0,
                    "best_c1": sbt_c1,
                    "best_c2": sbt_c2,
                    "best_principal_util": sbt_principal,
                    "best_effort": sbt_effort,
                    "agent_util": sbt_agent,
                }
                records.append(
                    {
                        "dataset": spec.name,
                        "delta": delta,
                        "monitor": monitor_type,
                        "contract": contract_type,
                        "n": n,
                        "regime": "SB_tilde",
                        "principal_utility": sbt_principal,
                        "agent_utility": sbt_agent,
                        "effort": sbt_effort,
                        "c0": sbt_c0,
                        "c1": sbt_c1,
                        "c2": sbt_c2,
                        "exp_para": exp_para,
                        "U_0": U_0,
                    }
                )

                sb_contract, sb_principal, sb_effort, sb_agent = solver_instance.SB_solve()
                sb_c0, sb_c1, sb_c2 = sb_contract if isinstance(sb_contract, tuple) else (None, None, None)
                results[(spec.name, monitor_type, contract_type, n, "SB")] = {
                    "best_c0": sb_c0,
                    "best_c1": sb_c1,
                    "best_c2": sb_c2,
                    "best_principal_util": sb_principal,
                    "best_effort": sb_effort,
                    "agent_util": sb_agent,
                }
                records.append(
                    {
                        "dataset": spec.name,
                        "delta": delta,
                        "monitor": monitor_type,
                        "contract": contract_type,
                        "n": n,
                        "regime": "SB",
                        "principal_utility": sb_principal,
                        "agent_utility": sb_agent,
                        "effort": sb_effort,
                        "c0": sb_c0,
                        "c1": sb_c1,
                        "c2": sb_c2,
                        "exp_para": exp_para,
                        "U_0": U_0,
                    }
                )
    return results, records


@click.command()
@click.option(
    "--delta",
    "deltas",
    multiple=True,
    type=float,
    default=DEFAULT_DELTAS,
    show_default=True,
    help="Disagreement probabilities to sweep for self-monitoring.",
)
@click.option(
    "--dataset",
    "datasets_",
    multiple=True,
    type=click.Choice(list(DATASETS.keys()), case_sensitive=False),
    default=list(DATASETS.keys()),
    show_default=True,
    help="Datasets/configs to include in the sweep.",
)
@click.option(
    "--monitor",
    "monitor_types",
    multiple=True,
    type=click.Choice(["self", "expert"], case_sensitive=False),
    default=["self", "expert"],
    show_default=True,
    help="Monitoring modes to compare.",
)
@click.option(
    "--contract",
    "contract_types",
    multiple=True,
    type=click.Choice(["linear", "binary"], case_sensitive=False),
    default=["linear", "binary"],
    show_default=True,
    help="Contract families to compare.",
)
@click.option(
    "--summary-csv",
    type=click.Path(dir_okay=False),
    default="fig_contract/delta_sweep_summary.csv",
    show_default=True,
    help="Where to dump a tidy CSV of all principal/agent utilities.",
)
@click.option(
    "--skip-plots",
    is_flag=True,
    default=False,
    help="If set, only write the CSV without regenerating EPS plots.",
)
@click.option(
    "--uniform-n-grid",
    is_flag=True,
    default=False,
    help="Use the base n grid for every dataset (ignoring PKU's larger default).",
)
def main(
    deltas: Tuple[float, ...],
    datasets_: Tuple[str, ...],
    monitor_types: Tuple[str, ...],
    contract_types: Tuple[str, ...],
    summary_csv: str,
    skip_plots: bool,
    uniform_n_grid: bool,
):
    set_random_seed(32)
    plt.rcParams["font.size"] = 24

    base_n_list = np.arange(1, 102, 10).tolist()
    effort_space = np.arange(0, 1.01, 0.01).tolist()
    G_func = lambda x: 2 * x ** 0.5  # noqa: E731
    E_func = lambda e: 0.18 * e ** 2  # noqa: E731
    mu_func_map = {
        "low": lambda e: 0.3 * (e ** 0.8),  # noqa: E731
        "high": lambda e: 0.5 * (e ** 0.8),  # noqa: E731
    }
    exp_para_list = [1, 0.5]
    U_0 = 0.0

    dataset_keys = [ds.lower() for ds in datasets_]
    summary_records: List[Dict[str, float]] = []

    logger.info(
        "Starting delta sweep: deltas=%s, datasets=%s, monitor=%s, contract=%s",
        deltas,
        dataset_keys,
        monitor_types,
        contract_types,
    )

    for exp_para in exp_para_list:
        for mu_name, mu_func in mu_func_map.items():
            for delta in deltas:
                logger.info("Running exp_para=%s mu=%s delta=%.3f", exp_para, mu_name, delta)
                for ds_key in dataset_keys:
                    spec = DATASETS[ds_key]
                    n_grid = spec.n_override if (spec.n_override and not uniform_n_grid) else base_n_list
                    results, records = _run_single_dataset(
                        spec=spec,
                        delta=delta,
                        monitor_types=monitor_types,
                        contract_types=contract_types,
                        n_list=n_grid,
                        effort_space=effort_space,
                        G_func=G_func,
                        E_func=E_func,
                        mu_func=mu_func,
                        exp_para=exp_para,
                        U_0=U_0,
                    )
                    summary_records.extend(
                        {
                            **row,
                            "mu_variant": mu_name,
                        }
                        for row in records
                    )

                    if not skip_plots:
                        suffix = f"delta_sweep_exp{exp_para}_mu_{mu_name}"
                        plot_func(
                            monitor_types,
                            contract_types,
                            n_grid,
                            results,
                            spec.name,
                            U_0,
                            delta,
                            plot_mode="gap",
                            save_name=suffix,
                        )
                        plot_func(
                            monitor_types,
                            contract_types,
                            n_grid,
                            results,
                            spec.name,
                            U_0,
                            delta,
                            plot_mode="effort",
                            save_name=suffix,
                        )
                        plot_func(
                            monitor_types,
                            contract_types,
                            n_grid,
                            results,
                            spec.name,
                            U_0,
                            delta,
                            plot_mode="agent_util",
                            save_name=suffix,
                        )

    if summary_records:
        summary_path = Path(summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_records).to_csv(summary_path, index=False)
        logger.info("Wrote %d rows to %s", len(summary_records), summary_path)


if __name__ == "__main__":
    main()
