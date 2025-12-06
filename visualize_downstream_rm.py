from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np

DATASET_ORDER = ["Helpsteer", "PKU"]
CONTRACT_ORDER = ["linear", "binary"]
MONITOR_ORDER = ["self", "expert"]


@dataclass(frozen=True)
class ScenarioMeta:
    dataset: str
    monitor: str
    contract: str
    eta: float | None
    name: str


@dataclass
class RunRecord:
    scenario: ScenarioMeta
    run_dir: Path
    metric_value: float


def parse_scenario_dir(path: Path) -> ScenarioMeta | None:
    """
    Training outputs created by downstream_contract_experiment.py are grouped as:
        bt_models/<Dataset>-<monitor>-<contract>-eta<val>/<run_name>/
    This helper parses the scenario metadata from the parent directory name.
    """
    if not path.is_dir():
        return None

    parts = path.name.split("-")
    if len(parts) < 3:
        return None

    dataset = parts[0]
    monitor = parts[1].lower()
    contract = parts[2].lower()

    eta = None
    for token in parts[3:]:
        if token.startswith("eta"):
            try:
                eta = float(token.replace("eta", ""))
            except ValueError:
                eta = None
            break

    return ScenarioMeta(
        dataset=dataset,
        monitor=monitor,
        contract=contract,
        eta=eta,
        name=path.name,
    )


def extract_metric_from_trainer_state(
    trainer_state_path: Path,
    metric_name: str,
    higher_is_better: bool,
) -> float | None:
    """
    trainer_state.json contains a log_history list with eval metrics emitted during training.
    Grab the best value for the requested metric.
    """
    try:
        data = json.loads(trainer_state_path.read_text())
    except json.JSONDecodeError:
        return None

    history = data.get("log_history", [])
    values: List[float] = []
    for entry in history:
        if metric_name in entry:
            value = entry[metric_name]
            if isinstance(value, (float, int)):
                values.append(float(value))
    if not values:
        return None

    return max(values) if higher_is_better else min(values)


def collect_run_records(
    results_root: Path,
    metric_name: str,
    higher_is_better: bool,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    for scenario_dir in sorted(results_root.iterdir()):
        scenario_meta = parse_scenario_dir(scenario_dir)
        if scenario_meta is None:
            continue

        for trainer_state_path in scenario_dir.rglob("trainer_state.json"):
            metric_value = extract_metric_from_trainer_state(
                trainer_state_path, metric_name, higher_is_better
            )
            if metric_value is None:
                continue
            records.append(
                RunRecord(
                    scenario=scenario_meta,
                    run_dir=trainer_state_path.parent,
                    metric_value=metric_value,
                )
            )
    return records


def summarize_by_group(
    records: Sequence[RunRecord],
) -> Dict[Tuple[str, str, str], List[float]]:
    aggregates: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for rec in records:
        key = (
            rec.scenario.dataset,
            rec.scenario.contract,
            rec.scenario.monitor,
        )
        aggregates[key].append(rec.metric_value)
    return aggregates


def build_group_order(
    groups: Iterable[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    dataset_priority = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    contract_priority = {name: idx for idx, name in enumerate(CONTRACT_ORDER)}

    unique_groups = set(groups)
    sorted_groups = sorted(
        unique_groups,
        key=lambda item: (
            dataset_priority.get(item[0], len(dataset_priority)),
            contract_priority.get(item[1], len(contract_priority)),
            item[0],
            item[1],
        ),
    )
    return sorted_groups


def plot_monitor_comparison(
    grouped_stats: Dict[Tuple[str, str, str], List[float]],
    metric_name: str,
    output_path: Path,
    show: bool,
    dpi: int,
    higher_is_better: bool,
):
    group_pairs = {(key[0], key[1]) for key in grouped_stats.keys()}
    group_order = build_group_order(group_pairs)

    if not group_order:
        raise click.ClickException("No valid (dataset, contract) groups found.")

    width = 0.35
    x = np.arange(len(group_order))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    all_heights = []
    for idx, monitor in enumerate(MONITOR_ORDER):
        heights = []
        errors = []
        counts = []
        for group in group_order:
            samples = grouped_stats.get((*group, monitor), [])
            if samples:
                arr = np.array(samples, dtype=float)
                heights.append(float(np.mean(arr)))
                errors.append(float(np.std(arr)) if arr.size > 1 else 0.0)
                counts.append(arr.size)
            else:
                heights.append(np.nan)
                errors.append(0.0)
                counts.append(0)
        all_heights.extend([h for h in heights if not np.isnan(h)])
        offset = (idx - (len(MONITOR_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            heights,
            width,
            label=f"{monitor.title()} monitor",
            yerr=errors,
            capsize=6,
            alpha=0.85,
        )
        for bar, count in zip(bars, counts):
            if np.isnan(bar.get_height()) or count == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ylabel = metric_name.replace("_", " ").title()
    if not higher_is_better:
        ylabel += " (lower is better)"
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{dataset}\n{contract.title()} contract" for dataset, contract in group_order]
    )
    if all_heights:
        lo, hi = min(all_heights), max(all_heights)
        pad = max(0.05 * (hi - lo), 0.05 * max(abs(lo), 1.0))
        ax.set_ylim(lo - pad, hi + pad)
    ax.legend()
    ax.set_title("Reward-model eval by monitoring regime")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    click.echo(f"Saved figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary_table(
    grouped_stats: Dict[Tuple[str, str, str], List[float]],
    metric_name: str,
):
    click.echo("\nAggregated evaluation metrics:")
    dataset_priority = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    contract_priority = {name: idx for idx, name in enumerate(CONTRACT_ORDER)}
    monitor_priority = {name: idx for idx, name in enumerate(MONITOR_ORDER)}

    sorted_keys = sorted(
        grouped_stats.keys(),
        key=lambda item: (
            dataset_priority.get(item[0], len(dataset_priority)),
            contract_priority.get(item[1], len(contract_priority)),
            monitor_priority.get(item[2], len(monitor_priority)),
            item[0],
            item[1],
            item[2],
        ),
    )

    for dataset, contract, monitor in sorted_keys:
        samples = grouped_stats[(dataset, contract, monitor)]
        arr = np.array(samples, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr)) if arr.size > 1 else 0.0
        click.echo(
            f"- {dataset} / {contract} / {monitor}: "
            f"{metric_name}={mean_val:.4f} (n={arr.size}, std={std_val:.4f})"
        )


@click.command()
@click.option(
    "--results-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("bt_models"),
    show_default=True,
    help="Directory containing reward-model training outputs.",
)
@click.option(
    "--metric",
    "metric_name",
    default="eval_oracle_CE_loss",
    show_default=True,
    help="Evaluation metric to aggregate from trainer_state.json.",
)
@click.option(
    "--higher-is-better/--lower-is-better",
    default=False,
    show_default=True,
    help="Whether larger metric values are better when selecting the best checkpoint.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("fig/downstream_reward_models.png"),
    show_default=True,
    help="Where to save the generated plot.",
)
@click.option("--dpi", default=300, show_default=True, help="Figure DPI.")
@click.option("--show/--no-show", default=False, show_default=True, help="Show the plot interactively.")
def main(
    results_root: Path,
    metric_name: str,
    higher_is_better: bool,
    output: Path,
    dpi: int,
    show: bool,
):
    """
    Visualize reward-model performance grouped by monitoring regime (self vs expert)
    for the downstream contract experiment described in readme.md.
    """
    records = collect_run_records(results_root, metric_name, higher_is_better)
    if not records:
        raise click.ClickException(
            f"No trainer_state.json files with '{metric_name}' found under {results_root}."
        )

    grouped_stats = summarize_by_group(records)
    print_summary_table(grouped_stats, metric_name)
    plot_monitor_comparison(
        grouped_stats, metric_name, output, show, dpi, higher_is_better
    )


if __name__ == "__main__":
    main()
