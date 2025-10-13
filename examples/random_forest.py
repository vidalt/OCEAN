import time
from argparse import ArgumentParser
from dataclasses import dataclass

import gurobipy as gp
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import (
    ConstraintProgrammingExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.abc import Mapper
from ocean.datasets import load_adult, load_compas, load_credit
from ocean.feature import Feature
from ocean.typing import Array1D, BaseExplainer

Loaded = tuple[tuple[pd.DataFrame, "pd.Series[int]"], Mapper[Feature]]


@dataclass
class Args:
    seed: int
    n_estimators: int
    max_depth: int
    n_examples: int
    dataset: str


def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        dest="n_estimators",
    )
    parser.add_argument("--max-depth", type=int, default=5, dest="max_depth")
    parser.add_argument(
        "--n-examples",
        type=int,
        default=100,
        dest="n_examples",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "compas", "credit"],
        default="compas",
    )
    args = parser.parse_args()
    return Args(
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_examples=args.n_examples,
        dataset=args.dataset,
    )


def load(dataset: str) -> Loaded:
    if dataset == "credit":
        return load_credit()
    if dataset == "adult":
        return load_adult()
    if dataset == "compas":
        return load_compas()
    msg = f"Unknown dataset: {dataset}"
    raise ValueError(msg)


ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()
CONSOLE = Console()


def main(explainers: dict[str, type[BaseExplainer]]) -> None:
    args = parse_args()
    data, target, mapper = load_data(args)
    rf = fit_model(args, data, target)
    times: dict[str, pd.Series[float]] = {}
    for name, explainer in explainers.items():
        exp = build_explainer(name, explainer, args, rf, mapper)
        queries = generate_queries(args, rf, data)
        times[name] = run_queries(exp, queries)
    display_statistics(times)


def load_data(
    args: Args,
) -> tuple[pd.DataFrame, "pd.Series[int]", Mapper[Feature]]:
    with CONSOLE.status("[bold blue]Loading the data[/bold blue]"):
        (data, target), mapper = load(args.dataset)
    CONSOLE.print("[bold green]Data loaded[/bold green]")
    return data, target, mapper


def fit_model(
    args: Args,
    data: pd.DataFrame,
    target: "pd.Series[int]",
) -> RandomForestClassifier:
    X_train, _, y_train, _ = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=args.seed,
    )
    with CONSOLE.status("[bold blue]Fitting a Random Forest model[/bold blue]"):
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            max_depth=args.max_depth,
        )
        rf.fit(X_train, y_train)
    CONSOLE.print("[bold green]Model fitted[/bold green]")
    return rf


def build_explainer(
    name: str,
    explainer: type[BaseExplainer],
    args: Args,
    rf: RandomForestClassifier,
    mapper: Mapper[Feature],
) -> BaseExplainer:
    with CONSOLE.status("[bold blue]Building the Explainer[/bold blue]"):
        start = time.time()
        if name == "mip":
            ENV.setParam("Seed", args.seed)
            exp = explainer(rf, mapper=mapper, env=ENV)  # type: ignore  # noqa: PGH003
        else:
            exp = explainer(rf, mapper=mapper)  # type: ignore  # noqa: PGH003
        end = time.time()
    CONSOLE.print(f"[bold green]{name.upper()} Explainer built[/bold green]")
    msg = f"Build time: {end - start:.2f} seconds"
    CONSOLE.print(f"\t[bold yellow]{msg}[/bold yellow]")
    return exp


def generate_queries(
    args: Args,
    rf: RandomForestClassifier,
    data: pd.DataFrame,
) -> list[tuple[Array1D, int]]:
    _, X_test = train_test_split(
        data,
        test_size=0.2,
        random_state=args.seed,
    )
    X_test = pd.DataFrame(X_test)
    y_pred = rf.predict(X_test)
    with CONSOLE.status("[bold blue]Generating queries[/bold blue]"):
        queries: list[tuple[Array1D, int]] = [
            (X_test.iloc[i].to_numpy().flatten(), 1 - y_pred[i])
            for i in range(min(args.n_examples, len(X_test)))
        ]
    CONSOLE.print("[bold green]Queries generated[/bold green]")
    return queries


def run_queries(
    explainer: BaseExplainer, queries: list[tuple[Array1D, int]]
) -> "pd.Series[float]":
    times: pd.Series[float] = pd.Series()
    for i, (x, y) in track(
        enumerate(queries),
        total=len(queries),
        description="[bold blue]Running queries[/bold blue]",
    ):
        start = time.time()
        explainer.explain(x, y=y, norm=1)
        explainer.cleanup()
        end = time.time()
        times[i] = end - start
    return times


def create_table_row(
    metric: str, times: dict[str, "pd.Series[float]"]
) -> list[str]:
    row = [metric]
    for t in times.values():
        if metric == "Number of queries":
            row.append(str(len(t)))
        elif metric == "Total time (seconds)":
            row.append(f"{t.sum():.2f}")
        elif metric == "Mean time per query (seconds)":
            row.append(f"{t.mean():.2f}")
        elif metric == "Std of time per query (seconds)":
            row.append(f"{t.std():.2f}")
        elif metric == "Maximum time per query (seconds)":
            row.append(f"{t.max():.2f}")
        elif metric == "Minimum time per query (seconds)":
            row.append(f"{t.min():.2f}")
        else:
            row.append("N/A")
    return row


def display_statistics(times: dict[str, "pd.Series[float]"]) -> None:
    CONSOLE.print("[bold blue]Statistics:[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=30)
    names = list(times.keys())
    for name in names:
        table.add_column(name.upper())
    metrics = [
        "Number of queries",
        "Total time (seconds)",
        "Mean time per query (seconds)",
        "Std of time per query (seconds)",
        "Maximum time per query (seconds)",
        "Minimum time per query (seconds)",
    ]
    for metric in metrics:
        row = create_table_row(metric, times)
        table.add_row(*row)
    CONSOLE.print(table)
    CONSOLE.print("[bold green]Done[/bold green]")


if __name__ == "__main__":
    explainers = {
        "mip": MixedIntegerProgramExplainer,
        "cp": ConstraintProgrammingExplainer,
    }
    main(explainers)
