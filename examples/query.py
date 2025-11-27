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
from xgboost import XGBClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.abc import Mapper
from ocean.datasets import load_adult, load_compas, load_credit
from ocean.feature import Feature
from ocean.typing import Array1D, BaseExplainer

# Global constants
ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()
CONSOLE = Console()
EXPLAINERS = {
    "mip": MixedIntegerProgramExplainer,
    "cp": ConstraintProgrammingExplainer,
    "maxsat": MaxSATExplainer,
}
MODELS = {
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}


@dataclass
class Args:
    seed: int
    n_estimators: int
    max_depth: int
    n_examples: int
    dataset: str
    explainers: list[str]
    models: list[str]


def create_argument_parser() -> ArgumentParser:
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
    parser.add_argument(
        "-e",
        "--exp",
        "--explainer",
        help="List of explainers to use",
        type=str,
        nargs="+",
        choices=["mip", "cp", "maxsat"],
        default=["mip", "cp", "maxsat"],
    )
    parser.add_argument(
        "-m",
        "--model",
        help="List of models to use",
        type=str,
        nargs="+",
        choices=["rf", "xgb"],
        default=["rf"],
    )
    return parser


def parse_args() -> Args:
    parser = create_argument_parser()
    args = parser.parse_args()
    return Args(
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_examples=args.n_examples,
        dataset=args.dataset,
        explainers=args.exp,
        models=args.model,
    )


def load_dataset(
    dataset: str,
) -> tuple[tuple[pd.DataFrame, pd.Series], Mapper[Feature]]:
    if dataset == "credit":
        return load_credit()
    if dataset == "adult":
        return load_adult()
    if dataset == "compas":
        return load_compas()
    msg = f"Unknown dataset: {dataset}"
    raise ValueError(msg)


def load_data(args: Args) -> tuple[pd.DataFrame, pd.Series, Mapper[Feature]]:
    with CONSOLE.status("[bold blue]Loading the data[/bold blue]"):
        (data, target), mapper = load_dataset(args.dataset)
    CONSOLE.print("[bold green]Data loaded[/bold green]")
    return data, target, mapper


def fit_model_with_console(
    args: Args,
    data: pd.DataFrame,
    target: pd.Series,
    model_class: type[RandomForestClassifier] | type[XGBClassifier],
    model_name: str,
    **model_kwargs: str | float | bool | None,
) -> RandomForestClassifier | XGBClassifier:
    X_train, _, y_train, _ = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=args.seed,
    )
    with CONSOLE.status(f"[bold blue]Fitting a {model_name} model[/bold blue]"):
        model = model_class(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            max_depth=args.max_depth,
            **model_kwargs,
        )
        model.fit(X_train, y_train)
    CONSOLE.print("[bold green]Model fitted[/bold green]")
    return model


def build_explainer(
    explainer_name: str,
    explainer_class: type[MixedIntegerProgramExplainer]
    | type[ConstraintProgrammingExplainer]
    | type[MaxSATExplainer],
    args: Args,
    model: RandomForestClassifier | XGBClassifier,
    mapper: Mapper[Feature],
) -> BaseExplainer:
    with CONSOLE.status("[bold blue]Building the Explainer[/bold blue]"):
        start = time.time()
        if explainer_class is MixedIntegerProgramExplainer:
            ENV.setParam("Seed", args.seed)
            exp = explainer_class(model, mapper=mapper, env=ENV)
        elif (
            explainer_class is ConstraintProgrammingExplainer
            or explainer_class is MaxSATExplainer
        ):
            exp = explainer_class(model, mapper=mapper)
        else:
            msg = f"Unknown explainer type: {explainer_class}"
            raise ValueError(msg)
        end = time.time()
    CONSOLE.print(
        f"[bold green]{explainer_name.upper()} Explainer built[/bold green]"
    )
    msg = f"Build time: {end - start:.2f} seconds"
    CONSOLE.print(f"\t[bold yellow]{msg}[/bold yellow]")
    return exp


def generate_queries(
    args: Args,
    model: RandomForestClassifier | XGBClassifier,
    data: pd.DataFrame,
) -> list[tuple[Array1D, int]]:
    _, X_test = train_test_split(
        data,
        test_size=0.2,
        random_state=args.seed,
    )
    X_test = pd.DataFrame(X_test)
    y_pred = model.predict(X_test)
    with CONSOLE.status("[bold blue]Generating queries[/bold blue]"):
        queries: list[tuple[Array1D, int]] = [
            (X_test.iloc[i].to_numpy().flatten(), 1 - y_pred[i])
            for i in range(min(args.n_examples, len(X_test)))
        ]
    CONSOLE.print("[bold green]Queries generated[/bold green]")
    return queries


def run_queries_verbose(
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
    """Display timing statistics in a table."""
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


def main() -> None:
    args = parse_args()
    data, target, mapper = load_data(args)
    explainers = {
        name: explainer
        for name, explainer in EXPLAINERS.items()
        if name in args.explainers
    }
    models = {
        name: model for name, model in MODELS.items() if name in args.models
    }
    for model_name, model_class in models.items():
        CONSOLE.print(
            f"[bold blue]Running experiment with {model_name}: [/bold blue]"
        )
        model = fit_model_with_console(
            args, data, target, model_class, model_name
        )
        for explainer_name, explainer_class in explainers.items():
            CONSOLE.print(
                f"[bold blue]Running for {explainer_name}[/bold blue]"
            )
            exp = build_explainer(
                explainer_name, explainer_class, args, model, mapper
            )
            queries = generate_queries(args, model, data)
            times = run_queries_verbose(exp, queries)
            display_statistics({explainer_name: times})


if __name__ == "__main__":
    main()
