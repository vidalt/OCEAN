import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gurobipy as gp
import pandas as pd
from rich.progress import track
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import MixedIntegerProgramExplainer
from ocean.abc import Mapper
from ocean.datasets import load_adult, load_compas, load_credit
from ocean.feature import Feature

Loaded = tuple[tuple[pd.DataFrame, "pd.Series[int]"], Mapper[Feature]]

if TYPE_CHECKING:
    from ocean.typing import Array1D


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


def main() -> None:
    args = parse_args()

    # Load the data
    print("Loading the data")
    (data, target), mapper = load(args.dataset)
    print("Data loaded")

    X_train, X_test, y_train, _ = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=args.seed,
    )

    # Fit the Random Forest model
    print("Fitting a Random Forest model")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        max_depth=args.max_depth,
    )
    rf.fit(X_train, y_train)
    print("Model fitted")

    # Evaluate the model
    print("Building the Explainer")
    ENV.setParam("Seed", args.seed)
    start = time.time()
    mip = MixedIntegerProgramExplainer(rf, mapper=mapper, env=ENV)
    end = time.time()
    print("Explainer built")
    print(f"Building the model took {end - start:.2f} seconds")

    # Generate multiple queries:
    X_test = pd.DataFrame(X_test)
    y_pred = rf.predict(X_test)

    print("Generating queries")
    queries: list[tuple[Array1D, int]] = [
        (X_test.iloc[i].to_numpy().flatten(), 1 - y_pred[i])
        for i in range(min(args.n_examples, len(X_test)))
    ]
    print("Queries generated")

    times: pd.Series[float] = pd.Series()
    for i, (x, y) in track(
        enumerate(queries),
        total=len(queries),
        description="Running queries",
    ):
        start = time.time()
        mip.explain(x, y=y, norm=1)
        mip.cleanup()
        end = time.time()

        times[i] = end - start

    print("Statistics:")
    print(f"Number of queries: {len(times)}")
    print(f"Total time: {times.sum():.2f} seconds")
    print(f"Mean time per query: {times.mean():.2f} seconds")
    print(f"Standard deviation of time per query: {times.std():.2f} seconds")
    print(f"Maximum time per query: {times.max():.2f} seconds")
    print(f"Minimum time per query: {times.min():.2f} seconds")
    print("Done")


if __name__ == "__main__":
    main()
