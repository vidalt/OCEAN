import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gurobipy as gp
import pandas as pd
from datasets import load_adult, load_compas, load_credit
from rich.progress import track
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import MIPExplainer

if TYPE_CHECKING:
    from ocean.typing import FloatArray1D


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
    parser.add_argument("--dataset", type=str, default="compas")
    args = parser.parse_args()
    return Args(
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_examples=args.n_examples,
        dataset=args.dataset,
    )


def load(dataset: str):  # noqa: ANN201
    if dataset == "credit":
        return load_credit()
    if dataset == "adult":
        return load_adult()
    if dataset == "compas":
        return load_compas()
    msg = f"Unknown dataset: {dataset}"
    raise ValueError(msg)


def main() -> None:  # noqa: PLR0914
    args = parse_args()

    # Load the data
    print("Loading the data")
    mapper, (X, y) = load(args.dataset)
    print("Data loaded")

    X_train, X_test, y_train, _ = train_test_split(
        X.to_numpy(),
        y.to_numpy().flatten(),
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
    print("Building the MIPExplainer")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.start()
    start = time.time()
    mip = MIPExplainer(rf, mapper=mapper, env=env)
    end = time.time()
    print("MIPExplainer built")
    print(f"Building the model took {end - start:.2f} seconds")

    # Generate multiple queries:
    queries: list[tuple[FloatArray1D, int]] = []

    y_pred = rf.predict(X_test)

    print("Generating queries")
    for x, y_ in zip(
        X_test[: min(args.n_examples, len(X_test))],
        y_pred[: min(args.n_examples, len(X_test))],
        strict=True,
    ):
        queries.append((x, 1 - y_))
    print("Queries generated")

    print("Running queries")
    times: list[float] = []
    for x, y_ in track(
        queries,
        total=len(queries),
        description="Running queries",
    ):
        start = time.time()
        mip.add_objective(x)
        mip.set_majority_class(y_)
        mip.optimize()
        mip.clear_majority_class()
        mip.cleanup()
        end = time.time()

        times.append(end - start)
    print("Queries run")
    series = pd.Series(times)
    print("Statistics:")
    print(f"Number of queries: {len(series)}")
    print(f"Total time: {series.sum():.2f} seconds")
    print(f"Mean time per query: {series.mean():.2f} seconds")
    print(f"Standard deviation of time per query: {series.std():.2f} seconds")
    print(f"Maximum time per query: {series.max():.2f} seconds")
    print(f"Minimum time per query: {series.min():.2f} seconds")
    print("Done")


if __name__ == "__main__":
    main()
