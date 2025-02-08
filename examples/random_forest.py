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
from ocean.datasets import load_adult, load_compas, load_credit

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
    (X, y), mapper = load(args.dataset)
    print("Data loaded")

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
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
    mip = MixedIntegerProgramExplainer(rf, mapper=mapper, env=env)
    end = time.time()
    print("MIPExplainer built")
    print(f"Building the model took {end - start:.2f} seconds")

    # Generate multiple queries:
    queries: list[tuple[Array1D, int]] = []
    X_test = pd.DataFrame(X_test)
    y_pred = rf.predict(X_test)

    print("Generating queries")
    for i in range(min(args.n_examples, len(X_test))):
        queries.append((X_test.iloc[i].to_numpy().flatten(), 1 - y_pred[i]))  # noqa: PERF401
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
