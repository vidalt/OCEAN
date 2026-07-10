from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.feature import parse_features

if TYPE_CHECKING:
    from ocean.abc import Mapper
    from ocean.feature import Feature


@dataclass(frozen=True)
class Dataset:
    raw: pd.DataFrame
    data: pd.DataFrame
    target: pd.Series[int]
    mapper: Mapper[Feature]


def build_dataset(
    seed: int = 7,
    n_samples: int = 120,
) -> Dataset:
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({
        "credit_lines": rng.choice([0, 1, 2, 3], size=n_samples),
        "owns_home": rng.integers(0, 2, size=n_samples),
        "has_guarantor": rng.integers(0, 2, size=n_samples),
        "job_type": rng.choice(
            ["office", "manual", "service", "student"],
            size=n_samples,
        ),
    })

    score = (
        (raw["credit_lines"] >= 2).astype(int)
        + raw["owns_home"].astype(int)
        + raw["has_guarantor"].astype(int)
        + raw["job_type"].isin(["office", "service"]).astype(int)
    )
    target = (score >= 3).astype(int).rename("approved")
    data, mapper = parse_features(
        raw,
        discretes=("credit_lines",),
        encoded=("job_type",),
        scale=False,
    )
    return Dataset(raw=raw, data=data, target=target, mapper=mapper)


def build_tie_break_dataset() -> Dataset:
    near_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    far_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    raw = pd.DataFrame(
        [
            {"near_score": near, "far_score": far}
            for near in near_values
            for far in far_values
        ]
    )
    target = (
        (raw["near_score"] >= 0.35) | (raw["far_score"] >= 0.75)
    ).astype(int)
    target = target.rename("approved")
    data, mapper = parse_features(raw, scale=False)
    return Dataset(raw=raw, data=data, target=target, mapper=mapper)


@dataclass(frozen=True)
class Result:
    series: pd.Series[float]
    counterfactual: np.ndarray[tuple[int], np.dtype[np.float64]]
    objective: float
    distance: float
    prediction: int


def predict_one(
    model: RandomForestClassifier,
    x: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> int:
    prediction = np.asarray(model.predict(x.reshape(1, -1)), dtype=np.int64)
    return int(prediction[0])


def explain_with_mip(
    model: RandomForestClassifier,
    mapper: Mapper[Feature],
    query: np.ndarray[tuple[int], np.dtype[np.float64]],
    weighted_norms: list[float],
    target_class: int,
) -> Result:
    explainer = MixedIntegerProgramExplainer(model, mapper=mapper)
    explanation = explainer.explain(
        query,
        y=target_class,
        weighted_norms=weighted_norms,
        max_time=20,
        num_workers=1,
        random_seed=7,
        clean_up=False,
    )
    if explanation is None:
        msg = "MIP did not find a counterfactual."
        raise RuntimeError(msg)
    counterfactual = explanation.to_numpy()
    result = Result(
        series=explanation.to_series(),
        counterfactual=counterfactual,
        objective=explainer.get_objective_value(),
        distance=explainer.get_distance(),
        prediction=predict_one(model, counterfactual),
    )
    explainer.cleanup()
    return result


def explain_with_cp(
    model: RandomForestClassifier,
    mapper: Mapper[Feature],
    query: np.ndarray[tuple[int], np.dtype[np.float64]],
    weighted_norms: list[float],
    target_class: int,
) -> Result:
    explainer = ConstraintProgrammingExplainer(model, mapper=mapper)
    explanation = explainer.explain(
        query,
        y=target_class,
        weighted_norms=weighted_norms,
        max_time=20,
        num_workers=1,
        random_seed=7,
        clean_up=False,
    )
    if explanation is None:
        msg = "CP did not find a counterfactual."
        raise RuntimeError(msg)
    counterfactual = explanation.to_numpy()
    result = Result(
        series=explanation.to_series(),
        counterfactual=counterfactual,
        objective=explainer.get_objective_value(),
        distance=explainer.get_distance(),
        prediction=predict_one(model, counterfactual),
    )
    explainer.cleanup()
    return result


def print_result(
    name: str,
    result: Result,
) -> None:
    print(f"\n{name}")
    print("Counterfactual prediction:", result.prediction)
    print("Solver objective:", result.objective)
    print("Weighted distance:", result.distance)
    print(result.series)


def train_forest(
    dataset: Dataset,
    *,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> tuple[
    RandomForestClassifier,
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
]:
    processed = dataset.data.to_numpy(dtype=np.float64)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        bootstrap=False,
    )
    model.fit(processed, dataset.target.to_numpy(dtype=np.int64))
    return model, processed


def run_comparison(
    title: str,
    dataset: Dataset,
    model: RandomForestClassifier,
    processed: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    query_index: int,
    weighted_norms: list[float],
    target_class: int,
) -> tuple[Result, Result]:
    query = processed[query_index].astype(np.float64, copy=True)

    print(f"\n\n=== {title} ===")
    print("Weighted norms [L0, L1, L2]:", weighted_norms)
    print("\nOriginal raw instance:")
    print(dataset.raw.loc[query_index])
    print("Original prediction:", predict_one(model, query))
    print("Target class:", target_class)

    mip = explain_with_mip(
        model,
        dataset.mapper,
        query,
        weighted_norms,
        target_class,
    )
    cp = explain_with_cp(
        model,
        dataset.mapper,
        query,
        weighted_norms,
        target_class,
    )
    print_result("MIP result", mip)
    print_result("CP result", cp)

    print("\nComparison")
    print("Objective gap:", abs(mip.objective - cp.objective))
    print("Distance gap:", abs(mip.distance - cp.distance))
    print(
        "Same counterfactual vector:",
        bool(np.allclose(mip.counterfactual, cp.counterfactual)),
    )
    return mip, cp


def run_categorical_examples() -> None:
    dataset = build_dataset()
    model, processed = train_forest(
        dataset,
        n_estimators=10,
        max_depth=3,
        random_state=7,
    )
    predicted = np.asarray(model.predict(processed), dtype=np.int64)
    predictions = pd.Series(predicted, index=dataset.data.index)
    denied = predictions[predictions == 0].index
    weighted_norms = [2.0, 1.0, 0.5]

    for query_index in [int(denied[0]), int(denied[10])]:
        run_comparison(
            "Categorical weighted-norm query",
            dataset,
            model,
            processed,
            query_index,
            weighted_norms,
            target_class=1,
        )


def run_continuous_tie_break_example() -> None:
    dataset = build_tie_break_dataset()
    model, processed = train_forest(
        dataset,
        n_estimators=1,
        max_depth=2,
        random_state=0,
    )
    query_index = 19

    l0_mip, l0_cp = run_comparison(
        "Pure L0 tie: two one-feature changes are optimal",
        dataset,
        model,
        processed,
        query_index,
        weighted_norms=[1.0],
        target_class=1,
    )
    weighted_mip, weighted_cp = run_comparison(
        "L0 tie broken by L1 and L2",
        dataset,
        model,
        processed,
        query_index,
        weighted_norms=[1.0, 1.0, 1.0],
        target_class=1,
    )

    print("\nTie-break summary")
    print("Pure L0 MIP vector:", l0_mip.counterfactual)
    print("Pure L0 CP vector:", l0_cp.counterfactual)
    print("Weighted MIP vector:", weighted_mip.counterfactual)
    print("Weighted CP vector:", weighted_cp.counterfactual)
    print("Weighted result changes the nearer continuous threshold.")


def main() -> None:
    run_categorical_examples()
    run_continuous_tie_break_example()


if __name__ == "__main__":
    main()
