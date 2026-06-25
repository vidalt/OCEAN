from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from ocean.abc import Mapper  # noqa: E402
from ocean.datasets import load_adult, load_compas, load_credit  # noqa: E402
from ocean.feature import (  # noqa: E402
    Feature,
    parse_features,
)
from ocean.ls import DLSExplainer, SLSExplainer  # noqa: E402
from ocean.ls.utils.costs import get_norm  # noqa: E402

if TYPE_CHECKING:
    from ocean.typing import BaseExplainableEnsemble

SEED = 0
type DemoDataset = tuple[pd.DataFrame, pd.Series[int], Mapper[Feature]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local-search explainers on an OCEAN dataset.",
    )
    parser.add_argument(
        "--data",
        choices=("custom", "adult", "compas", "credit"),
        default="custom",
        help=(
            "Dataset to use. 'custom' is generated locally; the other choices "
            "use ocean.datasets loaders and may require network access."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("rf", "xgb", "adaboost"),
        default="rf",
        help="Model to fit before running DLS and SLS.",
    )
    return parser.parse_args()


def build_custom_dataset(
    seed: int = SEED,
    n_samples: int = 280,
) -> DemoDataset:
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({
        "credit_lines": rng.choice([0, 1, 2, 3, 4], size=n_samples),
        "owns_home": rng.integers(0, 2, size=n_samples),
        "has_guarantor": rng.integers(0, 2, size=n_samples),
        "income_ratio": rng.uniform(-0.4, 0.8, size=n_samples),
        "debt_ratio": rng.uniform(0.0, 1.0, size=n_samples),
        "savings_ratio": rng.uniform(-0.5, 0.6, size=n_samples),
        "job_type": rng.choice(
            ["office", "manual", "service", "student"],
            size=n_samples,
        ),
        "region": rng.choice(
            ["north", "south", "east", "west"],
            size=n_samples,
        ),
    })

    score = (
        (raw["credit_lines"] >= 2).astype(int)
        + raw["owns_home"].astype(int)
        + raw["has_guarantor"].astype(int)
        + (raw["income_ratio"] > 0.1).astype(int)
        + (raw["savings_ratio"] > 0.0).astype(int)
        + raw["job_type"].isin(["office", "service"]).astype(int)
        + raw["region"].isin(["north", "east"]).astype(int)
        - (raw["debt_ratio"] > 0.55).astype(int)
    )
    target = (score >= 4).astype(int).rename("approved")
    data, mapper = parse_features(raw, discretes=("credit_lines",))
    return data, target, mapper


def load_dataset(
    name: str,
) -> DemoDataset:
    if name == "custom":
        return build_custom_dataset()
    if name == "adult":
        (data, target), mapper = load_adult(scale=True)
        return data, target, mapper
    if name == "compas":
        (data, target), mapper = load_compas(scale=True)
        return data, target, mapper
    if name == "credit":
        (data, target), mapper = load_credit(scale=True)
        return data, target, mapper

    msg = f"Unknown dataset: {name}"
    raise ValueError(msg)


def build_model(name: str) -> BaseExplainableEnsemble:
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=20,
            max_depth=4,
            random_state=SEED,
        )
    if name == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2,
                random_state=SEED,
            ),
            n_estimators=20,
            random_state=SEED,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=20,
            max_depth=4,
            random_state=SEED,
            eval_metric="logloss",
        )

    msg = f"Unknown model: {name}"
    raise ValueError(msg)


def to_frame(
    point: np.ndarray,
    columns: pd.Index | pd.MultiIndex,
) -> pd.DataFrame:
    return pd.DataFrame([point], columns=columns)


def predict_one(
    model: BaseExplainableEnsemble,
    point: np.ndarray,
    columns: pd.Index | pd.MultiIndex,
) -> int:
    prediction = np.asarray(
        model.predict(to_frame(point, columns)),
        dtype=np.int_,
    )
    return int(prediction.item())


def explain_with_dls(
    explainer: DLSExplainer,
    query: np.ndarray,
    query_class: int,
) -> np.ndarray:
    explanation = explainer.explain(
        x=query,
        query_class=query_class,
        norm=1,
        std=1.0,
        n_population=20,
        lambda_=1,
        tabu_size=100,
        return_callback=True,
        random_seed=SEED,
        n_iter=40,
        max_time_per_local_search=0.05,
        init_type="naive",
        flip_prob=0.2,
        perturb_ratio=1.0,
        total_timeout=5,
    )
    if explanation is None:
        msg = "DLS did not find a counterfactual for the demo query."
        raise RuntimeError(msg)
    return explanation.to_numpy()


def explain_with_sls(
    explainer: SLSExplainer,
    query: np.ndarray,
    query_class: int,
    n_features: int,
) -> np.ndarray:
    explanation = explainer.explain(
        x=query,
        query_class=query_class,
        norm=1,
        std=1.0,
        n_population=20,
        lambda_=1,
        n_faces=2 * n_features,
        n_samples_per_face=2,
        tabu_size=100,
        return_callback=True,
        random_seed=SEED,
        n_iter=40,
        max_time_per_local_search=0.05,
        init_type="naive",
        flip_prob=0.2,
        perturb_ratio=1.0,
        total_timeout=5,
    )
    if explanation is None:
        msg = "SLS did not find a counterfactual for the demo query."
        raise RuntimeError(msg)
    return explanation.to_numpy()


def main() -> None:  # noqa: PLR0914
    args = parse_args()
    data_name = cast("str", args.data)
    model_name = cast("str", args.model)
    data, target, mapper = load_dataset(data_name)
    split = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=target,
    )
    X_train_frame = cast("pd.DataFrame", split[0])
    X_test_frame = cast("pd.DataFrame", split[1])
    y_train = cast("pd.Series[int]", split[2])
    y_test = cast("pd.Series[int]", split[3])
    X_train = X_train_frame.to_numpy(dtype=np.float32)
    X_test = X_test_frame.to_numpy(dtype=np.float32)

    model = build_model(model_name)
    model.fit(X_train_frame, y_train)
    train_accuracy = float(model.score(X_train_frame, y_train))
    test_accuracy = float(model.score(X_test_frame, y_test))
    print("dataset=", data_name)
    print("model=", model_name)
    print(
        "accuracy_train=",
        train_accuracy,
        "accuracy_test=",
        test_accuracy,
    )

    query = X_test[0].astype(np.float32)
    query_class = predict_one(model, query, data.columns)

    dls_explainer = DLSExplainer(model, mapper, data)
    sls_explainer = SLSExplainer(model, mapper, data)

    dls_cf = explain_with_dls(dls_explainer, query, query_class)
    sls_cf = explain_with_sls(
        sls_explainer,
        query,
        query_class,
        int(X_train.shape[1]),
    )

    dls_label = predict_one(model, dls_cf, data.columns)
    sls_label = predict_one(model, sls_cf, data.columns)
    dls_distance = get_norm(1, query, dls_cf)
    sls_distance = get_norm(1, query, sls_cf)

    print("\nQuery")
    dls_explainer.print_explanation(query)
    print("Query label=", query_class)
    print("\nDLS explanation")
    dls_explainer.print_explanation(dls_cf)
    print("DLS label=", dls_label, "distance=", float(dls_distance))
    print("\nSLS explanation")
    sls_explainer.print_explanation(sls_cf)
    print("SLS label=", sls_label, "distance=", float(sls_distance))
    print("\nDLS comparison")
    dls_explainer.print_comparison(query, dls_cf)
    print("\nSLS comparison")
    sls_explainer.print_comparison(query, sls_cf)


if __name__ == "__main__":
    main()
