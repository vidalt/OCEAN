from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer
from ocean.feature import parse_features

if TYPE_CHECKING:
    from ocean.abc import Mapper
    from ocean.feature import Feature


def build_dataset(
    seed: int = 42,
    n_samples: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[int], Mapper[Feature]]:
    rng = np.random.default_rng(seed)

    raw = pd.DataFrame({
        "credit_lines": rng.choice([0, 1, 2, 4], size=n_samples),
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
    return raw, data, target, mapper


def main() -> None:
    raw, data, target, mapper = build_dataset()

    model = RandomForestClassifier(
        n_estimators=40,
        max_depth=4,
        random_state=42,
    )
    model.fit(data, target)

    predictions = pd.Series(model.predict(data), index=data.index)
    query_index = predictions[predictions == 0].index[0]
    query = data.loc[query_index].to_numpy(dtype=float).flatten()
    query_frame = data.loc[[query_index]]
    raw_query = raw.loc[query_index]

    explainer = ConstraintProgrammingExplainer(model, mapper=mapper)
    explanation = explainer.explain(
        query,
        y=1,
        norm=1,
        max_time=10,
        num_workers=1,
        random_seed=42,
    )
    if explanation is None:
        msg = "No counterfactual was found for the synthetic example."
        raise RuntimeError(msg)
    counterfactual_frame = explanation.to_numpy().reshape(1, -1)

    print("Original raw instance:")
    print(raw_query)
    print()
    print("Model prediction:", int(model.predict(query_frame).item()))
    print("Target class:", 1)
    print(
        "Counterfactual prediction:",
        int(model.predict(counterfactual_frame).item()),
    )
    print("Counterfactual values:")
    print(explanation)
    print()
    print("Counterfactual vector:")
    print(explanation.to_series())
    print()
    print("Objective value:", explainer.get_objective_value())


if __name__ == "__main__":
    main()
