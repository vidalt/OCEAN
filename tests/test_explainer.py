import sys

import gurobipy as gp
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.feature import parse_features

from .utils import ENV, generate_data


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [5])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_mip_explain(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
    num_workers: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    model = MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV)

    x = data.iloc[0, :].to_numpy().astype(float).flatten()
    # pyright: ignore[reportUnknownVariableType]
    y = clf.predict([x])[0]
    classes = np.unique(clf.predict(data.to_numpy())).astype(np.int64)  # pyright: ignore[reportUnknownArgumentType]
    for target in classes[classes != y]:
        try:
            exp = model.explain(
                x,
                y=target,
                norm=1,
                return_callback=True,
                num_workers=num_workers,
                random_seed=seed,
            )
            assert model.Status == gp.GRB.OPTIMAL
            assert len(model.callback.sollist) != 0
            assert exp is not None
            assert clf.predict([exp.to_numpy()])[0] == target
            model.cleanup()

        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [5])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_mip_explain_xgb(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
    num_workers: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = XGBClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    model = MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV)

    x = data.iloc[0, :].to_numpy().astype(float).flatten()
    # pyright: ignore[reportUnknownVariableType]
    y = clf.predict([x])[0]
    classes = np.unique(clf.predict(data.to_numpy())).astype(np.int64)  # pyright: ignore[reportUnknownArgumentType]
    for target in classes[classes != y]:
        try:
            exp = model.explain(
                x,
                y=target,
                norm=1,
                return_callback=True,
                num_workers=num_workers,
                random_seed=seed,
            )
            model.cleanup()
            assert model.Status == gp.GRB.OPTIMAL
            assert len(model.callback.sollist) != 0
            assert exp is not None
            assert clf.predict([exp.to_numpy()])[0] == target

        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [5])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_cp_explain(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
    num_workers: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    model = ConstraintProgrammingExplainer(clf, mapper=mapper)

    x = data.iloc[0, :].to_numpy().astype(float).flatten()
    # pyright: ignore[reportUnknownVariableType]
    y = clf.predict([x])[0]
    classes = np.unique(clf.predict(data.to_numpy())).astype(np.int64)  # pyright: ignore[reportUnknownArgumentType]
    for target in classes[classes != y]:
        try:
            exp = model.explain(
                x,
                y=target,
                norm=1,
                return_callback=True,
                num_workers=num_workers,
                random_seed=seed,
            )
            assert model.get_solving_status() == "OPTIMAL"
            assert model.callback is None or len(model.callback.sollist) != 0
            assert exp is not None
            assert clf.predict([exp.to_numpy()])[0] == target
            model.cleanup()
        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [5])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_cp_explain_xgb(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
    num_workers: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = XGBClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    model = ConstraintProgrammingExplainer(clf, mapper=mapper)

    x = data.iloc[0, :].to_numpy().astype(float).flatten()
    # pyright: ignore[reportUnknownVariableType]
    y = clf.predict([x])[0]
    classes = np.unique(clf.predict(data.to_numpy())).astype(np.int64)  # pyright: ignore[reportUnknownArgumentType]
    for target in classes[classes != y]:
        try:
            exp = model.explain(
                x,
                y=target,
                norm=1,
                return_callback=True,
                num_workers=num_workers,
                random_seed=seed,
            )
            assert model.get_solving_status() == "OPTIMAL"
            assert model.callback is None or len(model.callback.sollist) != 0
            assert exp is not None
            assert clf.predict([exp.to_numpy()])[0] == target
            model.cleanup()

        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")


@pytest.mark.skipif(
    sys.platform == "win32", reason="tests for non-windows platforms"
)
def test_explainers_return_same_distance_on_discrete_data() -> None:
    raw = pd.DataFrame({
        "age_bucket": [0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 0, 3],
        "owns_home": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        "job_type": [
            "office",
            "office",
            "manual",
            "manual",
            "service",
            "service",
            "office",
            "manual",
            "service",
            "office",
            "manual",
            "service",
        ],
    })
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    data, mapper = parse_features(
        raw,
        discretes=("age_bucket",),
        encoded=("job_type",),
        scale=False,
    )
    clf = RandomForestClassifier(
        random_state=7,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])

    mip = MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV)
    cp = ConstraintProgrammingExplainer(clf, mapper=mapper)
    maxsat = MaxSATExplainer(clf, mapper=mapper)

    exp_mip = mip.explain(x, y=target, norm=1, random_seed=7)
    exp_cp = cp.explain(x, y=target, norm=1, random_seed=7)
    exp_maxsat = maxsat.explain(x, y=target, norm=1, random_seed=7)

    assert exp_mip is not None
    assert exp_cp is not None
    assert exp_maxsat is not None
    assert clf.predict([exp_mip.to_numpy()])[0] == target
    assert clf.predict([exp_cp.to_numpy()])[0] == target
    assert clf.predict([exp_maxsat.to_numpy()])[0] == target
    assert mip.get_distance() == pytest.approx(cp.get_distance())
    assert mip.get_distance() == pytest.approx(maxsat.get_distance())
