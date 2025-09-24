import gurobipy as gp
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer

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

    try:
        model.explain(x, y=0, norm=1,
                      num_workers=num_workers,
                      random_seed=seed)
        assert model.Status == gp.GRB.OPTIMAL
        model.cleanup()
        model.explain(x, y=0, norm=1,
                      return_callback=True,
                      num_workers=num_workers,
                      random_seed=seed)
        assert len(model.callback.sollist) != 0

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

    try:
        _ = model.explain(x, y=0, norm=1,
                          return_callback=False,
                          num_workers=num_workers,
                          random_seed=seed)
        assert model.callback is None or len(model.callback.sollist) == 0
        model.cleanup()
        _ = model.explain(x, y=0, norm=1,
                          return_callback=True,
                          num_workers=num_workers,
                          random_seed=seed)
        assert model.callback is None or len(model.callback.sollist) != 0
    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")
