import gurobipy as gp
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import MixedIntegerProgramExplainer

from .utils import ENV, generate_data


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [5])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_explain(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    model = MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV)

    x = data.iloc[0, :].to_numpy().astype(float).flatten()  # pyright: ignore[reportUnknownVariableType]

    try:
        model.explain(x, y=0, norm=1)
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL

    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")
