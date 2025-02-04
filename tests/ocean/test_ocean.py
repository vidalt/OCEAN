import gurobipy as gp
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean.ocean import MIPExplainer

from ..utils import ENV, generate_data


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [8])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_ocean(
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
    clf.fit(data.to_numpy(), y)
    model = MIPExplainer(
        ensemble=clf,
        mapper=mapper,
        weights=None,
        model_type=MIPExplainer.Type.MIP,
        env=ENV,
    )

    assert model is not None
    assert model.n_trees == n_estimators
    assert model.n_classes == n_classes

    model.set_majority_class(m_class=0, output=0)
    x = data.iloc[0, :].to_numpy().astype(float).flatten()  # pyright: ignore[reportUnknownVariableType]
    model.add_objective(x, norm=2)  # pyright: ignore[reportArgumentType]

    try:
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        assert model.solution is not None

    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")
