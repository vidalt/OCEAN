import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import MaxSATExplainer

from .utils import generate_data


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [5])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_maxsat_explain(
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
    model = MaxSATExplainer(clf, mapper=mapper)

    x = data.iloc[0, :].to_numpy().astype(float).flatten()
    # pyright: ignore[reportUnknownVariableType]
    y = clf.predict([x])[0]
    classes = np.unique(clf.predict(data.to_numpy())).astype(np.int64)  # pyright: ignore[reportUnknownArgumentType]
    for target in classes[classes != y]:
        exp = model.explain(
            x,
            y=target,
            norm=1,
            random_seed=seed,
        )
        assert model.Status == "OPTIMAL"
        assert exp is not None
        assert clf.predict([exp.to_numpy()])[0] == target
        model.cleanup()
