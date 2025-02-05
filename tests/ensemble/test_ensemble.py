from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean.ensemble import Ensemble
from ocean.tree import Tree

from ..utils import generate_data


def test_random_forest() -> None:
    seed = 42
    n_samples = 100
    n_classes = 3

    data, y, mapper = generate_data(
        seed=seed,
        n_samples=n_samples,
        n_classes=n_classes,
    )

    rf = RandomForestClassifier(random_state=seed, n_estimators=10)
    rf.fit(data, y)

    ensemble = Ensemble(rf, mapper=mapper)

    assert len(ensemble) == 10
    assert isinstance(ensemble[0], Tree)
    assert isinstance(ensemble[:2], tuple)

    for tree in ensemble:
        assert tree.shape == (1, n_classes)


def test_isolation_forest() -> None:
    seed = 42
    n_samples = 100

    data, _, mapper = generate_data(
        seed=seed,
        n_samples=n_samples,
        n_classes=2,
    )

    iforest = IsolationForest(random_state=seed, n_estimators=10)
    iforest.fit(data)

    ensemble = Ensemble(iforest, mapper=mapper)

    assert len(ensemble) == 10
    assert isinstance(ensemble[0], Tree)
    assert isinstance(ensemble[:2], tuple)

    for tree in ensemble:
        assert tree.shape == (1, 1)
