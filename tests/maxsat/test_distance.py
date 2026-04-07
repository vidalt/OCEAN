import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import MaxSATExplainer

from ..distance_utils import manual_postprocessed_distance
from ..utils import generate_data

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="tests for non-windows platforms"
)


def test_get_distance_requires_explanation() -> None:
    data, y, mapper = generate_data(seed=42, n_samples=50, n_classes=2)
    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=3,
        max_depth=2,
    )
    clf.fit(data, y)
    model = MaxSATExplainer(clf, mapper=mapper)

    with pytest.raises(RuntimeError, match="No explanation has been computed"):
        model.get_distance()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_get_distance_matches_manual_postprocessed_distance(seed: int) -> None:
    data, y, mapper = generate_data(seed=seed, n_samples=100, n_classes=2)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)
    model = MaxSATExplainer(clf, mapper=mapper)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])
    explanation = model.explain(
        x,
        y=target,
        norm=1,
        random_seed=seed,
    )

    assert explanation is not None

    expected = manual_postprocessed_distance(explanation, norm=1)
    assert model.get_distance() == pytest.approx(expected)
    assert model.get_distance() == pytest.approx(expected)
