import numpy as np
import pytest

from ocean.abc import Mapper
from ocean.mip import Model
from ocean.tree import parse_ensembles, parse_trees

from ...utils import ENV
from ..utils import (
    MAX_DEPTH,
    MAX_SAMPLES,
    N_CLASSES,
    N_ESTIMATORS,
    N_ISOLATORS,
    N_SAMPLES,
    SEEDS,
    train_rf,
    train_rf_isolation,
)


def test_no_trees() -> None:
    msg = r"At least one tree is required."
    with pytest.raises(ValueError, match=msg):
        Model(trees=[], mapper=Mapper(), env=ENV)


def test_no_features() -> None:
    msg = r"At least one feature is required."
    rf, mapper = train_rf(42, 2, 2, 100, 2)
    trees = tuple(parse_trees(rf, mapper=mapper))
    with pytest.raises(ValueError, match=msg):
        Model(trees=trees, mapper=Mapper(), env=ENV)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_classes", N_CLASSES)
class TestNoIsolation:
    @staticmethod
    def test_no_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = parse_trees(clf, mapper=mapper)
        model = Model(trees=trees, mapper=mapper, env=ENV)
        expected_weights = np.ones(n_estimators, dtype=float)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == expected_weights.shape
        assert np.isclose(model.weights, expected_weights).all()

    @staticmethod
    def test_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = parse_trees(clf, mapper=mapper)
        generator = np.random.default_rng(seed)
        weights = generator.random(n_estimators).flatten()
        model = Model(trees=trees, mapper=mapper, weights=weights, env=ENV)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == weights.shape
        assert np.isclose(model.weights, weights).all()

    @staticmethod
    def test_invalid_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        generator = np.random.default_rng(seed)
        shapes = [generator.integers(n_estimators + 1, 2 * n_estimators + 1)]
        if n_estimators > 2:
            shapes += [generator.integers(1, n_estimators - 1)]
        for shape in shapes:
            weights = generator.random(shape).flatten()
            msg = r"The number of weights must match the number of trees."
            with pytest.raises(ValueError, match=msg):
                Model(trees=trees, mapper=mapper, weights=weights, env=ENV)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("n_isolators", N_ISOLATORS)
@pytest.mark.parametrize("max_samples", MAX_SAMPLES)
class TestIsolation:
    @staticmethod
    def test_no_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        n_isolators: int,
        max_samples: int,
    ) -> None:
        clf, ilf, mapper = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
        )
        trees = parse_ensembles(clf, ilf, mapper=mapper)
        model = Model(
            trees=trees,
            mapper=mapper,
            n_isolators=n_isolators,
            env=ENV,
        )
        expected_weights = np.ones(n_estimators, dtype=float)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == expected_weights.shape
        assert np.isclose(model.weights, expected_weights).all()
        assert model.n_isolators == n_isolators

    @staticmethod
    def test_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        n_isolators: int,
        max_samples: int,
    ) -> None:
        clf, ilf, mapper = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
        )
        trees = parse_ensembles(clf, ilf, mapper=mapper)
        generator = np.random.default_rng(seed)
        weights = generator.random(n_estimators).flatten()
        model = Model(
            trees=trees,
            mapper=mapper,
            n_isolators=n_isolators,
            weights=weights,
            env=ENV,
        )
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == weights.shape
        assert np.isclose(model.weights, weights).all()
        assert model.n_isolators == n_isolators

    @staticmethod
    def test_invalid_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        n_isolators: int,
        max_samples: int,
    ) -> None:
        clf, ilf, mapper = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
        )
        trees = tuple(parse_ensembles(clf, ilf, mapper=mapper))
        generator = np.random.default_rng(seed)
        shapes = [generator.integers(n_estimators + 1, 2 * n_estimators + 1)]
        if n_estimators > 2:
            shapes += [generator.integers(1, n_estimators - 1)]
        for shape in shapes:
            weights = generator.random(shape).flatten()
            msg = r"The number of weights must match the number of trees."
            with pytest.raises(ValueError, match=msg):
                Model(
                    trees=trees,
                    mapper=mapper,
                    n_isolators=n_isolators,
                    weights=weights,
                    env=ENV,
                )
