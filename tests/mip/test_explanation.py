from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from ocean.mip import Explanation

if TYPE_CHECKING:
    from ocean.mip._variables import FeatureVar


@dataclass
class _SolutionValue:
    X: float


class _ContinuousFeature:
    def __init__(self, *, x: float, mu: tuple[float, ...]) -> None:
        self._x = x
        self._mu = mu
        self._levels = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    @property
    def levels(self) -> np.ndarray:
        return self._levels

    def xget(self) -> _SolutionValue:
        return _SolutionValue(self._x)

    def mget(self, key: int) -> _SolutionValue:
        return _SolutionValue(self._mu[key])


class _TestExplanation(Explanation):
    def continuous_index(self, feature: FeatureVar) -> int:
        return self._continuous_index(feature)


def test_continuous_index_uses_right_interval_at_boundary() -> None:
    explanation = _TestExplanation.__new__(_TestExplanation)
    feature = _ContinuousFeature(x=1.0, mu=(1.0, 1e-6))

    idx = explanation.continuous_index(cast("FeatureVar", feature))

    assert idx == 1


def test_continuous_index_uses_left_interval_for_zero_boundary_mu() -> None:
    explanation = _TestExplanation.__new__(_TestExplanation)
    feature = _ContinuousFeature(x=1.0, mu=(1.0, 0.0))

    idx = explanation.continuous_index(cast("FeatureVar", feature))

    assert idx == 0
