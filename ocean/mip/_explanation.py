from collections.abc import Mapping

import gurobipy as gp
import numpy as np
import pandas as pd

from ..abc import Mapper
from ..typing import Array1D, BaseExplanation, Key, Number
from ._variables import FeatureVar


class Explanation(Mapper[FeatureVar], BaseExplanation):
    """Concrete explanation container returned by the MIP backend."""

    _atol: float = 1e-10
    _x: Array1D = np.zeros((0,), dtype=int)

    def vget(self, i: int) -> gp.Var:
        name = self.names[i]
        if self[name].is_one_hot_encoded:
            code = self.codes[i]
            return self[name].xget(code)
        return self[name].xget()

    def to_series(self) -> "pd.Series[float]":
        values = [v.X for v in map(self.vget, range(self.n_columns))]
        for i, f in enumerate(range(self.n_columns)):
            name = self.names[f]
            value = values[i]
            if self[name].is_continuous:
                idx = self._continuous_index(self[name])
                values[f] = self.format_value(f, idx, list(self[name].levels))
            elif self[name].is_discrete:
                values[f] = self.format_discrete_value(
                    f, value, self[name].thresholds
                )
        return pd.Series(values, index=self.columns)

    def to_numpy(self) -> Array1D:
        return (
            self
            .to_series()
            .to_frame()
            .T[self.columns]
            .to_numpy()
            .flatten()
            .astype(np.float64)
        )

    @property
    def x(self) -> Array1D:
        return self.to_numpy()

    def format_value(
        self,
        f: int,
        idx: int,
        levels: list[float],
    ) -> float:
        if self.query.shape[0] == 0:
            return float(levels[idx] + levels[idx + 1]) / 2
        j = 0
        query_arr = np.asarray(self.query, dtype=float).ravel()
        while query_arr[f] > levels[j + 1]:
            j += 1
        if j == idx:
            value = float(query_arr[f])
        elif j < idx:
            value = self._next_float32_up(levels[idx])
        else:
            value = self._next_float32_down(levels[idx + 1])
        return value

    def format_discrete_value(
        self,
        f: int,
        val: float,
        thresholds: Array1D,
    ) -> float:
        if self.query.shape[0] == 0:
            return val
        query_arr = np.asarray(self.query, dtype=float).ravel()
        j_x = np.searchsorted(thresholds, query_arr[f], side="left")
        j_val = np.searchsorted(thresholds, val, side="left")
        if j_x != j_val:
            return float(val)
        return float(query_arr[f])

    def _continuous_index(self, feature: FeatureVar) -> int:
        levels = np.asarray(feature.levels, dtype=float)
        n_intervals = len(levels) - 1
        x = float(feature.xget().X)
        idx = int(np.searchsorted(levels, x, side="left")) - 1

        close = np.flatnonzero(np.isclose(levels, x, rtol=0.0, atol=self._atol))
        if close.size > 0:
            k = int(close[0])
            if k <= 0:
                return 0
            if k >= n_intervals:
                return n_intervals - 1
            # A right branch only forces a small positive mu past the split.
            return k if self._atol < feature.mget(k).X else k - 1

        return max(0, min(idx, n_intervals - 1))

    @property
    def value(self) -> Mapping[Key, Key | Number]:
        def get(v: FeatureVar) -> Key | Number:
            if v.is_one_hot_encoded:
                for code in v.codes:
                    if np.isclose(v.xget(code).X, 1.0):
                        return code
            if v.is_continuous:
                f = list(self.values()).index(v)
                idx = self._continuous_index(v)
                return self.format_value(f, idx, list(v.levels))
            if v.is_discrete:
                f = list(self.values()).index(v)
                val = v.xget().X
                return self.format_discrete_value(f, val, v.thresholds)
            x = v.xget().X
            return 0 if np.isclose(x, 0.0) else x

        return self.reduce(get)

    def __repr__(self) -> str:
        mapping = self.value
        prefix = f"{self.__class__.__name__}:\n"
        root = self._repr(mapping)
        suffix = ""
        return prefix + root + suffix

    @property
    def query(self) -> Array1D:
        return self._x

    @query.setter
    def query(self, value: Array1D) -> None:
        self._x = value


__all__ = ["Explanation"]
