from collections.abc import Mapping

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model as cp

from ..abc import Mapper
from ..typing import Array1D, BaseExplanation, Key, Number
from ._env import ENV
from ._variables import FeatureVar


class Explanation(Mapper[FeatureVar], BaseExplanation):
    _epsilon: float = float(np.finfo(np.float32).eps)
    _x: Array1D = np.zeros((0,), dtype=int)

    def vget(self, i: int) -> cp.IntVar:
        name = self.names[i]
        if self[name].is_one_hot_encoded:
            code = self.codes[i]
            return self[name].xget(code)
        return self[name].xget()

    def to_series(self) -> "pd.Series[float]":
        values: list[float] = [
            ENV.solver.Value(v) for v in map(self.vget, range(self.n_columns))
        ]
        for f in range(self.n_columns):
            name = self.names[f]
            value = ENV.solver.Value(self.vget(f))
            if self[name].is_continuous:
                values[f] = self.format_value(
                    f, int(value), list(self[name].levels)
                )
            elif self[name].is_discrete:
                values[f] = self.format_discrete_value(
                    f, value, self[name].thresholds
                )
        return pd.Series(values, index=self.columns)

    def to_numpy(self) -> Array1D:
        return (
            self.to_series()
            .to_frame()
            .T[self.columns]
            .to_numpy()
            .flatten()
            .astype(np.float64)
        )

    @property
    def x(self) -> Array1D:
        return self.to_numpy()

    @property
    def value(self) -> Mapping[Key, Key | Number]:
        solver = ENV.solver

        def get(v: FeatureVar) -> Key | Number:
            if v.is_one_hot_encoded:
                for code in v.codes:
                    if np.isclose(solver.Value(v.xget(code)), 1.0):
                        return code
            if v.is_numeric:
                f = [val for _, val in self.items()].index(v)
                if v.is_discrete:
                    val = int(solver.Value(v.xget()))
                    return self.format_discrete_value(f, val, v.thresholds)
                idx = int(solver.Value(v.xget()))
                return self.format_value(
                    f,
                    idx,
                    list(v.levels),
                )
            x = v.xget()
            return solver.Value(x)

        return self.reduce(get)

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
            value = float(levels[idx]) + self._epsilon
        else:
            value = float(levels[idx + 1]) - self._epsilon
        return value

    def format_discrete_value(
        self,
        f: int,
        val: int,
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

    @property
    def query(self) -> Array1D:
        return self._x

    @query.setter
    def query(self, value: Array1D) -> None:
        self._x = value

    def __repr__(self) -> str:
        mapping = self.value
        prefix = f"{self.__class__.__name__}:\n"
        root = self._repr(mapping)
        suffix = ""

        return prefix + root + suffix


__all__ = ["Explanation"]
