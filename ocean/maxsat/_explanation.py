from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..abc import Mapper
from ..typing import Array1D, BaseExplanation, Key, Number
from ._variables import FeatureVar

if TYPE_CHECKING:
    import pandas as pd


class Explanation(Mapper[FeatureVar], BaseExplanation):
    _epsilon: float = float(np.finfo(np.float32).eps)
    _x: Array1D = np.zeros((0,), dtype=int)

    def vget(self, i: int) -> int:
        msg = "Not implemented."
        raise NotImplementedError(msg)

    def to_series(self) -> "pd.Series[float]":
        msg = "Not implemented."
        raise NotImplementedError(msg)

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
        msg = "Not implemented."
        raise NotImplementedError(msg)

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
