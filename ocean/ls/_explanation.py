from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..abc import Mapper
from ..feature import Feature
from ..typing import Array1D, BaseExplanation, Key, Number

if TYPE_CHECKING:
    from collections.abc import Mapping


class Explanation(Mapper[Feature], BaseExplanation):
    """Concrete explanation container returned by the LS heuristics."""

    _x: Array1D
    _query: Array1D

    def __init__(
        self,
        mapper: Mapper[Feature],
        x: Array1D,
        query: Array1D,
    ) -> None:
        super().__init__(mapper)
        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._query = np.asarray(query, dtype=np.float64).ravel()

    def to_series(self) -> pd.Series[float]:
        return pd.Series(self._x, index=self.columns)

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

    @property
    def value(self) -> Mapping[Key, Key | Number]:
        values: dict[Key, Key | Number] = {}
        for name, feature in self.items():
            if feature.is_one_hot_encoded:
                values[name] = self._active_code(name, feature)
            else:
                idx = self.idx.get(name)
                value = float(self._x[idx])
                values[name] = int(value) if feature.is_binary else value
        return values

    def _active_code(self, name: Key, feature: Feature) -> Key:
        for code in feature.codes:
            idx = self.idx.get(name, code)
            if np.isclose(self._x[idx], 1.0):
                return code
        return feature.codes[0]

    @property
    def query(self) -> Array1D:
        return self._query

    @query.setter
    def query(self, value: Array1D) -> None:
        self._query = np.asarray(value, dtype=np.float64).ravel()

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:  # noqa: PLW3201
        array: np.ndarray = np.asarray(self.x, dtype=dtype)
        return array

    def __repr__(self) -> str:
        mapping = self.value
        prefix = f"{self.__class__.__name__}:\n"
        root = str(self._repr(mapping))
        return prefix + root


__all__ = ["Explanation"]
