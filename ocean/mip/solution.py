import numpy as np
import pandas as pd

from ..abc import Indexer, Mapper
from ..typing import Array1D, Key, NonNegativeInt
from .variable import FeatureVar

type SolutionIndex = Key | tuple[Key, Key]
type SolutionIndexer = Indexer[SolutionIndex, NonNegativeInt]


class Solution(Mapper[FeatureVar]):
    def __init__(self, mapper: Mapper[FeatureVar]) -> None:
        super().__init__(mapper)
        self._idx = self._add_indexer()

    @property
    def idx(self) -> SolutionIndexer:
        return self._idx

    def to_series(self) -> "pd.Series[float]":
        def get(i: int) -> float:
            name = self.names[i]
            feature = self[name]
            if not feature.is_one_hot_encoded:
                return feature.X
            code = self.codes[i]
            return feature[code].X

        values = [get(i) for i in range(self.n_columns)]
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

    def _add_indexer(self) -> SolutionIndexer:
        def get(key: SolutionIndex) -> NonNegativeInt:
            if key in self.names:
                return self.names.index(key)
            if not isinstance(key, tuple):
                msg = f"Key {key} not found in names"
                raise KeyError(msg)
            name, code = key
            if name not in self.names:
                msg = f"Key {name} not found in names"
                raise KeyError(msg)
            names = np.array(self.names)
            indices = tuple(map(int, np.where(names == name)[0]))
            codes = tuple(self.codes[j] for j in indices)
            if code not in codes:
                msg = f"Key {code} not found in codes"
                raise KeyError(msg)
            idx = codes.index(code)
            return indices[idx]

        return Indexer(get)
