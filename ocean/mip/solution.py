from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..abc import Mapper
from ..typing import Array1D, Key
from .variable import FeatureVar


class Solution(Mapper[FeatureVar]):
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

    def __repr__(self) -> str:
        def get(v: FeatureVar) -> float | Key:
            if not v.is_one_hot_encoded:
                if v.is_binary:
                    return int(v.X)

                if np.isclose(v.X, 0.0):
                    return 0.0
                return v.X
            for code in v.codes:
                if v[code].X == 1:
                    return code
            msg = "Unreachable code"
            raise ValueError(msg)

        mapping = self.reduce(get)
        prefix = f"{self.__class__.__name__}:\n"
        root = self._repr(mapping)
        suffix = ""

        return prefix + root + suffix

    @staticmethod
    def _repr(mapping: Mapping[Key, float | Key]) -> str:
        length = max(len(str(k)) for k in mapping)
        lines = [
            f"{str(k).ljust(length + 1)} : {v}" for k, v in mapping.items()
        ]
        return "\n".join(lines)
