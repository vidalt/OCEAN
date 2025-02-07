import numpy as np
import pandas as pd

from ..abc import Mapper
from ..typing import Array1D
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
