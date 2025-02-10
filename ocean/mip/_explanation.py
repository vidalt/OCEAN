from collections.abc import Mapping

import gurobipy as gp
import numpy as np
import pandas as pd

from ..abc import Mapper
from ..typing import Array1D, BaseExplanation, Key, Number
from ._variables import FeatureVar


class Explanation(Mapper[FeatureVar], BaseExplanation):
    def vget(self, i: int) -> gp.Var:
        name = self.names[i]
        if self[name].is_one_hot_encoded:
            code = self.codes[i]
            return self[name].xget(code)
        return self[name].xget()

    def to_series(self) -> "pd.Series[float]":
        values = [v.X for v in map(self.vget, range(self.n_columns))]
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
        def get(v: FeatureVar) -> Key | Number:
            if v.is_one_hot_encoded:
                for code in v.codes:
                    if np.isclose(v.xget(code).X, 1.0):
                        return code
            x = v.xget().X
            return 0 if np.isclose(x, 0.0) else x

        return self.reduce(get)

    def __repr__(self) -> str:
        mapping = self.value
        prefix = f"{self.__class__.__name__}:\n"
        root = self._repr(mapping)
        suffix = ""

        return prefix + root + suffix


__all__ = ["Explanation"]
