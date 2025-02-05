from collections.abc import Hashable, Mapping

import gurobipy as gp
import numpy as np
import pandas as pd

from ..typing import FloatArray1D
from .variable import FeatureVar


class Solution:
    _features: Mapping[Hashable, FeatureVar]

    def __init__(self, features: Mapping[Hashable, FeatureVar]) -> None:
        self._features = features

    def to_series(self, *, attr: str = gp.GRB.Attr.X) -> "pd.Series[float]":
        index = []
        values: list[float] = []

        for name, feature in self._features.items():
            if not feature.is_one_hot_encoded:
                index.append(name)
                value = getattr(feature, attr)
                values.append(value)

        series = pd.Series(values, index=index).astype(float)

        index.clear()
        values.clear()

        for name, feature in self._features.items():
            if not feature.is_one_hot_encoded:
                continue

            for code in feature.codes:
                index.append((name, code))
                value = getattr(feature[code], attr)
                values.append(value)

        encoded = pd.Series(values, index=index).astype(float)
        if encoded.empty:
            return series

        series.index = pd.MultiIndex.from_product([series.index, [""]])
        return pd.concat([series, encoded]).astype(float)

    def to_numpy(
        self,
        *,
        columns: "pd.Index[str] | pd.MultiIndex",
        attr: str = gp.GRB.Attr.X,
    ) -> FloatArray1D:
        return (
            self.to_series(attr=attr)
            .loc[columns]
            .to_frame()
            .T.to_numpy()
            .flatten()
            .astype(np.float64)
        )
