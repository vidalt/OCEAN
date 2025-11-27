from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..abc import Mapper
from ..typing import Array1D, BaseExplanation, Key, Number
from ._env import ENV
from ._variables import FeatureVar


class Explanation(Mapper[FeatureVar], BaseExplanation):
    _epsilon: float = float(np.finfo(np.float32).eps)
    _x: Array1D = np.zeros((0,), dtype=int)

    def vget(self, i: int) -> int:
        name = self.names[i]
        if self[name].is_one_hot_encoded:
            code = self.codes[i]
            return self[name].xget(code=code)
        if self[name].is_numeric:
            j: int = int(
                np.searchsorted(self[name].levels, self._x[i], side="left")  # pyright: ignore[reportUnknownArgumentType]
            )
            return self[name].xget(mu=j)
        return self[name].xget()

    def _get_active_mu_index(
        self,
        name: Key,
        for_discrete: bool = False,  # noqa: FBT001, FBT002
    ) -> int:
        """
        Find which mu variable is set to true for a numeric feature.

        Returns:
            Index of the active mu variable, or 0 if none found.

        """
        if for_discrete:
            # For discrete: one mu per level
            n_vars = len(self[name].levels)
        else:
            # For continuous: one mu per interval
            n_vars = len(self[name].levels) - 1
        for mu_idx in range(n_vars):
            var = self[name].xget(mu=mu_idx)
            if ENV.solver.model(var) > 0:
                return mu_idx
        return 0  # Default to first if none found

    def to_series(self) -> "pd.Series[float]":
        values: list[float] = []
        for f in range(self.n_columns):
            name = self.names[f]
            if self[name].is_one_hot_encoded:
                code = self.codes[f]
                var = self[name].xget(code=code)
                values.append(ENV.solver.model(var))
            elif self[name].is_continuous:
                mu_idx = self._get_active_mu_index(name, for_discrete=False)
                values.append(
                    self.format_continuous_value(
                        f, mu_idx, list(self[name].levels)
                    )
                )
            elif self[name].is_discrete:
                # For discrete features, mu[i] means value == levels[i]
                mu_idx = self._get_active_mu_index(name, for_discrete=True)
                levels = list(self[name].levels)
                discrete_val = int(levels[mu_idx])
                values.append(
                    self.format_discrete_value(
                        f, discrete_val, self[name].levels
                    )
                )
            elif self[name].is_binary:
                var = self[name].xget()
                values.append(ENV.solver.model(var))
            else:
                var = self[name].xget()
                values.append(ENV.solver.model(var))
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
        msg = "Not implemented."
        raise NotImplementedError(msg)

    def format_continuous_value(
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
