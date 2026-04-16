from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..abc import Mapper
from ..typing import Array1D, BaseExplanation, Key, Number
from ._env import ENV
from ._variables import FeatureVar


class Explanation(Mapper[FeatureVar], BaseExplanation):
    """Concrete explanation container returned by the MaxSAT backend."""

    _x: Array1D = np.zeros((0,), dtype=int)

    def vget(self, i: int) -> int:
        name = self.names[i]
        if self[name].is_one_hot_encoded:
            code = self.codes[i]
            return self[name].xget(code=code)
        if self[name].is_numeric:
            if self[name].has_threshold_encoding:
                idx = self._get_threshold_index(name)
                if idx is None:
                    msg = "No threshold variable is available for this feature."
                    raise ValueError(msg)
                return self[name].xget(mu=idx)
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
                if self[name].has_threshold_encoding:
                    values.append(
                        self.format_threshold_continuous_value(
                            f,
                            name,
                        )
                    )
                else:
                    mu_idx = self._get_active_mu_index(name, for_discrete=False)
                    values.append(
                        self.format_continuous_value(
                            f,
                            mu_idx,
                            list(self[name].levels),
                        )
                    )
            elif self[name].is_discrete:
                if self[name].has_threshold_encoding:
                    values.append(self.format_threshold_discrete_value(f, name))
                else:
                    # For discrete features, mu[i] means value == levels[i]
                    mu_idx = self._get_active_mu_index(name, for_discrete=True)
                    levels = list(self[name].levels)
                    discrete_val = int(levels[mu_idx])
                    values.append(
                        self.format_discrete_value(
                            f,
                            discrete_val,
                            self[name].levels,
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
        def get(v: FeatureVar) -> Key | Number:
            if v.is_one_hot_encoded:
                for code in v.codes:
                    if ENV.solver.model(v.xget(code)) > 0:
                        return code
            if v.is_numeric:
                f = list(self.values()).index(v)
                if v.has_threshold_encoding:
                    if v.is_discrete:
                        return self.format_threshold_discrete_value(
                            f,
                            self.names[f],
                        )
                    return self.format_threshold_continuous_value(
                        f,
                        self.names[f],
                    )
                if v.is_discrete:
                    idx = self._get_active_mu_index(
                        self.names[f],
                        for_discrete=True,
                    )
                    val = int(v.levels[idx])
                    return self.format_discrete_value(
                        f,
                        val,
                        v.levels,
                    )
                idx = self._get_active_mu_index(
                    self.names[f],
                    for_discrete=False,
                )
                return self.format_continuous_value(
                    f,
                    idx,
                    list(v.levels),
                )
            x = v.xget()
            return int(ENV.solver.model(x))

        return self.reduce(get)

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
            value = self._next_float32_up(levels[idx])
        else:
            value = self._next_float32_down(levels[idx + 1])
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

    def _get_threshold_states(self, name: Key) -> list[bool]:
        thresholds = self[name].split_threshold_values
        return [
            bool(ENV.solver.model(self[name].xget(mu=idx)) > 0)
            for idx in range(len(thresholds))
        ]

    def _get_threshold_index(self, name: Key) -> int | None:
        states = self._get_threshold_states(name)
        for idx, state in enumerate(states):
            if state:
                return idx
        if states:
            return len(states) - 1
        return None

    def format_threshold_continuous_value(self, f: int, name: Key) -> float:
        thresholds = list(self[name].split_threshold_values)
        if len(thresholds) == 0:
            if self.query.shape[0] != 0:
                return float(np.asarray(self.query, dtype=float).ravel()[f])
            return float(self[name].levels[0] + self[name].levels[-1]) / 2

        query_arr = np.asarray(self.query, dtype=float).ravel()
        query_val = float(query_arr[f]) if self.query.shape[0] != 0 else None
        first_true = next(
            (
                idx
                for idx, state in enumerate(self._get_threshold_states(name))
                if state
            ),
            None,
        )
        result: float
        if first_true == 0:
            upper = thresholds[0]
            if query_val is not None and query_val <= upper:
                result = query_val
            else:
                result = self._next_float32_down(upper)
        elif first_true is None:
            lower = thresholds[-1]
            if query_val is not None and query_val > lower:
                result = query_val
            else:
                result = self._next_float32_up(lower)
        else:
            lower = thresholds[first_true - 1]
            upper = thresholds[first_true]
            if query_val is not None and lower < query_val <= upper:
                result = query_val
            elif query_val is not None and query_val <= lower:
                result = self._next_float32_up(lower)
            else:
                result = self._next_float32_down(upper)
        return result

    def format_threshold_discrete_value(self, f: int, name: Key) -> float:
        levels = np.asarray(self[name].levels, dtype=np.float64)
        thresholds = list(self[name].split_threshold_values)
        if len(thresholds) == 0:
            if self.query.shape[0] != 0:
                return float(np.asarray(self.query, dtype=float).ravel()[f])
            return float(levels[0])

        query_arr = np.asarray(self.query, dtype=float).ravel()
        query_val = float(query_arr[f]) if self.query.shape[0] != 0 else None
        first_true = next(
            (
                idx
                for idx, state in enumerate(self._get_threshold_states(name))
                if state
            ),
            None,
        )
        if first_true == 0:
            lower = float("-inf")
        elif first_true is None:
            lower = thresholds[-1]
        else:
            lower = thresholds[first_true - 1]

        upper = float("inf") if first_true is None else thresholds[first_true]
        candidates = levels[(levels > lower) & (levels <= upper)]
        if candidates.size == 0:
            if first_true == 0:
                candidates = np.array([levels[0]])
            elif first_true is None:
                candidates = np.array([levels[-1]])
            else:
                insertion = np.searchsorted(levels, upper, side="left")
                insertion = np.min(insertion, len(levels) - 1)
                candidates = np.array([levels[insertion]])
        if query_val is not None and np.any(np.isclose(candidates, query_val)):
            return query_val
        return float(candidates[0])

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
