from collections.abc import Mapping

import numpy as np

from ...feature import Feature
from ...feature._keeper import FeatureKeeper
from ...typing import Key
from .._base import BaseModel, Var


class FeatureVar(Var, FeatureKeeper):
    """MaxSAT variable bundle associated with a single parsed feature."""

    X_VAR_NAME_FMT: str = "x[{name}]"
    TH_VAR_NAME_FMT: str = "{name}[{idx}]"

    _x: int
    _u: Mapping[Key, int]
    _mu: Mapping[Key, int]
    _th: Mapping[Key, int]
    _threshold_values: tuple[float, ...]
    _threshold_encoding: bool = False
    _hard_voting: bool = False

    def __init__(self, feature: Feature, name: str) -> None:
        Var.__init__(self, name=name)
        FeatureKeeper.__init__(self, feature=feature)

    def build(self, model: BaseModel) -> None:
        self._hard_voting = bool(getattr(model, "_hard_voting", False))
        self._threshold_encoding = self.is_continuous or (
            self.is_discrete and self._hard_voting
        )
        if self.is_binary:
            self._x = self._add_x(model)
        if self.is_numeric:
            if self._threshold_encoding:
                self._th = self._add_th(model)
                self._add_threshold_order(model)
                if self.is_discrete:
                    self._add_discrete_feasibility(model)
            else:
                self._mu = self._add_mu(model)
                model.add_exactly_one(list(self._mu.values()))
        if self.is_one_hot_encoded:
            self._u = self._add_u(model)

    def xget(self, code: Key | None = None, mu: Key | None = None) -> int:
        if mu is not None and code is not None:
            msg = "Cannot get both 'mu' and 'code' at the same time"
            raise ValueError(msg)
        if self.is_one_hot_encoded:
            return self._xget_one_hot_encoded(code)
        if code is not None:
            msg = "Get by code is only supported for one-hot encoded features"
            raise ValueError(msg)
        if self.is_numeric:
            if self._threshold_encoding:
                return self._xget_numeric_threshold(mu)
            return self._xget_numeric(mu)
        if mu is not None:
            msg = "Get by 'mu' is only supported for numeric features"
            raise ValueError(msg)
        return self._x

    @property
    def has_threshold_encoding(self) -> bool:
        return self._threshold_encoding and self.is_numeric

    @property
    def split_threshold_values(self) -> tuple[float, ...]:
        if not self.is_numeric:
            msg = "Split thresholds are only available for numeric features."
            raise ValueError(msg)
        return self._threshold_values

    def threshold_index(self, value: float) -> int:
        if not self.has_threshold_encoding:
            msg = "Threshold indices are only available in hard-voting mode."
            raise ValueError(msg)
        values = np.asarray(self._threshold_values, dtype=np.float64)
        matches = np.flatnonzero(np.isclose(values, value))
        if matches.size == 0:
            msg = f"Threshold '{value}' not found for this feature."
            raise ValueError(msg)
        return int(matches[0])

    def _add_x(self, model: BaseModel) -> int:
        if not self.is_binary:
            msg = "The '_add_x' method is only supported for binary features"
            raise ValueError(msg)
        name = self.X_VAR_NAME_FMT.format(name=self._name)
        return self._add_binary(model, name)

    def _add_u(self, model: BaseModel) -> Mapping[Key, int]:
        name = self._name.format(name=self._name)
        u = self._add_one_hot_encoded(model=model, name=name)
        model.add_exactly_one(list(u.values()))
        return u

    def _add_one_hot_encoded(
        self,
        model: BaseModel,
        name: str,
    ) -> Mapping[Key, int]:
        return {
            code: model.add_var(
                name=f"{name}[{code}]",
            )
            for code in self.codes
        }

    def _add_mu(self, model: BaseModel) -> Mapping[Key, int]:
        name = self._name.format(name=self._name)
        if self.is_discrete:
            # For discrete features: one mu variable per level (value)
            # mu[i] means value == levels[i]
            n_values = len(self.levels)
            return {
                lv: model.add_var(
                    name=f"{name}[{lv}]",
                )
                for lv in range(n_values)
            }
        # For continuous features: n-1 mu variables for n levels (intervals)
        # mu[i] means value in interval (levels[i], levels[i+1]]
        n_intervals = len(self.levels) - 1
        return {
            lv: model.add_var(
                name=f"{name}[{lv}]",
            )
            for lv in range(n_intervals)
        }

    def _get_split_threshold_values(self) -> tuple[float, ...]:
        minimum_levels: int = 2
        if self.is_continuous:
            if len(self.levels) <= minimum_levels:
                return ()
            return tuple(map(float, self.levels[1:-1]))
        return tuple(map(float, self.thresholds))

    def _add_th(self, model: BaseModel) -> Mapping[Key, int]:
        self._threshold_values = self._get_split_threshold_values()
        name = self._name.format(name=self._name)
        return {
            idx: model.add_var(
                name=self.TH_VAR_NAME_FMT.format(name=name, idx=idx),
            )
            for idx in range(len(self._threshold_values))
        }

    def _add_threshold_order(self, model: BaseModel) -> None:
        for idx in range(len(self._threshold_values) - 1):
            model.add_hard([-self._th[idx], self._th[idx + 1]])

    def _add_discrete_feasibility(self, model: BaseModel) -> None:
        levels = np.asarray(self.levels, dtype=np.float64)
        thresholds = np.asarray(self._threshold_values, dtype=np.float64)
        for idx in range(len(thresholds) - 1):
            lower = thresholds[idx]
            upper = thresholds[idx + 1]
            feasible = np.any((levels > lower) & (levels <= upper))
            if not feasible:
                model.add_hard([self._th[idx], -self._th[idx + 1]])

    @staticmethod
    def _add_binary(model: BaseModel, name: str) -> int:
        return model.add_var(name=name)

    def _xget_one_hot_encoded(self, code: Key | None) -> int:
        if code is None:
            msg = "Code is required for one-hot encoded features get"
            raise ValueError(msg)
        if code not in self.codes:
            msg = f"Code '{code}' not found in the feature codes"
            raise ValueError(msg)
        return self._u[code]

    def _xget_numeric(self, mu: Key | None) -> int:
        if mu is None:
            msg = "mu is required to get numeric features"
            raise ValueError(msg)
        if self.is_discrete:
            # For discrete: mu[i] represents value levels[i]
            n_values = len(self.levels)
            if mu not in range(n_values):
                msg = f"mu '{mu}' not in values (0 to {n_values - 1})"
                raise ValueError(msg)
        else:
            # For continuous: mu[i] represents interval (levels[i], levels[i+1]]
            n_intervals = len(self.levels) - 1
            if mu not in range(n_intervals):
                msg = f"mu '{mu}' not in intervals (0 to {n_intervals - 1})"
                raise ValueError(msg)
        return self._mu[mu]

    def _xget_numeric_threshold(self, mu: Key | None) -> int:
        if mu is None:
            msg = "mu is required to get hard-voting numeric thresholds"
            raise ValueError(msg)
        n_thresholds = len(self._threshold_values)
        if mu not in range(n_thresholds):
            msg = f"mu '{mu}' not in thresholds (0 to {n_thresholds - 1})"
            raise ValueError(msg)
        return self._th[mu]
