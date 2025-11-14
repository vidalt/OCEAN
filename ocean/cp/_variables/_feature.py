from collections.abc import Mapping

import numpy as np
from ortools.sat.python import cp_model as cp

from ...feature import Feature
from ...feature._keeper import FeatureKeeper
from ...typing import Key
from .._base import BaseModel, Var


class FeatureVar(Var, FeatureKeeper):
    X_VAR_NAME_FMT: str = "x[{name}]"

    _x: cp.IntVar
    _u: Mapping[Key, cp.IntVar]
    _objvar: cp.IntVar

    def __init__(self, feature: Feature, name: str) -> None:
        Var.__init__(self, name=name)
        FeatureKeeper.__init__(self, feature=feature)

    def build(self, model: BaseModel) -> None:
        if not self.is_one_hot_encoded:
            self._x = self._add_x(model)
        if self.is_one_hot_encoded:
            self._u = self._add_u(model)

    def xget(self, code: Key | None = None) -> cp.IntVar:
        if self.is_one_hot_encoded:
            return self._xget_one_hot_encoded(code)
        if code is not None:
            msg = "Get by code is only supported for one-hot encoded features"
            raise ValueError(msg)
        return self._x

    def objvarget(self) -> cp.IntVar:
        if not self.is_discrete and not self.is_continuous:
            msg = (
                "The 'objvarget' method is only supported for continuous and discrete features"
            )
            raise ValueError(msg)
        return self._objvar

    def _add_x(self, model: BaseModel) -> cp.IntVar:
        name = self.X_VAR_NAME_FMT.format(name=self._name)

        # Case when the feature is one-hot encoded.
        if self.is_one_hot_encoded:
            msg = "One-hot encoded features are not for x"
            raise ValueError(msg)

        # Case when the feature is binary.
        if self.is_binary:
            return self._add_binary(model, name)

        # Case when the feature is continuous or discrete
        if self.is_continuous:
            return self._add_continuous(model, name)

        return self._add_discrete(model, name)

    def _add_u(self, model: BaseModel) -> Mapping[Key, cp.IntVar]:
        name = self._name.format(name=self._name)
        u = self._add_one_hot_encoded(model=model, name=name)
        model.AddExactlyOne(u.values())
        return u

    def _add_one_hot_encoded(
        self,
        model: BaseModel,
        name: str,
    ) -> Mapping[Key, cp.IntVar]:
        return {
            code: model.NewBoolVar(f"{name}[{code}]") for code in self.codes
        }

    @staticmethod
    def _add_binary(model: BaseModel, name: str) -> cp.IntVar:
        return model.NewBoolVar(name)

    def _add_continuous(self, model: BaseModel, name: str) -> cp.IntVar:
        self._objvar = model.NewIntVar(0, 42, f"u_{name}") # arbitrary, will be adapted for each query
        m = len(self.levels)
        return model.NewIntVar(0, m - 2, name)

    def _add_discrete(self, model: BaseModel, name: str) -> cp.IntVar:
        if len(self.thresholds) != 0:
            values = [
                val for v in self.thresholds for val in [int(v), int(v) + 1]
            ]
        else:
            values = np.asarray(self.levels).astype(int).tolist()
        #print("values for discrete feature ", self._name, ": ", values) #debug
        val = cp.Domain.FromValues(values)
        self._objvar = model.NewIntVar(0, int(self.levels[-1]), f"u_{name}")
        return model.NewIntVarFromDomain(val, name)

    def _xget_one_hot_encoded(self, code: Key | None) -> cp.IntVar:
        if code is None:
            msg = "Code is required for one-hot encoded features get"
            raise ValueError(msg)
        if code not in self.codes:
            msg = f"Code '{code}' not found in the feature codes"
            raise ValueError(msg)
        return self._u[code]
