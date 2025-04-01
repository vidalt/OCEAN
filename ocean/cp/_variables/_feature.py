from ortools.sat.python import cp_model as cp

from ...feature import Feature
from ...feature._keeper import FeatureKeeper
from ...typing import Key
from .._base import BaseModel, Var


class FeatureVar(Var, FeatureKeeper):
    X_VAR_NAME_FMT: str = "x[{name}]"

    _x: cp.IntVar
    _u: dict[Key, cp.IntVar]
    _mu: list[cp.IntVar]

    def __init__(self, feature: Feature, name: str) -> None:
        Var.__init__(self, name=name)
        FeatureKeeper.__init__(self, feature=feature)

    def build(self, model: BaseModel) -> None:
        self._x = self._add_x(model)
        if self.is_numeric:
            mu = self._set_mu(model)
            model.add_map_domain(self.xget(), mu)
            self._mu = mu
        elif self.is_one_hot_encoded:
            u = self._add_u(model)
            model.Add(cp.LinearExpr.Sum(u.values()) == 1)
            self._u = u
            return

    def xget(self, code: Key | None = None) -> cp.IntVar:
        if self.is_one_hot_encoded:
            if code is None:
                msg = "Code was not provided for one-hot encoded feature"
                raise ValueError(msg)
            return self._u[code]

        if code is not None:
            msg = "Get by code is only supported for one-hot encoded features"
            raise ValueError(msg)

        return self._x

    def mget(self, key: int) -> cp.IntVar:
        if not self.is_numeric:
            msg = "The 'mget' method is only supported for numeric features"
            raise ValueError(msg)
        return self._mu[key]

    def _add_x(self, model: BaseModel) -> cp.IntVar:
        name = self.X_VAR_NAME_FMT.format(name=self._name)

        # Case when the feature is binary.
        if self.is_binary:
            return self._add_binary(model, name)

        # Case when the feature is continuous or discrete.
        return self._add_numeric(model, name)

    def _add_u(self, model: BaseModel) -> dict[Key, cp.IntVar]:
        name = self.X_VAR_NAME_FMT.format(name=self._name)
        return self._add_one_hot_encoded(model=model, name=name)

    def _set_mu(self, model: BaseModel) -> list[cp.IntVar]:
        m = len(self.levels)
        return [model.NewBoolVar(f"{self._name}_mu_{i}") for i in range(m)]

    def _add_one_hot_encoded(
        self,
        model: BaseModel,
        name: str,
    ) -> dict[Key, cp.IntVar]:
        return {
            code: model.NewBoolVar(f"{name}[{code}]") for code in self.codes
        }

    @staticmethod
    def _add_binary(model: BaseModel, name: str) -> cp.IntVar:
        return model.NewBoolVar(name)

    def _add_numeric(self, model: BaseModel, name: str) -> cp.IntVar:
        m = len(self.levels)
        return model.NewIntVar(0, m - 1, name)
