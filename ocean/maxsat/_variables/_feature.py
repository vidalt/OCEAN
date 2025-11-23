from collections.abc import Mapping

from ...feature import Feature
from ...feature._keeper import FeatureKeeper
from ...typing import Key
from .._base import BaseModel, Var


class FeatureVar(Var, FeatureKeeper):
    X_VAR_NAME_FMT: str = "x[{name}]"

    _x: int
    _u: Mapping[Key, int]
    _mu: Mapping[Key, int]

    def __init__(self, feature: Feature, name: str) -> None:
        Var.__init__(self, name=name)
        FeatureKeeper.__init__(self, feature=feature)

    def build(self, model: BaseModel) -> None:
        if self.is_binary:
            self._x = self._add_x(model)
        if self.is_numeric:
            self._mu = self._add_mu(model)
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
            return self._xget_numeric(mu)
        if mu is not None:
            msg = "Get by 'mu' is only supported for numeric features"
            raise ValueError(msg)
        return self._x

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
            code: model.add_var(name=f"{name}[{code}]") for code in self.codes
        }

    def _add_mu(self, model: BaseModel) -> Mapping[Key, int]:
        name = self._name.format(name=self._name)
        return {
            lv: model.add_var(name=f"{name}[{lv}]")
            for lv in range(len(self.levels))
        }

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
            msg = "Mu is required for numeric features get"
            raise ValueError(msg)
        if mu not in range(len(self.levels)):
            msg = f"Mu '{mu}' not found in the feature levels"
            raise ValueError(msg)
        return self._mu[mu]
