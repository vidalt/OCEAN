import gurobipy as gp
import numpy as np

from ...feature import Feature
from ...feature._keeper import FeatureKeeper
from ...typing import Key
from .._base import BaseModel, Var


class FeatureVar(Var, FeatureKeeper):
    X_VAR_NAME_FMT: str = "x[{name}]"

    _x: gp.MVar
    _mu: gp.MVar

    def __init__(self, feature: Feature, name: str) -> None:
        Var.__init__(self, name=name)
        FeatureKeeper.__init__(self, feature=feature)

    def build(self, model: BaseModel) -> None:
        x = self._add_x(model)

        if self.is_numeric:
            mu = self._set_mu(model)
            model.addConstr(x.item() == self.weighted_x(mu=mu))
            self._mu = mu
        elif self.is_one_hot_encoded:
            model.addConstr(x.sum().item() == 1.0)

        self._x = x

    def xget(self, code: Key | None = None) -> gp.Var:
        if self.is_one_hot_encoded:
            return self._xget_one_hot_encoded(code)

        if code is not None:
            msg = "Get by code is only supported for one-hot encoded features"
            raise ValueError(msg)

        return self._x.item()

    def mget(self, key: int) -> gp.Var:
        if not self.is_numeric:
            msg = "The 'mget' method is only supported for numeric features"
            raise ValueError(msg)
        return self._mu[key].item()

    def _add_x(self, model: BaseModel) -> gp.MVar:
        name = self.X_VAR_NAME_FMT.format(name=self._name)

        # Case when the feature is one-hot encoded.
        if self.is_one_hot_encoded:
            return self._add_one_hot_encoded(model, name)

        # Case when the feature is binary.
        if self.is_binary:
            return self._add_binary(model, name)

        # Case when the feature is continuous or discrete.
        return self._add_numeric(model, name)

    def _set_mu(self, model: BaseModel) -> gp.MVar:
        vtype = gp.GRB.CONTINUOUS if self.is_continuous else gp.GRB.BINARY
        n = len(self.levels) - 1
        name = f"{self._name}_mu"
        lb, ub = 0.0, 1.0
        mu = model.addMVar(shape=n, vtype=vtype, lb=lb, ub=ub, name=name)

        for j in range(n - 1):
            model.addConstr(mu[j + 1] <= mu[j])

        return mu

    def _add_one_hot_encoded(self, model: BaseModel, name: str) -> gp.MVar:
        m = len(self.codes)
        vtype = gp.GRB.BINARY
        names = [f"{name}[{code}]" for code in self.codes]
        return model.addMVar(shape=m, vtype=vtype, name=names)

    @staticmethod
    def _add_binary(model: BaseModel, name: str) -> gp.MVar:
        vtype = gp.GRB.BINARY
        return model.addMVar(shape=1, vtype=vtype, name=name)

    @staticmethod
    def _add_numeric(model: BaseModel, name: str) -> gp.MVar:
        vtype = gp.GRB.CONTINUOUS
        lb = -gp.GRB.INFINITY
        return model.addMVar(shape=1, vtype=vtype, lb=lb, name=name)

    def weighted_x(self, mu: gp.MVar) -> gp.LinExpr:
        diff = np.diff(self.levels).astype(np.float64).flatten()
        return (np.min(self.levels) + (mu * diff).sum()).item()

    def _xget_one_hot_encoded(self, code: Key | None) -> gp.Var:
        if code is None:
            msg = "Code is required for one-hot encoded features get"
            raise ValueError(msg)
        if code not in self.codes:
            msg = f"Code '{code}' not found in the feature codes"
            raise ValueError(msg)
        j = self.codes.index(code)
        return self._x[j].item()
