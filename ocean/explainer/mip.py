from functools import partial
from itertools import chain

import gurobipy as gp
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ..ensemble import Ensemble
from ..feature import FeatureMapper
from ..mip import Model, TreeVar
from ..typing import FloatArray1D


class MIPExplainer(Model):
    _mapper: FeatureMapper
    _garbage: list[gp.Var | gp.MVar | gp.Constr | gp.MConstr]

    def __init__(
        self,
        ensemble: RandomForestClassifier,
        *,
        mapper: FeatureMapper,
        weights: FloatArray1D | None = None,
        isolation: IsolationForest | None = None,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: float = Model.DEFAULT_EPSILON,
        num_epsilon: float = Model.DEFAULT_NUM_EPSILON,
        model_type: Model.Type = Model.Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        self._mapper = mapper
        ensembles = (ensemble,) if isolation is None else (ensemble, isolation)
        n_isolators = len(isolation) if isolation is not None else 0
        func = partial(Ensemble, mapper=mapper)
        trees = chain.from_iterable(map(func, ensembles))
        Model.__init__(
            self,
            trees,
            mapper,
            weights=weights,
            n_isolators=n_isolators,
            name=name,
            env=env,
            epsilon=epsilon,
            num_epsilon=num_epsilon,
            model_type=model_type,
            flow_type=flow_type,
        )
        Model.build(self)
        self._garbage = []

    def add_objective(
        self,
        x: FloatArray1D,
        *,
        norm: int = 1,
        sense: int = gp.GRB.MINIMIZE,
    ) -> None:
        objective = self._add_objective(x=x, norm=norm)
        self.setObjective(objective, sense=sense)

    def cleanup(self) -> None:
        self.remove(self._garbage)
        self._garbage.clear()

    def _add_objective(
        self,
        x: FloatArray1D,
        norm: int,
    ) -> gp.LinExpr | gp.QuadExpr:
        series = pd.Series(x, index=self._mapper.columns)
        match norm:
            case 1:
                return self._add_l1(series)
            case 2:
                return self._add_l2(series)
            case _:
                msg = f"Unsupported norm: {norm}"
                raise ValueError(msg)

    def _add_l2(self, series: "pd.Series[float]") -> gp.QuadExpr:
        obj = gp.QuadExpr()
        value: float = 0.0
        for name, var in self.features.items():
            if not var.is_one_hot_encoded:
                value = series.xs(name, level=0).to_numpy()[0]
                obj += (var.x - value) ** 2
            else:
                for code in var.codes:
                    value = series.xs((name, code), level=(0, 1)).to_numpy()[0]
                    obj += (var[code] - value) ** 2
        return obj

    def _add_l1(self, series: "pd.Series[float]") -> gp.LinExpr:
        value: float = 0.0
        n_vars = len(series)
        u = self.addMVar(n_vars, name="u")
        self._garbage.append(u)
        for name, var in self.features.items():
            if not var.is_one_hot_encoded:
                value = series.xs(name, level=0).to_numpy()[0]
                if series.index.nlevels == 1:
                    idx = series.index.to_list().index(name)
                else:
                    idx = series.index.to_list().index((name, ""))
                cons = self.addConstr(var.x - value <= u[idx])
                self._garbage.append(cons)
                cons = self.addConstr(var.x - value >= -u[idx])
                self._garbage.append(cons)
            else:
                for code in var.codes:
                    value = series.xs((name, code), level=(0, 1)).to_numpy()[0]
                    idx = series.index.to_list().index((name, code))
                    cons = self.addConstr(var[code] - value <= u[idx])
                    self._garbage.append(cons)
                    cons = self.addConstr(var[code] - value >= -u[idx])
                    self._garbage.append(cons)
        return u.sum().item()
