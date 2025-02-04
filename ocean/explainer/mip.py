import gurobipy as gp
import pandas as pd

from ..ensemble import Ensemble
from ..feature import FeatureMapper
from ..mip import Model, TreeVar
from ..typing import BaseEnsemble, FloatArray1D


class MIPExplainer(Model):
    _ensemble: Ensemble
    _mapper: FeatureMapper

    def __init__(
        self,
        ensemble: BaseEnsemble,
        *,
        mapper: FeatureMapper,
        weights: FloatArray1D | None = None,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: float = Model.DEFAULT_EPSILON,
        num_epsilon: float = Model.DEFAULT_NUM_EPSILON,
        model_type: Model.Type = Model.Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        self._mapper = mapper
        self._ensemble = Ensemble(ensemble=ensemble, mapper=mapper)
        Model.__init__(
            self,
            self._ensemble,
            mapper,
            weights=weights,
            name=name,
            env=env,
            epsilon=epsilon,
            num_epsilon=num_epsilon,
            model_type=model_type,
            flow_type=flow_type,
        )
        Model.build(self)

    def add_objective(
        self,
        x: FloatArray1D,
        *,
        norm: int = 2,
        sense: int = gp.GRB.MINIMIZE,
    ) -> None:
        objective = self._add_objective(x=x, norm=norm)
        self.setObjective(objective, sense=sense)

    def _add_objective(
        self,
        x: FloatArray1D,
        norm: int,
    ) -> gp.LinExpr | gp.QuadExpr:
        series = pd.Series(x, index=self._mapper.columns)
        match norm:
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
