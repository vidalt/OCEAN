from functools import partial
from itertools import chain

import gurobipy as gp
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ..abc import Mapper
from ..ensemble import Ensemble
from ..feature import Feature
from ..mip import Model, Solution, TreeVar
from ..typing import Array1D, NonNegativeInt, PositiveInt


class MixedIntegerProgramExplainer(Model):
    _garbage: list[gp.Var | gp.MVar | gp.Constr | gp.MConstr]

    def __init__(
        self,
        ensemble: RandomForestClassifier,
        *,
        mapper: Mapper[Feature],
        weights: Array1D | None = None,
        isolation: IsolationForest | None = None,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: float = Model.DEFAULT_EPSILON,
        num_epsilon: float = Model.DEFAULT_NUM_EPSILON,
        model_type: Model.Type = Model.Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        ensembles = (ensemble,) if isolation is None else (ensemble, isolation)
        n_isolators = len(isolation) if isolation is not None else 0
        parser = partial(Ensemble, mapper=mapper)
        trees = chain.from_iterable(map(parser, ensembles))
        Model.__init__(
            self,
            trees,
            mapper=mapper,
            weights=weights,
            n_isolators=n_isolators,
            name=name,
            env=env,
            epsilon=epsilon,
            num_epsilon=num_epsilon,
            model_type=model_type,
            flow_type=flow_type,
        )
        self.build()

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: PositiveInt,
    ) -> Solution:
        self.add_objective(x, norm=norm)
        self.set_majority_class(y=y)
        self.optimize()
        if self.SolCount == 0:
            msg = "No solution found. Please check the model constraints."
            raise RuntimeError(msg)
        return self.solution
