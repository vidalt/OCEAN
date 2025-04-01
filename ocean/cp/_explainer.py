from ortools.sat.python import cp_model as cp

from ..abc import Mapper
from ..feature import Feature
from ..tree import parse_ensembles
from ..typing import (
    Array1D,
    BaseExplainableEnsemble,
    BaseExplainer,
    NonNegativeInt,
    PositiveInt,
)
from ._explanation import Explanation
from ._model import Model


class Explainer(Model, BaseExplainer):
    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
        *,
        mapper: Mapper[Feature],
        weights: Array1D | None = None,
        solver: cp.CpSolver | None = None,
        epsilon: int = Model.DEFAULT_EPSILON,
        model_type: Model.Type = Model.Type.CP,
    ) -> None:
        ensembles = (ensemble,)
        trees = parse_ensembles(*ensembles, mapper=mapper)
        Model.__init__(
            self,
            trees,
            mapper=mapper,
            weights=weights,
            epsilon=epsilon,
            model_type=model_type,
        )
        self.build()
        self.solver = solver if solver is not None else cp.CpSolver()

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: PositiveInt,
    ) -> Explanation:
        self.add_objective(x, norm=norm)
        self.set_majority_class(y=y)
        self.solver.Solve(self)
        status = self.solver.status_name()
        if status != "OPTIMAL":
            msg = f"Failed to optimize the model. Status: {status}"
            raise RuntimeError(msg)
        return self.explanation
