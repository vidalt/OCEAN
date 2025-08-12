import time
import traceback

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
from ._env import ENV
from ._explanation import Explanation
from ._model import Model


class Explainer(Model, BaseExplainer):
    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
        *,
        mapper: Mapper[Feature],
        weights: Array1D | None = None,
        epsilon: int = Model.DEFAULT_EPSILON,
        model_type: Model.Type = Model.Type.CP,
        n_threads: int | None = None,
        max_time: int = 3000,
        seed: int = 42,
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
        self.solver = ENV.solver
        self.solver.parameters.max_time_in_seconds = max_time
        self.solver.parameters.random_seed = seed
        if n_threads is not None:
            self.solver.parameters.num_workers = n_threads

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: PositiveInt,
        save_callback: bool = False,
    ) -> Explanation:
        self.add_objective(x, norm=norm)
        self.set_majority_class(y=y)
        self.callback: MySolCallback | None = (
            MySolCallback(starttime=time.time()) if save_callback else None
        )
        _ = self.solver.Solve(self, solution_callback=self.callback)
        status = self.solver.status_name()
        if status != "OPTIMAL":
            msg = f"Failed to optimize the model. Status: {status}"
            raise RuntimeError(msg)
        self.explanation.query = x
        return self.explanation


class MySolCallback(cp.CpSolverSolutionCallback):
    """Save intermediate solutions."""

    def __init__(self, starttime: float) -> None:
        cp.CpSolverSolutionCallback.__init__(self)
        self.sollist: list[dict[str, float]] = []
        self.__solution_count = 0
        self.starttime = starttime

    def on_solution_callback(self) -> None:
        try:
            self.__solution_count += 1
            t = time.time()
            objval = self.ObjectiveValue()
            self.addSol(objval, t - self.starttime)
        except Exception:
            traceback.print_exc()
            raise

    def solution_count(self) -> NonNegativeInt:
        return self.__solution_count

    def addSol(self, objval: float, time: float) -> None:
        self.sollist.append({"objective_value": objval, "time": time})
