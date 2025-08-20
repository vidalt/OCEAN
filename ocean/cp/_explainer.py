import time
import traceback
import warnings

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

    def get_objective_value(self) -> float:
        return self.solver.ObjectiveValue() / self._obj_scale

    def get_solving_status(self) -> str:
        return self.Status

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
        if self.callback is not None:
            return self.callback.sollist
        return None

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: PositiveInt,
        return_callback: bool = False,
        verbose: bool = False,
        max_time: int = 60,
        num_workers: int | None = None,
        random_seed: int = 42,
    ) -> Explanation | None:
        self.solver.parameters.log_search_progress = verbose
        self.solver.parameters.max_time_in_seconds = max_time
        self.solver.parameters.random_seed = random_seed
        if num_workers is not None:
            self.solver.parameters.num_workers = num_workers
        self.add_objective(x, norm=norm)
        self.set_majority_class(y=y)
        self.callback: MySolCallback | None = (
            MySolCallback(starttime=time.time(), _obj_scale=self._obj_scale)
            if return_callback
            else None
        )
        _ = self.solver.Solve(self, solution_callback=self.callback)
        status = self.solver.status_name()
        self.Status = status

        match status:
            case "OPTIMAL":
                pass
            case "FEASIBLE":
                msg = "A valid CF was found, but it might be "
                msg += "suboptimal as the constraint programming "
                msg += "solver could not prove optimality within "
                msg += "the given time frame. \n It can however certify"
                msg += " that no counterfactual can be closer than"
                msg += f" {self.solver.BestObjectiveBound()}."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
            case "INFEASIBLE":
                msg = "There are no feasible counterfactuals for this query."
                msg += " If there should be one, please check the model "
                msg += "constraints or report this issue to the developers."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                return None
            case "MODEL_INVALID":
                msg = "The constraint programming model is invalid. "
                msg += "Please check the model constraints or report"
                msg += " this issue to the developers."
                raise RuntimeError(msg)
            case "UNKNOWN":
                msg = "The constraint programming solver could "
                msg += "not find any valid CF within the given time frame."
                msg += " Try increasing the time limit."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                return None
            case _:
                msg = "Unexpected solver status: " + status
                raise RuntimeError(msg)
        self.explanation.query = x
        return self.explanation


class MySolCallback(cp.CpSolverSolutionCallback):
    """Save intermediate solutions."""

    def __init__(self, starttime: float, _obj_scale: float) -> None:
        cp.CpSolverSolutionCallback.__init__(self)
        self.sollist: list[dict[str, float]] = []
        self.__solution_count = 0
        self.starttime = starttime
        self._obj_scale = _obj_scale

    def on_solution_callback(self) -> None:
        try:
            self.__solution_count += 1
            t = time.time()
            objval = self.ObjectiveValue() / self._obj_scale
            self.addSol(objval, t - self.starttime)
        except Exception:
            traceback.print_exc()
            raise

    def solution_count(self) -> NonNegativeInt:
        return self.__solution_count

    def addSol(self, objval: float, time: float) -> None:
        self.sollist.append({"objective_value": objval, "time": time})
