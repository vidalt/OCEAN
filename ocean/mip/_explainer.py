import time
import warnings

import gurobipy as gp
from sklearn.ensemble import IsolationForest

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
from ._variables import TreeVar


class Explainer(Model, BaseExplainer):
    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
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
        n_isolators, max_samples = self._get_isolation_params(isolation)
        trees = parse_ensembles(*ensembles, mapper=mapper)
        Model.__init__(
            self,
            trees,
            mapper=mapper,
            weights=weights,
            n_isolators=n_isolators,
            max_samples=max_samples,
            name=name,
            env=env,
            epsilon=epsilon,
            num_epsilon=num_epsilon,
            model_type=model_type,
            flow_type=flow_type,
        )
        self.build()

    def get_objective_value(self) -> float:
        return self.ObjVal

    def get_solving_status(self) -> str:
        gurobi_statuses = {
            1: "LOADED",
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            6: "CUTOFF",
            7: "ITERATION_LIMIT",
            8: "NODE_LIMIT",
            9: "TIME_LIMIT",
            10: "SOLUTION_LIMIT",
            11: "INTERRUPTED",
            12: "NUMERIC",
            13: "SUBOPTIMAL",
            14: "INPROGRESS",
            15: "USER_OBJ_LIMIT",
            16: "WORK_LIMIT",
        }
        return gurobi_statuses[self.Status]

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
        return self.callback.sollist

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
        self.setParam("LogToConsole", int(verbose))
        self.setParam("TimeLimit", max_time)
        self.setParam("Seed", random_seed)
        if num_workers is not None:
            self.setParam("Threads", num_workers)
        self.add_objective(x, norm=norm)
        self.set_majority_class(y=y)
        if return_callback:
            self.callback = SolutionCallback(starttime=time.time())
            self.optimize(self.callback)
        else:
            self.optimize()
        status = self.get_solving_status()

        if status == "INFEASIBLE":
            msg = "There are no feasible counterfactuals for this query."
            msg += " If there should be one, please check the model "
            msg += "constraints or report this issue to the developers."
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return None
        if status != "OPTIMAL":
            if self.SolCount > 0:
                msg = "A valid CF was found, but it might be "
                msg += "suboptimal as the MILP "
                msg += "solver could not prove optimality within "
                msg += "the given time frame. \n It can however certify"
                msg += " that no counterfactual can be closer than"
                msg += f" {self.ObjBound}."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
            elif status == "TIME_LIMIT":
                msg = "The MILP solver could not find any"
                msg += " valid CF within the given time frame."
                msg += " Try increasing the time limit."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                return None
            elif status == "MEM_LIMIT":
                msg = "The MILP solver could not find any"
                msg += " valid CF within the given max memory."
                msg += " Try increasing the memory limit."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                return None
            else:
                msg = "The MILP solver could not find any"
                msg += " valid CF for an un-handled reason."
                msg += "Unexpected solver status: " + status
                raise RuntimeError(msg)
        return self.explanation

    @staticmethod
    def _get_isolation_params(
        isolation: IsolationForest | None,
    ) -> tuple[NonNegativeInt, NonNegativeInt]:
        if isolation is not None:
            return len(isolation), int(isolation.max_samples_)  # pyright: ignore[reportUnknownArgumentType]
        return 0, 0


class SolutionCallback:
    def __init__(self, starttime: float) -> None:
        self.starttime = starttime
        self.sollist: list[dict[str, float]] = []

    def __call__(self, model: gp.Model, where: int) -> None:
        if where == gp.GRB.Callback.MIPSOL:
            # Query the objective value of the new solution
            best_objective = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            self.sollist.append({
                "objective_value": best_objective,
                "time": time.time() - self.starttime,
            })
