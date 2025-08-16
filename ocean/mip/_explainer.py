import time

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
    ) -> Explanation:
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
        if self.SolCount == 0:
            msg = "No solution found. Please check the model constraints."
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
