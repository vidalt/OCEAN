import time
import warnings
from typing import cast

import gurobipy as gp
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, IsolationForest

from ..abc import Mapper
from ..feature import Feature
from ..tree import parse_ensembles
from ..typing import (
    Array1D,
    BaseExplainableEnsemble,
    BaseExplainer,
    NonNegativeInt,
)
from ._explanation import Explanation
from ._model import Model
from ._variables import TreeVar


class Explainer(Model, BaseExplainer):
    """Mixed-integer programming explainer for tree ensemble classifiers."""

    _output_values: Array1D | None

    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
        *,
        mapper: Mapper[Feature],
        weights: Array1D | None = None,
        isolation: IsolationForest | None = None,
        isolation_threshold: float | None = None,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: float = Model.DEFAULT_EPSILON,
        num_epsilon: float = Model.DEFAULT_NUM_EPSILON,
        model_type: "Model.Type" = Model.Type.MIP,
        flow_type: "TreeVar.FlowType" = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        ensembles = (ensemble,) if isolation is None else (ensemble, isolation)
        n_isolators, max_samples = self._get_isolation_params(isolation)
        trees = parse_ensembles(*ensembles, mapper=mapper)
        if isinstance(ensemble, AdaBoostClassifier):
            weights = ensemble.estimator_weights_
        Model.__init__(
            self,
            trees,
            mapper=mapper,
            weights=weights,
            n_isolators=n_isolators,
            max_samples=max_samples,
            isolation_threshold=isolation_threshold,
            name=name,
            env=env,
            epsilon=epsilon,
            num_epsilon=num_epsilon,
            model_type=model_type,
            flow_type=flow_type,
        )
        self._output_values = None
        self.build()

    def vget(self, i: int) -> gp.Var:
        var = super().vget(i)
        if self._output_values is None:
            return var
        return cast("gp.Var", _ValueProxy(var, float(self._output_values[i])))

    def get_objective_value(self) -> float:
        """
        Return the solver objective value of the last optimization run.

        Returns
        -------
        float
            Objective value reported by Gurobi for the latest solve.

        """
        return self.ObjVal

    def get_distance(self) -> float:
        """
        Return the post-processed distance of the last CF.

        Returns
        -------
        float
            Post-processed :math:`L_p` distance for the last successful solve.

        Raises
        ------
        RuntimeError
            If no explanation has been computed yet.

        """
        query = self.explanation.query
        if query.size == 0:
            msg = "No explanation has been computed yet."
            raise RuntimeError(msg)

        norm = getattr(self, "_distance_norm", None)
        if norm is None:
            msg = "No explanation has been computed yet."
            raise RuntimeError(msg)

        counterfactual = self.explanation.x
        distance = 0.0
        for name, feature in self.mapper.items():
            if feature.is_one_hot_encoded:
                feature_distance = 0.0
                for code in feature.codes:
                    idx = self.mapper.idx.get(name, code)
                    delta = float(counterfactual[idx]) - float(query[idx])
                    feature_distance += (
                        0.0
                        if norm == 0 and np.isclose(delta, 0.0)
                        else abs(delta) ** norm
                    )
                distance += feature_distance / 2.0
            else:
                idx = self.mapper.idx.get(name)
                delta = float(counterfactual[idx]) - float(query[idx])
                distance += (
                    0.0
                    if norm == 0 and np.isclose(delta, 0.0)
                    else abs(delta) ** norm
                )
        if norm not in {0, 1}:
            distance **= 1.0 / norm
        return float(distance)

    def get_solving_status(self) -> str:
        """
        Return the latest Gurobi solve status as a readable string.

        Returns
        -------
        str
            Current model status such as ``"OPTIMAL"`` or ``"TIME_LIMIT"``.

        """
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
        """
        Return incumbent solutions collected during the last solve.

        Returns
        -------
        list[dict[str, float]] | None
            Time-stamped incumbent objective values when ``return_callback``
            was enabled in :meth:`explain`, otherwise ``None``.

        """
        callback = cast(
            "SolutionCallback | None",
            getattr(self, "callback", None),
        )
        if callback is None:
            return None
        return callback.sollist

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: NonNegativeInt,
        return_callback: bool = False,
        verbose: bool = False,
        max_time: int = 60,
        num_workers: int | None = None,
        random_seed: int = 42,
        clean_up: bool = True,
    ) -> Explanation | None:
        """
        Solve one counterfactual query with the MIP backend.

        Parameters
        ----------
        x
            Query instance in the processed feature space.
        y
            Target class enforced by the counterfactual.
        norm
            Distance norm. The MIP backend supports ``0``, ``1``, and ``2``.
        return_callback
            Whether to collect incumbent solutions through a Gurobi callback.
        verbose
            Whether to print Gurobi logs.
        max_time
            Time limit in seconds.
        num_workers
            Optional Gurobi thread count.
        random_seed
            Random seed passed to Gurobi.
        clean_up
            Whether to remove query-specific constraints after the solve.

        Returns
        -------
        Explanation | None
            The decoded counterfactual, or ``None`` when no feasible
            counterfactual is found within the given limits.

        Raises
        ------
        RuntimeError
            If the solver stops for an unexpected status that is not handled
            by the explainer.

        """
        self._output_values = None
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
        self.explanation.query = x
        self._distance_norm = norm
        self._output_values = np.asarray(self.explanation.x, dtype=np.float64)
        if clean_up:
            self.cleanup()
        return self.explanation

    @staticmethod
    def _get_isolation_params(
        isolation: IsolationForest | None,
    ) -> tuple[NonNegativeInt, NonNegativeInt]:
        if isolation is not None:
            return len(isolation), int(isolation.max_samples_)  # pyright: ignore[reportUnknownArgumentType]
        return 0, 0


class SolutionCallback:
    """Collect incumbent solutions reported by Gurobi during optimization."""

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


class _ValueProxy:
    def __init__(self, var: gp.Var, value: float) -> None:
        self._var = var
        self._value = value

    @property
    def X(self) -> float:
        return self._value

    def __getattr__(self, name: str) -> object:
        return getattr(self._var, name)
