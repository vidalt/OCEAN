import time
import traceback
import warnings

from ortools.sat.python import cp_model as cp
from sklearn.ensemble import AdaBoostClassifier, IsolationForest

from ..abc import Mapper
from ..feature import Feature
from ..tree import parse_ensembles
from ..typing import (
    Array1D,
    BaseExplainableEnsemble,
    BaseExplainer,
    NonNegativeArray1D,
    NonNegativeInt,
    PositiveInt,
)
from ._env import ENV
from ._explanation import Explanation
from ._model import Model


class Explainer(Model, BaseExplainer):
    """Constraint programming explainer for tree ensemble classifiers."""

    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
        *,
        mapper: Mapper[Feature],
        weights: NonNegativeArray1D | None = None,
        isolation: IsolationForest | None = None,
        isolation_threshold: float | None = None,
        epsilon: int = Model.DEFAULT_EPSILON,
        model_type: "Model.Type" = Model.Type.CP,
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
            epsilon=epsilon,
            model_type=model_type,
        )
        self.build()
        self.solver = ENV.solver

    def get_objective_value(self) -> float:
        """
        Return the scaled objective value of the last CP-SAT solve.

        Returns
        -------
        float
            Objective value rescaled back to the user-facing distance units.

        """
        return self.solver.ObjectiveValue() / self._obj_scale

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
                    feature_distance += abs(delta) ** norm
                distance += feature_distance / 2.0
            else:
                idx = self.mapper.idx.get(name)
                delta = float(counterfactual[idx]) - float(query[idx])
                distance += abs(delta) ** norm
        if norm != 1:
            distance **= 1.0 / norm
        return float(distance)

    def get_solving_status(self) -> str:
        """
        Return the status string from the latest CP-SAT solve.

        Returns
        -------
        str
            Solver status such as ``"OPTIMAL"``, ``"FEASIBLE"``, or
            ``"INFEASIBLE"``.

        """
        return self.Status

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
        """
        Return intermediate solutions collected during the last solve.

        Returns
        -------
        list[dict[str, float]] | None
            Time-stamped incumbent objective values when ``return_callback``
            was enabled in :meth:`explain`, otherwise ``None``.

        """
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
        clean_up: bool = True,
    ) -> Explanation | None:
        """
        Solve one counterfactual query with the CP-SAT backend.

        Parameters
        ----------
        x
            Query instance in the processed feature space.
        y
            Target class enforced by the counterfactual.
        norm
            Integer distance norm used by the CP objective.
        return_callback
            Whether to record incumbent solutions during the search.
        verbose
            Whether to enable CP-SAT search logging.
        max_time
            Time limit in seconds.
        num_workers
            Optional number of CP-SAT workers.
        random_seed
            Random seed passed to CP-SAT.
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
            If CP-SAT reports an invalid model or an unexpected status.

        """
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
        cf_status_ok = True
        match status:
            case "OPTIMAL":
                pass
            case "FEASIBLE":
                msg = "A valid CF was found, but it might be "
                msg += "suboptimal as the constraint programming "
                msg += "solver could not prove optimality within "
                msg += "the given time frame. \n It can however certify"
                msg += " that no counterfactual can be closer than"
                msg += f" {self.solver.BestObjectiveBound() / self._obj_scale}."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
            case "INFEASIBLE":
                msg = "There are no feasible counterfactuals for this query."
                msg += " If there should be one, please check the model "
                msg += "constraints or report this issue to the developers."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                cf_status_ok = False
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
                cf_status_ok = False
            case _:
                msg = "Unexpected solver status: " + status
                raise RuntimeError(msg)

        if not cf_status_ok:
            self.cleanup()
            return None
        self.explanation.query = x
        self._distance_norm = norm
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
