from __future__ import annotations

import signal
import warnings
from typing import TYPE_CHECKING, Any

from sklearn.ensemble import AdaBoostClassifier

from ..tree import parse_ensembles
from ..typing import (
    Array1D,
    BaseExplainableEnsemble,
    BaseExplainer,
    NonNegativeInt,
    PositiveInt,
)
from ._env import ENV
from ._model import Model

if TYPE_CHECKING:
    from ..abc import Mapper
    from ..feature import Feature
    from ._explanation import Explanation


def handler(signum: Any, frame: Any) -> TimeoutError:  # noqa: ANN401, ARG001
    msg = "Timeout for maxsat!"
    raise TimeoutError(msg)


class Explainer(Model, BaseExplainer):
    """MaxSAT-based explainer for tree ensemble classifiers."""

    Status: str = "UNKNOWN"

    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
        *,
        mapper: Mapper[Feature],
        weights: Array1D | None = None,
        epsilon: int = Model.DEFAULT_EPSILON,
        model_type: Model.Type = Model.Type.MAXSAT,
    ) -> None:
        ensembles = (ensemble,)
        trees = parse_ensembles(*ensembles, mapper=mapper)
        if isinstance(ensemble, AdaBoostClassifier):
            weights = ensemble.estimator_weights_
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
        """
        Return the weighted MaxSAT objective value of the last solve.

        Returns
        -------
        float
            Objective value rescaled back to the user-facing distance units.

        """
        return self.solver.cost / self._obj_scale

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
        Return the status of the latest MaxSAT solve.

        Returns
        -------
        str
            Status string such as ``"OPTIMAL"`` or ``"INFEASIBLE"``.

        """
        return self.Status

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
        """
        Return the intermediate solution trace for the last MaxSAT solve.

        Returns
        -------
        None
            The MaxSAT backend currently exposes only the final solution.

        """
        _ = self.Status
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
        Solve one counterfactual query with the weighted MaxSAT backend.

        Parameters
        ----------
        x
            Query instance in the processed feature space.
        y
            Target class enforced by the counterfactual.
        norm
            Distance norm. The MaxSAT backend currently supports only ``1``.
        return_callback
            Accepted for API compatibility but ignored by this backend.
        verbose
            Whether to enable RC2 logging.
        max_time
            Time limit in seconds.
        num_workers
            Optional thread count forwarded to the MaxSAT solver.
        random_seed
            Accepted for API compatibility but currently ignored.
        clean_up
            Whether to remove query-specific clauses after the solve.

        Returns
        -------
        Explanation | None
            The decoded counterfactual, or ``None`` when no feasible
            counterfactual is found within the given limits.

        Raises
        ------
        RuntimeError
            If the MaxSAT solver raises an error that is not UNSAT or timeout.

        """
        if return_callback:
            default_seed = 42
            msg = "There are no callbacks for maxsat."
            if random_seed != default_seed:
                msg = "There are no callbacks/random_seed for maxsat."
            warnings.warn(msg, category=UserWarning, stacklevel=2)
        self.Status = "UNKNOWN"  # reset status from previous solves
        self.solver.TimeLimit = max_time
        self.solver.n_threads = num_workers if num_workers is not None else 1
        self.solver.verbose = verbose
        # Add objective soft clauses
        self.add_objective(x, norm=norm)

        # Add hard constraints for target class
        self.set_majority_class(y=y)
        signal.signal(signal.SIGALRM, handler=handler)
        signal.alarm(max_time)
        try:
            # Solve the MaxSAT problem
            self.solver.solve(self)
            self.Status = "OPTIMAL"
        except RuntimeError as e:
            if "UNSAT" in str(e):
                self.Status = "INFEASIBLE"
                msg = "There are no feasible counterfactuals for this query."
                msg += " If there should be one, please check the model "
                msg += "constraints or report this issue to the developers."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                if clean_up:
                    self.cleanup()
                return None
            raise
        except TimeoutError as exc:
            warnings.warn(str(exc), category=UserWarning, stacklevel=2)
            signal.alarm(0)
            if clean_up:
                self.cleanup()
            return None
        finally:
            signal.alarm(0)
        # Store the query in the explanation
        self.explanation.query = x
        self._distance_norm = norm

        # Clean up for next solve
        if clean_up:
            self.cleanup()
        return self.explanation
