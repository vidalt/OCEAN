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
        return self.solver.cost / self._obj_scale

    def get_solving_status(self) -> str:
        return self.Status

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
        """MaxSAT currently exposes only the final optimal solution."""
        raise NotImplementedError

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
        if return_callback:
            default_seed = 42
            msg = "There are no callbacks for maxsat."
            if random_seed != default_seed:
                msg = "There are no callbacks/random_seed for maxsat."
            warnings.warn(msg, category=UserWarning, stacklevel=2)
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
        finally:
            signal.alarm(0)
        # Store the query in the explanation
        self.explanation.query = x

        # Clean up for next solve
        if clean_up:
            self.cleanup()
        return self.explanation
