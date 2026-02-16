from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

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
        raise NotImplementedError

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: PositiveInt,
        verbose: bool = False,  # noqa: ARG002
        max_time: int = 60,  # noqa: ARG002
        random_seed: int = 42,  # noqa: ARG002
    ) -> Explanation | None:
        # Add objective soft clauses
        self.add_objective(x, norm=norm)

        # Add hard constraints for target class
        self.set_majority_class(y=y)

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
                self.cleanup()
                return None
            raise
        else:
            # Store the query in the explanation
            self.explanation.query = x

            # Clean up for next solve
            self.cleanup()

            return self.explanation
