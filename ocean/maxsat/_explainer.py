from __future__ import annotations

from typing import TYPE_CHECKING

from ..tree import parse_ensembles
from ..typing import (
    Array1D,
    BaseExplainableEnsemble,
    BaseExplainer,
    NonNegativeInt,
    PositiveInt,
)
from ._model import Model
from ._solver import MaxSATSolver

if TYPE_CHECKING:
    from ..abc import Mapper
    from ..feature import Feature
    from ._explanation import Explanation


class Explainer(Model, BaseExplainer):
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
        Model.__init__(
            self,
            trees,
            mapper=mapper,
            weights=weights,
            epsilon=epsilon,
            model_type=model_type,
        )
        self.build()
        self.solver = MaxSATSolver

    def get_objective_value(self) -> float:
        raise NotImplementedError

    def get_solving_status(self) -> str:
        raise NotImplementedError

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
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
    ) -> Explanation | None:
        raise NotImplementedError
