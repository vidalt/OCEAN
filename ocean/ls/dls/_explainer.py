from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .._explainer import BaseLocalSearchExplainer, StartLog
from ..utils import get_norm
from .multi_start import multi_start_deterministic

if TYPE_CHECKING:
    from ocean.typing import (
        Array1D,
        LocalSearchExplainer,
        NonNegativeInt,
        PositiveInt,
    )

    from .._explanation import Explanation


class DeterministicMultiStartExplainer(BaseLocalSearchExplainer):
    """
    Multi-start with deterministic neighborhood (DLS).

    init_type controls how the starting population is generated:
      - "simple"  : Gaussian perturbation, one feature type at a time
      - "gini"    : grid perturbation on the k most important features (Gini)
      - "naive"   : Gaussian noise applied to all feature types simultaneously
    """

    def explain(  # noqa: PLR0913, PLR0917
        self,
        x: Array1D,
        query_class: NonNegativeInt,
        norm: PositiveInt,
        std: float = 1,
        n_population: int = 10,
        lambda_: float = 1,
        tabu_size: int = 10,
        n_iter: PositiveInt = 100,
        best_distance: float | None = None,
        max_time_per_local_search: float = 0.1,
        return_callback: bool = False,  # noqa: FBT001, FBT002
        random_seed: PositiveInt = 42,
        init_type: str = "simple",
        k_features: int = 5,
        flip_prob: float = 0.5,
        perturb_ratio: float = 1.0,
        per_start_log: StartLog | None = None,
        total_timeout: float | None = None,
    ) -> Explanation | None:
        self.norm = norm
        self.max_distance = get_norm(norm, self.inf, self.sup)
        best_distance_value = (
            float(best_distance)
            if best_distance is not None
            else float(self.max_distance)
        )

        result = multi_start_deterministic(
            cast("LocalSearchExplainer", self),
            n_iter,
            std,
            n_population,
            x,
            query_class,
            lambda_,
            tabu_size,
            best_distance_value,
            random_seed,
            max_time_per_local_search,
            return_callback,
            norm=norm,
            init_type=init_type,
            k_features=k_features,
            flip_prob=flip_prob,
            perturb_ratio=perturb_ratio,
            per_start_log=per_start_log,
            total_timeout=total_timeout,
        )
        return self._process_result(
            result,
            x,
            query_class,
            return_callback,
        )


__all__ = ["DeterministicMultiStartExplainer"]
