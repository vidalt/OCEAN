from __future__ import annotations

import operator
import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from numba import types  # type: ignore[attr-defined]
from numba.typed.typedlist import List as NumbaList

from ._explanation import Explanation
from .dls.multi_start import (
    multi_start_deterministic,
    multi_start_simulated_annealing,
    multi_start_simulated_annealing_exhaustive,
)
from .dls.simulated_annealing import simulated_annealing
from .sls.multi_start import multi_start as multi_start_stochastic
from .utils import (
    build_cat_groups_ocean,
    get_norm,
    get_thresholds_ocean,
    hash_leaves_fnv1a,
    hypercube_cost,
    prepare_local_search_backend,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ocean.abc import Mapper
    from ocean.feature import Feature
    from ocean.typing import (
        Array1D,
        BaseExplainableEnsemble,
        LocalSearchExplainer,
        NonNegativeInt,
        PositiveInt,
    )

    from .utils.preprocess import RandomForestLike

type LeafBounds = tuple[np.ndarray, np.ndarray]
type Callback = list[dict[str, float]]
type StartLog = list[dict[str, object]]
type PlainSearchResult = tuple[
    LeafBounds | None, int | None, float | np.float32
]
type CallbackSearchResult = tuple[
    Callback,
    LeafBounds | None,
    int | None,
    float | np.float32,
]
type SearchResult = PlainSearchResult | CallbackSearchResult

DISPLAY_TOLERANCE = 1e-5


def _emit(line: object = "") -> None:
    sys.stdout.write(f"{line}\n")


def _group_to_index_array(group: object) -> np.ndarray:
    return np.asarray(list(cast("Iterable[int]", group)), dtype=int)


def _column_label(
    columns: pd.Index,
    index: int,
    level: int | None = None,
) -> str:
    column = columns[index]
    if level is not None and isinstance(column, tuple):
        values = cast("tuple[object, ...]", column)
        return str(values[level])
    return str(column)


# =============================================================================
# BASE CLASS
# =============================================================================


class BaseExplainer:
    def __init__(
        self,
        ensemble: BaseExplainableEnsemble,
        mapper: Mapper[Feature],
        data: pd.DataFrame,
    ) -> None:
        self.ensemble = ensemble
        self.mapper = mapper
        self.data = data
        backend = prepare_local_search_backend(ensemble, mapper)
        self.rf = backend.forest
        self.weights = backend.weights
        self.base_scores = backend.base_scores
        self.score_kind = backend.score_kind
        self.normalize_leaf_values = backend.normalize_leaf_values
        self.n_features = self.rf.n_features_in_
        self.callback: Callback | None = None
        self.norm: int = 2
        self.max_distance = np.float32(0.0)

        self._best_leaf: LeafBounds | None = None
        self._distance: float | np.float32 | None = None
        self._explanation: Explanation | None = None
        self._explanation_array: np.ndarray | None = None
        self._label: int | None = None

        self._load()

    def _load(self) -> None:
        (
            self.thresholds,
            self.thresh2idx,
            self.lengths_list,
            self.offsets,
            self.thresholds_concat,
            self.features_,
            self.thresholds_,
            self.values_,
            self.children_left_,
            self.children_right_,
        ) = get_thresholds_ocean(
            cast("RandomForestLike", self.rf),
            self.data,
            self.mapper,
            continuous_bounds="data",
        )

        self.continuous_col = np.array(
            [
                self.mapper.idx.get(name)
                for name, f in self.mapper.items()
                if f.is_continuous
            ],
            dtype=np.int32,
        )
        self.discrete_col = np.array(
            [
                self.mapper.idx.get(name)
                for name, f in self.mapper.items()
                if f.is_discrete
            ],
            dtype=np.int32,
        )
        self.binary_col = np.array(
            [
                self.mapper.idx.get(name)
                for name, f in self.mapper.items()
                if f.is_binary
            ],
            dtype=np.int32,
        )

        self.category_names = np.array(
            list(
                dict.fromkeys([
                    name
                    for name, f in self.mapper.items()
                    if f.is_one_hot_encoded
                ])
            )
        )
        self.one_hot_encoded_col = build_cat_groups_ocean(
            self.data.columns, self.category_names
        )

        self.inf = np.array(
            [self.thresholds[d][0] for d in range(self.n_features)],
            dtype=np.float32,
        )
        self.sup = np.array(
            [self.thresholds[d][-1] for d in range(self.n_features)],
            dtype=np.float32,
        )

        self._build_rank_maps()

    def _build_rank_maps(self) -> None:
        self.rank_maps = NumbaList.empty_list(  # type: ignore[no-untyped-call]
            types.int32[:]
        )
        self.inv_maps: list[list[int]] = []

        for est in self.rf.estimators_:
            tree = est.tree_
            node_count = tree.node_count

            rank_map: np.ndarray = -np.ones(node_count, dtype=np.int32)
            leaves = np.where(tree.children_left == -1)[0]
            rank_map[leaves] = np.arange(leaves.size, dtype=np.int32)

            self.rank_maps.append(rank_map)
            self.inv_maps.append(leaves.tolist())

    @staticmethod
    def encode(leaves_rank_array: np.ndarray) -> np.int64:
        return np.int64(hash_leaves_fnv1a(leaves_rank_array))

    def get_final_explanation(
        self,
        a: np.ndarray,
        b: np.ndarray,
        query: Array1D,
    ) -> tuple[np.float32, np.ndarray]:
        new_a = a.copy()
        new_b = b.copy()

        for i in self.continuous_col:
            new_a_i = np.nextafter(a[i], np.float32(self.sup[i] + 100))
            if new_a_i == b[i]:
                new_b_i = b[i]
            else:
                new_b_i = np.nextafter(b[i], np.float32(self.inf[i] - 100))
            new_a[i] = new_a_i
            new_b[i] = new_b_i

        dist, proj = hypercube_cost(
            new_a,
            new_b,
            query,
            self.continuous_col,
            self.binary_col,
            self.discrete_col,
            self.one_hot_encoded_col,
            self.max_distance,
            self.norm,
        )
        return np.float32(dist), proj

    def _process_result(
        self,
        result: SearchResult,
        x: Array1D,
        query_class: NonNegativeInt,
        return_callback: bool,  # noqa: FBT001
    ) -> Explanation | None:
        if return_callback:
            self.callback, best_leaf, _best_label, _best_distance = cast(
                "CallbackSearchResult",
                result,
            )
        else:
            best_leaf, _best_label, _best_distance = cast(
                "PlainSearchResult",
                result,
            )
            self.callback = None

        if best_leaf is None:
            _emit(f"[{self.__class__.__name__}] No explanation found")
            return None

        a, b = best_leaf
        self._best_leaf = (a, b)
        self._distance, explanation = self.get_final_explanation(a, b, x)
        self._explanation_array = explanation
        self._label = int(self.rf.predict([explanation])[0])

        if self._label == query_class:
            _emit(f"[{self.__class__.__name__}] Invalid explanation")
            return None

        self.verify_explanation(explanation)
        self._explanation = Explanation(self.mapper, explanation, x)
        return self._explanation

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_anytime_solutions(self) -> list[dict[str, float]] | None:
        return self.callback

    def get_distance(self) -> float | None:
        return None if self._distance is None else float(self._distance)

    def get_label(self) -> int | None:
        return self._label

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def print_feature_distribution(self) -> None:
        n_cont = len(self.continuous_col)
        n_disc = len(self.discrete_col)
        n_bin = len(self.binary_col)
        n_cat = len(self.one_hot_encoded_col)
        total_vars = n_cont + n_disc + n_bin + n_cat
        n_cat_cols = sum(len(group) for group in self.one_hot_encoded_col)
        total_encoded = n_cont + n_disc + n_bin + n_cat_cols

        _emit(f"{'Feature Type':<15} | {'Count':<5} | {'% (Vars)':<10}")
        _emit("-" * 36)
        if total_vars > 0:
            rows = (
                ("Continuous", n_cont),
                ("Discrete", n_disc),
                ("Binary", n_bin),
                ("Categorical", n_cat),
            )
            for label, count in rows:
                pct = count / total_vars * 100
                _emit(f"{label:<15} | {count:<5d} | {pct:5.1f}%")
        else:
            _emit("No features detected.")
        _emit("-" * 36)
        _emit(f"{'Total Variables':<15} | {total_vars:<5d}")
        _emit(
            f"{'Total Columns':<15} | {total_encoded:<5d} "
            f"(RF inputs: {self.n_features})"
        )

    def print_gini_importance_by_type(self) -> None:
        if not hasattr(self.rf, "feature_importances_"):
            _emit("Error: model has no 'feature_importances_' attribute.")
            return

        importances = self.rf.feature_importances_
        gini_cont = (
            np.sum(importances[self.continuous_col])
            if len(self.continuous_col) > 0
            else 0.0
        )
        gini_disc = (
            np.sum(importances[self.discrete_col])
            if len(self.discrete_col) > 0
            else 0.0
        )
        gini_bin = (
            np.sum(importances[self.binary_col])
            if len(self.binary_col) > 0
            else 0.0
        )
        gini_cat = 0.0
        if len(self.one_hot_encoded_col) > 0:
            for group in self.one_hot_encoded_col:
                group_indices = _group_to_index_array(group)
                gini_cat += np.sum(importances[group_indices])
        total_gini = gini_cont + gini_disc + gini_bin + gini_cat

        _emit(f"{'Feature Type':<15} | {'Gini Imp.':<10} | {'% Contrib.':<12}")
        _emit("-" * 43)
        if total_gini > 0:
            rows = (
                ("Continuous", gini_cont),
                ("Discrete", gini_disc),
                ("Binary", gini_bin),
                ("Categorical", gini_cat),
            )
            for label, value in rows:
                pct = value / total_gini * 100
                _emit(f"{label:<15} | {value:<10.4f} | {pct:5.1f}%")
        else:
            _emit("No importance available.")
        _emit("-" * 43)
        _emit(f"{'Total':<15} | {total_gini:<10.4f} | 100.0%")

    def print_explanation(self, explanation: np.ndarray) -> None:
        colnames = self.data.columns
        is_multi = isinstance(colnames, pd.MultiIndex)

        lines: list[tuple[str, str]] = []
        for i in np.concatenate([
            self.continuous_col,
            self.discrete_col,
            self.binary_col,
        ]).astype(int):
            idx = int(i)
            name = (
                _column_label(colnames, idx, 0)
                if is_multi
                else _column_label(colnames, idx)
            )
            lines.append((name, f"{float(explanation[i]):.6g}"))

        for j, cat_name in enumerate(self.category_names):
            oh_idx = _group_to_index_array(self.one_hot_encoded_col[j])
            idx_local = np.asarray(explanation)[oh_idx].argmax()
            idx_global = int(oh_idx[idx_local])
            if is_multi:
                val = _column_label(colnames, idx_global, 1)
            else:
                val = (
                    _column_label(colnames, idx_global)
                    .replace(f"{cat_name}_", "")
                    .replace(f"{cat_name}=", "")
                )
            lines.append((str(cat_name), val))

        lines.sort(key=operator.itemgetter(0))
        if lines:
            width = max(len(name) for name, _ in lines)
            for name, val in lines:
                _emit(f"{name:<{width}} : {val}")

    def print_comparison(
        self,
        original_point: np.ndarray,
        counterfactual_point: np.ndarray,
    ) -> None:
        original = np.asarray(original_point)
        candidate = np.asarray(counterfactual_point)
        colnames = self.data.columns
        is_multi = isinstance(colnames, pd.MultiIndex)

        _emit(
            f"{'Feature':<30} | {'Original':<20} -> "
            f"{'Counterfactual':<20} | {'Delta'}"
        )
        _emit("-" * 90)

        for i in np.concatenate([
            self.continuous_col,
            self.discrete_col,
            self.binary_col,
        ]).astype(int):
            if abs(candidate[i] - original[i]) > DISPLAY_TOLERANCE:
                idx = int(i)
                name = (
                    _column_label(colnames, idx, 0)
                    if is_multi
                    else _column_label(colnames, idx)
                )
                diff = candidate[i] - original[i]
                _emit(
                    f"{name:<30} | {original[i]:<20.4g} -> "
                    f"{candidate[i]:<20.4g} | {diff:+.4g}"
                )

        for j, cat_name in enumerate(self.category_names):
            oh_idx = _group_to_index_array(self.one_hot_encoded_col[j])
            idx_old = original[oh_idx].argmax()
            idx_new = candidate[oh_idx].argmax()
            if idx_old != idx_new:
                if is_multi:
                    old_global = int(oh_idx[idx_old])
                    new_global = int(oh_idx[idx_new])
                    name_old = _column_label(colnames, old_global, 1)
                    name_new = _column_label(colnames, new_global, 1)
                else:
                    prefix = str(cat_name)
                    name_old = (
                        _column_label(colnames, int(oh_idx[idx_old]))
                        .replace(f"{prefix}_", "")
                        .replace(f"{prefix}=", "")
                    )
                    name_new = (
                        _column_label(colnames, int(oh_idx[idx_new]))
                        .replace(f"{prefix}_", "")
                        .replace(f"{prefix}=", "")
                    )
                _emit(
                    f"{cat_name!s:<30} | {name_old:<20} -> "
                    f"{name_new:<20} | CHANGE"
                )

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify_explanation(self, explanation: np.ndarray) -> None:
        if len(self.continuous_col) > 0 and not (
            np.all(
                self.inf[self.continuous_col]
                <= explanation[self.continuous_col]
            )
            and np.all(
                explanation[self.continuous_col]
                <= self.sup[self.continuous_col]
            )
        ):
            _emit("continuous infeasability")

        if len(self.discrete_col) > 0 and not (
            np.all(self.inf[self.discrete_col] < explanation[self.discrete_col])
            and np.all(
                explanation[self.discrete_col] <= self.sup[self.discrete_col]
            )
            and np.all(
                np.floor(explanation[self.discrete_col])
                == explanation[self.discrete_col]
            )
        ):
            _emit("discrete infeasability")

        if len(self.binary_col) > 0 and not np.all(
            (explanation[self.binary_col] == 0)
            | (explanation[self.binary_col] == 1)
        ):
            _emit("binary infeasability")

        for category in self.one_hot_encoded_col:
            category_idx = _group_to_index_array(category)
            if np.sum(explanation[category_idx]) != 1:
                _emit("categorical infeasability")
                break


# =============================================================================
# CONCRETE EXPLAINERS
# =============================================================================


class MultiStartExplainer_deterministic(BaseExplainer):  # noqa: N801
    """
    Multi-Start with deterministic neighborhood (DLS).

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


class MultiStartExplainer_stochastic(BaseExplainer):  # noqa: N801
    """
    Multi-Start with stochastic neighborhood (SLS).

    Randomly samples neighbors on hypercube faces.
    init_type controls how the starting population is generated.
    """

    def explain(  # noqa: PLR0913, PLR0917
        self,
        x: Array1D,
        query_class: NonNegativeInt,
        norm: PositiveInt,
        std: float = 1,
        n_population: int = 10,
        lambda_: float = 1,
        n_faces: PositiveInt = 1,
        n_samples_per_face: PositiveInt = 1,
        n_iter: PositiveInt = 100,
        best_distance: float | None = None,
        max_time_per_local_search: float = 0.1,
        return_callback: bool = False,  # noqa: FBT001, FBT002
        random_seed: PositiveInt = 42,
        tabu_size: NonNegativeInt = 0,
        init_type: str = "gini",
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

        result = multi_start_stochastic(
            cast("LocalSearchExplainer", self),
            n_iter,
            std,
            n_population,
            x,
            query_class,
            lambda_,
            tabu_size,
            best_distance_value,
            n_faces,
            n_samples_per_face,
            random_seed,
            max_time_per_local_search,
            return_callback,
            norm,
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


class SimulatedAnnealingExplainer(BaseExplainer):
    """Simulated Annealing to escape local minima."""

    def explain(  # noqa: PLR0913, PLR0917
        self,
        x: Array1D,
        query_class: NonNegativeInt,
        norm: PositiveInt,
        std: float = 1,
        n_population: int = 10,
        lambda_: float = 1,
        n_iter: PositiveInt = 100,
        best_distance: float | None = None,
        max_time_per_sa: float = 0.1,
        return_callback: bool = False,  # noqa: FBT001, FBT002
        random_seed: PositiveInt = 42,
        T_init: float = 1.0,
        T_min: float = 0.001,
        alpha: float = 0.95,
        schedule: str = "exponential",
        M_k: int = 10,
        exhaustive: bool = False,  # noqa: FBT001, FBT002
    ) -> Explanation | None:
        self.norm = norm
        self.max_distance = get_norm(norm, self.inf, self.sup)
        best_distance_value = (
            float(best_distance)
            if best_distance is not None
            else float(self.max_distance)
        )

        if exhaustive:
            result = multi_start_simulated_annealing_exhaustive(
                cast("LocalSearchExplainer", self),
                n_iter,
                std,
                n_population,
                x,
                query_class,
                lambda_,
                best_distance_value,
                random_seed,
                max_time_per_sa,
                return_callback,
                norm,
                T_init=T_init,
                T_min=T_min,
                alpha=alpha,
                schedule=schedule,
            )
        else:
            result = multi_start_simulated_annealing(
                cast("LocalSearchExplainer", self),
                n_iter,
                std,
                n_population,
                x,
                query_class,
                lambda_,
                best_distance_value,
                random_seed,
                max_time_per_sa,
                return_callback,
                norm,
                T_init=T_init,
                T_min=T_min,
                alpha=alpha,
                schedule=schedule,
                M_k=M_k,
            )
        return self._process_result(
            result,
            x,
            query_class,
            return_callback,
        )

    def explain_single(
        self,
        x: Array1D,
        query_class: NonNegativeInt,
        norm: PositiveInt,
        lambda_: float = 1,
        n_iter: PositiveInt = 1000,
        best_distance: float | None = None,
        timeout: float | None = None,
        return_callback: bool = False,  # noqa: FBT001, FBT002
        random_seed: PositiveInt = 42,
        T_init: float = 1.0,
        T_min: float = 0.001,
        alpha: float = 0.95,
        schedule: str = "exponential",
        M_k: int = 10,
    ) -> Explanation | None:
        self.norm = norm
        self.max_distance = get_norm(norm, self.inf, self.sup)
        best_distance_value = (
            float(best_distance)
            if best_distance is not None
            else float(self.max_distance)
        )

        result = simulated_annealing(
            cast("LocalSearchExplainer", self),
            n_iter,
            x,
            x,
            query_class,
            norm,
            lambda_,
            best_distance_value,
            random_seed,
            timeout,
            T_init=T_init,
            T_min=T_min,
            alpha=alpha,
            schedule=schedule,
            M_k=M_k,
            store=return_callback,
        )

        if return_callback:
            self.callback, sols, _labels = cast(
                "tuple[Callback, list[LeafBounds], list[int]]",
                result,
            )
            if sols:
                a, b = sols[-1]
                self._best_leaf = (a, b)
                self._distance, explanation = self.get_final_explanation(
                    a, b, x
                )
                self._explanation_array = explanation
                self._label = int(self.rf.predict([explanation])[0])
                if self._label == query_class:
                    return None
                self.verify_explanation(explanation)
                self._explanation = Explanation(self.mapper, explanation, x)
                return self._explanation
            return None
        return self._process_result(
            cast("SearchResult", result),
            x,
            query_class,
            return_callback,
        )
