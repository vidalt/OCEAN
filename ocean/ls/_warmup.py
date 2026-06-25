"""
Precompile and cache every numba ``@njit`` function in ``ocean.ls``.

The public entry point is :func:`warmup_numba`, which is safe to call from an
LS explainer constructor. When this file is run as a script, it also verifies
that every discovered dispatcher has been compiled.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import time
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numba import types  # type: ignore[attr-defined]
from numba.typed.typedlist import List as NumbaList
from sklearn.ensemble import RandomForestClassifier

import ocean.ls.utils.costs as ls_costs
from ocean.abc import Mapper
from ocean.feature import Feature, parse_features
from ocean.ls.utils.costs import L0, L1, L2, fitness, get_final_explanation
from ocean.ls.utils.leaves import filtered_get_leaf_numba
from ocean.ls.utils.tools import (
    ceil_strict,
    cell_center,
    dot_product_int64,
    find_interval,
    floor_strict,
    idx2thresh,
    idx2thresh_vectorized,
    shuffle_typed_list,
    shuffled_copy,
    sum_numba_list,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from ocean.typing import Array1D

if __name__ == "__main__":
    sys.modules.setdefault("ocean.ls._warmup", sys.modules[__name__])

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGES = ["ocean.ls"]
IGNORE_MODULES: tuple[str, ...] = ()
KNOWN_UNREACHABLE: set[str] = set()
PYTHON_CACHE_TAG = f"py{sys.version_info.major}{sys.version_info.minor}"

MIN_CREDIT_LINES = 2
MIN_INCOME_RATIO = 0.1
MAX_DEBT_RATIO = 0.55
APPROVAL_SCORE = 4

_warmed_up: bool = False
_warming_up: bool = False
_warmup_lock = Lock()

type WarmupModel = tuple[
    RandomForestClassifier,
    Mapper[Feature],
    pd.DataFrame,
    np.ndarray,
]


def _emit(line: object = "") -> None:
    sys.stdout.write(f"{line}\n")


def _already_warmed_up() -> bool:
    return _warmed_up


def _currently_warming_up() -> bool:
    return _warming_up


def _numba_cache_exists(root: Path = HERE) -> bool:
    has_index = any(root.rglob(f"*.{PYTHON_CACHE_TAG}.nbi"))
    has_compiled = any(root.rglob(f"*.{PYTHON_CACHE_TAG}*.nbc"))
    return has_index and has_compiled


def clear_numba_cache(root: Path) -> int:
    removed = 0
    for path in root.rglob("*.nb[ic]"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def import_all_submodules(package_names: list[str]) -> list[ModuleType]:
    modules: list[ModuleType] = []
    for name in package_names:
        try:
            pkg = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            _emit(f"  [skip] package {name}: {type(exc).__name__}: {exc}")
            continue
        modules.append(pkg)
        module_path = getattr(pkg, "__path__", None)
        if module_path is None:
            continue
        for info in pkgutil.walk_packages(module_path, prefix=name + "."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception as exc:  # noqa: BLE001
                _emit(
                    f"  [skip] module {info.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
    return modules


def _is_numba_dispatcher(obj: object) -> bool:
    return hasattr(obj, "py_func") and hasattr(obj, "signatures")


def collect_dispatchers(modules: list[ModuleType]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for mod in modules:
        for obj in vars(mod).values():
            if not _is_numba_dispatcher(obj):
                continue
            dispatcher = obj
            pyf = dispatcher.py_func
            key = f"{pyf.__module__}.{pyf.__qualname__}"
            found[key] = dispatcher
    return found


def build_model() -> WarmupModel:
    rng = np.random.default_rng(0)
    n_samples = 240
    raw = pd.DataFrame({
        "credit_lines": rng.choice([0, 1, 2, 3, 4], size=n_samples),
        "owns_home": rng.integers(0, 2, size=n_samples),
        "has_guarantor": rng.integers(0, 2, size=n_samples),
        "income_ratio": rng.uniform(-0.4, 0.8, size=n_samples),
        "debt_ratio": rng.uniform(0.0, 1.0, size=n_samples),
        "savings_ratio": rng.uniform(-0.5, 0.6, size=n_samples),
        "job_type": rng.choice(
            ["office", "manual", "service", "student"], size=n_samples
        ),
        "region": rng.choice(
            ["north", "south", "east", "west"], size=n_samples
        ),
    })
    score = (
        (raw["credit_lines"] >= MIN_CREDIT_LINES).astype(int)
        + raw["owns_home"].astype(int)
        + raw["has_guarantor"].astype(int)
        + (raw["income_ratio"] > MIN_INCOME_RATIO).astype(int)
        + (raw["savings_ratio"] > 0.0).astype(int)
        + raw["job_type"].isin(["office", "service"]).astype(int)
        + raw["region"].isin(["north", "east"]).astype(int)
        - (raw["debt_ratio"] > MAX_DEBT_RATIO).astype(int)
    )
    target = (score >= APPROVAL_SCORE).astype(int).rename("approved")

    data, mapper = parse_features(raw, discretes=("credit_lines",))
    rf = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=0)
    rf.fit(data, target)
    x_values = data.to_numpy(dtype=np.float32)
    return rf, mapper, data, x_values


def _warmup_cost_helpers() -> None:
    format_discrete_cols = cast(
        "Callable[..., object]",
        getattr(ls_costs, "_format" + "_discrete_cols"),
    )
    format_ohe_cols = cast(
        "Callable[..., object]",
        getattr(ls_costs, "_format" + "_ohe_cols"),
    )

    x = np.array([0.0, 1.0], dtype=np.float32)
    y = np.array([1.0, 1.0], dtype=np.float32)
    L0(x, y)
    L1(x, y)
    L2(x, y)
    fitness(np.float32(1.0), np.float32(0.5), 0, 1, 1.0, np.float32(2.0))

    discrete_col = np.array([0], dtype=np.int32)
    a_small = np.array([0.0, 0.0], dtype=np.float32)
    b_small = np.array([2.0, 1.0], dtype=np.float32)
    q_small = np.array([1.0, 1.0], dtype=np.float32)
    proj_small = np.zeros(2, dtype=np.float32)
    format_discrete_cols(
        discrete_col,
        a_small,
        b_small,
        q_small,
        proj_small,
    )

    category = cast(
        "Any",
        NumbaList.empty_list(types.int64),  # type: ignore[no-untyped-call]
    )
    category.append(np.int64(0))
    category.append(np.int64(1))
    format_ohe_cols(category, a_small, b_small, q_small, proj_small)


def _warmup_tool_helpers(exp: object) -> None:
    exp_obj = cast("Any", exp)
    floor_strict(np.float32(1.2))
    ceil_strict(np.float32(1.2))
    dot_product_int64(
        np.array([1, 2], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
    )

    typed_list = cast(
        "Any",
        NumbaList.empty_list(types.int64),  # type: ignore[no-untyped-call]
    )
    typed_list.append(np.int64(1))
    typed_list.append(np.int64(2))
    shuffle_typed_list(typed_list)
    shuffled_copy(typed_list)
    sum_numba_list(np.array([1.0, 2.0], dtype=np.float32))

    n_features = int(exp_obj.n_features)
    idx = [1] * n_features
    idx_array = np.ones(n_features, dtype=np.int64)
    thresholds = tuple(
        np.asarray(exp_obj.thresholds[feature], dtype=np.float32)
        for feature in range(n_features)
    )
    idx2thresh(idx, thresholds)
    idx2thresh_vectorized(idx_array, exp_obj.offsets, exp_obj.thresholds_concat)
    cell_center(
        idx,
        exp_obj.offsets,
        exp_obj.thresholds_concat,
        exp_obj.lengths_list,
    )
    find_interval(np.float32(0.0), thresholds[0])


def _warmup_leaf_helpers(exp: object, query: np.ndarray) -> None:
    exp_obj = cast("Any", exp)
    node_indicator, tree_ptrs = exp_obj.rf.decision_path([query])
    global_nodes = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]
    local_nodes = global_nodes[
        (tree_ptrs[0] <= global_nodes) & (global_nodes < tree_ptrs[1])
    ] - tree_ptrs[0]
    local_nodes_list = cast(
        "Any",
        NumbaList.empty_list(types.int64),  # type: ignore[no-untyped-call]
    )
    for node_id in local_nodes:
        local_nodes_list.append(np.int64(node_id))

    filtered_get_leaf = cast("Any", filtered_get_leaf_numba)
    try:
        filtered_get_leaf(
            query,
            local_nodes_list,
            exp_obj.features_[0],
            exp_obj.thresholds_[0],
            exp_obj.values_[0],
            exp_obj.children_left_[0],
            exp_obj.children_right_[0],
            exp_obj.inf,
            exp_obj.sup,
            getattr(exp_obj, "normalize_leaf_values", True),
        )
    except TypeError as exc:
        if "too many arguments" not in str(exc):
            raise
        filtered_get_leaf(
            query,
            local_nodes_list,
            exp_obj.features_[0],
            exp_obj.thresholds_[0],
            exp_obj.values_[0],
            exp_obj.children_left_[0],
            exp_obj.children_right_[0],
            exp_obj.inf,
            exp_obj.sup,
        )

    get_final_explanation(
        exp_obj.inf,
        exp_obj.sup,
        query,
        exp_obj.continuous_col,
        exp_obj.binary_col,
        exp_obj.discrete_col,
        exp_obj.one_hot_encoded_col,
        np.float32(1.0),
        1,
    )


def run_direct_kernel_warmup(exp: object, query: np.ndarray) -> None:
    _warmup_cost_helpers()
    _warmup_tool_helpers(exp)
    _warmup_leaf_helpers(exp, query)


def run_workload(
    rf: RandomForestClassifier,
    mapper: Mapper[Feature],
    data: pd.DataFrame,
    x_values: np.ndarray,
    *,
    verbose: bool = False,
) -> None:
    """Exercise DLS / SLS / SA on real call paths to trigger compilation."""
    ls_module = importlib.import_module("ocean.ls")
    dls_cls = ls_module.DLSExplainer
    sls_cls = ls_module.SLSExplainer
    sa_cls = ls_module.SimulatedAnnealingExplainer

    dls = dls_cls(rf, mapper, data)
    sls = sls_cls(rf, mapper, data)
    sa = sa_cls(rf, mapper, data)

    query = np.asarray(x_values[0], dtype=np.float32)
    query_for_explain = cast("Array1D", query)
    prediction: object = rf.predict(data.iloc[[0]])
    query_class = int(np.asarray(prediction, dtype=np.int_).item())

    attempts: list[tuple[str, Callable[[], object]]] = [
        (
            "DLS simple",
            lambda: dls.explain(
                x=query_for_explain,
                query_class=query_class,
                norm=1,
                n_population=4,
                n_iter=5,
                max_time_per_local_search=0.05,
                random_seed=0,
                init_type="simple",
            ),
        ),
        (
            "DLS naive",
            lambda: dls.explain(
                x=query_for_explain,
                query_class=query_class,
                norm=2,
                n_population=4,
                n_iter=5,
                max_time_per_local_search=0.05,
                random_seed=0,
                init_type="naive",
            ),
        ),
        (
            "SLS simple",
            lambda: sls.explain(
                x=query_for_explain,
                query_class=query_class,
                norm=1,
                n_population=4,
                n_iter=5,
                max_time_per_local_search=0.05,
                random_seed=0,
                init_type="simple",
                n_faces=2,
            ),
        ),
        (
            "SA",
            lambda: sa.explain(
                x=query_for_explain,
                query_class=query_class,
                norm=2,
                n_population=4,
                n_iter=5,
                max_time_per_sa=0.05,
                random_seed=0,
            ),
        ),
        (
            "SA exhaustive",
            lambda: sa.explain(
                x=query_for_explain,
                query_class=query_class,
                norm=1,
                n_population=4,
                n_iter=5,
                max_time_per_sa=0.05,
                random_seed=0,
                exhaustive=True,
            ),
        ),
    ]

    for label, fn in attempts:
        t0 = time.time()
        try:
            fn()
            if verbose:
                _emit(f"  [ok]   {label:<20} ({time.time() - t0:5.1f}s)")
        except Exception as exc:  # noqa: BLE001 - failed solves still compile
            if verbose:
                _emit(f"  [warn] {label:<20} {type(exc).__name__}: {exc}")

    run_direct_kernel_warmup(dls, query)


def warmup_numba(*, verbose: bool = False, force: bool = False) -> bool:
    """
    Compile and cache LS numba kernels once for the current process.

    Existing cache files for the active Python version are treated as already
    warmed unless ``force`` is set.

    Returns
    -------
    bool
        ``True`` when this call found an existing cache, completed, or had
        already completed the warmup.
        ``False`` when called re-entrantly while another warmup is in progress.

    """
    global _warmed_up, _warming_up  # noqa: PLW0603

    if not force and _already_warmed_up():
        return True
    if _currently_warming_up():
        return False

    with _warmup_lock:
        if not force and _already_warmed_up():
            return True
        if _currently_warming_up():
            return False
        if not force and _numba_cache_exists():
            _warmed_up = True
            return True

        _warming_up = True
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if verbose:
                    rf, mapper, data, x_values = build_model()
                    run_workload(rf, mapper, data, x_values, verbose=True)
                else:
                    with redirect_stdout(StringIO()):
                        rf, mapper, data, x_values = build_model()
                        run_workload(rf, mapper, data, x_values)
        finally:
            _warming_up = False

        _warmed_up = True
        return True


def main() -> int:
    warnings.filterwarnings("ignore")

    _emit("== 0. Clearing numba cache under ocean/ls (cold run) ==")
    removed = clear_numba_cache(ROOT / "ocean" / "ls")
    _emit(f"   removed {removed} cache file(s)")

    _emit("\n== 1. Importing all submodules ==")
    modules = import_all_submodules(PACKAGES)
    dispatchers = collect_dispatchers(modules)
    _emit(f"   found {len(dispatchers)} @njit functions in {PACKAGES}")

    _emit("\n== 2. Running warm-up workload (compiles + caches) ==")
    t0 = time.time()
    try:
        warmup_numba(verbose=True, force=True)
    except Exception as exc:  # noqa: BLE001
        _emit(f"  [error] workload failed early: {type(exc).__name__}: {exc}")
    _emit(f"   workload done in {time.time() - t0:.1f}s")

    _emit("\n== 3. Verifying every @njit function is compiled ==")

    def in_ignored_module(key: str) -> bool:
        module = key.rsplit(".", 1)[0]
        return any(module.endswith(m) for m in IGNORE_MODULES)

    compiled = sorted(
        k for k, dispatcher in dispatchers.items() if dispatcher.signatures
    )
    missing = [
        k for k, dispatcher in dispatchers.items() if not dispatcher.signatures
    ]
    ignored = sorted(k for k in missing if in_ignored_module(k))
    dead = sorted(k for k in missing if k in KNOWN_UNREACHABLE)
    unexpected = sorted(
        k
        for k in missing
        if not in_ignored_module(k) and k not in KNOWN_UNREACHABLE
    )

    _emit(f"   compiled            : {len(compiled)}/{len(dispatchers)}")
    ignored_label = ", ".join(IGNORE_MODULES) if IGNORE_MODULES else "none"
    _emit(f"   ignored modules     : {len(ignored)} ({ignored_label})")
    _emit(
        f"   known-unreachable   : {len(dead)} "
        "(dead code, skipped on purpose)"
    )

    if unexpected:
        _emit(
            f"\n   /!\\ {len(unexpected)} function(s) NOT compiled and "
            "NOT on the allowlist:"
        )
        for key in unexpected:
            _emit(f"        - {key}")
        _emit("\n   -> extend run_workload() so these get called.")
        return 1

    _emit("\n   OK: every live @njit function is compiled and cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
