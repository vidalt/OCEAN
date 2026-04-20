# ruff: noqa: E402
# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

# %% [markdown]
# # Isolation Forest Example
#
# A random forest can sometimes learn a tiny approved pocket around an unusual
# training point. If we optimize only for the prediction flip and the shortest
# move, the counterfactual may head straight for that pocket.
#
# Adding an isolation forest changes the question. We still want a valid
# counterfactual, but we also ask that it stays in a region that looks typical
# with respect to the training data.

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.parent))

import gurobipy as gp
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.feature import parse_features
from ocean.mip import Model as MixedIntegerProgramModel
from ocean.tree import parse_ensembles
from ocean.tree._utils import minimum_average_length  # noqa: PLC2701

if TYPE_CHECKING:
    from ocean.abc import Mapper
    from ocean.feature import Feature

mpl.use("Agg")

from matplotlib import pyplot as plt

FIGURES_DIR = ROOT / "docs" / "_static" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = FIGURES_DIR / "isolation-forest-example-2d.svg"
EXAMPLE_SEED = 20
THRESHOLDS = (0.9, 0.51, 0.5, 0.1)
TABLE_THRESHOLDS = (0.9, 0.51, 0.5, 0.1)

ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()

BACKGROUND = ListedColormap(["#eff6ff", "#fee2e2"])
POINTS = ListedColormap(["#1d4ed8", "#dc2626"])


@dataclass(frozen=True)
class ExampleData:
    raw: pd.DataFrame
    data: pd.DataFrame
    mapper: Mapper[Feature]
    target: np.ndarray[tuple[int], np.dtype[np.int64]]
    model: RandomForestClassifier
    isolation: IsolationForest
    query: np.ndarray[tuple[int], np.dtype[np.float64]]
    outlier_point: np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass(frozen=True)
class CounterfactualResult:
    backend: str
    isolation_enabled: bool
    threshold: float | None
    status: str
    counterfactual: np.ndarray[tuple[int], np.dtype[np.float64]] | None
    distance: float | None
    isolation_score: float | None


@dataclass(frozen=True)
class ThresholdCase:
    threshold: float
    decision_level: float
    required_length: float
    max_target_length: float
    mip_plain: CounterfactualResult
    mip_iso: CounterfactualResult
    cp_plain: CounterfactualResult
    cp_iso: CounterfactualResult


# %% [markdown]
# ## Build a small 2D dataset
#
# We create:
# - a dense denied region on the left,
# - a dense approved region on the right,
# - a bridge of denied points close to the query,
# - and a single approved outlier that leaves behind a tiny approved pocket.


def build_example_dataset(seed: int = EXAMPLE_SEED) -> ExampleData:  # noqa: PLR0914
    rng = np.random.default_rng(seed)

    denied_main = rng.normal(
        loc=[-1.3, -0.45],
        scale=[0.30, 0.30],
        size=(150, 2),
    )
    denied_bridge = rng.normal(
        loc=[-0.2, 0.55],
        scale=[0.20, 0.20],
        size=(60, 2),
    )
    approved_main = rng.normal(
        loc=[1.65, 1.25],
        scale=[0.28, 0.25],
        size=(160, 2),
    )
    approved_island = rng.normal(
        loc=[0.3, 0.95],
        scale=[0.03, 0.04],
        size=(1, 2),
    )

    features = np.vstack([
        denied_main,
        denied_bridge,
        approved_main,
        approved_island,
    ])
    target = np.array(
        [0] * len(denied_main)
        + [0] * len(denied_bridge)
        + [1] * len(approved_main)
        + [1] * len(approved_island),
        dtype=np.int64,
    )

    raw = pd.DataFrame(features, columns=["income_shift", "stability_shift"])
    data, mapper = parse_features(raw, scale=False)

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        random_state=seed,
    )
    model.fit(data, target)

    isolation = IsolationForest(
        random_state=seed,
        n_estimators=100,
        max_samples=16,  # pyright: ignore[reportArgumentType]
    )
    isolation.fit(data)

    query = np.array([0.0, 0.65], dtype=np.float64)
    query_frame = pd.DataFrame([query], columns=data.columns)
    predicted = int(model.predict(query_frame)[0])
    if predicted != 0:
        msg = "The fixed query is expected to be rejected."
        raise RuntimeError(msg)

    return ExampleData(
        raw=raw,
        data=data,
        mapper=mapper,
        target=target,
        model=model,
        isolation=isolation,
        query=query,
        outlier_point=approved_island[0].astype(np.float64),
    )


# %% [markdown]
# ## Solve the same query under four isolation thresholds
#
# We compare four thresholds:
# - ``0.9`` where the isolation constraint is weak enough to match the plain CF,
# - ``0.51`` where the pocket is no longer enough and the CF moves inward,
# - ``0.5`` where it pushes the CF into the dense approved region,
# - ``0.1`` where it makes the target class infeasible.


def _solve_backend(
    data: ExampleData,
    *,
    backend: str,
    use_isolation: bool,
    seed: int,
    threshold: float | None = None,
) -> CounterfactualResult:
    explainer: (
        MixedIntegerProgramExplainer | ConstraintProgrammingExplainer
    )
    if backend == "MIP":
        if use_isolation:
            explainer = MixedIntegerProgramExplainer(
                data.model,
                mapper=data.mapper,
                isolation=data.isolation,
                isolation_threshold=threshold,
                env=ENV,
            )
        else:
            explainer = MixedIntegerProgramExplainer(
                data.model,
                mapper=data.mapper,
                env=ENV,
            )
    elif backend == "CP":
        if use_isolation:
            explainer = ConstraintProgrammingExplainer(
                data.model,
                mapper=data.mapper,
                isolation=data.isolation,
                isolation_threshold=threshold,
            )
        else:
            explainer = ConstraintProgrammingExplainer(
                data.model,
                mapper=data.mapper,
            )
    else:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="There are no feasible counterfactuals for this query.*",
        )
        explanation = explainer.explain(
            data.query,
            y=1,
            norm=1,
            max_time=10,
            num_workers=1,
            random_seed=seed,
        )
    status = explainer.get_solving_status()
    if explanation is None:
        return CounterfactualResult(
            backend=backend,
            isolation_enabled=use_isolation,
            threshold=threshold,
            status=status,
            counterfactual=None,
            distance=None,
            isolation_score=None,
        )

    counterfactual = explanation.to_numpy().copy()
    distance = float(explainer.get_distance())
    frame = pd.DataFrame([counterfactual], columns=data.data.columns)
    isolation_score = float(data.isolation.decision_function(frame)[0])

    return CounterfactualResult(
        backend=backend,
        isolation_enabled=use_isolation,
        threshold=threshold,
        status=status,
        counterfactual=counterfactual,
        distance=distance,
        isolation_score=isolation_score,
    )


def _decision_level(example: ExampleData, threshold: float) -> float:
    return float(-threshold - example.isolation.offset_)


def _maximize_target_length(data: ExampleData) -> float:
    trees = parse_ensembles(data.model, data.isolation, mapper=data.mapper)
    model = MixedIntegerProgramModel(
        trees=trees,
        mapper=data.mapper,
        n_isolators=len(data.isolation),
        max_samples=int(data.isolation.max_samples_),
        isolation_threshold=1.0,
        env=ENV,
    )
    model.build()
    model.set_majority_class(y=1)
    model.setObjective(model.length, gp.GRB.MAXIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        msg = "Failed to maximize the target-class isolation length."
        raise RuntimeError(msg)
    return float(model.ObjVal)


def solve_example(
    data: ExampleData,
    seed: int = EXAMPLE_SEED,
) -> tuple[ThresholdCase, ...]:
    mip_plain = _solve_backend(
        data,
        backend="MIP",
        use_isolation=False,
        seed=seed,
    )
    cp_plain = _solve_backend(
        data,
        backend="CP",
        use_isolation=False,
        seed=seed,
    )
    max_target_length = _maximize_target_length(data)
    cases: list[ThresholdCase] = []

    for threshold in THRESHOLDS:
        required_length = len(data.isolation) * minimum_average_length(
            int(data.isolation.max_samples_),
            threshold=threshold,
        )
        cases.append(
            ThresholdCase(
                threshold=threshold,
                decision_level=_decision_level(data, threshold),
                required_length=float(required_length),
                max_target_length=max_target_length,
                mip_plain=mip_plain,
                mip_iso=_solve_backend(
                    data,
                    backend="MIP",
                    use_isolation=True,
                    seed=seed,
                    threshold=threshold,
                ),
                cp_plain=cp_plain,
                cp_iso=_solve_backend(
                    data,
                    backend="CP",
                    use_isolation=True,
                    seed=seed,
                    threshold=threshold,
                ),
            )
        )

    return tuple(cases)


# %% [markdown]
# ## Plot the decision regions and the MIP moves
#
# The background is the random-forest decision boundary. The dashed contour is
# the isolation-forest cutoff induced by the chosen threshold. Points below
# that contour are too isolated for the corresponding run.


def _plot_background(
    ax: plt.Axes,
    example: ExampleData,
    *,
    decision_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = example.data["income_shift"]
    y = example.data["stability_shift"]

    x_pad = 0.30 * (float(x.max()) - float(x.min()) or 1.0)
    y_pad = 0.25 * (float(y.max()) - float(y.min()) or 1.0)
    x_range = np.linspace(float(x.min()) - x_pad, float(x.max()) + x_pad, 320)
    y_range = np.linspace(float(y.min()) - y_pad, float(y.max()) + y_pad, 320)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = pd.DataFrame({
        "income_shift": xx.ravel(),
        "stability_shift": yy.ravel(),
    })
    pred = example.model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, pred, alpha=0.88, cmap=BACKGROUND)
    ax.scatter(
        example.data["income_shift"],
        example.data["stability_shift"],
        c=example.target,
        cmap=POINTS,
        edgecolor="white",
        linewidth=0.45,
        s=16,
        alpha=0.9,
    )

    iso_values = example.isolation.decision_function(grid).reshape(xx.shape)
    ax.contour(
        xx,
        yy,
        iso_values,
        levels=[decision_level],
        colors=["#0f172a"],
        linewidths=1.6,
        linestyles=["--"],
    )

    ax.set_xlabel("income shift")
    ax.set_ylabel("stability shift")
    ax.grid(alpha=0.18, linewidth=0.6)
    return xx, yy


def _draw_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    arrow = FancyArrowPatch(
        posA=(start[0], start[1]),
        posB=(end[0], end[1]),
        arrowstyle="->",
        mutation_scale=11,
        linewidth=1.8,
        color=color,
        label=label,
    )
    ax.add_patch(arrow)


def _same_counterfactual(
    left: CounterfactualResult,
    right: CounterfactualResult,
) -> bool:
    if left.counterfactual is None or right.counterfactual is None:
        return False
    return bool(np.allclose(left.counterfactual, right.counterfactual))


def _format_counterfactual(
    counterfactual: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
) -> str:
    if counterfactual is None:
        return "-"
    rounded = np.round(counterfactual, 6).tolist()
    return str(rounded)


def _format_scalar(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _require_counterfactual(
    result: CounterfactualResult,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    if result.counterfactual is None:
        msg = (
            "Expected a counterfactual for "
            f"{result.backend} {result.threshold}."
        )
        raise RuntimeError(msg)
    return result.counterfactual


def _panel_result_text(case: ThresholdCase) -> str:
    plain = (
        f"plain: L1={case.mip_plain.distance:.3f}, "
        f"IF={case.mip_plain.isolation_score:.3f}"
    )
    if case.mip_iso.counterfactual is None:
        return (
            f"{plain}\n"
            "iso: infeasible\n"
            f"required length={case.required_length:.1f}"
        )
    return (
        f"{plain}\n"
        f"iso: L1={case.mip_iso.distance:.3f}, "
        f"IF={case.mip_iso.isolation_score:.3f}"
    )


def plot_example(
    example: ExampleData,
    cases: tuple[ThresholdCase, ...],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.4, 9.2),
        sharex=True,
        sharey=True,
    )

    for ax, case in zip(axes.flat, cases, strict=True):
        _plot_background(ax, example, decision_level=case.decision_level)
        same_mip_solution = _same_counterfactual(case.mip_plain, case.mip_iso)
        plain_cf = _require_counterfactual(case.mip_plain)

        ax.scatter(
            example.query[0],
            example.query[1],
            marker="*",
            s=120,
            c="#0f172a",
            zorder=5,
        )
        ax.scatter(
            plain_cf[0],
            plain_cf[1],
            marker="X",
            s=58,
            c="#d97706",
            zorder=6,
        )

        _draw_arrow(
            ax,
            example.query,
            plain_cf,
            color="#d97706",
            label="MIP move without isolation",
        )

        if case.mip_iso.counterfactual is not None:
            iso_cf = case.mip_iso.counterfactual
            ax.scatter(
                iso_cf[0],
                iso_cf[1],
                marker="D",
                s=46,
                facecolors="none",
                edgecolors="#b91c1c",
                linewidths=1.5,
                zorder=7,
            )
            if not same_mip_solution:
                _draw_arrow(
                    ax,
                    example.query,
                    iso_cf,
                    color="#b91c1c",
                    label="MIP move with isolation",
                )

        ax.text(
            0.02,
            0.98,
            f"decision_function >= {case.decision_level:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        )
        ax.text(
            0.02,
            0.02,
            _panel_result_text(case),
            transform=ax.transAxes,
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        )

        if case.threshold == 0.9:
            ax.annotate(
                "single approved outlier",
                xy=(example.outlier_point[0], example.outlier_point[1]),
                xytext=(-42, 16),
                textcoords="offset points",
                fontsize=8.6,
                color="#7c2d12",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#fb923c",
                },
            )
            ax.annotate(
                "plain and isolation match",
                xy=(
                    plain_cf[0],
                    plain_cf[1],
                ),
                xytext=(-68, 20),
                textcoords="offset points",
                fontsize=8.6,
                color="#92400e",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#f59e0b",
                },
            )
        elif case.threshold == 0.51:
            iso_cf = _require_counterfactual(case.mip_iso)
            ax.annotate(
                "the tiny pocket is no longer enough,\nso the CF moves inward",
                xy=(
                    iso_cf[0],
                    iso_cf[1],
                ),
                xytext=(-82, 18),
                textcoords="offset points",
                fontsize=8.6,
                color="#92400e",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#f59e0b",
                },
            )
        elif case.threshold == 0.5 and case.mip_iso.counterfactual is not None:
            ax.annotate(
                "plain falls in the tiny pocket",
                xy=(
                    plain_cf[0],
                    plain_cf[1],
                ),
                xytext=(-72, 20),
                textcoords="offset points",
                fontsize=8.6,
                color="#92400e",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#f59e0b",
                },
            )
            ax.annotate(
                "isolation pushes to the dense region",
                xy=(
                    case.mip_iso.counterfactual[0],
                    case.mip_iso.counterfactual[1],
                ),
                xytext=(-8, -26),
                textcoords="offset points",
                fontsize=8.6,
                color="#7f1d1d",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#ef4444",
                },
            )
        else:
            ax.text(
                0.04,
                0.75,
                (
                    "no isolation-aware target point\n"
                    f"max target length = {case.max_target_length:.1f}\n"
                    f"required = {case.required_length:.1f}"
                ),
                transform=ax.transAxes,
                fontsize=8.7,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "fc": "white",
                    "ec": "#ef4444",
                },
            )

        outcome = "plain = isolation"
        if case.mip_iso.counterfactual is None:
            outcome = "isolation infeasible"
        elif not same_mip_solution:
            outcome = "plain != isolation"
        ax.set_title(f"threshold = {case.threshold}\n{outcome}")

    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=10,
            markerfacecolor="#0f172a",
            markeredgecolor="#0f172a",
            label="query",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markersize=7,
            markerfacecolor="#d97706",
            markeredgecolor="#d97706",
            label="MIP CF without isolation",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=6.4,
            markerfacecolor="white",
            markeredgecolor="#b91c1c",
            label="MIP CF with isolation",
        ),
        Line2D(
            [0],
            [0],
            color="#0f172a",
            linestyle="--",
            linewidth=1.6,
            label="threshold contour",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Isolation forest example", y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Print the backend comparison
#
# The figure shows the MIP geometry. The table below lets us compare the
# decoded explanations returned by both MIP and CP on the same query.


def build_results_table(
    cases: tuple[ThresholdCase, ...],
) -> pd.DataFrame:
    plain_case = cases[0]
    by_threshold = {case.threshold: case for case in cases}
    rows: list[dict[str, object]] = [
        {
            "case": "plain",
            "mip_status": plain_case.mip_plain.status,
            "mip_cf": _format_counterfactual(
                plain_case.mip_plain.counterfactual
            ),
            "mip_distance": _format_scalar(plain_case.mip_plain.distance),
            "mip_if_score": _format_scalar(
                plain_case.mip_plain.isolation_score
            ),
            "cp_status": plain_case.cp_plain.status,
            "cp_cf": _format_counterfactual(plain_case.cp_plain.counterfactual),
            "cp_distance": _format_scalar(plain_case.cp_plain.distance),
            "cp_if_score": _format_scalar(plain_case.cp_plain.isolation_score),
        }
    ]
    rows.extend(
        (
            {
                "case": f"isolation = {threshold}",
                "mip_status": by_threshold[threshold].mip_iso.status,
                "mip_cf": _format_counterfactual(
                    by_threshold[threshold].mip_iso.counterfactual
                ),
                "mip_distance": _format_scalar(
                    by_threshold[threshold].mip_iso.distance
                ),
                "mip_if_score": _format_scalar(
                    by_threshold[threshold].mip_iso.isolation_score
                ),
                "cp_status": by_threshold[threshold].cp_iso.status,
                "cp_cf": _format_counterfactual(
                    by_threshold[threshold].cp_iso.counterfactual
                ),
                "cp_distance": _format_scalar(
                    by_threshold[threshold].cp_iso.distance
                ),
                "cp_if_score": _format_scalar(
                    by_threshold[threshold].cp_iso.isolation_score
                ),
            }
        )
        for threshold in TABLE_THRESHOLDS
    )
    return pd.DataFrame(rows)


def print_summary(
    example: ExampleData,
    cases: tuple[ThresholdCase, ...],
) -> None:
    columns = example.data.columns
    query_frame = pd.DataFrame([example.query], columns=columns)
    by_threshold = {case.threshold: case for case in cases}
    infeasible_case = by_threshold[0.1]

    print("Query:", example.query)
    print(
        "Query prediction:",
        int(example.model.predict(query_frame)[0]),
        "proba=",
        np.round(example.model.predict_proba(query_frame)[0], 4),
    )
    print()
    print(build_results_table(cases).to_string(index=False))
    print()
    print("Infeasibility proof at threshold", infeasible_case.threshold)
    print(
        "maximum target-class isolation length:",
        round(infeasible_case.max_target_length, 6),
    )
    print(
        "required isolation length:",
        round(infeasible_case.required_length, 6),
    )
    print(
        "inequality:",
        f"{infeasible_case.max_target_length:.6f} < "
        f"{infeasible_case.required_length:.6f}",
    )
    print()
    print("Saved figure to", OUTPUT_PATH)


def main() -> None:
    example = build_example_dataset(seed=EXAMPLE_SEED)
    cases = solve_example(example, seed=EXAMPLE_SEED)
    plot_example(example, cases, OUTPUT_PATH)
    print_summary(example, cases)


if __name__ == "__main__":
    main()
