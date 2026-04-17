# ruff: noqa: E402

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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.parent))

import matplotlib as mpl
import gurobipy as gp
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.feature import parse_features

mpl.use("Agg")

from matplotlib import pyplot as plt

FIGURES_DIR = ROOT / "docs" / "_static" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = FIGURES_DIR / "isolation-forest-example-2d.svg"

ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()

BACKGROUND = ListedColormap(["#eff6ff", "#fee2e2"])
POINTS = ListedColormap(["#1d4ed8", "#dc2626"])


@dataclass(frozen=True)
class ExampleData:
    raw: pd.DataFrame
    data: pd.DataFrame
    mapper: object
    target: np.ndarray[tuple[int], np.dtype[np.int64]]
    model: RandomForestClassifier
    isolation: IsolationForest
    query: np.ndarray[tuple[int], np.dtype[np.float64]]
    outlier_point: np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass(frozen=True)
class CounterfactualResult:
    backend: str
    isolation_enabled: bool
    counterfactual: np.ndarray[tuple[int], np.dtype[np.float64]]
    distance: float
    isolation_score: float


@dataclass(frozen=True)
class ExampleSolutions:
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


def build_example_dataset(seed: int = 11) -> ExampleData:
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
        max_samples=16,
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
# ## Solve the same query four ways
#
# We compare two backends, and for each backend we solve once without the
# isolation forest and once with it.


def _solve_backend(
    data: ExampleData,
    *,
    backend: str,
    use_isolation: bool,
    seed: int,
) -> CounterfactualResult:
    kwargs: dict[str, object] = {"mapper": data.mapper}
    if backend == "MIP":
        kwargs["env"] = ENV
        explainer_cls = MixedIntegerProgramExplainer
    elif backend == "CP":
        explainer_cls = ConstraintProgrammingExplainer
    else:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)

    if use_isolation:
        kwargs["isolation"] = data.isolation

    explainer = explainer_cls(data.model, **kwargs)
    explanation = explainer.explain(
        data.query,
        y=1,
        norm=1,
        max_time=10,
        num_workers=1,
        random_seed=seed,
    )
    if explanation is None:
        mode = "with isolation" if use_isolation else "without isolation"
        msg = f"{backend} could not find a counterfactual {mode}."
        raise RuntimeError(msg)

    counterfactual = explanation.to_numpy().copy()
    distance = float(explainer.get_distance())
    frame = pd.DataFrame([counterfactual], columns=data.data.columns)
    isolation_score = float(data.isolation.decision_function(frame)[0])

    return CounterfactualResult(
        backend=backend,
        isolation_enabled=use_isolation,
        counterfactual=counterfactual,
        distance=distance,
        isolation_score=isolation_score,
    )


def solve_example(data: ExampleData, seed: int = 11) -> ExampleSolutions:
    return ExampleSolutions(
        mip_plain=_solve_backend(
            data,
            backend="MIP",
            use_isolation=False,
            seed=seed,
        ),
        mip_iso=_solve_backend(
            data,
            backend="MIP",
            use_isolation=True,
            seed=seed,
        ),
        cp_plain=_solve_backend(
            data,
            backend="CP",
            use_isolation=False,
            seed=seed,
        ),
        cp_iso=_solve_backend(
            data,
            backend="CP",
            use_isolation=True,
            seed=seed,
        ),
    )


# %% [markdown]
# ## Plot the decision regions and the MIP moves
#
# The background is the random-forest decision boundary. The dashed contour is
# the zero level of the isolation-forest decision function. Points outside that
# contour are treated as too isolated by the auxiliary forest.


def _plot_background(
    ax: plt.Axes,
    example: ExampleData,
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
        levels=[0.0],
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


def plot_example(
    example: ExampleData,
    solution: ExampleSolutions,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    _plot_background(ax, example)

    ax.scatter(
        example.query[0],
        example.query[1],
        marker="*",
        s=180,
        c="#0f172a",
        zorder=5,
        label="query",
    )
    ax.scatter(
        solution.mip_plain.counterfactual[0],
        solution.mip_plain.counterfactual[1],
        marker="X",
        s=110,
        c="#d97706",
        zorder=6,
        label="MIP CF without isolation",
    )
    ax.scatter(
        solution.mip_iso.counterfactual[0],
        solution.mip_iso.counterfactual[1],
        marker="D",
        s=86,
        c="#b91c1c",
        zorder=6,
        label="MIP CF with isolation",
    )

    _draw_arrow(
        ax,
        example.query,
        solution.mip_plain.counterfactual,
        color="#d97706",
        label="MIP move without isolation",
    )
    _draw_arrow(
        ax,
        example.query,
        solution.mip_iso.counterfactual,
        color="#b91c1c",
        label="MIP move with isolation",
    )

    ax.annotate(
        "single approved outlier",
        xy=(example.outlier_point[0], example.outlier_point[1]),
        xytext=(-54, 18),
        textcoords="offset points",
        fontsize=9,
        color="#7c2d12",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#fb923c"},
    )
    ax.annotate(
        "tiny approved pocket",
        xy=(
            solution.mip_plain.counterfactual[0],
            solution.mip_plain.counterfactual[1],
        ),
        xytext=(-72, 26),
        textcoords="offset points",
        fontsize=9,
        color="#92400e",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#f59e0b"},
    )
    ax.annotate(
        "dense approved region",
        xy=(
            solution.mip_iso.counterfactual[0],
            solution.mip_iso.counterfactual[1],
        ),
        xytext=(12, -28),
        textcoords="offset points",
        fontsize=9,
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#ef4444"},
    )

    ax.text(
        0.02,
        0.02,
        (
            f"MIP without isolation: L1 = {solution.mip_plain.distance:.3f}, "
            f"IF score = {solution.mip_plain.isolation_score:.4f}\n"
            f"MIP with isolation: L1 = {solution.mip_iso.distance:.3f}, "
            f"IF score = {solution.mip_iso.isolation_score:.4f}"
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1"},
    )
    ax.set_title("Isolation forest example")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Print the backend comparison
#
# The figure shows the MIP geometry. The table below lets us compare the
# decoded explanations returned by both MIP and CP on the same query.


def build_results_table(solution: ExampleSolutions) -> pd.DataFrame:
    rows = []
    for result in (
        solution.mip_plain,
        solution.mip_iso,
        solution.cp_plain,
        solution.cp_iso,
    ):
        rows.append({
            "backend": result.backend,
            "isolation": "yes" if result.isolation_enabled else "no",
            "counterfactual": np.round(result.counterfactual, 6).tolist(),
            "distance": round(result.distance, 6),
            "if_score": round(result.isolation_score, 6),
        })
    return pd.DataFrame(rows)


def print_summary(example: ExampleData, solution: ExampleSolutions) -> None:
    columns = example.data.columns
    query_frame = pd.DataFrame([example.query], columns=columns)

    print("Query:", example.query)
    print(
        "Query prediction:",
        int(example.model.predict(query_frame)[0]),
        "proba=",
        np.round(example.model.predict_proba(query_frame)[0], 4),
    )
    print()
    print(build_results_table(solution).to_string(index=False))
    print()
    print("Saved figure to", OUTPUT_PATH)


def main() -> None:
    example = build_example_dataset()
    solution = solve_example(example)
    plot_example(example, solution, OUTPUT_PATH)
    print_summary(example, solution)


if __name__ == "__main__":
    main()
