# ruff: noqa: E402

from __future__ import annotations

import os
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.parent))

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer
from ocean.feature import parse_features

if TYPE_CHECKING:
    from ocean.abc import Mapper
    from ocean.feature import Feature
    from ocean.typing import Array1D

mpl.use("Agg")

from matplotlib import pyplot as plt

FIGURES_DIR = ROOT / "docs" / "_static" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND = ListedColormap(["#eff6ff", "#fee2e2"])
POINTS = ListedColormap(["#1d4ed8", "#dc2626"])


@dataclass(frozen=True)
class ExampleResult:
    raw: pd.DataFrame
    data: pd.DataFrame
    target: pd.Series[int]
    model: RandomForestClassifier
    query_index: int
    query: np.ndarray[tuple[int], np.dtype[np.float64]]
    counterfactual: np.ndarray[tuple[int], np.dtype[np.float64]]
    objective_value: float


def _choose_counterfactual(
    data: pd.DataFrame,
    model: RandomForestClassifier,
    mapper: Mapper[Feature],
    *,
    target_class: int = 1,
    max_checks: int = 24,
) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.float64]], Array1D, float]:
    explainer = ConstraintProgrammingExplainer(model, mapper=mapper)
    predictions = pd.Series(model.predict(data), index=data.index)
    probabilities = model.predict_proba(data)

    ranked_candidates: list[tuple[int, float]] = []
    for idx, predicted_class in predictions.items():
        if predicted_class == target_class:
            continue
        confidence = float(probabilities[idx, int(predicted_class)])
        ranked_candidates.append((int(idx), confidence))
    ranked_candidates.sort(key=itemgetter(1), reverse=True)

    best: (
        tuple[
            int,
            np.ndarray[tuple[int], np.dtype[np.float64]],
            Array1D,
            float,
            float,
        ]
        | None
    ) = None
    for idx, confidence in ranked_candidates[:max_checks]:
        query = data.loc[idx].to_numpy(dtype=float).flatten()
        explanation = explainer.explain(
            query,
            y=target_class,
            norm=1,
            max_time=10,
            num_workers=1,
            random_seed=42,
        )
        if explanation is None:
            continue

        objective_value = explainer.get_objective_value()
        counterfactual = explanation.to_numpy().copy()
        score = confidence * objective_value
        if best is None or score > best[4]:
            best = (
                idx,
                query.copy(),
                counterfactual,
                objective_value,
                score,
            )

    if best is None:
        msg = "Unable to generate a counterfactual for the plotting example."
        raise RuntimeError(msg)
    return best[:4]


def build_continuous_example(seed: int = 42) -> ExampleResult:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.4, 2.4, 260)
    x1 = rng.uniform(-2.0, 2.0, 260)
    score = 0.9 * x0 - 0.55 * x1 + 0.65 * np.sin(1.35 * x0)
    target = (score > 0.45).astype(int)

    raw = pd.DataFrame({
        "feature_0": x0,
        "feature_1": x1,
    })
    data, mapper = parse_features(raw, scale=False)

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=4,
        random_state=seed,
    )
    model.fit(data, target)

    query_index, query, counterfactual, objective_value = (
        _choose_counterfactual(
            data,
            model,
            mapper,
        )
    )
    return ExampleResult(
        raw=raw,
        data=data,
        target=pd.Series(target, name="target"),
        model=model,
        query_index=query_index,
        query=query,
        counterfactual=counterfactual,
        objective_value=objective_value,
    )


def build_ordinal_example(seed: int = 7) -> ExampleResult:
    rng = np.random.default_rng(seed)
    credit_lines = rng.choice([0, 1, 2, 4], size=280)
    income_ratio = rng.uniform(-0.45, 0.85, size=280)
    score = (
        (credit_lines >= 2).astype(int)
        + (credit_lines >= 4).astype(int)
        + (income_ratio > 0.15).astype(int)
        + (income_ratio > 0.45).astype(int)
    )
    target = (score >= 2).astype(int)

    raw = pd.DataFrame({
        "credit_lines": credit_lines,
        "income_ratio": income_ratio,
    })
    data, mapper = parse_features(
        raw,
        discretes=("credit_lines",),
        scale=False,
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=seed,
    )
    model.fit(data, target)

    query_index, query, counterfactual, objective_value = (
        _choose_counterfactual(
            data,
            model,
            mapper,
        )
    )
    return ExampleResult(
        raw=raw,
        data=data,
        target=pd.Series(target, name="target"),
        model=model,
        query_index=query_index,
        query=query,
        counterfactual=counterfactual,
        objective_value=objective_value,
    )


def _plot_background(
    ax: plt.Axes,
    example: ExampleResult,
    *,
    x_label: str,
    y_label: str,
) -> None:
    x = example.data[x_label]
    y = example.data[y_label]

    x_pad = 0.35 * (float(x.max()) - float(x.min()) or 1.0)
    y_pad = 0.25 * (float(y.max()) - float(y.min()) or 1.0)
    x_range = np.linspace(float(x.min()) - x_pad, float(x.max()) + x_pad, 320)
    y_range = np.linspace(float(y.min()) - y_pad, float(y.max()) + y_pad, 320)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = pd.DataFrame({x_label: xx.ravel(), y_label: yy.ravel()})
    pred = example.model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, pred, alpha=0.88, cmap=BACKGROUND)
    ax.scatter(
        x,
        y,
        c=example.target,
        cmap=POINTS,
        edgecolor="white",
        linewidth=0.5,
        s=26,
        alpha=0.9,
    )
    ax.set_xlabel(x_label.replace("_", " "))
    ax.set_ylabel(y_label.replace("_", " "))
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_continuous_example(path: Path) -> None:
    example = build_continuous_example()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _plot_background(ax, example, x_label="feature_0", y_label="feature_1")

    qx, qy = example.query
    cx, cy = example.counterfactual
    ax.scatter(qx, qy, marker="*", s=220, c="#0f172a", label="query")
    ax.scatter(cx, cy, marker="X", s=170, c="#b45309", label="counterfactual")
    ax.annotate(
        "",
        xy=(cx, cy),
        xytext=(qx, qy),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#b45309"},
    )
    ax.annotate(
        "query",
        xy=(qx, qy),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#94a3b8"},
    )
    ax.annotate(
        "counterfactual",
        xy=(cx, cy),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=10,
        color="#92400e",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#f59e0b"},
    )
    ax.set_title("2D continuous example: crossing the decision region")
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.01,
        0.01,
        f"L1 objective = {example.objective_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ordinal_example(path: Path) -> None:
    example = build_ordinal_example()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _plot_background(
        ax,
        example,
        x_label="credit_lines",
        y_label="income_ratio",
    )

    qx, qy = example.query
    cx, cy = example.counterfactual
    for level in [0, 1, 2, 4]:
        ax.axvline(
            level,
            color="#64748b",
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
        )
    ax.scatter(qx, qy, marker="*", s=220, c="#0f172a", label="query")
    ax.scatter(cx, cy, marker="X", s=170, c="#b45309", label="counterfactual")
    ax.annotate(
        "",
        xy=(cx, cy),
        xytext=(qx, qy),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#b45309"},
    )
    ax.annotate(
        "query",
        xy=(qx, qy),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#94a3b8"},
    )
    ax.annotate(
        "counterfactual",
        xy=(cx, cy),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=10,
        color="#92400e",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#f59e0b"},
    )
    ax.set_xticks([0, 1, 2, 4])
    ax.set_title(
        "2D ordinal example: counterfactual over valid discrete levels"
    )
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.01,
        0.01,
        f"L1 objective = {example.objective_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_continuous_example(FIGURES_DIR / "continuous-counterfactual-2d.svg")
    plot_ordinal_example(FIGURES_DIR / "ordinal-counterfactual-2d.svg")
    print("Saved figures to", FIGURES_DIR)


if __name__ == "__main__":
    main()
