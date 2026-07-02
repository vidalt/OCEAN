"""Stochastic local-search helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._explainer import StochasticMultiStartExplainer

__all__ = ["StochasticMultiStartExplainer"]


def __getattr__(name: str) -> object:
    if name == "StochasticMultiStartExplainer":
        module = importlib.import_module("._explainer", __name__)
        return module.StochasticMultiStartExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
