"""Deterministic local-search helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._explainer import DeterministicMultiStartExplainer

__all__ = ["DeterministicMultiStartExplainer"]


def __getattr__(name: str) -> object:
    if name == "DeterministicMultiStartExplainer":
        module = importlib.import_module("._explainer", __name__)
        return module.DeterministicMultiStartExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
