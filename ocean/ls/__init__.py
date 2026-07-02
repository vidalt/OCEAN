"""Local-search heuristic backend for counterfactual search."""

from ._explainer import (
    BaseLocalSearchExplainer,
    SimulatedAnnealingExplainer,
)
from ._explanation import Explanation
from .dls._explainer import DeterministicMultiStartExplainer
from .sls._explainer import StochasticMultiStartExplainer

DLSExplainer = DeterministicMultiStartExplainer
SLSExplainer = StochasticMultiStartExplainer

__all__ = [
    "BaseLocalSearchExplainer",
    "DLSExplainer",
    "Explanation",
    "SLSExplainer",
    "SimulatedAnnealingExplainer",
]
