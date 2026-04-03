"""Convenience loaders for the example datasets used in OCEAN."""

from functools import partial

from ._load import Loader

loader = Loader()


load_credit = partial(loader.load, name="Credit")
load_adult = partial(loader.load, name="Adult")
load_compas = partial(loader.load, name="COMPAS")

load_credit.__doc__ = (
    "Load the Credit dataset as ``((data, target), mapper)`` by default."
)
load_adult.__doc__ = (
    "Load the Adult dataset as ``((data, target), mapper)`` by default."
)
load_compas.__doc__ = (
    "Load the COMPAS dataset as ``((data, target), mapper)`` by default."
)


__all__ = ["load_adult", "load_compas", "load_credit"]
