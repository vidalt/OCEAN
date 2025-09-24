from functools import partial

from ._load import Loader

loader = Loader()


load_credit = partial(loader.load, name="Credit")
load_adult = partial(loader.load, name="Adult")
load_compas = partial(loader.load, name="COMPAS")


__all__ = ["load_adult", "load_compas", "load_credit"]
