from .load import Loaded, Loader


def load_credit() -> Loaded:
    return Loader("Credit").load()


def load_adult() -> Loaded:
    return Loader("Adult").load()


def load_compas() -> Loaded:
    return Loader("COMPAS").load()


__all__ = ["load_adult", "load_compas", "load_credit"]
