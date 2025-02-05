from .adult import AdultLoader
from .compas import COMPASLoader
from .credit import CreditLoader
from .load import Loaded


def load_credit() -> Loaded:
    return CreditLoader().load()


def load_adult() -> Loaded:
    return AdultLoader().load()


def load_compas() -> Loaded:
    return COMPASLoader().load()


__all__ = ["load_adult", "load_compas", "load_credit"]
