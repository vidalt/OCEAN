"""Shared validation for query-specific CP and MIP actionability rules."""

import numpy as np

from .abc import Mapper
from .feature._keeper import FeatureKeeper
from .typing import Actionability, ActionabilityRule, Array1D, Key

_RULES: tuple[ActionabilityRule, ...] = ("fixed", "increase", "decrease")


def validate_actionability[V: FeatureKeeper](
    mapper: Mapper[V], x: Array1D, actionability: Actionability | None
) -> None:
    """
    Validate every rule before either backend mutates its model.

    Raises
    ------
    ValueError
        If a feature, rule, or constrained query value is invalid.

    """
    if actionability is None:
        return
    values = np.asarray(x, dtype=float)
    if values.shape != (mapper.n_columns,) or not np.isfinite(values).all():
        msg = "Actionability requires a finite 1D query matching the mapper"
        raise ValueError(msg)
    for name, rule in actionability.items():
        if name not in mapper:
            msg = f"Unknown actionability feature: {name!r}"
            raise ValueError(msg)
        if rule not in _RULES:
            msg = f"Invalid actionability rule for {name!r}: {rule!r}"
            raise ValueError(msg)
        feature = mapper[name]
        if feature.is_one_hot_encoded:
            if rule != "fixed":
                msg = f"One-hot feature {name!r} supports only 'fixed'"
                raise ValueError(msg)
            block = values[
                [mapper.idx.get(name, code) for code in feature.codes]
            ]
            if not np.isin(block, [0, 1]).all() or block.sum() != 1:
                msg = f"Expected exactly one active category for {name!r}"
                raise ValueError(msg)
            continue
        value = float(values[mapper.idx.get(name)])
        _validate_value(feature, name, value)


def _validate_value(feature: FeatureKeeper, name: Key, value: float) -> None:
    if feature.is_binary and value not in {0, 1}:
        msg = f"Expected a binary query value for {name!r}"
        raise ValueError(msg)
    if feature.is_discrete and not value.is_integer():
        msg = f"Expected an integer query value for {name!r}"
        raise ValueError(msg)
    if (
        feature.is_numeric
        and not feature.levels[0] <= value <= feature.levels[-1]
    ):
        msg = f"Query value for {name!r} is outside the feature bounds"
        raise ValueError(msg)
