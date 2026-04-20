from math import log2

from sklearn.ensemble._iforest import (  # noqa: PLC2701
    _average_path_length,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportPrivateUsage, reportArgumentType]
)

from ..typing import NonNegativeInt, NonNegativeNumber


def average_length(n: NonNegativeInt) -> NonNegativeNumber:
    return float(_average_path_length([n])[0])  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def minimum_average_length(
    n: NonNegativeInt,
    *,
    threshold: float | None = None,
) -> NonNegativeNumber:
    baseline = average_length(n)
    if threshold is None:
        return baseline
    if not 0.0 < threshold <= 1.0:
        msg = "The isolation threshold must satisfy 0 < threshold <= 1."
        raise ValueError(msg)
    return float(-baseline * log2(threshold))
