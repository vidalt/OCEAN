from sklearn.ensemble._iforest import (  # noqa: PLC2701
    _average_path_length,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportPrivateUsage, reportArgumentType]
)

from ..typing import NonNegativeInt, NonNegativeNumber


def average_length(n: NonNegativeInt) -> NonNegativeNumber:
    return float(_average_path_length([n])[0])  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
