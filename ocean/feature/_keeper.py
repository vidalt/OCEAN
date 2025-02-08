from ..typing import Array1D, Key
from ._feature import Feature


class FeatureKeeper:
    _feature: Feature

    def __init__(self, feature: Feature) -> None:
        self._feature = feature

    @property
    def is_continuous(self) -> bool:
        return self._feature.is_continuous

    @property
    def is_discrete(self) -> bool:
        return self._feature.is_discrete

    @property
    def is_numeric(self) -> bool:
        return self._feature.is_numeric

    @property
    def is_binary(self) -> bool:
        return self._feature.is_binary

    @property
    def is_one_hot_encoded(self) -> bool:
        return self._feature.is_one_hot_encoded

    @property
    def levels(self) -> Array1D:
        return self._feature.levels

    @property
    def codes(self) -> tuple[Key, ...]:
        return self._feature.codes
