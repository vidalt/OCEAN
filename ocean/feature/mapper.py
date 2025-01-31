from collections.abc import Hashable, Iterable, Iterator, Mapping

import pandas as pd

from .feature import Feature


class FeatureMapper(Mapping[Hashable, Feature]):
    _columns: pd.Index
    _map: dict[Hashable, Feature]

    _codes: tuple[Hashable, ...] | None = None
    _names: tuple[Hashable, ...]

    def __init__(
        self,
        names: Iterable[Hashable],
        features: Iterable[Feature],
        columns: pd.Index,
    ) -> None:
        self._columns = columns
        self._map = dict(zip(names, features, strict=True))
        self._names = tuple(columns.get_level_values(0))

    @property
    def columns(self) -> pd.Index:
        return self._columns

    @property
    def names(self) -> tuple[Hashable, ...]:
        return self._names

    @property
    def codes(self) -> tuple[Hashable, ...]:
        if self._codes is None:
            if self._columns.nlevels == 1:
                msg = "No one-hot encoded features found"
                raise ValueError(msg)
            self._codes = tuple(self._columns.get_level_values(1))
        return self._codes

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._map)

    def __getitem__(self, key: Hashable) -> Feature:
        return self._map[key]
