from collections.abc import Hashable, Iterable, Iterator, Mapping

import pandas as pd

from .feature import Feature


class FeatureMapper(Mapping[Hashable, Feature]):
    _columns: pd.MultiIndex
    _map: dict[Hashable, Feature]

    def __init__(
        self,
        names: Iterable[Hashable],
        features: Iterable[Feature],
        columns: pd.MultiIndex,
    ) -> None:
        self._columns = columns
        self._map = dict(zip(names, features, strict=True))

    @property
    def columns(self) -> list[tuple[Hashable, Hashable]]:
        return self._columns.to_list()

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._map)

    def __getitem__(self, key: Hashable) -> Feature:
        return self._map[key]
