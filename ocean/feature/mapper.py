from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from .feature import Feature

if TYPE_CHECKING:
    import pandas as pd


class FeatureMapper(Mapping[Hashable, Feature]):
    _columns: "pd.Index[str] | pd.MultiIndex"
    _map: dict[Hashable, Feature]

    _names: tuple[Hashable, ...] | None = None
    _codes: tuple[Hashable, ...] | None = None

    def __init__(
        self,
        names: Iterable[Hashable],
        features: Iterable[Feature],
        columns: "pd.Index[str] | pd.MultiIndex",
    ) -> None:
        self._columns = columns
        self._map = dict(zip(names, features, strict=True))

    @property
    def columns(self) -> "pd.Index[str] | pd.MultiIndex":
        return self._columns

    @property
    def names(self) -> tuple[Hashable, ...]:
        if self._names is None:
            names: pd.Index[str] = self._columns.get_level_values(0)
            self._names = tuple(names)
        return self._names

    @property
    def codes(self) -> tuple[Hashable, ...]:
        if self._codes is None:
            n_levels = 2
            if self._columns.nlevels < n_levels:
                msg = "No one-hot encoded features found"
                raise ValueError(msg)
            codes: pd.Index[str] = self._columns.get_level_values(1)
            self._codes = tuple(codes)
        return self._codes

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._map)

    def __getitem__(self, key: Hashable) -> Feature:
        return self._map[key]
