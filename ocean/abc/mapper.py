from collections.abc import Callable, Iterator, Mapping
from typing import overload

import pandas as pd

from ..typing import Index, Index1L, Key, NonNegativeInt


class Mapper[T](Mapping[Key, T]):
    _columns: Index
    _map: dict[Key, T]

    _names: tuple[Key, ...] | None = None
    _codes: tuple[Key, ...] | None = None

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, mapping: Mapping[Key, T], *, columns: Index) -> None: ...

    @overload
    def __init__(self, mapping: "Mapper[T]") -> None: ...

    def __init__(
        self,
        mapping: "Mapping[Key, T] | None" = None,
        *,
        columns: Index | None = None,
    ) -> None:
        mapping, columns = self._get_args(mapping, columns)
        self._validate_args(mapping, columns)
        self._columns = columns
        self._map = dict(mapping)

    @property
    def n_columns(self) -> NonNegativeInt:
        return len(self.columns)

    @property
    def n_levels(self) -> NonNegativeInt:
        return self.columns.nlevels

    @property
    def is_multi_level(self) -> bool:
        return self.n_levels > 1

    @property
    def columns(self) -> Index:
        return self._columns

    @property
    def names(self) -> tuple[Key, ...]:
        if self._names is None:
            names: Index1L = self.columns.get_level_values(0)
            self._names = tuple(names)
        return self._names

    @property
    def codes(self) -> tuple[Key, ...]:
        if self._codes is None:
            if not self.is_multi_level:
                msg = "No one-hot encoded features found"
                raise ValueError(msg)
            codes: Index1L = self.columns.get_level_values(1)
            self._codes = tuple(codes)
        return self._codes

    def apply[S](self, func: Callable[[T], S]) -> "Mapper[S]":
        mapping = {name: func(value) for name, value in self.items()}
        return Mapper(mapping, columns=self.columns)

    def transform[S](self, func: Callable[[Key, T], S]) -> "Mapper[S]":
        mapping = {name: func(name, value) for name, value in self.items()}
        return Mapper(mapping, columns=self.columns)

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[Key]:
        return iter(self._map)

    def __getitem__(self, key: Key) -> T:
        return self._map[key]

    @staticmethod
    def _get_args(
        mapping: Mapping[Key, T] | None,
        columns: Index | None,
    ) -> tuple[Mapping[Key, T], Index]:
        if mapping is None:
            mapping = {}

        if isinstance(mapping, Mapper):
            columns = mapping.columns
        elif columns is None:
            columns = pd.MultiIndex([])

        return mapping, columns

    @staticmethod
    def _validate_args(mapping: Mapping[Key, T], columns: Index) -> None:
        names: Index1L = columns.get_level_values(0)
        if set(mapping.keys()) != set(names):
            msg = "Mapping keys must match column names"
            raise ValueError(msg)
