from collections.abc import Callable, Iterator, Mapping
from typing import Concatenate, Literal, Protocol, overload

import pandas as pd

from ..typing import Index, Index1L, Key, NonNegativeInt, Number, PositiveInt


class Value(Protocol):
    @property
    def is_one_hot_encoded(self) -> bool: ...

    @property
    def codes(self) -> tuple[Key, ...]: ...


type Getter[K, V] = Callable[Concatenate[K, ...], V]
type Args[V] = tuple[Mapping[Key, V], Index]


class Indexer[K, V]:
    _getters: tuple[Getter[K, V], ...]
    _memo: dict[tuple[K, ...], V]

    def __init__(self, *getters: Getter[K, V]) -> None:
        self._getters = getters
        self._memo = {}

    def _get(self, *keys: K) -> V:
        return self._getters[len(keys) - 1](*keys)

    def get(self, *keys: K) -> V:
        if keys not in self._memo:
            self._memo[*keys] = self._get(*keys)
        return self._memo[*keys]


class Mapper[V: Value](Mapping[Key, V]):
    NAME_LEVEL: Literal[0] = 0
    CODE_LEVEL: Literal[1] = 1

    _columns: Index
    _mapping: dict[Key, V]
    _indexer: Indexer[Key, NonNegativeInt] | None = None

    _names: tuple[Key, ...] | None = None
    _codes: tuple[Key, ...] | None = None

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self,
        mapping: Mapping[Key, V],
        *,
        columns: Index,
        validate: bool = True,
    ) -> None: ...

    @overload
    def __init__(self, mapping: "Mapper[V]") -> None: ...

    def __init__(
        self,
        mapping: "Mapping[Key, V] | None" = None,
        *,
        columns: Index | None = None,
        validate: bool = True,
    ) -> None:
        mapping, columns = self._get_args(mapping, columns)
        self._validate_args(mapping, columns, validate=validate)
        self._columns = columns
        self._mapping = dict(mapping)

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
            names: Index1L = self.columns.get_level_values(self.NAME_LEVEL)
            self._names = tuple(names)
        return self._names

    @property
    def codes(self) -> tuple[Key, ...]:
        if self._codes is None:
            if not self.is_multi_level:
                msg = "No one-hot encoded features found"
                raise ValueError(msg)
            codes: Index1L = self.columns.get_level_values(self.CODE_LEVEL)
            self._codes = tuple(codes)
        return self._codes

    @property
    def idx(self) -> Indexer[Key, NonNegativeInt]:
        if self._indexer is None:
            self._indexer = self._add_indexer()
        return self._indexer

    def reduce[S](self, reducer: Callable[[V], S]) -> Mapping[Key, S]:
        return {name: reducer(value) for name, value in self.items()}

    def apply[U: Value](self, func: Callable[[Key, V], U]) -> "Mapper[U]":
        mapping = {name: func(name, value) for name, value in self.items()}
        return Mapper(mapping, columns=self.columns, validate=False)

    def __len__(self) -> NonNegativeInt:
        return len(self._mapping)

    def __iter__(self) -> Iterator[Key]:
        return iter(self._mapping)

    def __getitem__(self, key: Key) -> V:
        return self._mapping[key]

    @staticmethod
    def _get_args(
        mapping: Mapping[Key, V] | None,
        columns: Index | None,
    ) -> Args[V]:
        if mapping is None:
            mapping = {}

        if isinstance(mapping, Mapper):
            columns = mapping.columns
        elif columns is None:
            columns = pd.Index([])

        return mapping, columns

    def _validate_args(
        self,
        mapping: Mapping[Key, V],
        columns: Index,
        *,
        validate: bool = True,
    ) -> None:
        if not validate:
            return

        if isinstance(mapping, Mapper):
            return

        names: Index1L = columns.get_level_values(self.NAME_LEVEL)
        if set(mapping.keys()) != set(names):
            msg = "Mapping keys must match column names"
            raise ValueError(msg)

        if columns.nlevels <= 1:
            return

        codes: Index1L = columns.get_level_values(self.CODE_LEVEL)
        for name, value in mapping.items():
            if not value.is_one_hot_encoded:
                continue
            matched = names == name
            if set(value.codes) != set(codes[matched]):
                msg = "Mapping codes must match column codes"
                raise ValueError(msg)

    def _add_indexer(self) -> Indexer[Key, NonNegativeInt]:
        n = self.columns.nlevels
        return Indexer(*map(self._add_getter, range(1, n + 1)))

    def _add_getter(self, n: PositiveInt) -> Getter[Key, NonNegativeInt]:
        match n:
            case 1:
                return self._get_with_name
            case 2:
                return self._get_with_code
            case _:
                msg = f"Unsupported number of keys: {n}"
                raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Mapper({self._mapping!r}, columns={self.columns!r})"

    def _get_with_name(self, name: Key) -> NonNegativeInt:
        if name not in self.names:
            msg = f"Name {name} not found"
            raise KeyError(msg)
        return self.names.index(name)

    def _get_with_code(self, name: Key, code: Key) -> NonNegativeInt:
        if name not in self.names:
            msg = f"Name {name} not found in names"
            raise KeyError(msg)
        codes = self[name].codes
        if code not in codes:
            msg = f"Code {code} not found in codes associatedwith {name}"
            raise KeyError(msg)
        indices = [j for j, n in enumerate(self.names) if n == name]
        codes = tuple(self.codes[j] for j in indices)
        i = codes.index(code)
        return indices[i]

    @staticmethod
    def _repr(mapping: Mapping[Key, Key | Number]) -> str:
        length = max(len(str(k)) for k in mapping)
        lines = [
            f"{str(k).ljust(length + 1)} : {v}" for k, v in mapping.items()
        ]
        return "\n".join(lines)
