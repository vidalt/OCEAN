from collections.abc import Sequence
from typing import overload

from ..abc import Mapper
from ..feature import Feature
from ..tree import Tree, parse_trees
from ..typing import BaseEnsemble, NonNegativeInt


class Ensemble(Sequence[Tree]):
    _trees: tuple[Tree, ...]

    def __init__(
        self,
        ensemble: BaseEnsemble,
        *,
        mapper: Mapper[Feature],
    ) -> None:
        self._trees = tuple(self._parse_trees(ensemble, mapper=mapper))

    @overload
    def __getitem__(self, i: int) -> Tree: ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[Tree]: ...

    def __getitem__(self, i: int | slice) -> Tree | Sequence[Tree]:
        return self._trees[i]

    def __len__(self) -> NonNegativeInt:
        return len(self._trees)

    @staticmethod
    def _parse_trees(
        ensemble: BaseEnsemble,
        *,
        mapper: Mapper[Feature],
    ) -> tuple[Tree, ...]:
        return parse_trees(ensemble, mapper=mapper)
