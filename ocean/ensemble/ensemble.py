import operator
from collections.abc import Iterable, Sequence
from functools import partial
from typing import overload

from ..feature import FeatureMapper
from ..tree import Tree, parse_tree
from ..typing import BaseEnsemble


class Ensemble(Sequence[Tree]):
    _trees: tuple[Tree, ...]

    def __init__(
        self,
        ensemble: BaseEnsemble,
        *,
        mapper: FeatureMapper,
    ) -> None:
        self._trees = tuple(self._parse_trees(ensemble, mapper=mapper))

    @overload
    def __getitem__(self, i: int) -> Tree: ...

    @overload
    def __getitem__(
        self,
        i: "slice[int | None, int | None, int | None]",
    ) -> Sequence[Tree]: ...

    def __getitem__(self, i: int | slice) -> Tree | Sequence[Tree]:
        return self._trees[i]

    def __len__(self) -> int:
        return len(self._trees)

    @staticmethod
    def _parse_trees(
        ensemble: BaseEnsemble,
        *,
        mapper: FeatureMapper,
    ) -> Iterable[Tree]:
        f = partial(parse_tree, mapper=mapper)
        g = operator.attrgetter("tree_")
        return tuple(map(f, map(g, ensemble)))
