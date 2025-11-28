from collections.abc import Iterator, Mapping

from pydantic import validate_call

from ...tree._keeper import TreeKeeper, TreeLike
from ...typing import NonNegativeInt
from .._base import BaseModel, Var


class TreeVar(Var, TreeKeeper, Mapping[NonNegativeInt, object]):
    PATH_VAR_NAME_FMT: str = "{name}_path"

    _path: Mapping[NonNegativeInt, int]

    def __init__(
        self,
        tree: TreeLike,
        name: str,
    ) -> None:
        Var.__init__(self, name=name)
        TreeKeeper.__init__(self, tree=tree)

    def build(self, model: BaseModel) -> None:
        name = self.PATH_VAR_NAME_FMT.format(name=self._name)
        self._path = self._add_path(model=model, name=name)
        model.add_exactly_one(list(self._path.values()))

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[NonNegativeInt]:
        return iter(range(self.n_nodes))

    @validate_call
    def __getitem__(self, node_id: NonNegativeInt) -> int:
        return self._path[node_id]

    def _add_path(
        self,
        model: BaseModel,
        name: str,
    ) -> Mapping[NonNegativeInt, int]:
        return {
            leaf.node_id: model.add_var(name=f"{name}[{leaf.node_id}]")
            for leaf in self.leaves
        }
