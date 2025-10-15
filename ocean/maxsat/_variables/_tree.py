from collections.abc import Iterator, Mapping

from pydantic import validate_call

from ...tree._keeper import TreeKeeper, TreeLike
from ...typing import NonNegativeInt
from .._base import BaseModel, Var


class TreeVar(Var, TreeKeeper, Mapping[NonNegativeInt, object]):
    PATH_VAR_NAME_FMT: str = "{name}_path"

    def __init__(
        self,
        tree: TreeLike,
        name: str,
    ) -> None:
        Var.__init__(self, name=name)
        TreeKeeper.__init__(self, tree=tree)

    def build(self, model: BaseModel) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[NonNegativeInt]:
        return iter(range(self.n_nodes))

    @validate_call
    def __getitem__(self, node_id: NonNegativeInt) -> None:
        raise NotImplementedError

    def _add_path(
        self,
        model: BaseModel,
        name: str,
    ) -> None:
        raise NotImplementedError
