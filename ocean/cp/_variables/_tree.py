from collections.abc import Iterator, Mapping

from ortools.sat.python import cp_model as cp
from pydantic import validate_call

from ...tree._keeper import TreeKeeper, TreeLike
from ...typing import NonNegativeInt
from .._base import BaseModel, Var


class TreeVar(Var, TreeKeeper, Mapping[NonNegativeInt, cp.IntVar]):
    PATH_VAR_NAME_FMT: str = "{name}_path"

    _path: dict[NonNegativeInt, cp.IntVar]

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
        model.Add(cp.LinearExpr.Sum(*self._path.values()) == 1)

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[NonNegativeInt]:
        return iter(range(self.n_nodes))

    @validate_call
    def __getitem__(self, node_id: NonNegativeInt) -> cp.IntVar:
        return self._path[node_id]

    def _add_path(
        self,
        model: BaseModel,
        name: str,
    ) -> dict[NonNegativeInt, cp.IntVar]:
        return {
            leaf.node_id: model.NewBoolVar(name=f"{name}[{leaf.node_id}]")
            for leaf in self.leaves
        }
