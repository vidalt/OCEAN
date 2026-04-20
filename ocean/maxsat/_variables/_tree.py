from collections.abc import Iterator, Mapping

from pydantic import validate_call

from ...tree._keeper import TreeKeeper, TreeLike
from ...typing import NonNegativeInt
from .._base import BaseModel, Var


class TreeVar(Var, TreeKeeper, Mapping[NonNegativeInt, object]):
    """MaxSAT literals encoding the active leaf of one parsed tree."""

    PATH_VAR_NAME_FMT: str = "{name}_path"
    CLASS_VAR_NAME_FMT: str = "{name}_class"

    _path: Mapping[NonNegativeInt, int]
    _class: Mapping[NonNegativeInt, int]
    _hard_voting: bool = False

    def __init__(
        self,
        tree: TreeLike,
        name: str,
    ) -> None:
        Var.__init__(self, name=name)
        TreeKeeper.__init__(self, tree=tree)

    def build(self, model: BaseModel) -> None:
        self._hard_voting = bool(getattr(model, "_hard_voting", False))
        if self._hard_voting:
            name = self.CLASS_VAR_NAME_FMT.format(name=self._name)
            self._class = self._add_class(model=model, name=name)
            model.add_exactly_one(list(self._class.values()))
        else:
            name = self.PATH_VAR_NAME_FMT.format(name=self._name)
            self._path = self._add_path(model=model, name=name)
            model.add_exactly_one(list(self._path.values()))

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[NonNegativeInt]:
        return iter(range(self.n_nodes))

    @validate_call
    def __getitem__(self, node_id: NonNegativeInt) -> int:
        if self._hard_voting:
            msg = "Leaf variables are not available in hard-voting mode."
            raise ValueError(msg)
        return self._path[node_id]

    @validate_call
    def cget(self, class_id: NonNegativeInt) -> int:
        if not self._hard_voting:
            msg = "Class variables are only available in hard-voting mode."
            raise ValueError(msg)
        return self._class[class_id]

    def _add_path(
        self,
        model: BaseModel,
        name: str,
    ) -> Mapping[NonNegativeInt, int]:
        return {
            leaf.node_id: model.add_var(name=f"{name}[{leaf.node_id}]")
            for leaf in self.leaves
        }

    def _add_class(
        self,
        model: BaseModel,
        name: str,
    ) -> Mapping[NonNegativeInt, int]:
        n_classes = self.shape[-1]
        return {
            class_id: model.add_var(name=f"{name}[{class_id}]")
            for class_id in range(n_classes)
        }
