import gurobipy as gp

from .._base import BaseModel


class GarbageManager:
    """Track temporary MIP variables and constraints tied to one query."""

    type GarbageObject = gp.Var | gp.MVar | gp.Constr | gp.MConstr

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[GarbageObject]

    def __init__(self) -> None:
        """Initialize the container used for query-specific garbage objects."""
        self._garbage = []

    def add_garbage(self, *args: GarbageObject) -> None:
        """Register temporary objects created for the current query."""
        self._garbage.extend(args)

    def remove_garbage(self, model: BaseModel) -> None:
        """Delete all registered temporary objects from the backend model."""
        model.remove(self._garbage)
        self._garbage.clear()
