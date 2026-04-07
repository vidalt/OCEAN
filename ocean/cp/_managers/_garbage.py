from ortools.sat.python import cp_model as cp


class GarbageManager:
    """Track temporary CP variables and constraints tied to one query."""

    type GarbageObject = cp.IntVar | cp.Constraint

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

    def remove_garbage(self) -> None:
        """Clear all registered temporary CP objects from the model state."""
        for garbage in self._garbage:
            garbage.Proto().Clear()
        self._garbage.clear()
