from ortools.sat.python import cp_model as cp


class GarbageManager:
    """Track temporary CP variables and constraints tied to one query."""

    type GarbageObject = cp.IntVar | cp.Constraint

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to deactivate the removable ones when the model is cleared.
    _garbage: list[GarbageObject]

    def __init__(self) -> None:
        """Initialize the container used for query-specific garbage objects."""
        self._garbage = []

    def add_garbage(self, *args: GarbageObject) -> None:
        """Register temporary objects created for the current query."""
        self._garbage.extend(args)

    def remove_garbage(self) -> None:
        """Clear registered temporary constraints from the model state."""
        for garbage in self._garbage:
            if isinstance(garbage, cp.IntVar):
                # CpModel does not support removing variables. Clearing an
                # IntVar proto leaves a variable slot with an empty domain,
                # which makes the next CP-SAT solve report MODEL_INVALID.
                continue
            garbage.Proto().Clear()
        self._garbage.clear()
