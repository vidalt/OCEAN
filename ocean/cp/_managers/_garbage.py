from ortools.sat.python import cp_model as cp


class GarbageManager:
    type GarbageObject = cp.IntVar | cp.Constraint

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[GarbageObject]

    def __init__(self) -> None:
        self._garbage = []

    def add_garbage(self, *args: GarbageObject) -> None:
        self._garbage.extend(args)

    def remove_garbage(self) -> None:
        for garbage in self._garbage:
            garbage.Proto().Clear()
        self._garbage.clear()
