import gurobipy as gp


class GarbageManager:
    type GurobiObject = gp.Var | gp.MVar | gp.Constr | gp.MConstr

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[GurobiObject]

    def __init__(self) -> None:
        self._garbage = []

    def add_garbage(self, *args: GurobiObject) -> None:
        self._garbage.extend(args)

    def cleanup(self) -> None:
        self._garbage.clear()
