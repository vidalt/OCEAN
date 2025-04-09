from ortools.sat.python import cp_model as cp


class Env:
    _solver: cp.CpSolver

    def __init__(self) -> None:
        self._solver = cp.CpSolver()

    @property
    def solver(self) -> cp.CpSolver:
        return self._solver

    @solver.setter
    def solver(self, solver: cp.CpSolver) -> None:
        self._solver = solver


ENV = Env()
