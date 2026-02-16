from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pysat.examples.rc2 import RC2

if TYPE_CHECKING:
    from pysat.formula import WCNF


class Env:
    _solver: MaxSATSolver

    def __init__(self) -> None:
        self._solver = MaxSATSolver()

    @property
    def solver(self) -> MaxSATSolver:
        return self._solver

    @solver.setter
    def solver(self, solver: MaxSATSolver) -> None:
        self._solver = solver


class MaxSATSolver:
    """Thin RC2 wrapper to keep a stable interface."""

    _model: list[int] | None = None
    _cost: float = float("inf")

    def __init__(
        self,
        solver_name: str = "cadical195",
        TimeLimit: int = 60,
        n_threads: int = 1,
    ) -> None:
        self.solver_name = solver_name
        self.TimeLimit = TimeLimit
        self.n_threads = n_threads

    def solve(self, w: WCNF) -> list[int]:
        with RC2(
            w,
            solver=self.solver_name,
            adapt=True,
            exhaust=False,
            minz=True,
        ) as rc2:
            model = cast("list[int] | None", rc2.compute())
            if model is None:
                msg = "UNSAT: no counterfactual found."
                raise RuntimeError(msg)
            self._model = model
            self._cost = rc2.cost
            return model

    def model(self, v: int) -> float:
        """
        Return 1.0 if variable v is true in the model, 0.0 otherwise.

        Args:
            v: Variable to check in the model.

        Returns:
            1.0 if variable v is true in the model, 0.0 otherwise.

        Raises:
            ValueError: If no model has been found solve() must be called first.

        """
        if self._model is None:
            msg = "No model found, please run 'solve' first."
            raise ValueError(msg)
        # The model is a list of signed literals.
        # Variable v is true if v is in the model, false if -v is in the model.
        if v in self._model:
            return 1.0
        return 0.0

    @property
    def cost(self) -> float:
        return self._cost


ENV = Env()
