from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pysat.examples.rc2 import RC2

try:
    from pysat.examples.rc2 import RC2Stratified
except ImportError:
    RC2Stratified = None

if TYPE_CHECKING:
    from pysat.formula import WCNF

Clause = list[int]


class Env:
    """Container exposing the shared MaxSAT solver instance."""

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
    _rc2: RC2 | None = None
    _formula: WCNF | None = None
    _formula_epoch: int | None = None
    _synced_hard: int = 0
    _synced_soft: int = 0
    _active_solver_name: str | None = None
    _state_version: int = 0

    def __init__(
        self,
        solver_name: str = "cadical195",
        TimeLimit: int = 60,
        n_threads: int = 1,
    ) -> None:
        self.solver_name = solver_name
        self.TimeLimit = TimeLimit
        self.n_threads = n_threads
        self.verbose = False

    def solve(self, w: WCNF) -> list[int]:
        if self._needs_rebuild(w):
            self._rebuild(w)
        else:
            self._sync(w)

        if self._rc2 is None:  # pragma: no cover
            msg = "Internal RC2 solver was not initialized."
            raise RuntimeError(msg)

        self._rc2.verbose = int(self.verbose)
        model = cast("list[int] | None", self._rc2.compute())
        if model is None:
            msg = "UNSAT: no counterfactual found."
            raise RuntimeError(msg)
        self._model = model
        self._cost = self._rc2.cost
        return model

    def delete(self) -> None:
        """Release any persistent RC2 state held by this wrapper."""
        if self._rc2 is not None:
            self._rc2.delete()
        self._rc2 = None
        self._formula = None
        self._formula_epoch = None
        self._synced_hard = 0
        self._synced_soft = 0
        self._active_solver_name = None
        self._model = None
        self._cost = float("inf")

    @property
    def state_token(self) -> int | None:
        """Opaque token identifying the currently cached RC2 instance."""
        if self._rc2 is None:
            return None
        return id(self._rc2)

    @property
    def synced_counts(self) -> tuple[int, int]:
        """Number of hard and soft clauses synchronized into RC2."""
        return self._synced_hard, self._synced_soft

    @property
    def state_version(self) -> int:
        """Monotonic version incremented whenever RC2 is rebuilt."""
        return self._state_version

    def _needs_rebuild(self, w: WCNF) -> bool:
        if self._rc2 is None:
            return True
        if self._formula is not w:
            return True
        if self._formula_epoch != getattr(w, "solver_epoch", 0):
            return True
        return self._active_solver_name != self.solver_name

    def _rebuild(self, w: WCNF) -> None:
        self.delete()
        self._state_version += 1
        hard = cast("list[Clause]", w.hard)
        soft = cast("list[Clause]", w.soft)
        if RC2Stratified is not None and len(soft) > 0:
            self._rc2 = RC2Stratified(
                w,
                solver=self.solver_name,
                blo="full",
                exhaust=False,
                minz=True,
                verbose=int(self.verbose),
            )
        else:
            self._rc2 = RC2(
                w,
                solver=self.solver_name,
                adapt=True,
                exhaust=False,
                minz=True,
                verbose=int(self.verbose),
            )
        self._formula = w
        self._formula_epoch = getattr(w, "solver_epoch", 0)
        self._synced_hard = len(hard)
        self._synced_soft = len(soft)
        self._active_solver_name = self.solver_name

    def _sync(self, w: WCNF) -> None:
        if self._rc2 is None:  # pragma: no cover
            return
        hard = cast("list[Clause]", w.hard)
        soft = cast("list[Clause]", w.soft)
        weights = cast("list[int]", w.wght)

        for clause in hard[self._synced_hard :]:
            self._rc2.add_clause(clause)

        new_soft = zip(
            soft[self._synced_soft :],
            weights[self._synced_soft :],
            strict=True,
        )
        for clause, weight in new_soft:
            self._rc2.add_clause(clause, weight=weight)

        self._synced_hard = len(hard)
        self._synced_soft = len(soft)

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
