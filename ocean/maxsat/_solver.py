from __future__ import annotations

try:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
except Exception as e:
    msg = "PySAT not available."
    msg += " Install it using pip or use another method (CP, MIP)."
    raise ImportError(msg) from e


class MaxSATSolver:
    """Thin RC2 wrapper to keep a stable interface."""

    def __init__(self, solver_name: str = "glucose3") -> None:
        self.solver_name = solver_name

    @staticmethod
    def new_wcnf() -> WCNF:
        return WCNF()

    @staticmethod
    def add_hard(w: WCNF, clause: list[int]) -> None:
        w.append(clause, weight=-1)

    @staticmethod
    def add_soft(w: WCNF, clause: list[int], weight: int) -> None:
        w.append(clause, weight=weight)

    def solve(self, w: WCNF) -> list[int]:
        with RC2(w, solver=self.solver_name, adapt=True) as rc2:
            model = rc2.compute()
            if model is None:
                msg = "UNSAT: no counterfactual found."
                raise RuntimeError(msg)
            return model
