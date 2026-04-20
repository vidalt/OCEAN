from abc import ABC
from typing import Any, Protocol

from pysat.formula import WCNF, IDPool

try:
    from pysat.card import CardEnc
    from pysat.card import EncType as CardEncType
except ImportError:
    CardEnc = None
    CardEncType = None


class BaseModel(ABC, WCNF):
    """Base weighted CNF model used by the MaxSAT backend."""

    vpool: IDPool
    _solver_epoch: int

    def __init__(self) -> None:
        WCNF.__init__(self)
        self.vpool = IDPool()  # Create new pool for each instance
        self._solver_epoch = 0

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        object.__setattr__(self, name, value)

    def build_vars(self, *variables: "Var") -> None:
        for variable in variables:
            variable.build(model=self)

    def add_var(self, name: str) -> int:
        if name in self.vpool.obj2id:  # var has been already created
            msg = f"Variable with name '{name}' already exists."
            raise ValueError(msg)
        return self.vpool.id(f"{name}")  # type: ignore[no-any-return]

    def get_var(self, name: str) -> int:
        if name not in self.vpool.obj2id:  # var has not been created
            msg = f"Variable with name '{name}' does not exist."
            raise ValueError(msg)
        return self.vpool.obj2id[name]  # type: ignore[no-any-return]

    def add_hard(self, lits: list[int], return_id: bool = False) -> int:  # noqa: FBT001, FBT002
        """
        Add a hard clause (must be satisfied).

        Returns:
            The clause ID if return_id is True, otherwise -1.

        """
        # weight=None => hard clause in WCNF
        self.append(lits)
        if return_id:
            return len(self.hard) - 1  # pyright: ignore[reportUnknownArgumentType]
        return -1

    def add_soft(self, lits: list[int], weight: float = 1.0) -> None:
        """Add a soft clause with a given weight."""
        self.append(lits, weight=weight)

    def add_exactly_one(self, lits: list[int]) -> None:
        """Add constraint that exactly one path is selected."""
        if len(lits) == 0:
            self.add_hard([])
            return
        if len(lits) == 1:
            self.add_hard([lits[0]])
            return
        if CardEnc is None or CardEncType is None:
            self.add_hard(lits)  # at least one
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self.add_hard([-lits[i], -lits[j]])  # at most one
            return

        card = CardEnc.equals(
            lits=lits,
            bound=1,
            vpool=self.vpool,
            encoding=CardEncType.seqcounter,
        )
        for clause in card.clauses:
            self.add_hard(clause)  # pyright: ignore[reportUnknownArgumentType]

    def _clean_soft(self) -> None:
        """Reset the model to only contain hard constraints."""
        self.soft.clear()
        self.wght.clear()
        self.topw = 1

    @property
    def solver_epoch(self) -> int:
        return self._solver_epoch

    def invalidate_solver_state(self) -> None:
        """Mark the formula as having changed non-monotonically."""
        self._solver_epoch += 1


class Var(Protocol):
    _name: str

    def __init__(self, name: str) -> None:
        self._name = name

    def build(self, model: BaseModel) -> None: ...
