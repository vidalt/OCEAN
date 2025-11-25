from abc import ABC
from typing import Any, Protocol

from pysat.formula import WCNF, IDPool


class BaseModel(ABC, WCNF):
    vpool: IDPool = IDPool()

    def __init__(self) -> None:
        WCNF.__init__(self)

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

    def add_hard(self, lits: list[int]) -> None:
        """Add a hard clause (must be satisfied)."""
        # weight=None => hard clause in WCNF
        self.append(lits)

    def add_soft(self, lits: list[int], weight: int = 1) -> None:
        """Add a soft clause with a given weight."""
        self.append(lits, weight=weight)

    def add_exactly_one(self, lits: list[int]) -> None:
        """Add constraint that exactly one path is selected."""
        self.add_hard(lits)  # at least one
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                self.add_hard([-lits[i], -lits[j]])  # at most one


class Var(Protocol):
    _name: str

    def __init__(self, name: str) -> None:
        self._name = name

    def build(self, model: BaseModel) -> None: ...
