from abc import ABC
from typing import Any, Protocol

from ortools.sat.python import cp_model as cp


class BaseModel(ABC, cp.CpModel):
    def __init__(self) -> None:
        cp.CpModel.__init__(self)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        object.__setattr__(self, name, value)

    def build_vars(self, *variables: "Var") -> None:
        for variable in variables:
            variable.build(model=self)


class Var(Protocol):
    _name: str

    def __init__(self, name: str) -> None:
        self._name = name

    def build(self, model: BaseModel) -> None: ...
