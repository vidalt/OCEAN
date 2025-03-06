from abc import ABC
from typing import Any, Protocol

import gurobipy as gp


class BaseModel(ABC, gp.Model):
    def __init__(self, name: str = "", env: gp.Env | None = None) -> None:
        gp.Model.__init__(self, name=name, env=env)

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
