from abc import ABC

import gurobipy as gp


class BaseModel(ABC, gp.Model):
    def __init__(self, name: str = "", env: gp.Env | None = None) -> None:
        gp.Model.__init__(self, name=name, env=env)

    def __setattr__(self, name: str, value: int) -> None:
        object.__setattr__(self, name, value)
