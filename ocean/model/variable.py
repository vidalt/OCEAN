from typing import Protocol

from .base import BaseModel


class Var(Protocol):
    _name: str

    def __init__(self, name: str) -> None:
        self._name = name

    def build(self, model: BaseModel) -> None: ...
