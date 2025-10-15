from ...feature import Feature
from ...feature._keeper import FeatureKeeper
from ...typing import Key
from .._base import BaseModel, Var


class FeatureVar(Var, FeatureKeeper):
    X_VAR_NAME_FMT: str = "x[{name}]"

    def __init__(self, feature: Feature, name: str) -> None:
        Var.__init__(self, name=name)
        FeatureKeeper.__init__(self, feature=feature)

    def build(self, model: BaseModel) -> None:
        raise NotImplementedError

    def xget(self, code: Key | None = None) -> None:
        raise NotImplementedError

    def mget(self, key: int) -> None:
        raise NotImplementedError

    def objvarget(self) -> None:
        raise NotImplementedError

    def _add_x(self, model: BaseModel) -> None:
        raise NotImplementedError

    def _add_u(self, model: BaseModel) -> None:
        raise NotImplementedError

    def _set_mu(self, model: BaseModel, m: int) -> None:
        raise NotImplementedError

    def _add_mu(self, model: BaseModel) -> None:
        raise NotImplementedError

    def _add_one_hot_encoded(
        self,
        model: BaseModel,
        name: str,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _add_binary(model: BaseModel, name: str) -> None:
        raise NotImplementedError

    def _add_continuous(self, model: BaseModel, name: str) -> None:
        raise NotImplementedError

    def _add_discrete(self, model: BaseModel, name: str) -> None:
        raise NotImplementedError

    def _xget_one_hot_encoded(self, code: Key | None) -> None:
        raise NotImplementedError
