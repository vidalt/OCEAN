import gurobipy as gp

from ...abc import Mapper
from ...feature import Feature
from ...typing import (
    Key,
    PositiveInt,
)
from .._base import BaseModel
from .._explanation import Explanation
from .._variables import FeatureVar


class FeatureManager:
    FEATURE_VAR_FMT: str = "feature[{key}]"

    _mapper: Explanation

    def __init__(self, mapper: Mapper[Feature]) -> None:
        self._set_mapper(mapper)

    def build_features(self, model: BaseModel) -> None:
        model.build_vars(*self.mapper.values())

    @property
    def n_columns(self) -> PositiveInt:
        return self.mapper.n_columns

    @property
    def n_features(self) -> PositiveInt:
        return len(self.mapper)

    @property
    def mapper(self) -> Explanation:
        return self._mapper

    @property
    def explanation(self) -> Explanation:
        return self.mapper

    def vget(self, i: int) -> gp.Var:
        return self.mapper.vget(i)

    def _set_mapper(self, mapper: Mapper[Feature]) -> None:
        def create(key: Key, feature: Feature) -> FeatureVar:
            name = self.FEATURE_VAR_FMT.format(key=key)
            return FeatureVar(feature, name=name)

        if len(mapper) == 0:
            msg = "At least one feature is required."
            raise ValueError(msg)

        self._mapper = Explanation(mapper.apply(create))
