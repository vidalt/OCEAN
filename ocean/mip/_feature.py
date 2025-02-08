import gurobipy as gp

from ..abc import Mapper
from ..feature import Feature
from ..typing import (
    Key,
    PositiveInt,
)
from ._base import BaseModel
from ._solution import Solution
from ._variable import FeatureVar


class FeatureBuilder:
    FEATURE_VAR_FMT: str = "feature[{key}]"

    # Mapper for the features in the model.
    # this also is the solution of the model.
    _mapper: Solution

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
    def mapper(self) -> Solution:
        return self._mapper

    @property
    def solution(self) -> Solution:
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

        self._mapper = Solution(mapper.apply(create))
