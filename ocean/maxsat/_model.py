from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ._base import BaseModel
from ._builder.model import ModelBuilder, ModelBuilderFactory
from ._managers import FeatureManager, GarbageManager, TreeManager

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..abc import Mapper
    from ..feature import Feature
    from ..tree import Tree
    from ..typing import NonNegativeArray1D, NonNegativeInt


@dataclass
class Model(BaseModel, FeatureManager, TreeManager, GarbageManager):
    # Model builder for the ensemble.
    _builder: ModelBuilder
    DEFAULT_EPSILON: int = 1

    class Type(Enum):
        MAXSAT = "MAXSAT"

    def __init__(
        self,
        trees: Iterable[Tree],
        mapper: Mapper[Feature],
        *,
        weights: NonNegativeArray1D | None = None,
        max_samples: NonNegativeInt = 0,
        epsilon: int = DEFAULT_EPSILON,
        model_type: Type = Type.MAXSAT,
    ) -> None:
        BaseModel.__init__(self)
        TreeManager.__init__(
            self,
            trees=trees,
            weights=weights,
        )
        FeatureManager.__init__(self, mapper=mapper)
        GarbageManager.__init__(self)

        self._set_weights(weights=weights)
        self._max_samples = max_samples
        self._epsilon = epsilon
        self._set_builder(model_type=model_type)

    def build(self) -> None:
        self.build_features(self)
        self.build_trees(self)
        self._builder.build(self, trees=self.trees, mapper=self.mapper)

    def _set_builder(self, model_type: Type) -> None:
        match model_type:
            case Model.Type.MAXSAT:
                self._builder = ModelBuilderFactory.MAXSAT()
