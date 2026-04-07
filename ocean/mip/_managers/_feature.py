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
    r"""
    Manage MIP feature variables for the counterfactual point :math:`x`.

    This manager owns the backend-specific feature variables that represent the
    processed coordinates of the counterfactual explanation. The query
    :math:`\hat{x}` only appears later when the model objective is built.
    """

    FEATURE_VAR_FMT: str = "feature[{key}]"

    _mapper: Explanation

    def __init__(self, mapper: Mapper[Feature]) -> None:
        """Wrap the parsed feature mapper in MIP-specific feature variables."""
        self._set_mapper(mapper)

    def build_features(self, model: BaseModel) -> None:
        r"""Create backend variables for the coordinates of :math:`x`."""
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
        """
        Return the solver variable for processed coordinate ``i`` of ``x``.

        Returns
        -------
        gp.Var
            Solver variable representing the requested coordinate.

        """
        return self.mapper.vget(i)

    def _set_mapper(self, mapper: Mapper[Feature]) -> None:
        """
        Convert parsed feature metadata into backend-specific feature variables.

        Raises
        ------
        ValueError
            If ``mapper`` is empty.

        """
        def create(key: Key, feature: Feature) -> FeatureVar:
            name = self.FEATURE_VAR_FMT.format(key=key)
            return FeatureVar(feature, name=name)

        if len(mapper) == 0:
            msg = "At least one feature is required."
            raise ValueError(msg)

        self._mapper = Explanation(mapper.apply(create))
