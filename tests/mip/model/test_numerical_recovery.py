"""Regression coverage for Gurobi's numerical recovery path."""

import gurobipy as gp
import numpy as np

from ocean.mip import Model
from ocean.tree import parse_trees

from ...utils import ENV
from ..utils import train_rf, validate_sklearn_paths, validate_sklearn_pred


def test_numerical_recovery_preserves_settings_and_callback() -> None:
    clf, mapper, data = train_rf(45, 8, 3, 100, 4, return_data=True)
    query = np.random.default_rng(45).integers(len(data))
    x = data.to_numpy()[query].astype(np.float64)
    callback_limits: list[float] = []

    def callback(model: gp.Model, where: int) -> None:
        if where == gp.GRB.Callback.MIPSOL:
            callback_limits.append(model.Params.TimeLimit)

    model = Model(trees=parse_trees(clf, mapper=mapper), mapper=mapper, env=ENV)
    with model:
        model.build()
        time_limit = 60.0
        model.Params.TimeLimit = time_limit
        original_focus = model.Params.NumericFocus
        # The final target triggers SUBOPTIMAL on Gurobi 14/Linux before
        # recovery. Earlier solves exercise reuse of the same model.
        for target in range(4):
            model.set_majority_class(y=target)
            model.add_objective(x=x, norm=1)
            model.optimize(callback)
            assert model.Status == gp.GRB.OPTIMAL
            assert model.MaxVio <= model.Params.FeasibilityTol
            assert model.Params.NumericFocus == original_focus
            assert model.Params.TimeLimit == time_limit
            validate_sklearn_paths(clf, model.explanation, model.estimators)
            validate_sklearn_pred(clf, model.explanation, target, model)
            model.cleanup()

    assert callback_limits
    assert all(0 < limit <= time_limit for limit in callback_limits)
