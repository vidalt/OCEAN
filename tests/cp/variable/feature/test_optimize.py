import numpy as np
import pytest
from ortools.sat.python import cp_model as cp

from ocean.cp import ENV, BaseModel, FeatureVar
from ocean.feature import Feature

from ....feature.utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_binary(seed: int) -> None:
    generator = np.random.default_rng(seed)
    model = BaseModel()
    feature = Feature(Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    solver = ENV.solver
    v = var.xget()

    val = generator.choice(range(2, 10))
    objective = val + v
    model.Minimize(objective)
    solver.Solve(model)
    value = solver.Value(v)
    assert value == 0.0

    val = generator.choice(range(5, 10))
    objective = val - v
    model.Minimize(objective)
    solver.Solve(model)
    value = solver.Value(v)
    assert value == 1.0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_discrete(
    seed: int,
    n_levels: int,
    lower: int,
    upper: int,
) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    levels = np.sort(levels)
    model = BaseModel()
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    v = var.xget()
    val = generator.choice(levels)
    j = int(np.searchsorted(levels, val, side="left"))  # pyright: ignore[reportUnknownArgumentType]
    u = model.NewIntVar(0, n_levels - 1, f"u_{j}")
    model.AddAbsEquality(u, v - j)
    objective = u
    model.Minimize(objective)
    solver = ENV.solver
    solver.Solve(model)
    value = levels[solver.Value(v)]
    assert np.isclose(value, val), (
        f"Expected {val}, but got {value}, with levels {levels} and j={j}"
        f" and u={solver.Value(u)} and v={solver.Value(v)}"
        f" and solver status {solver.StatusName()}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_continuous(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = np.sort(generator.uniform(lower, upper, n_levels))
    lb, ub = np.min(levels), np.max(levels)
    # add lb - 1 and ub + 1 to the levels
    levels = np.concatenate(([lb - 1], levels, [ub + 1]))

    def solve_and_assert(val: float, expected: float) -> None:
        model = BaseModel()
        solver = ENV.solver
        feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
        var = FeatureVar(feature=feature, name="x")
        var.build(model)
        scale = int(1e6)
        v = var.xget()
        variables = [var.mget(i) for i in range(len(levels) - 1)]
        intervals_cost = np.zeros(len(levels) - 1, dtype=int)
        for i in range(len(intervals_cost)):
            if levels[i] < val <= levels[i + 1]:
                continue
            if levels[i] > val:
                intervals_cost[i] = int(abs(val - levels[i]) * scale)
            elif levels[i + 1] < val:
                intervals_cost[i] = int(abs(val - levels[i + 1]) * scale)

        objective = cp.LinearExpr.WeightedSum(variables, intervals_cost)
        model.Minimize(objective)
        solver.Solve(model)
        value = levels[solver.Value(v) + 1]
        mu_values = [solver.Value(mu) for mu in variables]
        assert sum(mu_values) == 1.0, (
            f"Solver status {solver.StatusName()}, "
            f"mu values {mu_values}, "
            f"model variables {model.Proto().variables}, "
        )
        assert np.isclose(value, expected), (
            f"Expected {expected}, but got {value}, with levels {levels} "
            f"and val={val} and solver status {solver.StatusName()}"
            f" v = {solver.Value(v)} and mu_values {mu_values}"
        )

    # Test within bounds
    val = generator.uniform(lb, ub)
    j = int(np.searchsorted(levels, val, side="left"))
    solve_and_assert(val, levels[j])

    # Test below lower bound
    val = generator.uniform(lb - 0.5, lb - 0.1)
    solve_and_assert(val, lb)

    # Test above upper bound
    val = generator.uniform(ub + 0.1, ub + 0.5)
    solve_and_assert(val, ub + 1.0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_codes", N_CODES)
def test_one_hot_encoded(seed: int, n_codes: int) -> None:
    generator = np.random.default_rng(seed)
    codes = generator.choice(CHOICES, n_codes)
    model = BaseModel()
    solver = ENV.solver
    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=codes)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    code = generator.choice(codes)
    v = var.xget(code)
    val = generator.uniform(0.0, 0.4)
    objective = v - val
    model.Minimize(objective)
    solver.Solve(model)
    value = solver.Value(v)
    assert np.isclose(value, 0)
    assert (
        sum(1 for code in var.codes if solver.Value(var.xget(code)) == 1.0)
        == 1.0
    )

    model = BaseModel()
    var.build(model)
    val = generator.uniform(0.6, 1.0)
    objective = val - v
    model.Minimize(objective)
    solver.Solve(model)
    value = solver.Value(v)
    assert np.isclose(value, 1.0)
    assert (
        sum(1 for code in var.codes if solver.Value(var.xget(code)) == 1.0)
        == 1.0
    )
