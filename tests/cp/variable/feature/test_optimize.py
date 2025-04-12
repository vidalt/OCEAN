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
    j = int(np.searchsorted(levels, val, side="left"))  # type: ignore[reportUnknownArgumentType]
    u = model.NewIntVar(0, len(levels) - 1, f"u_{j}")
    model.AddAbsEquality(u, v - j)
    objective = u
    model.Minimize(objective)
    solver = ENV.solver
    solver.Solve(model)
    value = levels[solver.Value(v)]
    assert np.isclose(value, val), (
        f"Expected {val}, but got {value}, with levels {levels} and j {j}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_continuous(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = np.sort(generator.uniform(lower, upper, n_levels))
    lb, ub = np.min(levels), np.max(levels)

    def solve_and_assert(val: float, expected: float) -> None:
        model = BaseModel()
        solver = ENV.solver
        feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
        var = FeatureVar(feature=feature, name="x")
        var.build(model)
        v = var.xget()
        variables = [var.mget(i) for i in range(len(levels))]
        intervals_cost = [
            0 if i > 0 and levels[i - 1] < val <= L else abs(val - L)
            for i, L in enumerate(levels)
        ]
        objective = cp.LinearExpr.WeightedSum(variables, intervals_cost)
        model.Minimize(objective)
        solver.Solve(model)
        value = levels[solver.Value(v)]
        mu_values = [solver.Value(mu) for mu in variables]
        assert sum(mu_values) == 1.0
        assert np.isclose(value, expected)

    # Test within bounds
    val = generator.uniform(lb, ub)
    j = int(np.searchsorted(levels, val, side="right"))
    solve_and_assert(val, levels[j])

    # Test below lower bound
    val = generator.uniform(lb - 1.0, lb - 0.5)
    solve_and_assert(val, lb)

    # Test above upper bound
    val = generator.uniform(ub + 0.5, ub + 1.0)
    solve_and_assert(val, ub)


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
