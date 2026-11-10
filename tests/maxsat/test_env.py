import sys
from itertools import product
from math import fsum
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pysat.formula import WCNF

from ocean.maxsat import ENV, MaxSATSolver, Model
from ocean.tree import parse_trees

from .utils import train_rf

if TYPE_CHECKING:
    from ocean.typing import Array1D

Clause = list[int]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="tests for non-windows platforms"
)


def test_solver_reuses_rc2_for_append_only_updates() -> None:
    old_solver = ENV.solver
    solver = MaxSATSolver()
    ENV.solver = solver

    try:
        clf, mapper, data = train_rf(
            43,
            4,
            3,
            100,
            2,
            return_data=True,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        model = Model(trees=trees, mapper=mapper)
        model.build()

        x = np.asarray(data.to_numpy()[0], dtype=np.float64).flatten()
        target = int(
            np.asarray(cast("Array1D", clf.predict(data)), dtype=np.int64)[0]
        )

        model.add_objective(x)
        solver.solve(model)
        first_version = solver.state_version

        model.set_majority_class(y=target)
        solver.solve(model)

        assert solver.state_version == first_version
        assert solver.synced_counts == (
            len(cast("list[Clause]", model.hard)),
            len(cast("list[Clause]", model.soft)),
        )
    finally:
        solver.delete()
        ENV.solver = old_solver


def test_solver_rebuilds_after_cleanup() -> None:
    old_solver = ENV.solver
    solver = MaxSATSolver()
    ENV.solver = solver

    try:
        clf, mapper, data = train_rf(
            43,
            4,
            3,
            100,
            2,
            return_data=True,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        model = Model(trees=trees, mapper=mapper)
        model.build()

        x = np.asarray(data.to_numpy()[0], dtype=np.float64).flatten()
        model.add_objective(x)
        solver.solve(model)
        first_version = solver.state_version

        model.cleanup()
        model.add_objective(x)
        solver.solve(model)

        assert solver.state_version > first_version
    finally:
        solver.delete()
        ENV.solver = old_solver


def test_solver_rebuilds_for_distinct_model_instances() -> None:
    old_solver = ENV.solver
    solver = MaxSATSolver()
    ENV.solver = solver

    try:
        clf, mapper, data = train_rf(
            44,
            1,
            3,
            100,
            2,
            return_data=True,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        x = np.asarray(data.to_numpy()[0], dtype=np.float64).flatten()
        target = int(
            np.asarray(cast("Array1D", clf.predict(data)), dtype=np.int64)[0]
        )

        first = Model(trees=trees, mapper=mapper)
        first.build()
        first.add_objective(x)
        first.set_majority_class(y=target)
        solver.solve(first)
        first_version = solver.state_version

        second = Model(trees=trees, mapper=mapper)
        second.build()
        second.add_objective(x)
        second.set_majority_class(y=target)
        solver.solve(second)

        assert solver.state_version > first_version
    finally:
        solver.delete()
        ENV.solver = old_solver


def test_fractional_weights_preserve_feasibility() -> None:
    formula = WCNF()
    # A unique feasible assignment that violates every soft clause. The
    # stratified solver incorrectly hardens a clause after accumulating 1.1,
    # because the remaining budget 1.3 - 1.1 is slightly smaller than 0.2.
    for literal in [-1, -2, -3, 4]:
        formula.append([literal])
    for literal, weight in [(1, 0.3), (2, 0.2), (3, 0.3), (-4, 0.5)]:
        formula.append([literal], weight=weight)

    solver = MaxSATSolver()
    try:
        model = solver.solve(formula)
        assert set(model) == {-1, -2, -3, 4}
        assert solver.cost == pytest.approx(1.3)
    finally:
        solver.delete()


@pytest.mark.parametrize("seed", range(20))
def test_fractional_objective_matches_exhaustive_search(seed: int) -> None:
    generator = np.random.default_rng(seed)
    hard = [[1, 2], [-2, 3], [-3, 4]]
    soft = [[1], [-1], [2], [-2], [3], [-3], [4], [-4], [-1, -4]]
    weights = [
        float(generator.choice([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])) for _ in soft
    ]
    formula = WCNF()
    for clause in hard:
        formula.append(clause)
    for clause, weight in zip(soft, weights, strict=True):
        formula.append(clause, weight=weight)

    def cost(assignment: set[int]) -> float:
        return fsum(
            weight
            for clause, weight in zip(soft, weights, strict=True)
            if not assignment.intersection(clause)
        )

    feasible = [
        {
            variable if value else -variable
            for variable, value in enumerate(values, start=1)
        }
        for values in product([False, True], repeat=4)
    ]
    feasible = [
        assignment
        for assignment in feasible
        if all(assignment.intersection(clause) for clause in hard)
    ]
    optimum = min(map(cost, feasible))
    solver = MaxSATSolver()
    try:
        assignment = set(solver.solve(formula))
        assert assignment in feasible
        assert cost(assignment) == pytest.approx(optimum)
        assert solver.cost == pytest.approx(optimum)
    finally:
        solver.delete()
