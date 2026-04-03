import sys
from typing import cast

import numpy as np
import pytest

from ocean.maxsat import ENV, MaxSATSolver, Model
from ocean.tree import parse_trees

from .utils import train_rf

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
        target = int(np.asarray(clf.predict(data), dtype=np.int64)[0])

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
        target = int(np.asarray(clf.predict(data), dtype=np.int64)[0])

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
