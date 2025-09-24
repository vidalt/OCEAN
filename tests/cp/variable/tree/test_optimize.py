import numpy as np
import pytest
from ortools.sat.python import cp_model as cp

from ocean.cp import ENV, BaseModel, TreeVar
from ocean.tree import Node, Tree


@pytest.mark.parametrize("seed", range(10))
def test_optimize(seed: int) -> None:
    generator = np.random.default_rng(seed)
    values = generator.uniform(0.0, 1.0, (2, 2))
    values /= values.sum(axis=1, keepdims=True)
    left = Node(1, value=values[0])
    right = Node(2, value=values[1])
    root = Node(0, threshold=0.5, feature="x", left=left, right=right)
    tree = Tree(root=root)
    treevar = TreeVar(tree=tree, name="tree")
    solver = ENV.solver

    def solve_and_assert(
        objective: cp.LinearExpr,
        obj_expected: float,
        leaf1_expected: float,
        leaf2_expected: float,
    ) -> None:
        # Test optimization
        model.Minimize(objective)
        solver.Solve(model)
        v1 = solver.Value(treevar[1])
        v2 = solver.Value(treevar[2])
        assert solver.ObjectiveValue() == obj_expected
        assert v1 == leaf1_expected
        assert v2 == leaf2_expected
        assert v1 + v2 == 1.0

    model = BaseModel()
    treevar.build(model)
    solve_and_assert(treevar[1] - treevar[2], -1.0, 0.0, 1.0)

    model = BaseModel()
    treevar.build(model)
    solve_and_assert(treevar[2] - treevar[1], -1.0, 1.0, 0.0)

    model = BaseModel()
    treevar.build(model)
    solve_and_assert(treevar[1], 0.0, 0.0, 1.0)

    model = BaseModel()
    treevar.build(model)
    solve_and_assert(treevar[2], 0.0, 1.0, 0.0)
