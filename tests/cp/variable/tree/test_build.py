import numpy as np
import pytest
from ortools.sat.python import cp_model as cp

from ocean.cp import BaseModel, TreeVar
from ocean.tree import Node, Tree


@pytest.mark.parametrize("seed", range(10))
def test_init(seed: int) -> None:
    # Create a simple tree
    generator = np.random.default_rng(seed)
    model = BaseModel()
    values = generator.uniform(0.0, 1.0, (2, 2))
    values /= values.sum(axis=1, keepdims=True)

    left = Node(1, value=values[0])
    right = Node(2, value=values[1])
    root = Node(0, threshold=0.5, feature="x", left=left, right=right)
    tree = Tree(root=root)

    # Test basic initialization
    treevar = TreeVar(tree=tree, name="tree")
    treevar.build(model)
    assert treevar.root.node_id == 0
    assert treevar.n_nodes == 3
    assert len(treevar.leaves) == 2
    assert isinstance(treevar[2], cp.IntVar)
    assert isinstance(treevar[1], cp.IntVar)
    assert len(treevar) == 3
