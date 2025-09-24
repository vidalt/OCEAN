import pytest
from sklearn.tree import DecisionTreeClassifier

from ocean.mip import BaseModel, TreeVar
from ocean.tree import Node, parse_tree

from ..utils import ENV, generate_data


def _check_node(tree: TreeVar, node: Node) -> Node:
    if node.is_leaf:
        return node
    left, right = node.left, node.right
    assert (tree[left.node_id].X + tree[right.node_id].X) == 1.0
    if tree[left.node_id].X == 1.0:
        return _check_node(tree, left)
    return _check_node(tree, right)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("max_depth", [2, 3, 4])
def test_variable(
    seed: int,
    n_classes: int,
    n_samples: int,
    max_depth: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    dt = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
    dt.fit(data.to_numpy(), y)
    tree = parse_tree(dt, mapper=mapper)
    tree_var = TreeVar(tree=tree, name="tree")

    assert tree_var.root.node_id == 0
    assert tree_var.n_nodes == tree.n_nodes
    assert len(tree_var) == tree.n_nodes
    assert tuple(iter(tree_var)) == tuple(iter(range(tree.n_nodes)))

    model = BaseModel(env=ENV)

    tree_var.build(model=model)
    assert tree_var.shape == (1, n_classes)

    model.optimize()

    node = _check_node(tree_var, tree_var.root)
    assert node.is_leaf
    assert (node.value == tree_var.value.getValue()).all()

    tree_var = TreeVar(
        tree=tree,
        name="tree",
        flow_type=TreeVar.FlowType.BINARY,
    )
    tree_var.build(model=model)

    model.optimize()

    node = _check_node(tree_var, tree_var.root)
    assert node.is_leaf
    assert (node.value == tree_var.value.getValue()).all()
