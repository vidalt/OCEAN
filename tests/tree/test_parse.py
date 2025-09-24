import pytest
from pydantic import ValidationError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ocean.abc import Mapper
from ocean.feature import Feature
from ocean.tree import Node, parse_tree
from ocean.typing import SKLearnTree

from ..utils import generate_data


def _check_tree(
    root: Node,
    tree: SKLearnTree,
    *,
    mapper: Mapper[Feature],
) -> None:
    def _dfs(node: Node) -> None:
        if node.is_leaf:
            node_id = node.node_id
            left_id = tree.children_left[node_id]
            right_id = tree.children_right[node_id]
            assert left_id == right_id == -1
            assert (node.value == tree.value[node_id][0]).all()
        else:
            node_id = node.node_id
            left_id = tree.children_left[node_id]
            right_id = tree.children_right[node_id]
            assert node.feature is not None
            assert node.feature in mapper
            assert node.left is not None
            assert node.right is not None
            assert node.left.node_id == left_id
            assert node.right.node_id == right_id
            assert len(node.children) == 2
            feature = mapper[node.feature]
            if feature.is_numeric:
                assert node.threshold == tree.threshold[node_id]
            if feature.is_one_hot_encoded:
                assert node.code in feature.codes

            _dfs(node.left)
            _dfs(node.right)

    _dfs(root)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("max_depth", [2, 3, 4])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_parse_classifier(
    seed: int,
    n_classes: int,
    n_samples: int,
    max_depth: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    dt = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
    dt.fit(data.to_numpy(), y)
    tree = parse_tree(dt, mapper=mapper)
    assert tree is not None
    assert tree.root is not None
    assert tree.root.node_id == 0
    assert tree.n_nodes == dt.tree_.node_count
    assert tree.max_depth == dt.tree_.max_depth  # pyright: ignore[reportAttributeAccessIssue]
    assert tree.shape == (1, n_classes)
    _check_tree(tree.root, dt.tree_, mapper=mapper)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]

    with pytest.raises(ValidationError):
        tree.nodes_at(-1)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("max_depth", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_parse_regressor(seed: int, n_samples: int, max_depth: int) -> None:
    data, y, mapper = generate_data(seed, n_samples, -1)
    dt = DecisionTreeRegressor(random_state=seed, max_depth=max_depth)
    dt.fit(data.to_numpy(), y)
    tree = parse_tree(dt, mapper=mapper)
    assert tree is not None
    assert tree.root is not None
    assert tree.root.node_id == 0
    assert tree.n_nodes == dt.tree_.node_count
    assert tree.max_depth == dt.tree_.max_depth  # pyright: ignore[reportAttributeAccessIssue]
    assert tree.shape == (1, 1)
    _check_tree(tree.root, dt.tree_, mapper=mapper)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
