import operator
from functools import partial

import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ocean.feature import FeatureMapper
from ocean.tree import Node, parse_tree
from ocean.tree.protocol import SKLearnTreeProtocol

from ..utils import generate_data


def _check_tree(
    root: Node,
    protocol: SKLearnTreeProtocol,
    *,
    mapper: FeatureMapper,
) -> None:
    def _dfs(node: Node) -> None:
        if node.is_leaf:
            node_id = node.node_id
            assert protocol.left[node_id] == protocol.right[node_id] == -1
            assert (node.value == protocol.value[node_id][0]).all()
        else:
            node_id = node.node_id
            left_id = protocol.left[node_id]
            right_id = protocol.right[node_id]
            assert node.feature is not None
            assert node.feature in mapper
            assert node.left is not None
            assert node.right is not None
            assert node.left.node_id == left_id
            assert node.right.node_id == right_id
            assert len(node.children) == 2
            feature = mapper[node.feature]
            if feature.is_numeric:
                assert node.threshold == protocol.threshold[node_id]
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
    f = partial(parse_tree, mapper=mapper)
    g = operator.attrgetter("tree_")
    sklearn_tree = SKLearnTreeProtocol(g(dt))
    tree = f(g(dt))
    assert tree is not None
    assert tree.root is not None
    assert tree.root.node_id == 0
    assert tree.n_nodes == sklearn_tree.n_nodes
    assert tree.max_depth == sklearn_tree.max_depth
    assert tree.shape == (1, n_classes)
    _check_tree(tree.root, sklearn_tree, mapper=mapper)

    with pytest.raises(ValueError, match=r"The depth must be non-negative."):
        tree.nodes_at(-1)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("max_depth", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_parse_regressor(seed: int, n_samples: int, max_depth: int) -> None:
    data, y, mapper = generate_data(seed, n_samples, -1)
    dt = DecisionTreeRegressor(random_state=seed, max_depth=max_depth)
    dt.fit(data.to_numpy(), y)
    f = partial(parse_tree, mapper=mapper)
    g = operator.attrgetter("tree_")
    sklearn_tree = SKLearnTreeProtocol(g(dt))
    tree = f(g(dt))
    assert tree is not None
    assert tree.root is not None
    assert tree.root.node_id == 0
    assert tree.n_nodes == sklearn_tree.n_nodes
    assert tree.max_depth == sklearn_tree.max_depth
    assert tree.shape == (1, 1)
    _check_tree(tree.root, sklearn_tree, mapper=mapper)
