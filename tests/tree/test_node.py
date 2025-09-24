import numpy as np
import pytest

from ocean.tree import Node


def test_leaf() -> None:
    leaf = Node(0, value=np.array([1.0, 0.0, 0.0]))
    assert leaf.is_leaf
    assert np.all(leaf.value == [1.0, 0.0, 0.0])
    assert leaf.node_id == 0
    with pytest.raises(AttributeError):
        leaf.left  # noqa: B018
    with pytest.raises(AttributeError):
        leaf.right  # noqa: B018
    with pytest.raises(AttributeError):
        leaf.feature  # noqa: B018
    with pytest.raises(AttributeError):
        leaf.threshold  # noqa: B018
    with pytest.raises(AttributeError):
        leaf.code  # noqa: B018

    leaf = Node(1)
    with pytest.raises(AttributeError):
        leaf.value  # noqa: B018


def test_node() -> None:
    left = Node(0, value=np.array([1.0, 0.0, 0.0]))
    right = Node(1, value=np.array([0.0, 1.0, 0.0]))
    node = Node(2, feature="test", left=left, right=right)

    assert not node.is_leaf
    with pytest.raises(AttributeError):
        node.value  # noqa: B018

    with pytest.raises(AttributeError):
        node.threshold  # noqa: B018
    with pytest.raises(AttributeError):
        node.code  # noqa: B018

    assert node.left.node_id == left.node_id
    assert node.right.node_id == right.node_id
    assert node.feature == "test"

    new_left = Node(3, value=np.array([0.0, 0.0, 1.0]))
    node.left = new_left

    assert node.left.node_id == new_left.node_id
    assert node.right.node_id == right.node_id
    assert len(node.children) == 2
    assert left.is_root

    new_right = Node(4, value=np.array([0.0, 0.0, 1.0]))
    node.right = new_right

    assert node.left.node_id == new_left.node_id
    assert node.right.node_id == new_right.node_id
    assert len(node.children) == 2
    assert right.is_root
