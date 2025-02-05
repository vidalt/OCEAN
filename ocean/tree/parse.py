from ..feature import FeatureMapper
from ..typing import NonNegativeInt
from .node import Node
from .protocol import SKLearnTree, SKLearnTreeProtocol, TreeProtocol
from .tree import Tree


def _build_leaf(tree: TreeProtocol, node_id: NonNegativeInt) -> Node:
    value = tree.value[node_id]
    return Node(node_id, value=value)


def _build_node(
    tree: TreeProtocol,
    node_id: NonNegativeInt,
    *,
    mapper: FeatureMapper,
) -> Node:
    idx = tree.feature[node_id]
    name = mapper.names[idx]
    children = tree.left[node_id], tree.right[node_id]
    left_id, right_id = map(int, children)
    threshold, code = None, None
    if mapper[name].is_numeric:
        threshold = float(tree.threshold[node_id])
        if mapper[name].is_continuous:
            mapper[name].add(threshold)
    elif mapper[name].is_one_hot_encoded:
        code = mapper.codes[idx]

    node = Node(node_id, feature=name, threshold=threshold, code=code)
    node.left = _parse_node(tree, left_id, mapper=mapper)
    node.right = _parse_node(tree, right_id, mapper=mapper)
    return node


def _parse_node(
    tree: TreeProtocol,
    node_id: NonNegativeInt,
    *,
    mapper: FeatureMapper,
) -> Node:
    left_id, right_id = map(int, (tree.left[node_id], tree.right[node_id]))
    if left_id == right_id:
        return _build_leaf(tree, node_id)
    return _build_node(tree, node_id, mapper=mapper)


def parse_tree(sklearn_tree: SKLearnTree, *, mapper: FeatureMapper) -> Tree:
    tree = SKLearnTreeProtocol(sklearn_tree)
    root = _parse_node(tree, 0, mapper=mapper)
    return Tree(root=root)
