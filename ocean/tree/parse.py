from ..feature import FeatureMapper
from .protocol import SKLearnTree, SKLearnTreeProtocol, TreeProtocol
from .tree import Node, Tree


def _build_leaf(protocol: TreeProtocol, node_id: int) -> Node:
    value = protocol.value[node_id]
    return Node(node_id, value=value)


def _build_node(
    protocol: TreeProtocol,
    node_id: int,
    *,
    mapper: FeatureMapper,
) -> Node:
    idx = int(protocol.feature[node_id])
    name = mapper.names[idx]
    left_id, right_id = (
        int(protocol.left[node_id]),
        int(protocol.right[node_id]),
    )
    threshold, code = None, None
    if mapper[name].is_numeric:
        threshold = float(protocol.threshold[node_id])
        if mapper[name].is_continuous:
            mapper[name].add(threshold)
    elif mapper[name].is_one_hot_encoded:
        code = mapper.codes[idx]

    node = Node(node_id, feature=name, threshold=threshold, code=code)
    node.left = _parse_node(protocol, left_id, mapper=mapper)
    node.right = _parse_node(protocol, right_id, mapper=mapper)
    return node


def _parse_node(
    protocol: TreeProtocol,
    node_id: int,
    *,
    mapper: FeatureMapper,
) -> Node:
    left_id, right_id = (
        int(protocol.left[node_id]),
        int(protocol.right[node_id]),
    )
    if left_id == right_id:
        return _build_leaf(protocol, node_id)
    return _build_node(protocol, node_id, mapper=mapper)


def parse_tree(sklearn_tree: SKLearnTree, *, mapper: FeatureMapper) -> Tree:
    protocol = SKLearnTreeProtocol(sklearn_tree)
    root = _parse_node(protocol, 0, mapper=mapper)
    return Tree(root=root)
