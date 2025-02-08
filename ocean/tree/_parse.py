import operator
from collections.abc import Iterable
from functools import partial
from itertools import chain

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..abc import Mapper
from ..feature import Feature
from ..typing import NonNegativeInt, ParsableEnsemble
from ._node import Node
from ._protocol import SKLearnTree, SKLearnTreeProtocol, TreeProtocol
from ._tree import Tree

type DecisionTree = DecisionTreeClassifier | DecisionTreeRegressor


def _build_leaf(tree: TreeProtocol, node_id: NonNegativeInt) -> Node:
    value = tree.value[node_id, :]
    n_samples = int(tree.n_samples[node_id])
    return Node(node_id, n_samples=n_samples, value=value)


def _build_node(
    tree: TreeProtocol,
    node_id: NonNegativeInt,
    *,
    mapper: Mapper[Feature],
) -> Node:
    idx = int(tree.feature[node_id])
    name = mapper.names[idx]
    children = map(int, (tree.left[node_id], tree.right[node_id]))
    left_id, right_id = children
    threshold, code = None, None
    n_samples = int(tree.n_samples[node_id])

    if mapper[name].is_numeric:
        threshold = float(tree.threshold[node_id])
        if mapper[name].is_continuous:
            mapper[name].add(threshold)
    elif mapper[name].is_one_hot_encoded:
        code = mapper.codes[idx]

    node = Node(
        node_id,
        feature=name,
        threshold=threshold,
        code=code,
        n_samples=n_samples,
    )
    node.left = _parse_node(tree, left_id, mapper=mapper)
    node.right = _parse_node(tree, right_id, mapper=mapper)
    return node


def _parse_node(
    tree: TreeProtocol,
    node_id: NonNegativeInt,
    *,
    mapper: Mapper[Feature],
) -> Node:
    left_id, right_id = map(int, (tree.left[node_id], tree.right[node_id]))
    if left_id == right_id:
        return _build_leaf(tree, node_id)
    return _build_node(tree, node_id, mapper=mapper)


def _parse_tree(sklearn_tree: SKLearnTree, *, mapper: Mapper[Feature]) -> Tree:
    tree = SKLearnTreeProtocol(sklearn_tree)
    root = _parse_node(tree, 0, mapper=mapper)
    return Tree(root=root)


def parse_tree(tree: DecisionTree, *, mapper: Mapper[Feature]) -> Tree:
    getter = operator.attrgetter("tree_")
    return _parse_tree(getter(tree), mapper=mapper)


def parse_trees(
    trees: Iterable[DecisionTree],
    *,
    mapper: Mapper[Feature],
) -> tuple[Tree, ...]:
    parser = partial(parse_tree, mapper=mapper)
    return tuple(map(parser, trees))


def parse_ensembles(
    *ensembles: ParsableEnsemble,
    mapper: Mapper[Feature],
) -> tuple[Tree, ...]:
    parser = partial(parse_trees, mapper=mapper)
    return tuple(chain.from_iterable(map(parser, ensembles)))
