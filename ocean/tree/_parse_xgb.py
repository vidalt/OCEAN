import json
from collections.abc import Iterable

import numpy as np
import xgboost as xgb

from ..abc import Mapper
from ..feature import Feature
from ..typing import NonNegativeInt, XGBTree
from ._node import Node
from ._tree import Tree


def _get_column_value(
    xgb_tree: XGBTree, node_id: NonNegativeInt, column: str
) -> str | float | int:
    mask = xgb_tree["Node"] == node_id
    return xgb_tree.loc[mask, column].values[0]  # type: ignore[no-any-return]


def _build_xgb_leaf(
    xgb_tree: XGBTree,
    *,
    node_id: NonNegativeInt,
    tree_id: NonNegativeInt,
    num_trees_per_round: NonNegativeInt,
) -> Node:
    weight = float(_get_column_value(xgb_tree, node_id, "Gain"))

    if num_trees_per_round == 1:
        value = np.array([[0.0, weight]])
    else:
        k = int(tree_id % num_trees_per_round)
        value = np.zeros((1, int(num_trees_per_round)), dtype=float)
        value[0, k] += weight
    return Node(node_id, n_samples=0, value=value)


def _parse_feature_info(
    feature_name: str, mapper: Mapper[Feature]
) -> tuple[str, str | None]:
    words = feature_name.split(" ")
    name = words[0] if words else feature_name
    code = words[1] if len(words) > 1 and words[1] else None

    if name not in mapper.names:
        msg = f"feature '{name}' not found in mapper '{mapper.names}'"
        raise KeyError(msg)

    return name, code


def _validate_feature_format(
    name: str,
    code: str | None,
    mapper: Mapper[Feature],
    node_id: NonNegativeInt,
) -> None:
    if mapper[name].is_numeric and code:
        msg = f"invalid numeric feature {name} for node {node_id}"
        raise ValueError(msg)

    if mapper[name].is_one_hot_encoded:
        if not code:
            msg = f"invalid one-hot encoded feature {name} for node {node_id}"
            raise ValueError(msg)
        if code not in mapper.codes:
            msg = f"code '{code}' not found in mapper codes '{mapper.codes}'"
            raise KeyError(msg)


def _get_child_id(
    xgb_tree: XGBTree, node_id: NonNegativeInt, column: str
) -> int:
    raw = str(_get_column_value(xgb_tree, node_id, column))
    return int(raw.rsplit("-", 1)[-1])


def _build_xgb_node(
    xgb_tree: XGBTree,
    *,
    node_id: NonNegativeInt,
    tree_id: NonNegativeInt,
    num_trees_per_round: NonNegativeInt,
    mapper: Mapper[Feature],
) -> Node:
    feature_name = str(_get_column_value(xgb_tree, node_id, "Feature"))
    name, code = _parse_feature_info(feature_name, mapper)
    _validate_feature_format(name, code, mapper, node_id)

    threshold = None
    if mapper[name].is_numeric:
        threshold = float(_get_column_value(xgb_tree, node_id, "Split")) - 1e-8
        mapper[name].add(threshold)

    left_id = _get_child_id(xgb_tree, node_id, "Yes")
    right_id = _get_child_id(xgb_tree, node_id, "No")

    node = Node(
        node_id, feature=name, threshold=threshold, code=code, n_samples=0
    )
    node.left = _parse_xgb_node(
        xgb_tree,
        node_id=left_id,
        tree_id=tree_id,
        num_trees_per_round=num_trees_per_round,
        mapper=mapper,
    )
    node.right = _parse_xgb_node(
        xgb_tree,
        node_id=right_id,
        tree_id=tree_id,
        num_trees_per_round=num_trees_per_round,
        mapper=mapper,
    )
    return node


def _parse_xgb_node(
    xgb_tree: XGBTree,
    node_id: NonNegativeInt,
    *,
    tree_id: NonNegativeInt,
    num_trees_per_round: NonNegativeInt,
    mapper: Mapper[Feature],
) -> Node:
    mask = xgb_tree["Node"] == node_id
    feature_val = str(xgb_tree.loc[mask, "Feature"].to_numpy().item())

    if feature_val == "Leaf":
        return _build_xgb_leaf(
            xgb_tree,
            node_id=node_id,
            tree_id=tree_id,
            num_trees_per_round=num_trees_per_round,
        )

    return _build_xgb_node(
        xgb_tree,
        node_id=node_id,
        tree_id=tree_id,
        num_trees_per_round=num_trees_per_round,
        mapper=mapper,
    )


def _parse_xgb_tree(
    xgb_tree: XGBTree,
    *,
    tree_id: NonNegativeInt,
    num_trees_per_round: NonNegativeInt,
    mapper: Mapper[Feature],
    base_score_prob: float = 0.0,
) -> Tree:
    root = _parse_xgb_node(
        xgb_tree,
        node_id=0,
        tree_id=tree_id,
        num_trees_per_round=num_trees_per_round,
        mapper=mapper,
    )
    tree = Tree(root=root)
    tree.logit = np.log(base_score_prob / (1 - base_score_prob))
    tree.xgboost = True
    return tree


def parse_xgb_tree(
    xgb_tree: XGBTree,
    *,
    tree_id: NonNegativeInt,
    num_trees_per_round: NonNegativeInt,
    mapper: Mapper[Feature],
    base_score_prob: float = 0.0,
) -> Tree:
    return _parse_xgb_tree(
        xgb_tree,
        tree_id=tree_id,
        num_trees_per_round=num_trees_per_round,
        mapper=mapper,
        base_score_prob=base_score_prob,
    )


def parse_xgb_trees(
    trees: Iterable[XGBTree],
    *,
    num_trees_per_round: NonNegativeInt,
    mapper: Mapper[Feature],
    base_score_prob: float = 0.0,
) -> tuple[Tree, ...]:
    return tuple(
        parse_xgb_tree(
            tree,
            tree_id=tree_id,
            num_trees_per_round=num_trees_per_round,
            mapper=mapper,
            base_score_prob=base_score_prob,
        )
        for tree_id, tree in enumerate(trees)
    )


def parse_xgb_ensemble(
    ensemble: xgb.Booster, *, mapper: Mapper[Feature]
) -> tuple[Tree, ...]:
    df = ensemble.trees_to_dataframe()
    cfg = json.loads(ensemble.save_config())
    groups = df.groupby("Tree")
    trees = tuple(
        groups.get_group(tree_id).reset_index(drop=True)
        for tree_id in groups.groups
    )

    num_rounds = ensemble.num_boosted_rounds() or 1
    num_trees_per_round = max(1, len(trees) // num_rounds)
    base_score_prob = float(cfg["learner"]["learner_model_param"]["base_score"])
    return parse_xgb_trees(
        trees,
        num_trees_per_round=num_trees_per_round,
        mapper=mapper,
        base_score_prob=base_score_prob,
    )
