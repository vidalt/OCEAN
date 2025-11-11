import pytest
import xgboost as xgb
from pydantic import ValidationError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ocean.abc import Mapper
from ocean.feature import Feature
from ocean.tree import Node, parse_ensembles, parse_tree
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


def _check_xgb_tree(
    root: Node,
    booster: xgb.Booster,
    *,
    tree_id: int,
    mapper: Mapper[Feature],
) -> None:
    df = booster.trees_to_dataframe()
    tree_df = df[df["Tree"] == tree_id].reset_index(drop=True)

    def _dfs(node: Node) -> None:
        row = tree_df[tree_df["Node"] == node.node_id]
        assert not row.empty, f"node {node.node_id} not found in tree {tree_id}"

        if node.is_leaf:
            assert row["Feature"].values[0] == "Leaf"
            assert (node.value == row["Gain"].values[0]).any()
        else:
            assert node.feature is not None
            assert node.feature in mapper
            assert node.left is not None
            assert node.right is not None
            assert len(node.children) == 2

            feature = mapper[node.feature]
            feature_name = str(row["Feature"].values[0]).strip()
            if feature.is_numeric:
                assert feature_name == node.feature
                assert node.threshold == float(row["Split"].values[0] - 1e-8)
            if feature.is_one_hot_encoded:
                assert feature_name == f"{node.feature} {node.code}"
                assert node.code in feature.codes

            left_id = int(str(row["Yes"].values[0]).split("-")[-1])
            right_id = int(str(row["No"].values[0]).split("-")[-1])
            assert node.left.node_id == left_id
            assert node.right.node_id == right_id

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


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("n_estimators", [3, 5, 4])
def test_parse_xgb_classifier(
    seed: int,
    n_classes: int,
    n_samples: int,
    n_estimators: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(data, y)
    assert model is not None
    booster = model.get_booster()
    assert booster is not None
    trees = parse_ensembles(model, mapper=mapper)
    assert len(trees) == n_estimators * (1 if n_classes == 2 else n_classes)
    for i, tree in enumerate(trees):
        assert tree.root is not None
        assert tree.root.node_id == 0
        assert tree.max_depth >= 1

        _check_xgb_tree(
            tree.root,
            booster,
            tree_id=i,
            mapper=mapper,
        )
