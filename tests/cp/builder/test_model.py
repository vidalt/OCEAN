import numpy as np
import pandas as pd
import pytest

from ocean.abc import Mapper
from ocean.cp import (
    BaseModel,
    ConstraintProgramBuilder,
    FeatureVar,
    TreeVar,
)
from ocean.feature import Feature
from ocean.tree import Node, Tree

from ..utils import N_CLASSES, SEEDS


def create_binary_feature() -> Feature:
    return Feature(Feature.Type.BINARY)


def create_continuous_feature(
    levels: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
) -> Feature:
    if levels is None:
        levels = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    return Feature(Feature.Type.CONTINUOUS, levels=levels)


def create_discrete_feature(
    levels: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
) -> Feature:
    if levels is None:
        levels = np.array([1, 2, 3])
    return Feature(Feature.Type.DISCRETE, levels=levels)


def create_one_hot_feature(
    n_codes: int = 3,
) -> tuple[Feature, np.ndarray[tuple[int, ...], np.dtype[np.str_]]]:
    codes = np.array([f"cat_{i}" for i in range(n_codes)])
    return Feature(Feature.Type.ONE_HOT_ENCODED, codes=codes), codes


def create_simple_tree(
    seed: int,
    n_classes: int,
    thresholds: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
    codes: np.ndarray[tuple[int, ...], np.dtype[np.str_]] | None = None,
) -> Tree:
    generator = np.random.default_rng(seed)
    values = generator.uniform(0.0, 1.0, (4, n_classes))
    values /= values.sum(axis=1, keepdims=True)
    values = values.reshape(4, 1, n_classes)
    if thresholds is None:
        thresholds = generator.uniform(0.0, 1.0, 3)
    left = Node(2, value=values[0])
    right = Node(3, value=values[1])
    root1 = Node(
        1,
        threshold=thresholds[0],
        feature="x",
        left=left,
        right=right,
        code=codes[0] if codes is not None else None,
    )

    left = Node(5, value=values[2])
    right = Node(6, value=values[3])
    root2 = Node(
        4,
        threshold=thresholds[1],
        feature="x",
        left=left,
        right=right,
        code=codes[1] if codes is not None else None,
    )

    root = Node(
        0,
        threshold=thresholds[2],
        feature="x",
        left=root1,
        right=root2,
        code=codes[0] if codes is not None else None,
    )
    return Tree(root=root)


def create_multiple_features_tree(
    seed: int,
    n_classes: int,
    codes: np.ndarray[tuple[int, ...], np.dtype[np.str_]],
) -> Tree:
    generator = np.random.default_rng(seed)
    values = generator.uniform(0.0, 1.0, (8, n_classes))
    values /= values.sum(axis=1, keepdims=True)
    values = values.reshape(8, 1, n_classes)
    left = Node(3, value=values[0])
    right = Node(4, value=values[1])
    root1 = Node(2, threshold=0.5, feature="b", left=left, right=right)

    left = Node(6, value=values[2])
    right = Node(7, value=values[3])
    root2 = Node(5, threshold=0.7, feature="c", left=left, right=right)

    left = Node(9, value=values[4])
    right = Node(10, value=values[5])
    root3 = Node(8, threshold=2, feature="d", left=left, right=right)

    root4 = Node(
        1,
        threshold=0.5,
        feature="e",
        left=root1,
        right=root2,
        code=codes[0],
    )
    root = Node(
        0,
        threshold=0.5,
        feature="e",
        left=root4,
        right=root3,
        code=codes[1],
    )
    return Tree(root=root)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
class TestConstraintProgramBuilder:
    @staticmethod
    def test_build_binary_feature(
        seed: int,
        n_classes: int,
    ) -> None:
        """Test building constraints with binary feature."""
        model = BaseModel()
        builder = ConstraintProgramBuilder()
        thresholds = np.array([0.5, 0.5, 0.5])

        feature = create_binary_feature()
        tree = create_simple_tree(seed, n_classes, thresholds=thresholds)

        feature_var = FeatureVar(feature=feature, name="x")
        tree_var = TreeVar(tree=tree, name="tree")

        feature_var.build(model)  # 1 variable
        tree_var.build(model)  # 1 constraint and 4 variables

        mapper = Mapper[FeatureVar]({"x": feature_var}, columns=pd.Index(["x"]))

        builder.build(model, trees=[tree_var], mapper=mapper)  # 8 constraints

        # Verify model has constraints
        assert len(model.Proto().constraints) == 9
        assert len(model.Proto().variables) == 5

    @staticmethod
    def test_build_continuous_feature(
        seed: int,
        n_classes: int,
    ) -> None:
        """Test building constraints with continuous feature."""
        model = BaseModel()
        builder = ConstraintProgramBuilder()

        feature = create_continuous_feature()
        tree = create_simple_tree(seed, n_classes)

        feature_var = FeatureVar(feature=feature, name="x")
        tree_var = TreeVar(tree=tree, name="tree")

        feature_var.build(model)  # 3 variables and 6 constraints
        tree_var.build(model)  # 4 variables and 1 constraint

        mapper = Mapper[FeatureVar]({"x": feature_var}, columns=pd.Index(["x"]))

        builder.build(model, trees=[tree_var], mapper=mapper)  # 8 constraints

        assert len(model.Proto().constraints) == 13
        assert len(model.Proto().variables) == 7

    @staticmethod
    def test_build_discrete_feature(
        seed: int,
        n_classes: int,
    ) -> None:
        """Test building constraints with discrete feature."""
        model = BaseModel()
        builder = ConstraintProgramBuilder()

        feature = create_discrete_feature()
        tree = create_simple_tree(seed, n_classes)

        feature_var = FeatureVar(feature=feature, name="x")
        tree_var = TreeVar(tree=tree, name="tree")

        feature_var.build(model)  # 4 variables and 6 constraints
        tree_var.build(model)  # 4 variables and 1 constraint

        mapper = Mapper[FeatureVar]({"x": feature_var}, columns=pd.Index(["x"]))

        builder.build(model, trees=[tree_var], mapper=mapper)  # 8 constraints

        assert len(model.Proto().constraints) == 15
        assert len(model.Proto().variables) == 9

    @staticmethod
    def test_build_one_hot_feature(
        seed: int,
        n_classes: int,
    ) -> None:
        """Test building constraints with one-hot encoded feature."""
        model = BaseModel()
        builder = ConstraintProgramBuilder()

        feature, codes = create_one_hot_feature()
        tree = create_simple_tree(seed, n_classes, codes=codes)

        feature_var = FeatureVar(feature=feature, name="x")
        tree_var = TreeVar(tree=tree, name="tree")

        feature_var.build(model)  # 3 variables and 1 constraint
        tree_var.build(model)  # 4 variables and 1 constraint

        mapper = Mapper[FeatureVar]({"x": feature_var}, columns=pd.Index(["x"]))

        builder.build(model, trees=[tree_var], mapper=mapper)  # 8 constraints

        assert len(model.Proto().constraints) == 10
        assert len(model.Proto().variables) == 7

    @staticmethod
    def test_build_multiple_features(
        seed: int,
        n_classes: int,
    ) -> None:
        """Test building constraints with multiple features."""
        model = BaseModel()
        builder = ConstraintProgramBuilder()

        feature1 = create_binary_feature()
        feature2 = create_continuous_feature(levels=np.array([0.7]))
        feature3 = create_discrete_feature(levels=np.array([2]))
        feature4, codes = create_one_hot_feature(n_codes=2)
        tree = create_multiple_features_tree(seed, n_classes, codes=codes)

        feature_vars = [
            FeatureVar(feature=feature1, name="b"),
            FeatureVar(feature=feature2, name="c"),
            FeatureVar(feature=feature3, name="d"),
            FeatureVar(feature=feature4, name="e"),
        ]

        tree_var = TreeVar(tree=tree, name="tree")
        for feature_var in feature_vars:
            # 6 variables and 3 constraints
            feature_var.build(model)

        tree_var.build(model)  # 6 variables and 1 constraint

        mapper = Mapper[FeatureVar](
            {
                "b": feature_vars[0],
                "c": feature_vars[1],
                "d": feature_vars[2],
                "e": feature_vars[3],
            },
            columns=pd.Index(["b", "c", "d", "e"]),
        )
        builder.build(model, trees=[tree_var], mapper=mapper)  # 16 constraints
        assert len(model.Proto().constraints) == 20
        assert len(model.Proto().variables) == 13

    @staticmethod
    def test_build_multiple_trees(
        seed: int,
        n_classes: int,
    ) -> None:
        """Test building constraints with multiple trees."""
        model = BaseModel()
        builder = ConstraintProgramBuilder()

        feature1 = create_binary_feature()
        feature2 = create_continuous_feature(levels=np.array([0.7]))
        feature3 = create_discrete_feature(levels=np.array([2]))
        feature4, codes = create_one_hot_feature(n_codes=2)
        trees = [
            create_multiple_features_tree(seed + i, n_classes, codes=codes)
            for i in range(3)
        ]

        feature_vars = [
            FeatureVar(feature=feature1, name="b"),
            FeatureVar(feature=feature2, name="c"),
            FeatureVar(feature=feature3, name="d"),
            FeatureVar(feature=feature4, name="e"),
        ]
        tree_vars = [
            TreeVar(tree=tree, name=f"tree_{i}") for i, tree in enumerate(trees)
        ]
        for feature_var in feature_vars:
            # 7 variables and 3 constraints
            feature_var.build(model)
        for tree_var in tree_vars:
            # 18 variables and 3 constraint
            tree_var.build(model)

        mapper = Mapper[FeatureVar](
            {
                "b": feature_vars[0],
                "c": feature_vars[1],
                "d": feature_vars[2],
                "e": feature_vars[3],
            },
            columns=pd.Index(["b", "c", "d", "e"]),
        )
        builder.build(model, trees=tree_vars, mapper=mapper)
        # 48 constraints
        assert len(model.Proto().constraints) == 54
        assert len(model.Proto().variables) == 25
