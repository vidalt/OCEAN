from enum import Enum
eps = 1e-5


class TreeConstraintsType(Enum):
    BigM = 1
    ExtendedFormulation = 2
    LinearCombinationOfPlanes = 3


class BinaryDecisionVariables(Enum):
    LeftRight_lambda = 1
    PathFlow_y = 2


class FeatureType(Enum):
    Numeric = 1
    Binary = 2
    Discrete = 3
    Categorical = 4
    CategoricalNonOneHot = 5


class FeatureActionnability(Enum):
    Free = 1
    Fixed = 2
    Increasing = 3
    Predict = 4


def getFeatureType(name):
    if name == 'N':
        return FeatureType.Numeric
    elif name == 'B':
        return FeatureType.Binary
    elif name == 'D':
        return FeatureType.Discrete
    elif name == 'C':
        return FeatureType.Categorical
    else:
        print("Unknown feature type", name)
        return None


def isFeatureTypeScalable(featureType):
    if featureType == FeatureType.Categorical:
        return False
    return True


def getFeatureActionnability(name):
    if name == "FREE":
        return FeatureActionnability.Free
    elif name == "FIXED":
        return FeatureActionnability.Fixed
    elif name == "INC":
        return FeatureActionnability.Increasing
    elif name == "PREDICT":
        return FeatureActionnability.Predict
    elif name == "PROBLEM":
        print("Problematic feature treated as free")
        return FeatureActionnability.Free
    else:
        print("Unknown actionnability", name)
        return None


class ObjectiveType(Enum):
    L0 = 0
    L1 = 1
    L2 = 2
