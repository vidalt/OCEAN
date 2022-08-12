import unittest
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
# Import local functions to test
from src.RandomForestCounterfactual import RandomForestCounterfactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType


class test_RandomForestCounterfactualMilp(unittest.TestCase):
    # Create a toy data set and train a random forest
    X, y = make_moons(noise=0.3, random_state=0)
    classifier = RandomForestClassifier(max_depth=5,
                                        n_estimators=10,
                                        max_features=1)
    classifier.fit(X, y)
    outputDesired = 1

    def test_defaultParametersIsBestFormulation(self):
        """
        Several implementations have been tested, the best
        performing one should use:
         constraintsType=TreeConstraintsType.LinearCombinationOfPlanes
        and
         binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda
        """
        rfCfMilp = RandomForestCounterfactualMilp()
        self.assertEqual(rfCfMilp.constraintsType,
                         TreeConstraintsType.LinearCombinationOfPlanes)
        self.assertEqual(rfCfMilp.binaryDecisionVariables,
                         BinaryDecisionVariables.LeftRight_lambda)
