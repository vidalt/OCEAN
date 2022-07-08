import unittest
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
# Import local functions to test
from src.DecisionTreeCounterFactual import DecisionTreeCounterFactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType


class test_DecisionTreeCounterFactualMilp(unittest.TestCase):
    # Create a toy data set and train a random forest
    X, y = make_moons(noise=0.01, random_state=0)
    dtClassifier = DecisionTreeClassifier(max_depth=5)
    dtClassifier.fit(X, y)
    initSample = np.array([0, 1]).reshape(1, -1)
    outputDesired = 1

    def test_emptyClassInstantiationRaisesTypeError(self):
        self.assertRaises(TypeError, DecisionTreeCounterFactualMilp)

    def test_simpleClassInstance(self):
        DecisionTreeCounterFactualMilp(
            self.dtClassifier, self.initSample, self.outputDesired)

    def test_simpleCounterfactualOnToyData(self):
        dtCfMilp = DecisionTreeCounterFactualMilp(
            self.dtClassifier, self.initSample, self.outputDesired)
        dtCfMilp.buildModel()
        dtCfMilp.solveModel()
        # Check counterfactual is valid
        counterfactualExplanation = dtCfMilp.x_sol
        self.assertEqual(self.dtClassifier.predict(counterfactualExplanation),
                         self.outputDesired)

    def test_defaultParameters(self):
        # Note that he LinearCombinationOfPlanes is not implemented for a DT
        dtCfMilp = DecisionTreeCounterFactualMilp(
            self.dtClassifier, self.initSample, self.outputDesired)
        self.assertEqual(dtCfMilp.constraintsType,
                         TreeConstraintsType.ExtendedFormulation)
        self.assertEqual(dtCfMilp.binaryDecisionVariables,
                         BinaryDecisionVariables.LeftRight_lambda)
