import unittest
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
# Import local functions to test
from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType


class test_ClassifierCounterFactualMilp(unittest.TestCase):
    # Create a toy data set and train a random forest
    X, y = make_moons(noise=0.3, random_state=0)
    classifier = RandomForestClassifier(max_depth=5,
                                        n_estimators=10,
                                        max_features=1)
    classifier.fit(X, y)
    outputDesired = 1

    def test_emptyClassInstantiationRaisesTypeError(self):
        self.assertRaises(TypeError, ClassifierCounterFactualMilp)

    def test_ClassifierClassOnToyData(self):
        """ Simple call should not raise errors. """
        # - Test for single counterfactual input -
        # Init sample for counterfactual
        sample = self.X[0, :].reshape(1, -1)
        # Instantiation should not raise any error
        ClassifierCounterFactualMilp(
            self.classifier, sample, self.outputDesired)
        # - Test for multiple counterfactual input -
        sample = self.X[0:5, :]
        ClassifierCounterFactualMilp(
            self.classifier, sample, self.outputDesired)

    def test_defaultParametersIsBestFormulation(self):
        """
        Several implementations have been tested, the best
        performing one should use:
         constraintsType=TreeConstraintsType.LinearCombinationOfPlanes
        and
         binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda
        """
        classCfMilp = ClassifierCounterFactualMilp(
            self.classifier, self.X[0:5, :], self.outputDesired)
        self.assertEqual(classCfMilp.constraintsType,
                         TreeConstraintsType.LinearCombinationOfPlanes)
        self.assertEqual(classCfMilp.binaryDecisionVariables,
                         BinaryDecisionVariables.LeftRight_lambda)

    def test_InitializeSolutionDecisionVariables(self):
        classCfMilp = ClassifierCounterFactualMilp(
            self.classifier, self.X[0:5, :], self.outputDesired)
        classCfMilp.initSolution()
        # Test size of solution decision variables
        self.assertEqual(len(classCfMilp.x_var_sol), 2)
        self.assertEqual(len(classCfMilp.discreteFeaturesLevel_var), 0)
