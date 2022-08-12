import unittest
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
# Import local functions to test
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.RandomForestCounterfactual import RandomForestCounterfactualMilp
from src.TreeMilpManager import TreeInMilpManager


class test_TreeMilpManager(unittest.TestCase):
    # Create a toy data set and train a random forest
    X, y = make_moons(noise=0.3, random_state=0)
    randomForest = RandomForestClassifier(max_depth=5,
                                          n_estimators=10,
                                          max_features=1)
    randomForest.fit(X, y)
    outputDesired = 1
    # Instantiate a ClassifierCounterFactual
    classCfMilp = ClassifierCounterFactualMilp(randomForest,
                                               X[0:5, :],
                                               outputDesired)

    def test_emptyClassInstantiationRaisesTypeError(self):
        self.assertRaises(TypeError, TreeInMilpManager)

    def test_ClassOnToyData(self):
        """ Simple call should not raise errors. """
        rfCfMilp = RandomForestCounterfactualMilp()
        # Initialize x_var_sol
        self.classCfMilp.initSolution()
        # Create a TreeMilpManager
        TreeInMilpManager(self.randomForest.estimators_[0].tree_,
                          self.classCfMilp.model, self.classCfMilp.x_var_sol,
                          self.outputDesired, self.classCfMilp.featuresType,
                          rfCfMilp.constraintsType,
                          rfCfMilp.binaryDecisionVariables)

    def test_addTreeVariablesAndConstraintsToMilp(self):
        # Initialize x_var_sol
        self.classCfMilp.initSolution()
        # -- Default arguments: LinComb + LeftRight --
        treeMng = TreeInMilpManager(
            self.randomForest.estimators_[0].tree_,
            self.classCfMilp.model, self.classCfMilp.x_var_sol,
            self.outputDesired, self.classCfMilp.featuresType,
            TreeConstraintsType.LinearCombinationOfPlanes,
            BinaryDecisionVariables.LeftRight_lambda)
        treeMng.addTreeVariablesAndConstraintsToMilp()
        # -- Var1: Extended + LeftRight --
        treeMng = TreeInMilpManager(
            self.randomForest.estimators_[0].tree_,
            self.classCfMilp.model, self.classCfMilp.x_var_sol,
            self.outputDesired, self.classCfMilp.featuresType,
            TreeConstraintsType.ExtendedFormulation,
            BinaryDecisionVariables.LeftRight_lambda)
        treeMng.addTreeVariablesAndConstraintsToMilp()
        # -- Var2: LinComb + PathFlow --
        treeMng = TreeInMilpManager(
            self.randomForest.estimators_[0].tree_,
            self.classCfMilp.model, self.classCfMilp.x_var_sol,
            self.outputDesired, self.classCfMilp.featuresType,
            TreeConstraintsType.LinearCombinationOfPlanes,
            BinaryDecisionVariables.PathFlow_y)
        treeMng.addTreeVariablesAndConstraintsToMilp()
        # -- Var3: Extended + PathFlow --
        treeMng = TreeInMilpManager(
            self.randomForest.estimators_[0].tree_,
            self.classCfMilp.model, self.classCfMilp.x_var_sol,
            self.outputDesired, self.classCfMilp.featuresType,
            TreeConstraintsType.ExtendedFormulation,
            BinaryDecisionVariables.PathFlow_y)
        treeMng.addTreeVariablesAndConstraintsToMilp()

    def test_FeaturesTypeIsARequiredArgument(self):
        # Initialize x_var_sol
        self.classCfMilp.initSolution()
        self.assertRaises(TypeError, TreeInMilpManager,
                          self.randomForest.estimators_[0].tree_,
                          self.classCfMilp.model, self.classCfMilp.x_var_sol,
                          self.outputDesired)

    def test_defaultParametersIsBestFormulation(self):
        """
        Several implementations have been tested, the best
        performing one should use:
         constraintsType=TreeConstraintsType.LinearCombinationOfPlanes
        and
         binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda
        """
        # Initialize x_var_sol
        self.classCfMilp.initSolution()
        # - Default parameters from class TreeInMilpManager -
        # Create a TreeMilpManager
        treeMng = TreeInMilpManager(self.randomForest.estimators_[0].tree_,
                                    self.classCfMilp.model,
                                    self.classCfMilp.x_var_sol,
                                    self.outputDesired,
                                    self.classCfMilp.featuresType)
        # Test value of default parameters
        self.assertEqual(treeMng.constraintsType,
                         TreeConstraintsType.LinearCombinationOfPlanes)
        self.assertEqual(treeMng.binaryDecisionVariables,
                         BinaryDecisionVariables.LeftRight_lambda)
        # - Default parameters from class RandomForestCounterfactualMilp -
        rfCfMilp = RandomForestCounterfactualMilp()
        # Create a TreeMilpManager
        treeMng = TreeInMilpManager(self.randomForest.estimators_[0].tree_,
                                    self.classCfMilp.model,
                                    self.classCfMilp.x_var_sol,
                                    self.outputDesired,
                                    self.classCfMilp.featuresType,
                                    rfCfMilp.constraintsType,
                                    rfCfMilp.binaryDecisionVariables)
        # Test value of default parameters
        self.assertEqual(treeMng.constraintsType,
                         TreeConstraintsType.LinearCombinationOfPlanes)
        self.assertEqual(treeMng.binaryDecisionVariables,
                         BinaryDecisionVariables.LeftRight_lambda)
