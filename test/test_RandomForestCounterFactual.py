import unittest
import numpy as np
from pathlib import Path
from gurobipy import GurobiError
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
# Import local functions
from src.dataProcessing import DatasetReader
from src.RandomForestCounterFactual import RandomForestCounterFactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType


class test_RandomForestCounterFactualMilpOnToyData(unittest.TestCase):
    # Create a toy data set and train a random forest
    X, y = make_moons(noise=0.01, random_state=0)
    rfClassifier = RandomForestClassifier(max_depth=4)
    rfClassifier.fit(X, y)
    initSample = np.array([0, 1]).reshape(1, -1)
    outputDesired = 1

    def test_emptyClassInstantiationRaisesTypeError(self):
        self.assertRaises(TypeError, RandomForestCounterFactualMilp)

    def test_simpleClassInstance(self):
        RandomForestCounterFactualMilp(
            self.rfClassifier, self.initSample, self.outputDesired)

    def test_simpleCounterfactualOnToyData(self):
        rfCfMilp = RandomForestCounterFactualMilp(
            self.rfClassifier, self.initSample, self.outputDesired,
            constraintsType=TreeConstraintsType.ExtendedFormulation,)
        rfCfMilp.buildModel()
        try:
            modelStatus = rfCfMilp.solveModel()
            gurobiLicenseAvailable = True
        except GurobiError:
            gurobiLicenseAvailable = False
            print("Warning: Gurobi license not found:"
                  " cannot run integration test that solves MILP.")
        if gurobiLicenseAvailable:
            self.assertTrue(modelStatus)
            # Check counterfactual is valid
            counterfactualExplanation = rfCfMilp.x_sol
            self.assertEqual(self.rfClassifier.predict(counterfactualExplanation),
                             self.outputDesired)

    def test_defaultParametersGiveBestFormulation(self):
        """
        Several implementations have been tested, the best
        performing one should use:
         constraintsType=TreeConstraintsType.LinearCombinationOfPlanes
        and
         binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda
        """
        rfCfMilp = RandomForestCounterFactualMilp(
            self.rfClassifier, self.initSample, self.outputDesired)
        self.assertEqual(rfCfMilp.constraintsType,
                         TreeConstraintsType.LinearCombinationOfPlanes)
        self.assertEqual(rfCfMilp.binaryDecisionVariables,
                         BinaryDecisionVariables.LeftRight_lambda)


class test_RandomForestCounterFactualMilpOnRealData(unittest.TestCase):
    # Test parameters
    dataset = 'datasets/German-Credit.csv'
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset

    def test_simpleCounterfactualOnRealData(self):
        reader = DatasetReader(self.datasetFile)
        # Train random forest
        clf = RandomForestClassifier(max_depth=4,
                                     random_state=0,
                                     n_estimators=50)
        clf.fit(reader.X_train.values, reader.y_train.values)
        # Train isolation forest
        ilf = IsolationForest(random_state=0,
                              max_samples=32,
                              n_estimators=50,
                              contamination=0.1)
        ilf.fit(reader.XwithGoodPoint.values)
        # Test RandomForestCounterFactualMilp
        outputDesired = 1
        xInit = reader.X_train.values[0, :].reshape(1, -1)
        rfCfMilp = RandomForestCounterFactualMilp(
            clf, xInit, outputDesired,
            isolationForest=ilf,
            featuresType=reader.featuresType,
            featuresPossibleValues=reader.featuresPossibleValues,
            featuresActionnability=reader.featuresActionnability)
        rfCfMilp.buildModel()
        try:
            rfCfMilp.solveModel()
            gurobiLicenseAvailable = True
        except GurobiError:
            gurobiLicenseAvailable = False
            print("Warning: Gurobi license not found:"
                  " cannot run integration test that solves MILP.")
        if gurobiLicenseAvailable:
            # Check counterfactual is valid
            counterfactualExplanation = rfCfMilp.x_sol
            self.assertEqual(clf.predict(counterfactualExplanation),
                             outputDesired)
