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
    initSample = np.array([1, 0]).reshape(1, -1)
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
    # Get data reader and train forests
    reader = DatasetReader(datasetFile)
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
    yVals = (reader.y_train.values == 0)
    hasZeroClass = reader.X_train.values[yVals]
    xInit = hasZeroClass[0:1, :]

    def __solveModelAndCheckCfIsValid(self, rfCfMilp):
        try:
            rfCfMilp.solveModel()
            self.gurobiLicenseAvailable = True
        except GurobiError:
            self.gurobiLicenseAvailable = False
            print("Warning: Gurobi license not found:"
                  " cannot run integration test that solves MILP.")
            counterfactualExplanation = False
        if self.gurobiLicenseAvailable:
            # Check counterfactual is valid
            counterfactualExplanation = rfCfMilp.x_sol
            self.assertEqual(self.clf.predict(counterfactualExplanation),
                             self.outputDesired)
        return counterfactualExplanation

    def test_simpleCounterfactualOnRealData(self):
        # l0-norm
        rfCfMilp = RandomForestCounterFactualMilp(
            self.clf, self.xInit, self.outputDesired,
            objectiveNorm=0,
            isolationForest=self.ilf,
            featuresType=self.reader.featuresType,
            featuresPossibleValues=self.reader.featuresPossibleValues,
            featuresActionnability=self.reader.featuresActionnability)
        rfCfMilp.buildModel()
        _ = self.__solveModelAndCheckCfIsValid(rfCfMilp)
        # l1-norm
        rfCfMilp = RandomForestCounterFactualMilp(
            self.clf, self.xInit, self.outputDesired,
            objectiveNorm=1,
            isolationForest=self.ilf,
            featuresType=self.reader.featuresType,
            featuresPossibleValues=self.reader.featuresPossibleValues,
            featuresActionnability=self.reader.featuresActionnability)
        rfCfMilp.buildModel()
        _ = self.__solveModelAndCheckCfIsValid(rfCfMilp)
        # l2-norm
        rfCfMilp = RandomForestCounterFactualMilp(
            self.clf, self.xInit, self.outputDesired,
            objectiveNorm=2,
            isolationForest=self.ilf,
            featuresType=self.reader.featuresType,
            featuresPossibleValues=self.reader.featuresPossibleValues,
            featuresActionnability=self.reader.featuresActionnability)
        rfCfMilp.buildModel()
        _ = self.__solveModelAndCheckCfIsValid(rfCfMilp)

    def __getCounterfactualWithModelFormulation(self,
                                                constraintsType,
                                                binaryVars):
        rfCfMilp = RandomForestCounterFactualMilp(
            self.clf, self.xInit, self.outputDesired,
            isolationForest=self.ilf,
            featuresType=self.reader.featuresType,
            featuresPossibleValues=self.reader.featuresPossibleValues,
            featuresActionnability=self.reader.featuresActionnability,
            constraintsType=constraintsType,
            binaryDecisionVariables=binaryVars)
        rfCfMilp.buildModel()
        counterfactualExplanation = self.__solveModelAndCheckCfIsValid(
            rfCfMilp)
        return counterfactualExplanation

    def __assertCounterfactualsEqual(self, input1, input2):
        nbCfs = len(input1)
        for cf in range(nbCfs):
            for x in range(len(input1[cf])):
                self.assertEqual(round(input1[cf][x], 6),
                                 round(input2[cf][x], 6))

    def test_allModelFormulationsHaveSameSolution(self):
        # -- Default arguments: LinComb + LeftRight --
        cf1 = self.__getCounterfactualWithModelFormulation(
            TreeConstraintsType.LinearCombinationOfPlanes,
            BinaryDecisionVariables.LeftRight_lambda)
        # -- Var1: Extended + LeftRight --
        cf2 = self.__getCounterfactualWithModelFormulation(
             TreeConstraintsType.ExtendedFormulation,
             BinaryDecisionVariables.LeftRight_lambda)
        # -- Var2: LinComb + PathFlow --
        # ! The combination of TreeConstraintsType.LinearCombinationOfPlanes
        # and BinaryDecisionVariables.PathFlow_y is not supported !
        # cf3 = self.__getCounterfactualWithModelFormulation(
        #     TreeConstraintsType.LinearCombinationOfPlanes,
        #     BinaryDecisionVariables.PathFlow_y)
        # -- Var3: Extended + PathFlow --
        cf4 = self.__getCounterfactualWithModelFormulation(
            TreeConstraintsType.ExtendedFormulation,
            BinaryDecisionVariables.PathFlow_y)
        if self.gurobiLicenseAvailable:
            # Check all solutions are equal
            self.__assertCounterfactualsEqual(cf1, cf2)
            # self.__assertCounterfactualsEqual(cf1, cf3)
            self.__assertCounterfactualsEqual(cf1, cf4)
