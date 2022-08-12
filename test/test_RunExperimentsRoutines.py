import unittest
import os
import pandas as pd
from gurobipy import GurobiError
from pathlib import Path
# Import local functions to test
from src.experiment_routines import train_model_and_solve_counterfactual
from src.build_counterfactual_set import build_counterfactual_file


class test_train_model_and_solve_counterfactual(unittest.TestCase):
    # Test parameters
    dataset = 'datasets/German-Credit.csv'
    desiredOutcome = 1
    nbCf = 2
    # Manage path to files
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    pathToCounterfactual = datasetFile.parent / 'counterfactuals'
    datasetName = datasetFile.name
    oneHotDataset = "OneHot_" + datasetName
    counterfactualOneHotDatasetFile = pathToCounterfactual / oneHotDataset
    resultFile = THIS_DIR.parent / 'NumericalResults.csv'
    # Create a counterfactual seeked set
    build_counterfactual_file(datasetFile, desiredOutcome, nbCf)

    def __remove_numerical_results_file(self):
        """
        Remove file 'test/NumericalResults.csv' if it exists.
        """
        try:
            os.remove(self.resultFile)
        except OSError:
            pass

    def test_emptyCallHasTypeError(self):
        """ Check train_model_and_solve_counterfactual requires two arguments """
        self.assertRaises(TypeError, train_model_and_solve_counterfactual)

    def test_simpleFunctionCall(self):
        """ Simple call to function on a single data set. """
        self.__remove_numerical_results_file()
        try:
            train_model_and_solve_counterfactual(
                self.datasetFile, self.counterfactualOneHotDatasetFile,
                rf_max_depth=4, rf_n_estimators=20, ilf_n_estimators=20)
            gurobiLicenseAvailable = True
        except GurobiError:
            gurobiLicenseAvailable = False
            print("Warning: Gurobi license not found:"
                  " cannot run integration test that solves MILP.")
        # Check results if model could be solved
        if gurobiLicenseAvailable:
            # Check that the result file exists and is in same directory
            self.assertTrue(self.resultFile.exists())
            # Check that length of result Df is equal to nbCfs+1
            resultDf = pd.read_csv(self.resultFile)
            nbRows, nbCols = resultDf.shape
            self.assertEqual(nbRows+1, self.nbCf)
            self.__remove_numerical_results_file()

    def test_incorrectObjNormRaisesValueError(self):
        self.assertRaises(ValueError, train_model_and_solve_counterfactual,
                          self.datasetFile,
                          self.counterfactualOneHotDatasetFile,
                          rf_max_depth=4, objectiveNorm=3)
