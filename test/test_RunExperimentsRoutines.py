import unittest
import os
import pandas as pd
from pathlib import Path
# Import local functions to test
from src.RunExperimentsRoutines import trainModelAndSolveCounterFactuals
from src.BuildCounterFactualSeekedSet import buildCounterFactualSeekedFile


class test_trainModelAndSolveCounterFactuals(unittest.TestCase):
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
    buildCounterFactualSeekedFile(datasetFile, desiredOutcome, nbCf)

    def __remove_numerical_results_file(self):
        """
        Remove file 'test/NumericalResults.csv' if it exists.
        """
        try:
            os.remove(self.resultFile)
        except OSError:
            pass

    def test_emptyCallHasTypeError(self):
        """ Check trainModelAndSolveCounterFactuals requires two arguments """
        self.assertRaises(TypeError, trainModelAndSolveCounterFactuals)

    def test_simpleFunctionCall(self):
        """ Simple call to function on a single data set. """
        self.__remove_numerical_results_file()
        trainModelAndSolveCounterFactuals(
            self.datasetFile, self.counterfactualOneHotDatasetFile,
            rf_max_depth=4)
        # Check that the result file exists and is in same directory
        self.assertTrue(self.resultFile.exists())
        # Check that length of result Df is equal to nbCfs+1
        resultDf = pd.read_csv(self.resultFile)
        nbRows, nbCols = resultDf.shape
        self.assertEqual(nbRows+1, self.nbCf)
        self.__remove_numerical_results_file()
