import unittest
from pathlib import Path
# Import local functions to test
from src.BuildCounterFactualSeekedSet import buildCounterFactualSeekedFile


class test_BuildCounterFactualSeekedSet(unittest.TestCase):
    # Test parameters
    dataset = 'datasets/Students-Performance-MAT.csv'
    desiredOutcome = 1
    nbCounterFactuals = 5
    # Manage path to dataset files
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    print(datasetFile)

    def test_buildCounterFactualSeekedFile(self):
        """
        Simple function call to build counterfactual set
        for a single dataset.
        """
        buildCounterFactualSeekedFile(self.datasetFile, self.desiredOutcome,
                                      self.nbCounterFactuals)

    def test_folderAndFilesCreated(self):
        # Test if the 'counterfactuals' folder has been created
        pathToCounterfactual = self.datasetFile.parent / 'counterfactuals'
        self.assertTrue(pathToCounterfactual.exists())
        # Test that the counterfactual dataset has been created
        datasetName = self.datasetFile.name
        counterfactualDatasetFile = pathToCounterfactual / datasetName
        self.assertTrue(counterfactualDatasetFile.exists())
        # Test that the One-Hot counterfactual dataset has been created
        oneHotDataset = "OneHot_" + datasetName
        counterfactualOneHotDatasetFile = pathToCounterfactual / oneHotDataset
        self.assertTrue(counterfactualOneHotDatasetFile.exists())
