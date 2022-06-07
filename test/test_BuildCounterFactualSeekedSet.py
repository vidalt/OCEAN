import unittest
from pathlib import Path
import pandas as pd
# Import local functions to test
from src.BuildCounterFactualSeekedSet import buildCounterFactualSeekedFile


class test_BuildCounterFactualSeekedSet(unittest.TestCase):
    # Test parameters
    dataset = 'datasets/Students-Performance-MAT.csv'
    desiredOutcome = 1
    nbCounterFactuals = 5
    # Manage path to dataset and counterfactuals files
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    pathToCounterfactual = datasetFile.parent / 'counterfactuals'
    datasetName = datasetFile.name
    counterfactualDatasetFile = pathToCounterfactual / datasetName
    oneHotDataset = "OneHot_" + datasetName
    counterfactualOneHotDatasetFile = pathToCounterfactual / oneHotDataset

    def test_buildCounterFactualSeekedFile(self):
        """
        Simple function call to build counterfactual set
        for a single dataset.
        """
        buildCounterFactualSeekedFile(self.datasetFile, self.desiredOutcome,
                                      self.nbCounterFactuals)

    def test_folderAndFilesCreated(self):
        # Test if the 'counterfactuals' folder has been created
        self.assertTrue(self.pathToCounterfactual.exists())
        # Test that the counterfactual dataset has been created
        self.assertTrue(self.counterfactualDatasetFile.exists())
        # Test that the One-Hot counterfactual dataset has been created
        self.assertTrue(self.counterfactualOneHotDatasetFile.exists())

    def test_counterfactualDataset(self):
        """ Test size of the counterfactual dataset built."""
        # Dataset
        counterfactualsData = pd.read_csv(self.counterfactualDatasetFile)
        nbFeaturesInStudentDataSet = 30
        self.assertTrue(counterfactualsData.size
                        == self.nbCounterFactuals * (nbFeaturesInStudentDataSet+1))
        # One-hot encoded dataset
        oneHotCounterfactualsData = pd.read_csv(
            self.counterfactualOneHotDatasetFile)
        nbFeaturesInOneHotStudentDataSet = 43
        self.assertTrue(oneHotCounterfactualsData.size
                        == self.nbCounterFactuals * (nbFeaturesInOneHotStudentDataSet+1))
