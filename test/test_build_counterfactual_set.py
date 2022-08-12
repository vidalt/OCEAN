import unittest
from pathlib import Path
import pandas as pd
# Import local functions to test
from src.build_counterfactual_set import build_counterfactual_file


class test_build_counterfactual_set(unittest.TestCase):
    # Test parameters
    dataset = 'datasets/Students-Performance-MAT.csv'
    desiredOutcome = 1
    nbCounterFactuals = 1
    # Manage path to dataset and counterfactuals files
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    pathToCounterfactual = datasetFile.parent / 'counterfactuals'
    datasetName = datasetFile.name
    counterfactualDatasetFile = pathToCounterfactual / datasetName
    oneHotDataset = "OneHot_" + datasetName
    counterfactualOneHotDatasetFile = pathToCounterfactual / oneHotDataset

    def test_build_counterfactual_file(self):
        """
        Simple function call to build counterfactual set
        for a single dataset.
        """
        build_counterfactual_file(self.datasetFile, self.desiredOutcome,
                                  self.nbCounterFactuals)

    def test_folderAndFilesCreated(self):
        # Test if the 'counterfactuals' folder has been created
        self.assertTrue(self.pathToCounterfactual.exists())
        # Test that the counterfactual dataset has been created
        self.assertTrue(self.counterfactualDatasetFile.exists())
        # Test that the One-Hot counterfactual dataset has been created
        self.assertTrue(self.counterfactualOneHotDatasetFile.exists())

    def test_noFolderCreatedInTestFolder(self):
        pathToCfFolderInTestFolder = self.THIS_DIR / 'counterfactuals'
        self.assertFalse(pathToCfFolderInTestFolder.exists())

    def test_counterfactualDataset(self):
        """ Test size of the counterfactual dataset built."""
        # Dataset
        counterfactualsData = pd.read_csv(self.counterfactualDatasetFile)
        nbFeaturesInStudentSet = 30
        self.assertEqual(counterfactualsData.size,
                         self.nbCounterFactuals * (nbFeaturesInStudentSet+1))
        # One-hot encoded dataset
        oneHotCounterfactualsData = pd.read_csv(
            self.counterfactualOneHotDatasetFile)
        nbFeatureOneHotStudent = 43
        self.assertEqual(oneHotCounterfactualsData.size,
                         self.nbCounterFactuals * (nbFeatureOneHotStudent+1))
