import unittest
from pathlib import Path
# Import local functions to test
from src.dataProcessing import DatasetReader
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import FeatureActionnability


class test_DatasetReader(unittest.TestCase):
    dataset = 'datasets/German-Credit.csv'
    # Manage path to files
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    # Get dataReader on selected data set
    assert(datasetFile.name == 'German-Credit.csv')
    reader = DatasetReader(datasetFile)

    def test_emptyCallFails(self):
        self.assertRaises(TypeError, DatasetReader)

    def test_simpleFunctionCall(self):
        _ = DatasetReader(self.datasetFile)

    def test_dataReaderIsCorrect(self):
        """
        Test that first row of data reader is correct.
            Dataset: German-Credit
        """
        # - Test features -
        self.assertEqual(self.reader.X.values[0, 0], 0.8571428571428571)
        self.assertEqual(self.reader.X.values[0, 3], 0.3333333333333333)
        self.assertEqual(self.reader.X.values[0, 5], 0.029411764705882353)
        # - Test label -
        self.assertEqual(self.reader.y.values[0], 1)
        self.assertEqual(self.reader.y.values[1], 0)
        self.assertEqual(self.reader.y.values[2], 1)

    def test_sizeOfTestAndTrainSets(self):
        """
        Test dimensions of data in data reader are correct.
            Dataset: German-Credit
        """
        # - Test features -
        nbRows, nbCols = self.reader.X.shape
        self.assertEqual(nbRows, 1000)
        self.assertEqual(len(self.reader.X.values), 1000)
        self.assertEqual(nbCols, 19)
        self.assertTrue('Class' not in self.reader.X.columns)
        self.assertTrue('Desired Outcome' not in self.reader.X.columns)
        # - Test label -
        self.assertEqual(len(self.reader.y.values), 1000)
        self.assertFalse(len(self.reader.y.values) == 999)
        self.assertTrue(set(self.reader.y.values) == set([0, 1]))
        self.assertFalse(set(self.reader.y.values) == set([0, 1, 2]))

    def test_featureActionabilityIsCorrect(self):
        self.assertEqual(self.reader.featuresActionnability[0],
                         FeatureActionnability.Increasing)  # Age
        self.assertEqual(self.reader.featuresActionnability[1],
                         FeatureActionnability.Free)  # Job
        self.assertEqual(self.reader.featuresActionnability[2],
                         FeatureActionnability.Free)  # Housing

    def test_featureTypesIsCorrect(self):
        self.assertEqual(self.reader.featuresType[0], FeatureType.Discrete)
        self.assertEqual(self.reader.featuresType[1], FeatureType.Discrete)
        self.assertEqual(self.reader.featuresType[4], FeatureType.Numeric)
        self.assertTrue(FeatureType.Discrete in self.reader.featuresType)
        self.assertTrue(
            FeatureType.Categorical not in self.reader.featuresType)

    def test_featurePossibleValues(self):
        self.assertEqual(set(self.reader.featuresPossibleValues[1]),
                         set([0, 1/3, 2/3, 3/3]))
        self.assertEqual(self.reader.featuresPossibleValues[4], [])
        self.assertEqual(self.reader.featuresPossibleValues[6], [])

    def test_upperAndLowerBounds(self):
        self.assertEqual(self.reader.lowerBounds['Age'], 0)
        self.assertEqual(self.reader.upperBounds['Age'], 75)
        self.assertEqual(self.reader.lowerBounds['Job'], 0)
        self.assertEqual(self.reader.upperBounds['Job'], 3)
