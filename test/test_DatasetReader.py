import unittest
from pathlib import Path
# Import local functions to test
from src.DatasetReader import DatasetReader
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
        MIN_DISCRETE = 0.  # 0, by choice for all discrete features
        MAX_AGE = 75.
        MAX_JOB = 3.
        MAX_SAVING = 4.
        MAX_CHECKING = 3.
        MAX_CREDIT = 18424.
        MIN_CREDIT = 250.
        MAX_DURATION = 72.
        # Test upper and lower bounds
        self.assertEqual(self.reader.upperBoundsList[0:6],
                         [MAX_AGE, MAX_JOB, MAX_SAVING,
                          MAX_CHECKING, MAX_CREDIT, MAX_DURATION])
        self.assertEqual(self.reader.lowerBoundsList[0:6],
                         [MIN_DISCRETE, MIN_DISCRETE, MIN_DISCRETE,
                          MIN_DISCRETE, MIN_CREDIT, MIN_DISCRETE])
        # Test min/max scaling
        self.assertEqual(self.reader.X.values[0, 0],
                         (67 - MIN_DISCRETE) / (MAX_AGE - MIN_DISCRETE))
        self.assertEqual(self.reader.X.values[0, 1],
                         (2 - MIN_DISCRETE) / (MAX_JOB - MIN_DISCRETE))
        self.assertEqual(self.reader.X.values[0, 2],
                         (0 - MIN_DISCRETE) / (MAX_SAVING - MIN_DISCRETE))
        self.assertEqual(self.reader.X.values[0, 3],
                         (1 - MIN_DISCRETE) / (MAX_CHECKING - MIN_DISCRETE))
        self.assertEqual(self.reader.X.values[0, 4],
                         (1169 - MIN_CREDIT) / (MAX_CREDIT - MIN_CREDIT))
        self.assertEqual(self.reader.X.values[0, 5],
                         (6 - MIN_DISCRETE) / (MAX_DURATION - MIN_DISCRETE))
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
