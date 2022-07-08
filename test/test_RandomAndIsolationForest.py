import unittest
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
# Import local functions to test
from src.RandomAndIsolationForest import RandomAndIsolationForest


class test_RandomAndIsolationForest(unittest.TestCase):
    # Train a random forest and an isolation forest on toy data
    X, y = make_moons(noise=0.3, random_state=0)
    randomForestSize = 8
    rf = RandomForestClassifier(max_depth=5,
                                n_estimators=randomForestSize,
                                max_features=1)
    rf.fit(X, y)
    isolationForestSize = 11
    ilf = IsolationForest(random_state=1, max_samples=100,
                          n_estimators=isolationForestSize)
    ilf.fit(X)

    def test_emptyCallHasTypeError(self):
        """ Check trainModelAndSolveCounterFactuals requires two arguments """
        self.assertRaises(TypeError, RandomAndIsolationForest)

    def test_onlyRandomForest(self):
        completeForest = RandomAndIsolationForest(self.rf, False)
        self.assertEqual(completeForest.n_estimators, self.randomForestSize)
        self.assertEqual(completeForest.estimators_[self.randomForestSize-1],
                         self.rf[-1])

    def test_randomForestAndIsolationForest(self):
        completeForest = RandomAndIsolationForest(self.rf, self.ilf)
        self.assertEqual(completeForest.n_estimators,
                         self.randomForestSize + self.isolationForestSize)
        self.assertEqual(completeForest.estimators_[self.randomForestSize-1],
                         self.rf[-1])
        self.assertEqual(completeForest.estimators_[self.randomForestSize],
                         self.ilf[0])
        self.assertEqual(completeForest.isolationForestEstimatorsIndices,
                         list(range(self.randomForestSize,
                                    self.randomForestSize
                                    + self.isolationForestSize)))
