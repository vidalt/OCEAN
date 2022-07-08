# Author: Moises Henrique Pereira
# this class handles to train models and give the corresponding prediction

from numpy.lib.arraysetops import isin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from .CounterfactualEngineEnums import CounterfactualEngineEnums

class CounterfactualEngine():

    # this function trains and return a trained random forest
    @staticmethod
    def trainRandomForestClassifier(xTrain, yTrain):
        assert xTrain is not None
        assert yTrain is not None
        assert len(xTrain) != 0
        assert len(yTrain) != 0

        randomForestClassifier = RandomForestClassifier(max_leaf_nodes=CounterfactualEngineEnums.RandomForestClassifierParameters.MAX_LEAF_NODES.value, 
                                                        random_state=CounterfactualEngineEnums.RandomForestClassifierParameters.RANDOM_STATE.value,
                                                        n_estimators=CounterfactualEngineEnums.RandomForestClassifierParameters.N_ESTIMATORS.value)
        randomForestClassifier.fit(xTrain, yTrain)

        return randomForestClassifier

    # this function trains and return a trained isolation forest
    @staticmethod
    def trainIsolationForest(xTrain):
        assert xTrain is not None
        assert len(xTrain) != 0

        isolationForest = IsolationForest(random_state=CounterfactualEngineEnums.IsolationForestParameters.RANDOM_STATE.value, 
                                          max_samples=CounterfactualEngineEnums.IsolationForestParameters.MAX_SAMPLES.value, 
                                          n_estimators=CounterfactualEngineEnums.IsolationForestParameters.N_ESTIMATORS.value)
        isolationForest.fit(xTrain)

        return isolationForest

    # this function returns the prediction given a model and a point
    @staticmethod
    def randomForestClassifierPredict(randomForestClassifier, xValue):
        assert randomForestClassifier is not None
        assert isinstance(xValue, list)

        return randomForestClassifier.predict(xValue)
