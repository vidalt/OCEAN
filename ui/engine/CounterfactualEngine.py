# Author: Moises Henrique Pereira

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest


class CounterfactualEngine():
    """
    Train models and give the corresponding prediction.
    """

    @staticmethod
    def trainRandomForestClassifier(xTrain, yTrain):
        """
        Return a trained random forest classifier.
        """
        RF_MAX_LEAF_NODES = 32
        RF_RANDOM_STATE = 1
        RF_N_ESTIMATORS = 100
        assert xTrain is not None
        assert yTrain is not None
        assert len(xTrain) != 0
        assert len(yTrain) != 0

        randomForestClassifier = RandomForestClassifier(
            max_leaf_nodes=RF_MAX_LEAF_NODES,
            random_state=RF_RANDOM_STATE,
            n_estimators=RF_N_ESTIMATORS)
        randomForestClassifier.fit(xTrain, yTrain)

        return randomForestClassifier

    @staticmethod
    def trainIsolationForest(xTrain):
        """
        Return a trained isolation forest.
        """
        IF_RANDOM_STATE = 2
        IF_MAX_SAMPLES = 24
        IF_N_ESTIMATORS = 100
        assert xTrain is not None
        assert len(xTrain) != 0

        isolationForest = IsolationForest(
            random_state=IF_RANDOM_STATE,
            max_samples=IF_MAX_SAMPLES,
            n_estimators=IF_N_ESTIMATORS)
        isolationForest.fit(xTrain)

        return isolationForest

    @staticmethod
    def randomForestClassifierPredict(rfClassifier, xValue):
        """
        Return class prediction for new observation.
        """
        assert rfClassifier is not None
        assert isinstance(xValue, list)
        return rfClassifier.predict(xValue)

    @staticmethod
    def randomForestClassifierPredictProbabilities(rfClassifier, xValue):
        """
        Return random forest prediction score for new observation.
        """
        assert rfClassifier is not None
        assert isinstance(xValue, list)
        return rfClassifier.predict_proba(xValue)
