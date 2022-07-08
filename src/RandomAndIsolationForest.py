class RandomAndIsolationForest:
    def __init__(self, randomForest, isolationForest=None):
        """
        Combines a randomForest and an isolationForest inputs into
        a completeForest object.
        """
        self.randomForest = randomForest
        self.isolationForest = isolationForest
        # Store the completeForest size, the indices of the subforests
        # and the tree estimators.
        self.n_estimators = randomForest.n_estimators
        self.randomForestEstimatorsIndices = list(
            range(randomForest.n_estimators))
        self.estimators_ = [est for est in randomForest.estimators_]
        # Add the isolation forest to the completeForest if input non empty
        if isolationForest:
            self.n_estimators += isolationForest.n_estimators
            self.isolationForestEstimatorsIndices = [
                i + randomForest.n_estimators
                for i in range(isolationForest.n_estimators)]
            for est in isolationForest.estimators_:
                self.estimators_.append(est)
