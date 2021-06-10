class RandomAndIsolationForest:
    def __init__(self, randomForest, isolationForest=None):
        self.randomForest = randomForest
        self.isolationForest = isolationForest

        self.n_estimators = self.randomForest.n_estimators
        self.randomForestEstimatorsIndices = [i for i in range(self.randomForest.n_estimators)]
        self.estimators_ = [est for est in self.randomForest.estimators_]

        if isolationForest:
            self.n_estimators += self.isolationForest.n_estimators
            self.isolationForestEstimatorsIndices = [i + self.randomForest.n_estimators for i in range(self.isolationForest.n_estimators)]
            for est in self.isolationForest.estimators_:
                self.estimators_.append(est)

