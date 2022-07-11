# Author: Moises Henrique Pereira

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
# Import OCEAN functions
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.RandomForestCounterFactual import RandomForestCounterFactualMilp


class IterativeWorker(QObject):
    """
    Handles the counterfactual generation.

    It is needed because this process can take enough time enough
    to freeze the interface,  so, this class is used to
    be instantiated in another thread
    """

    finished = pyqtSignal()
    counterfactualDataframe = pyqtSignal(object)
    counterfactualSteps = pyqtSignal(str)
    counterfactualError = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.__controller = controller
        self.__values = []

    def run(self):
        randomForestMilp = RandomForestCounterFactualMilp(self.__controller.randomForestClassifier,
            [self.__controller.transformedChosenDataPoint],
            1-self.__controller.predictedOriginalClass[0],
            isolationForest=None,
            # isolationForest=self.__controller.isolationForest,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=0,
            mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True,
            verbose=False,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            featuresActionnability=self.__controller.model.transformedFeaturesActionability,
            featuresType=self.__controller.model.transformedFeaturesType,
            featuresPossibleValues=self.__controller.model.transformedFeaturesPossibleValues)

        randomForestMilp.buildModel()

        contradictoryFeatures = []
        counterfactualFound = False
        # while True:
        for i in range(10):
            self.counterfactualSteps.emit('Counterfactual Iteration number '+str(i+1)+' out of 10')

            randomForestMilp.solveModel()
            counterfactualResult = randomForestMilp.x_sol

            # getting the counterfactual to the current datapoint
            counterfactualResult = randomForestMilp.x_sol

            if (np.array(counterfactualResult) == np.array([self.__controller.transformedChosenDataPoint])).all():
                counterfactualFound = False
                print('!'*75)
                print('ERROR: Model is infeasible')
                print('!'*75)
                self.counterfactualError.emit()
                # self.finished.emit()
                break

            elif counterfactualResult is not None:
                counterfactualFound = True
                counterfactualResultClass = self.__controller.randomForestClassifier.predict(counterfactualResult)

                result = self.__controller.model.invertTransformedDataPoint(counterfactualResult[0])
                result = np.append(result, counterfactualResultClass[0])

                constraintIndex = 0
                contradictoryFeatures = []
                for index, feature in enumerate(self.__controller.model.features):
                    if feature != 'Class':
                        if self.__controller.model.featuresType[feature] is FeatureType.Binary:
                            constraintIndex += 1

                        elif self.__controller.model.featuresType[feature] is FeatureType.Discrete or self.__controller.model.featuresType[feature] is FeatureType.Numeric:
                            content = self.__controller.dictControllersSelectedPoint[feature].getContent()
                            minimumValue = float(content['minimumValue'])
                            maximumValue = float(content['maximumValue'])

                            if float(result[index]) < minimumValue:
                                contradictoryFeatures.append(feature)

                                randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] >= minimumValue, feature+' minimum constraint')

                            if float(result[index]) > maximumValue:
                                contradictoryFeatures.append(feature)

                                randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] <= maximumValue, feature+' maximum constraint')

                            constraintIndex += 1

                        elif self.__controller.model.featuresType[feature] is FeatureType.Categorical:
                            content = self.__controller.dictControllersSelectedPoint[feature].getContent()
                            notAllowedValues = content['notAllowedValues']

                            if result[index] in notAllowedValues:
                                contradictoryFeatures.append(feature)

                            for value in self.__controller.model.featuresInformations[feature]['possibleValues']:
                                if value == result[index]:
                                    randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 0, feature+'_'+value+' not allowed')

                                constraintIndex += 1

                if len(contradictoryFeatures) == 0:
                    # sending the counterfactual
                    self.counterfactualDataframe.emit(result)
                    break

        if len(contradictoryFeatures) > 0 and counterfactualFound:
            print('!'*75)
            print('ERROR: It was not possible to find the counterfactual with those constraints')
            print('!'*75)
            self.counterfactualError.emit()

        self.finished.emit()
