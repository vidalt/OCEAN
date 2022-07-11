# Author: Moises Henrique Pereira

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
# Import ui functions
from ui.interface.InterfaceEnums import InterfaceEnums
# Import OCEAN functions
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.RandomForestCounterFactual import RandomForestCounterFactualMilp


class InterfaceWorker(QObject):
    """ Run the counterfactual generation.

    It is needed because this process takes time enough to freeze the
    interface, so this class is used to be instantiated in another thread
    """

    progress = pyqtSignal(str)
    couterfactualClass = pyqtSignal(str)
    tableCounterfactualValues = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, controller):
        super().__init__()

        import sys
        self.__controller = controller

    def run(self):
        # showing the steps
        self.progress.emit(InterfaceEnums.Status.STEP3.value)

        # instantiating the optimization model
        randomForestMilp = RandomForestCounterFactualMilp(self.__controller.randomForestClassifier,
            [self.__controller.transformedChosenDataPoint],
            1-self.__controller.predictedOriginalClass[0],
            isolationForest=self.__controller.isolationForest,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=0,
            mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True,
            verbose=True,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            featuresActionnability=self.__controller.model.transformedFeaturesActionability,
            featuresType=self.__controller.model.transformedFeaturesType,
            featuresPossibleValues=self.__controller.model.transformedFeaturesPossibleValues)

        randomForestMilp.buildModel()

        # adding the user constraints over the optimization model
        constraintIndex = 0
        for feature in self.__controller.model.features:
            if feature != 'Class':
                if self.__controller.model.featuresType[feature] is FeatureType.Binary:
                    notAllowedValue = self.__controller.featuresConstraints[feature]['notAllowedValue']
                    if notAllowedValue == self.__controller.model.featuresInformations[feature]['value0']:
                        randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 1, notAllowedValue+' not allowed')
                    elif notAllowedValue == self.__controller.model.featuresInformations[feature]['value1']:
                        randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 0, notAllowedValue+' not allowed')

                    constraintIndex += 1

                elif self.__controller.model.featuresType[feature] is FeatureType.Discrete or self.__controller.model.featuresType[feature] is FeatureType.Numeric:
                    selectedMinimum = self.__controller.featuresConstraints[feature]['selectedMinimum']
                    selectedMaximum = self.__controller.featuresConstraints[feature]['selectedMaximum']

                    randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] >= selectedMinimum, feature+' minimum constraint')
                    randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] <= selectedMaximum, feature+' maximum constraint')

                    constraintIndex += 1

                elif self.__controller.model.featuresType[feature] is FeatureType.Categorical:
                    notAllowedValues = self.__controller.featuresConstraints[feature]['notAllowedValues']

                    for value in self.__controller.model.featuresInformations[feature]['possibleValues']:
                        if value in notAllowedValues:
                            randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 0, feature+'_'+value+' not allowed')

                        constraintIndex += 1

        randomForestMilp.solveModel()
        counterfactualResult = randomForestMilp.x_sol

        if (np.array(counterfactualResult) == np.array([self.__controller.transformedChosenDataPoint])).all():
            self.progress.emit('Model is infeasible')
        elif counterfactualResult is not None:
            counterfactualResultClass = self.__controller.randomForestClassifier.predict(counterfactualResult)

            result = self.__controller.model.invertTransformedDataPoint(counterfactualResult[0])

            counterfactualComparison = []
            for index, feature in enumerate(self.__controller.model.features):
                if feature != 'Class':
                    item1 = self.__controller.chosenDataPoint[index]
                    item2 = result[index]
                    if isinstance(item2, float):
                        item1 = float(item1)
                    counterfactualComparison.append([feature, str(item1), str(item2)])

            # showing the steps
            self.progress.emit(InterfaceEnums.Status.STEP4.value)
            # showing the counterfactual class
            self.couterfactualClass.emit(str(counterfactualResultClass[0]))
            # showing the steps
            self.progress.emit(InterfaceEnums.Status.STEP5.value)
            # showing the comparisson between the selected and the counterfactual values
            self.tableCounterfactualValues.emit(counterfactualComparison)

        else:
            # showing the steps
            self.progress.emit(InterfaceEnums.Status.ERROR_MSG.value)

        self.finished.emit()
