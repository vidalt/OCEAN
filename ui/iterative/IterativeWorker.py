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

    It is needed because this process can take a long time
    and freeze the interface. Hence, this class is
    instantiated in another thread.
    """
    # Initialize pyqt signals
    finished = pyqtSignal()
    counterfactualDataframe = pyqtSignal(object)
    counterfactualSteps = pyqtSignal(str)
    counterfactualError = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.__controller = controller
        self.initialPoint = self.__controller.transformedChosenDataPoint

    def __add_custom_constraints(self, result, oceanMilp):
        """
        Add custom constraints to the optimization model until
        the counterfactual explanation respects all customs
        constraints of the user.
        """
        constraintIndex = 0
        contradictoryFeatures = []
        initialPoint = self.__controller.initPointFeatures
        featureInformations = self.__controller.model.featuresInformations
        for index, feature in enumerate(self.__controller.model.features):
            if feature != 'Class':
                featureType = self.__controller.model.featuresType[feature]
                content = initialPoint[feature].getContent()
                if featureType is FeatureType.Binary:
                    constraintIndex += 1
                elif featureType in [FeatureType.Discrete,
                                     FeatureType.Numeric]:
                    minVal = float(content['minimumValue'])
                    maxVal = float(content['maximumValue'])
                    if float(result[index]) < minVal:
                        contradictoryFeatures.append(feature)
                        oceanMilp.model.addConstr(
                            oceanMilp.x_var_sol[constraintIndex] >= minVal,
                            feature + ' minimum constraint')
                    if float(result[index]) > maxVal:
                        contradictoryFeatures.append(feature)
                        oceanMilp.model.addConstr(
                            oceanMilp.x_var_sol[constraintIndex] <= maxVal,
                            feature+' maximum constraint')
                    constraintIndex += 1
                elif featureType is FeatureType.Categorical:
                    notAllowedValues = content['notAllowedValues']
                    if result[index] in notAllowedValues:
                        contradictoryFeatures.append(feature)
                    for val in featureInformations[feature]['possibleValues']:
                        if val == result[index]:
                            oceanMilp.model.addConstr(
                                oceanMilp.x_var_sol[constraintIndex] == 0,
                                feature + '_' + val + ' not allowed')
                        constraintIndex += 1
        return contradictoryFeatures

    def run(self):
        model = self.__controller.model
        actionability = model.transformedFeaturesActionability
        possibleValues = model.transformedFeaturesPossibleValues
        featuresType = model.transformedFeaturesType

        # Build MILP model
        oceanMilp = RandomForestCounterFactualMilp(
            self.__controller.rfClassifier,
            [self.initialPoint], 1-self.__controller.predictedOriginalClass[0],
            isolationForest=None,
            # isolationForest=self.__controller.isolationForest,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=0,
            mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True,
            verbose=False,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            featuresActionnability=actionability,
            featuresType=featuresType,
            featuresPossibleValues=possibleValues)
        oceanMilp.buildModel()

        # ---- Generate counterfactual explanation ----
        counterfactualFound = False
        for i in range(10):
            self.counterfactualSteps.emit(
                'Counterfactual Iteration number '+str(i+1)+' out of 10')
            oceanMilp.solveModel()
            # Get the counterfactual explanation of the current datapoint
            cfExplanation = oceanMilp.x_sol

            # Check results
            counterfactualNotFound = (
                np.array(cfExplanation) == np.array([self.initialPoint])).all()
            if counterfactualNotFound:
                print('!'*75)
                print('ERROR: Could not find a counterfactual explanations.')
                print('The model is infeasible.')
                print('!'*75)
                self.counterfactualError.emit()
                break

            elif cfExplanation is not None:
                counterfactualFound = True
                # Predict class of counterfactual
                counterfactualClass = self.__controller.rfClassifier.predict(
                    cfExplanation)
                assert (counterfactualClass == 1
                        - self.__controller.predictedOriginalClass[0])
                result = self.__controller.model.invertTransformedDataPoint(
                    cfExplanation[0])
                result = np.append(result, counterfactualClass[0])

                # Check that the user's custom constraints are
                # respected. Add the constraints if not.
                contradictoryFeatures = self.__add_custom_constraints(
                    result, oceanMilp)

                if len(contradictoryFeatures) == 0:
                    # Success: send the counterfactual
                    self.counterfactualDataframe.emit(result)
                    break

        if len(contradictoryFeatures) > 0 and counterfactualFound:
            print('!'*75)
            print('ERROR: It was not possible to find the counterfactual '
                  'with those constraints')
            print('after 10 iterations of solving the model '
                  'with added constraints.')
            print('!'*75)
            self.counterfactualError.emit()

        self.finished.emit()
