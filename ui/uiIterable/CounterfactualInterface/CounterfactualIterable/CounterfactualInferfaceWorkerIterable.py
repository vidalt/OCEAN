# Author: Moises Henrique Pereira
# this class handles to run the counterfactual generation
# it is needed because this process takes time enough to freeze the interface, 
# so, this class is used to be instantiated in another thread

from ..CounterfactualInterfaceEnums import CounterfactualInterfaceEnums

from CounterFactualParameters import FeatureType
from CounterFactualParameters import BinaryDecisionVariables, TreeConstraintsType
from RandomForestCounterFactual import RandomForestCounterFactualMilp

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

class CounterfactualInferfaceWorkerIterable(QObject):

    finished = pyqtSignal()
    counterfactualDataframe = pyqtSignal(object)
    counterfactualError = pyqtSignal()

    def __init__(self, controller):
        super().__init__()

        # import sys

        # modulename = '.CounterfactualInterfaceControllerIterable'
        # if modulename not in sys.modules:
        #     from .CounterfactualInterfaceControllerIterable import CounterfactualInterfaceControllerIterable

        # assert isinstance(controller, CounterfactualInterfaceControllerIterable)

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
        
        ###########################################################################################
        # IDEIA: INCLUIR RESTRIÇÕES DINAMICAMENTE
        # I0 -> SEM RESTRIÇÕES E VERIFICA SE O CONTRAFACTUAL FERE ALGUMA RESTRIÇÃO, SE FERIR
        # I1 -> INCLUI APENAS AS RESTRIÇÕES QUE FORAM FERIDAS E TENTA DE NOVO
        # Ii -> REPETE O PROCESSO DE INCLUSÃO PARCIAL ATÉ QUE NÃO FIRA NENHUMA RESTRIÇÃO
        ###########################################################################################

        # # adding the user constraints over the optimization model
        # constraintIndex = 0
        # for feature in self.__controller.model.features:
        #     if feature != 'Class':
        #         if self.__controller.model.featuresType[feature] is FeatureType.Binary:
        #             # content = self.__controller.dictControllersSelectedPoint[feature].getContent()
        #             # notAllowedValue = content['notAllowedValue']
        #             # if notAllowedValue == self.__controller.model.featuresInformations[feature]['value0']:
        #             #     randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 1, notAllowedValue+' not allowed')
        #             # elif notAllowedValue == self.__controller.model.featuresInformations[feature]['value1']:
        #             #     randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 0, notAllowedValue+' not allowed')

        #             constraintIndex += 1

        #         elif self.__controller.model.featuresType[feature] is FeatureType.Discrete or self.__controller.model.featuresType[feature] is FeatureType.Numeric:
        #             content = self.__controller.dictControllersSelectedPoint[feature].getContent()
        #             minimumValue = content['minimumValue']
        #             maximumValue = content['maximumValue']

        #             randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] >= minimumValue, feature+' minimum constraint')
        #             randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] <= maximumValue, feature+' maximum constraint')

        #             constraintIndex += 1

        #         elif self.__controller.model.featuresType[feature] is FeatureType.Categorical:
        #             content = self.__controller.dictControllersSelectedPoint[feature].getContent()
        #             notAllowedValues = content['notAllowedValues']

        #             for value in self.__controller.model.featuresInformations[feature]['possibleValues']:
        #                 if value in notAllowedValues:
        #                     randomForestMilp.model.addConstr(randomForestMilp.x_var_sol[constraintIndex] == 0, feature+'_'+value+' not allowed')

        #                 constraintIndex += 1

        randomForestMilp.solveModel()
        counterfactualResult = randomForestMilp.x_sol

        # getting the counterfactual to the current datapoint
        counterfactualResult = randomForestMilp.x_sol

        if (np.array(counterfactualResult) == np.array([self.__controller.transformedChosenDataPoint])).all():
            print('!'*75)
            print('ERROR: Model is infeasible')
            print('!'*75)
            self.counterfactualError.emit()
            
        elif counterfactualResult is not None:
            counterfactualResultClass = self.__controller.randomForestClassifier.predict(counterfactualResult)

            result = self.__controller.model.invertTransformedDataPoint(counterfactualResult[0])
            result = np.append(result, counterfactualResultClass[0])
            
            # sending the counterfactual
            self.counterfactualDataframe.emit(result)

        # sending the objective function values        
        self.finished.emit()
