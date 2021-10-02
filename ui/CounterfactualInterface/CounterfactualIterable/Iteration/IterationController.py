# Author: Moises Henrique Pereira
# this class handle the logic over the interface, interacting with model, view and worker
# also taking the selected dataset informations from model to send to counterfactual generator in worker class

from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from .IterationView import IterationView
from .IterationEnums import IterationEnums

from CounterFactualParameters import FeatureType

from CounterfactualEngine.CounterfactualEngine import CounterfactualEngine

from ...ComboboxList.ComboboxListController import ComboboxListController
from ...DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ...Slider3Ranges.Slider3RangesController import Slider3RangesController

from ...CounterfactualInterfaceEnums import CounterfactualInterfaceEnums
from ..CounterfactualInferfaceWorkerIterable import CounterfactualInferfaceWorkerIterable

import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian

class IterationController():

    def __init__(self, original, parent, model, randomForestClassifier, isolationForest):
        self.original = original
        self.parent = parent

        self.view = IterationView()
        self.model = model

        self.iterationName = None

        self.randomForestClassifier = randomForestClassifier
        self.isolationForest = isolationForest

        self.__initializeView()

        self.dictControllersSelectedPoint = {}

        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None

        self.updatedCurrentPoint = None

        self.predictedCurrentClass = None
        self.predictedCurrentClassPercentage = None

        self.counterfactualToPlot = None

        self.__canvas = self.view.getCanvas()
        self.__canvas.updatedPoint.connect(self.__onUpdatedCurrentPoint)
        self.__canvas.errorPlot.connect(self.__errorPlotHandler)
        
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None
        self.transformedSamplesClassesPercentage = None

        self.__suggestedFeaturesToPlot = None

        self.view.selectedFeatures.connect(lambda: self.__updateGraph())
        self.view.nextIteration.connect(lambda: self.__handlerNextIteration())
        self.view.finishIteration.connect(lambda: self.__handlerFinishIteration())


    # this function takes the dataframe names and send them to interface
    def __initializeView(self):
        self.view.initializeView()

    # this function sets the previous values to the current view
    def setFeaturesAndValues(self, dictNextFeaturesInformation):
        self.iterationName = dictNextFeaturesInformation['iterationName']

        for feature in self.model.features:
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                content = dictNextFeaturesInformation[feature]

                if featureType is FeatureType.Binary:
                    actionable = content['actionable']
                    value0 = content['value0']
                    value1 = content['value1']
                    value = content['value']

                    componentController = DoubleRadioButtonController(self.view)
                    componentController.initializeView(feature, str(value0), str(value1))
                    componentController.setActionable(actionable)
                    componentController.setSelectedValue(value)
                    componentController.disableComponent()

                elif featureType is FeatureType.Discrete:
                    actionable = content['actionable']
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    componentController = Slider3RangesController(self.view, smaller=True)
                    componentController.initializeView(feature, minValue, maxValue, decimalPlaces=0)
                    componentController.setActionable(actionable)
                    componentController.setSelectedValue(value)
                    componentController.disableComponent()

                elif featureType is FeatureType.Numeric:
                    actionable = content['actionable']
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    componentController = Slider3RangesController(self.view, smaller=True)
                    componentController.initializeView(feature, minValue, maxValue)
                    componentController.setActionable(actionable)
                    componentController.setSelectedValue(value)
                    componentController.disableComponent()
                    
                elif featureType is FeatureType.Categorical:
                    actionable = content['actionable']
                    allowedValues = content['allowedValues']
                    notAllowedValues = content['notAllowedValues']
                    allPossibleValues = allowedValues + notAllowedValues
                    value = content['value']
                    
                    componentController = ComboboxListController(self.view, allPossibleValues)
                    componentController.initializeView(feature, allowedValues)
                    componentController.setActionable(actionable)
                    componentController.setSelectedValue(value)
                    componentController.disableComponent()

                # adding the view to selectedPoint component
                self.view.addFeatureWidget(componentController.view)
                # saving the controller to facilitate the access to components
                self.dictControllersSelectedPoint[feature] = componentController

        self.view.addFeaturesOptions(list(self.model.features[:-1]))
        self.__calculateClass()

    # this function draw the suggested feature
    def setSuggestedFeaturesToPlot(self, suggestedFeatures):
        self.__suggestedFeaturesToPlot = suggestedFeatures

        self.__updateGraph(suggestedFeatures)
        self.view.selectFeatures(suggestedFeatures)

    # listen the updated point to redraw the graph
    def __onUpdatedCurrentPoint(self, updatedPoint):
        self.waitCursor()

        selectedFeatures = self.view.getSelectedFeatures()
        if len(selectedFeatures) == len(updatedPoint):

            # current datapoint
            currentDataPoint = self.chosenDataPoint.copy()
            currentDataPoint = np.append(currentDataPoint, self.predictedCurrentClass)
            currentDataframe = pd.DataFrame(data=[currentDataPoint], columns=self.model.features)

            # updating the values
            for i, f in enumerate(selectedFeatures):
                currentDataframe[f] = updatedPoint[i]

            transformedCurrent = self.model.transformDataPoint(currentDataframe.to_numpy()[0][:-1])
            predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [transformedCurrent])
            predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [transformedCurrent])

            # adding the updated prediction class, and percentage
            currentDataframe['Class'] = predictedCurrentClass[0]
            currentDataframe['prob1'] = predictedCurrentClassPercentage[0][1]

            # saving the updated current point values
            # only needs the features values without the Class and prob1
            self.updatedCurrentPoint = currentDataframe.to_numpy()[0][:-2]

            # getting the initial datapoint, keeping a historic
            parentDataPoint = self.original.chosenDataPoint.copy()
            parentDataPoint = np.append(parentDataPoint, self.original.predictedOriginalClass)
            # transforming the parent datapoint to predict its class
            transformedParentDataPoint = self.model.transformDataPoint(parentDataPoint[:-1])
            # predicting the parent datapoint class probabilities
            predictedParentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [transformedParentDataPoint])
            # building the parent dataframe
            parentDataframe = pd.DataFrame(data=[parentDataPoint], columns=self.model.features)
            # adding the prediction percentage
            parentDataframe['prob1'] = predictedParentClassPercentage[0][1]

            lastScenarioDataframe = None
            if hasattr(self.parent, 'updatedCurrentPoint'):
                # getting the last scenario datapoint, keeping a historic
                lastScenarioDataPoint = self.parent.updatedCurrentPoint.copy()
                # transforming the parent datapoint to predict its class
                transformedLastScenarioDataPoint = self.model.transformDataPoint(lastScenarioDataPoint)
                # predicting the parent datapoint class and probabilities
                predictedLastScenarioClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [transformedLastScenarioDataPoint])
                predictedLastScenarioClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [transformedLastScenarioDataPoint])
                # adding class
                lastScenarioDataPoint = np.append(lastScenarioDataPoint, predictedLastScenarioClass[0])
                # building the parent dataframe
                lastScenarioDataframe = pd.DataFrame(data=[lastScenarioDataPoint], columns=self.model.features)
                # adding the prediction percentage
                lastScenarioDataframe['prob1'] = predictedLastScenarioClassPercentage[0][1]

            lastScenarioName = None
            if hasattr(self.parent, 'iterationName'):
                lastScenarioName = self.parent.iterationName

            #parameters to update graph
            parameters = {'controller':self, 'currentPoint':currentDataframe, 'originalPoint':parentDataframe, 'lastScenarioPoint':lastScenarioDataframe, 'lastScenarioName':lastScenarioName, 'selectedFeatures':selectedFeatures}
            self.__canvas.updateGraph(parameters)

        self.restorCursor()

    # this function takes the selected data point and calculate the respective class
    def __calculateClass(self):
        self.view.clearClass()
        
        # getting the datapoint
        auxiliarDataPoint = []
        for feature in self.model.features:
            if feature != 'Class':
                content = self.dictControllersSelectedPoint[feature].getContent()
                auxiliarDataPoint.append(content['value'])
                
        self.chosenDataPoint = np.array(auxiliarDataPoint)

        # transforming the datapoint to predict its class
        self.transformedChosenDataPoint = self.model.transformDataPoint(self.chosenDataPoint)
        
        # predicting the datapoint class and showing its value
        self.predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [self.transformedChosenDataPoint])
        self.predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [self.transformedChosenDataPoint])
        
        self.view.showCurrentClass(self.predictedCurrentClass[0])      

    def __updateGraph(self, suggestedFeatures=None):
        self.waitCursor()
        
        selectedFeatures = None
        if suggestedFeatures is not None:
            selectedFeatures = suggestedFeatures
        else:
            selectedFeatures = self.view.getSelectedFeatures()

        if len(selectedFeatures) != 0:
            # getting the current datapoint
            auxiliarDataPoint = []
            for feature in self.model.features:
                if feature != 'Class':
                    content = self.dictControllersSelectedPoint[feature].getContent()
                    auxiliarDataPoint.append(content['value'])
                    
            self.chosenDataPoint = np.array(auxiliarDataPoint)

            # saving the updated current point values
            self.updatedCurrentPoint = self.chosenDataPoint.copy()

            # transforming the datapoint to predict its class
            self.transformedChosenDataPoint = self.model.transformDataPoint(self.chosenDataPoint)
        
            # predicting the datapoint class and showing its value
            self.predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [self.transformedChosenDataPoint])
            self.predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [self.transformedChosenDataPoint])
            # SE A MUDANÇA DE VALOR ESTIVER DESABILITADA, ESSA PARTE É DESNECESSÁRIA
            self.view.showCurrentClass(self.predictedCurrentClass[0])

            # current datapoint
            currentDataPoint = self.chosenDataPoint.copy()
            currentDataPoint = np.append(currentDataPoint, self.predictedCurrentClass)
            currentDataframe = pd.DataFrame(data=[currentDataPoint], columns=self.model.features)
            # adding the prediction percentage
            currentDataframe['prob1'] = self.predictedCurrentClassPercentage[0][1]

            # getting the initial datapoint, keeping a historic
            parentDataPoint = self.original.chosenDataPoint.copy()
            parentDataPoint = np.append(parentDataPoint, self.original.predictedOriginalClass)
            # transforming the parent datapoint to predict its class
            transformedParentDataPoint = self.model.transformDataPoint(parentDataPoint[:-1])
            # predicting the parent datapoint class probabilities
            # predictedParentClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [transformedParentDataPoint])
            predictedParentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [transformedParentDataPoint])
            # building the parent dataframe
            parentDataframe = pd.DataFrame(data=[parentDataPoint], columns=self.model.features)
            # adding the prediction percentage
            parentDataframe['prob1'] = predictedParentClassPercentage[0][1]

            lastScenarioDataframe = None
            if hasattr(self.parent, 'updatedCurrentPoint'):
                # getting the last scenario datapoint, keeping a historic
                lastScenarioDataPoint = self.parent.updatedCurrentPoint.copy()
                # transforming the parent datapoint to predict its class
                transformedLastScenarioDataPoint = self.model.transformDataPoint(lastScenarioDataPoint)
                # predicting the parent datapoint class and probabilities
                predictedLastScenarioClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [transformedLastScenarioDataPoint])
                predictedLastScenarioClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [transformedLastScenarioDataPoint])
                # adding class
                lastScenarioDataPoint = np.append(lastScenarioDataPoint, predictedLastScenarioClass[0])
                # building the parent dataframe
                lastScenarioDataframe = pd.DataFrame(data=[lastScenarioDataPoint], columns=self.model.features)
                # adding the prediction percentage
                lastScenarioDataframe['prob1'] = predictedLastScenarioClassPercentage[0][1]

            lastScenarioName = None
            if hasattr(self.parent, 'iterationName'):
                lastScenarioName = self.parent.iterationName

            #parameters to update graph
            parameters = {'controller':self, 'currentPoint':currentDataframe, 'originalPoint':parentDataframe, 'lastScenarioPoint':lastScenarioDataframe, 'lastScenarioName':lastScenarioName, 'selectedFeatures':selectedFeatures}
            self.__canvas.updateGraph(parameters)

        self.restorCursor()

    def __handlerNextIteration(self):
        self.waitCursor()

        dictNextFeaturesInformation = {}
        for i, feature in enumerate(self.model.features):
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                currentValue = self.updatedCurrentPoint[i]
                actionable = self.dictControllersSelectedPoint[feature].getActionable()
                content = self.dictControllersSelectedPoint[feature].getContent()

                if featureType is FeatureType.Binary:
                    value0 = content['value0']
                    value1 = content['value1']

                    dictNextFeaturesInformation[feature] = {'actionable': actionable,
                                                            'value0': value0, 
                                                            'value1': value1, 
                                                            'value': currentValue}

                elif featureType is FeatureType.Discrete or featureType is FeatureType.Numeric:
                    minimumValue = content['minimumValue']
                    maximumValue = content['maximumValue']

                    dictNextFeaturesInformation[feature] = {'actionable': actionable,
                                                            'minimumValue': minimumValue, 
                                                            'maximumValue': maximumValue, 
                                                            'value': currentValue}

                elif featureType is FeatureType.Categorical:
                    allowedValues = content['allowedValues']
                    notAllowedValues = content['notAllowedValues']
                    dictNextFeaturesInformation[feature] = {'actionable': actionable,
                                                            'allowedValues': allowedValues, 
                                                            'notAllowedValues': notAllowedValues,
                                                            'value': currentValue}

        nextIteration = IterationController(original=self.original, parent=self, model=self.model, randomForestClassifier=self.randomForestClassifier, isolationForest=self.isolationForest)
        iterationName = self.original.view.addNewIterationTab(nextIteration.view)
        dictNextFeaturesInformation['iterationName'] = iterationName
        nextIteration.setFeaturesAndValues(dictNextFeaturesInformation)
        nextIteration.setSuggestedFeaturesToPlot(self.__suggestedFeaturesToPlot)

        self.restorCursor()

    def __handlerFinishIteration(self):
        pass

    # this function is used to change the default cursor to wait cursor
    def waitCursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
    
    # this function is used to restor the default cursor
    def restorCursor(self):
        # updating cursor
        QApplication.restoreOverrideCursor()

    def __errorPlotHandler(self, featureError):
        QMessageBox.information(self.view, 'User Information', 'The graph could not be updated, because some constraint is contradictory at feature '+featureError, QMessageBox.Ok)
