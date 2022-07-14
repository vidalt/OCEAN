# Author: Moises Henrique Pereira

import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread
# Load UI functions
from ui.interface.InterfaceController import InterfaceController
from .IterationView import IterationView
from .FinalIteration.FinalIterationController import FinalIterationController
from ui.engine.CounterfactualEngine import CounterfactualEngine
from ui.interface.ComboboxList.ComboboxListController import ComboboxListController
from ui.interface.DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ui.interface.Slider3Ranges.Slider3RangesController import Slider3RangesController
from ..IterativeWorker import IterativeWorker
# Load OCEAN functions
from src.CounterFactualParameters import FeatureType


class IterationController(InterfaceController):
    """
    Handle the logic over the interface, interacting with model,
    view and worker also taking the selected dataset informations
    from model to send to counterfactual generator in worker class
    """

    def __init__(self, original, parent, model, rfClassifier, isolationForest):
        super().__init__()
        self.original = original
        self.parent = parent
        self.rfClassifier = rfClassifier
        self.isolationForest = isolationForest

        self.view = IterationView()
        self.view.outdatedGraph.connect(self.__onOutdatedGraph)
        self.model = model

        self.iterationName = None

        self.updatedCurrentPoint = None
        self.updatedCurrentClass = None
        self.predictedOriginalClassPercentage = None

        self.counterfactualToPlot = None

        self.__counterfactualStep = None

        self.__canvas = self.view.getCanvas()
        self.__canvas.updatedPoint.connect(self.__onUpdatedCurrentPoint)
        self.__canvas.lastFeatureClicked.connect(
            self.__lastFeatureClickedHandler)
        self.__canvas.errorPlot.connect(self.__errorPlotHandler)

        self.__canvasDistribution = self.view.getCanvasDistribution()

        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None
        self.transformedSamplesClassesPercentage = None

        self.__suggestedFeaturesToPlot = None

        self.view.selectedFeatures.connect(lambda: self.__updateGraph())
        self.view.nextIteration.connect(lambda: self.__handlerNextIteration())
        self.view.finishIteration.connect(
            lambda: self.__handlerFinishIteration())

        self.__runningCounterfactual = False

    def __onOutdatedGraph(self):
        self.__updateGraph()

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

                    componentController = DoubleRadioButtonController(
                        self.view)
                    componentController.initializeView(
                        feature, str(value0), str(value1))
                    componentController.setSelectedValue(value)
                    componentController.setActionable(actionable)

                elif featureType is FeatureType.Discrete:
                    actionable = content['actionable']
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    componentController = Slider3RangesController(
                        self.view, smaller=True)
                    componentController.initializeView(
                        feature, minValue, maxValue, decimalPlaces=0)
                    componentController.setSelectedValue(value)
                    componentController.setActionable(actionable)

                elif featureType is FeatureType.Numeric:
                    actionable = content['actionable']
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    componentController = Slider3RangesController(
                        self.view, smaller=True)
                    componentController.initializeView(
                        feature, minValue, maxValue)
                    componentController.setSelectedValue(value)
                    componentController.setActionable(actionable)

                elif featureType is FeatureType.Categorical:
                    actionable = content['actionable']
                    allowedValues = content['allowedValues']
                    allPossibleValues = content['allPossibleValues']
                    value = content['value']

                    componentController = ComboboxListController(
                        self.view, allPossibleValues)
                    componentController.initializeView(feature, allowedValues)
                    componentController.setSelectedValue(value)
                    componentController.setActionable(actionable)

                # disabling the edition
                componentController.disableComponent()
                # connecting the signals
                componentController.outdatedGraph.connect(
                    self.__onOutdatedGraph)
                # hiding the components
                componentController.view.checkBoxActionability.hide()
                self.view.addFeatureWidget(feature, componentController.view)
                # saving the controller to facilitate the access to components
                self.initPointFeatures[feature] = componentController

        self.view.addFeaturesOptions(list(self.model.features[:-1]))
        self.__calculateClass()

    def setSuggestedFeaturesToPlot(self, suggestedFeatures):
        """Draw the suggested features."""
        self.__suggestedFeaturesToPlot = suggestedFeatures
        self.view.selectFeatures(self.__suggestedFeaturesToPlot)
        self.__updateGraph(self.__suggestedFeaturesToPlot)

    def setCounterfactual(self, countefactual):
        self.counterfactualToPlot = countefactual

    def __setActionable(self, features):
        """Set the actionability to the components."""
        for feature in self.model.features:
            if feature != 'Class':
                self.initPointFeatures[feature].setActionable(False)
        for feature in features:
            self.initPointFeatures[feature].setActionable(True)

    def getCurrentDataframeAllowance(self, selectedFeatures, currentDataframe):
        """
        Checks if the current datapoint is allowed
        considering the user's constraints.
        """
        allowed = True
        allowedDict = {}
        for feature in selectedFeatures:
            featureType = self.model.featuresInformations[feature]['featureType']

            currentValue = currentDataframe[feature].tolist()[0]
            content = self.initPointFeatures[feature].getContent()

            if featureType is FeatureType.Binary:
                allowedDict[feature] = True

            elif featureType in [FeatureType.Discrete, FeatureType.Numeric]:
                minValue = float(content['minimumValue'])
                maxValue = float(content['maximumValue'])
                currentValue = float(currentValue)

                allowedDict[feature] = True
                if currentValue < minValue or currentValue > maxValue:
                    allowed = False
                    allowedDict[feature] = False

            elif featureType is FeatureType.Categorical:
                allowedValues = content['allowedValues']

                allowedDict[feature] = True
                if currentValue not in allowedValues:
                    allowed = False
                    allowedDict[feature] = False

        # enable/disable nextIteration
        if self.__runningCounterfactual:
            allowed = False

        self.view.enabledNextIteration(allowed)

        return allowedDict

    # listen the updated point to redraw the graph
    def __onUpdatedCurrentPoint(self, updatedPoint):
        self.waitCursor()

        selectedFeatures = self.view.getSelectedFeatures()
        if len(selectedFeatures) == len(updatedPoint):
            self.__suggestedFeaturesToPlot = selectedFeatures
            # current datapoint
            currentDataPoint = self.chosenDataPoint.copy()
            currentDataPoint = np.append(
                currentDataPoint, self.predictedOriginalClass)
            currentDataframe = pd.DataFrame(
                data=[currentDataPoint], columns=self.model.features)
            # updating the values
            for i, f in enumerate(selectedFeatures):
                currentDataframe[f] = updatedPoint[i]

            transformedCurrent = self.model.transformDataPoint(
                currentDataframe.to_numpy()[0][:-1])
            predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(
                self.rfClassifier, [transformedCurrent])
            predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                self.rfClassifier, [transformedCurrent])

            # adding the updated prediction class, and percentage
            currentDataframe['Class'] = predictedCurrentClass[0]
            currentDataframe['prob1'] = predictedCurrentClassPercentage[0][1]

            # saving the updated current point values
            # only needs the features values without the Class and prob1
            self.updatedCurrentPoint = currentDataframe.to_numpy()[0][:-2]
            self.updatedCurrentClass = predictedCurrentClass[0]

            # getting the initial datapoint, keeping a historic
            parentDataPoint = self.original.chosenDataPoint.copy()
            parentDataPoint = np.append(
                parentDataPoint, self.original.predictedOriginalClass)
            # transforming the parent datapoint to predict its class
            transformedParentDataPoint = self.model.transformDataPoint(
                parentDataPoint[:-1])
            # predicting the parent datapoint class probabilities
            predictedParentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                self.rfClassifier, [transformedParentDataPoint])
            # building the parent dataframe
            parentDataframe = pd.DataFrame(
                data=[parentDataPoint], columns=self.model.features)
            # adding the prediction percentage
            parentDataframe['prob1'] = predictedParentClassPercentage[0][1]

            lastScenarioDataframe = None
            if hasattr(self.parent, 'updatedCurrentPoint'):
                # getting the last scenario datapoint, keeping a historic
                lastScenarioDataPoint = self.parent.updatedCurrentPoint.copy()
                # transforming the parent datapoint to predict its class
                transformedLastScenarioDataPoint = self.model.transformDataPoint(
                    lastScenarioDataPoint)
                # predicting the parent datapoint class and probabilities
                predictedLastScenarioClass = CounterfactualEngine.randomForestClassifierPredict(
                    self.rfClassifier, [transformedLastScenarioDataPoint])
                predictedLastScenarioClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                    self.rfClassifier, [transformedLastScenarioDataPoint])
                # adding class
                lastScenarioDataPoint = np.append(
                    lastScenarioDataPoint, predictedLastScenarioClass[0])
                # building the parent dataframe
                lastScenarioDataframe = pd.DataFrame(
                    data=[lastScenarioDataPoint], columns=self.model.features)
                # adding the prediction percentage
                lastScenarioDataframe['prob1'] = predictedLastScenarioClassPercentage[0][1]

            lastScenarioName = None
            if hasattr(self.parent, 'iterationName'):
                lastScenarioName = self.parent.iterationName

            # adding the counterfactual
            counterfactualToPlotDataframe = pd.DataFrame(
                data=[self.counterfactualToPlot], columns=self.model.features)
            transformedCounterfactualDataPoint = self.model.transformDataPoint(
                self.counterfactualToPlot[:-1])
            predictedCounterfactualClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                self.rfClassifier, [transformedCounterfactualDataPoint])
            counterfactualToPlotDataframe['prob1'] = predictedCounterfactualClassPercentage[0][1]

            # get current point allowance
            currentDataframeAllowance = self.getCurrentDataframeAllowance(
                selectedFeatures, currentDataframe)

            # parameters to update graph
            parameters = {'controller': self,
                          'currentPoint': currentDataframe,
                          'currentDataframeAllowance': currentDataframeAllowance,
                          'originalPoint': parentDataframe,
                          'lastScenarioPoint': lastScenarioDataframe,
                          'lastScenarioName': lastScenarioName,
                          'counterfactualPoint': counterfactualToPlotDataframe,
                          'selectedFeatures': selectedFeatures}
            self.__canvas.updateGraph(parameters)

        self.restorCursor()

    # listen the last feature clicked and draw the distribution graph, and show the feature informations
    def __lastFeatureClickedHandler(self, featureIndex):
        selectedFeatures = self.view.getSelectedFeatures()

        if featureIndex is None:
            return
        if featureIndex < 0 or featureIndex > len(selectedFeatures)-1:
            return

        feature = selectedFeatures[featureIndex]

        parameters = {'controller': self, 'featureToPlot': feature}
        self.__canvasDistribution.updateGraphDistribution(parameters)

        self.view.showItemByFeature(feature)

    # this function takes the selected data point and calculate the respective class
    def __calculateClass(self):
        # getting the datapoint
        auxiliarDataPoint = []
        for feature in self.model.features:
            if feature != 'Class':
                content = self.initPointFeatures[feature].getContent(
                )
                auxiliarDataPoint.append(content['value'])

        self.chosenDataPoint = np.array(auxiliarDataPoint)

        # transforming the datapoint to predict its class
        self.transformedChosenDataPoint = self.model.transformDataPoint(
            self.chosenDataPoint)

        # predicting the datapoint class and showing its value
        self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(
            self.rfClassifier, [self.transformedChosenDataPoint])
        self.predictedOriginalClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
            self.rfClassifier, [self.transformedChosenDataPoint])

    # this function updates the graph with the original, current, last, and the counterfactual points
    def __updateGraph(self, suggestedFeatures=None):
        self.waitCursor()

        if self.counterfactualToPlot is not None:
            selectedFeatures = None
            if suggestedFeatures is not None:
                selectedFeatures = suggestedFeatures
            else:
                selectedFeatures = self.view.getSelectedFeatures()

            if len(selectedFeatures) != 0:
                self.__suggestedFeaturesToPlot = selectedFeatures

                # getting the current datapoint
                auxiliarDataPoint = []
                for feature in self.model.features:
                    if feature != 'Class':
                        content = self.initPointFeatures[feature].getContent(
                        )
                        auxiliarDataPoint.append(content['value'])

                self.chosenDataPoint = np.array(auxiliarDataPoint)

                # transforming the datapoint to predict its class
                self.transformedChosenDataPoint = self.model.transformDataPoint(
                    self.chosenDataPoint)

                # predicting the datapoint class and showing its value
                self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(
                    self.rfClassifier, [self.transformedChosenDataPoint])
                self.predictedOriginalClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                    self.rfClassifier, [self.transformedChosenDataPoint])

                # saving the updated current point values and class
                if self.updatedCurrentPoint is None:
                    self.updatedCurrentPoint = self.chosenDataPoint.copy()
                    self.updatedCurrentClass = self.predictedOriginalClass[0]

                # current datapoint
                currentDataPoint = self.updatedCurrentPoint.copy()
                currentDataPoint = np.append(
                    currentDataPoint, self.predictedOriginalClass)
                currentDataframe = pd.DataFrame(
                    data=[currentDataPoint], columns=self.model.features)
                # adding the prediction percentage
                if self.updatedCurrentPoint is None:
                    currentDataframe['prob1'] = self.predictedOriginalClassPercentage[0][1]
                else:
                    transformedCurrentDataPoint = self.model.transformDataPoint(
                        self.updatedCurrentPoint)
                    predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(
                        self.rfClassifier, [transformedCurrentDataPoint])
                    predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                        self.rfClassifier, [transformedCurrentDataPoint])

                    currentDataframe['Class'] = predictedCurrentClass[0]
                    currentDataframe['prob1'] = predictedCurrentClassPercentage[0][1]

                # getting the initial datapoint, keeping a historic
                parentDataPoint = self.original.chosenDataPoint.copy()
                parentDataPoint = np.append(
                    parentDataPoint, self.original.predictedOriginalClass)
                # transforming the parent datapoint to predict its class
                transformedParentDataPoint = self.model.transformDataPoint(
                    parentDataPoint[:-1])
                # predicting the parent datapoint class probabilities
                predictedParentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                    self.rfClassifier, [transformedParentDataPoint])
                # building the parent dataframe
                parentDataframe = pd.DataFrame(
                    data=[parentDataPoint], columns=self.model.features)
                # adding the prediction percentage
                parentDataframe['prob1'] = predictedParentClassPercentage[0][1]

                # getting the last scenario datapoint
                lastScenarioDataframe = None
                if hasattr(self.parent, 'updatedCurrentPoint'):
                    # getting the last scenario datapoint, keeping a historic
                    lastScenarioDataPoint = self.parent.updatedCurrentPoint.copy()
                    # transforming the parent datapoint to predict its class
                    transformedLastScenarioDataPoint = self.model.transformDataPoint(
                        lastScenarioDataPoint)
                    # predicting the parent datapoint class and probabilities
                    predictedLastScenarioClass = CounterfactualEngine.randomForestClassifierPredict(
                        self.rfClassifier, [transformedLastScenarioDataPoint])
                    predictedLastScenarioClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                        self.rfClassifier, [transformedLastScenarioDataPoint])
                    # adding class
                    lastScenarioDataPoint = np.append(
                        lastScenarioDataPoint, predictedLastScenarioClass[0])
                    # building the parent dataframe
                    lastScenarioDataframe = pd.DataFrame(
                        data=[lastScenarioDataPoint], columns=self.model.features)
                    # adding the prediction percentage
                    lastScenarioDataframe['prob1'] = predictedLastScenarioClassPercentage[0][1]

                lastScenarioName = None
                if hasattr(self.parent, 'iterationName'):
                    lastScenarioName = self.parent.iterationName

                # adding the counterfactual
                counterfactualToPlotDataframe = pd.DataFrame(
                    data=[self.counterfactualToPlot], columns=self.model.features)
                transformedCounterfactualDataPoint = self.model.transformDataPoint(
                    self.counterfactualToPlot[:-1])
                predictedCounterfactualClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(
                    self.rfClassifier, [transformedCounterfactualDataPoint])
                counterfactualToPlotDataframe['prob1'] = predictedCounterfactualClassPercentage[0][1]

                # get current point allowance
                currentDataframeAllowance = self.getCurrentDataframeAllowance(
                    selectedFeatures, currentDataframe)

                # parameters to update graph
                parameters = {'controller': self,
                              'currentPoint': currentDataframe,
                              'currentDataframeAllowance': currentDataframeAllowance,
                              'originalPoint': parentDataframe,
                              'lastScenarioPoint': lastScenarioDataframe,
                              'lastScenarioName': lastScenarioName,
                              'counterfactualPoint': counterfactualToPlotDataframe,
                              'selectedFeatures': selectedFeatures}
                self.__canvas.updateGraph(parameters)

        self.restorCursor()

    def getCounterfactualExplanation(self, counterfactual):
        self.__runningCounterfactual = False

        self.view.enabledNextIteration(True)

        dictNextFeaturesInformation = {}
        for i, feature in enumerate(self.model.features):
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                actionable = self.initPointFeatures[feature].getActionable(
                )
                content = self.initPointFeatures[feature].getContent(
                )
                currentValue = self.updatedCurrentPoint[i]

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
                    allPossibleValues = content['allPossibleValues']
                    dictNextFeaturesInformation[feature] = {'actionable': actionable,
                                                            'allowedValues': allowedValues,
                                                            'notAllowedValues': notAllowedValues,
                                                            'allPossibleValues': allPossibleValues,
                                                            'value': currentValue}

        self.view.blockSignals(True)
        nextIteration = IterationController(original=self.original, parent=self, model=self.model,
                                            rfClassifier=self.rfClassifier, isolationForest=self.isolationForest)
        iterationName = self.original.view.addNewIterationTab(
            nextIteration.view)
        dictNextFeaturesInformation['iterationName'] = iterationName
        nextIteration.setFeaturesAndValues(dictNextFeaturesInformation)
        nextIteration.setCounterfactual(counterfactual)

        suggestedFeatures = []
        for ind, feature in enumerate(self.model.features):
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                if featureType is FeatureType.Discrete or featureType is FeatureType.Numeric:
                    if float(self.chosenDataPoint[ind]) != float(counterfactual[ind]):
                        suggestedFeatures.append(feature)

                else:
                    if str(self.chosenDataPoint[ind]) != str(counterfactual[ind]):
                        suggestedFeatures.append(feature)

        self.__suggestedFeaturesToPlot = suggestedFeatures
        nextIteration.setSuggestedFeaturesToPlot(
            self.__suggestedFeaturesToPlot)
        self.view.blockSignals(False)
        self.restorCursor()

    def handlerCounterfactualSteps(self, step=None):
        if step is None and self.__counterfactualStep is not None:
            self.__counterfactualStep.done(1)
            self.__counterfactualStep = None

        elif self.__counterfactualStep is None:
            self.__counterfactualStep = QMessageBox(self.view)
            self.__counterfactualStep.setWindowTitle(
                'Counterfactual information')
            self.__counterfactualStep.setStandardButtons(QMessageBox.Ok)
            self.__counterfactualStep.setText(step)
            result = self.__counterfactualStep.exec()

            if result == QMessageBox.Ok:
                self.__counterfactualStep = None

        else:
            self.__counterfactualStep.setText(step)

    def handlerCounterfactualError(self):
        self.__runningCounterfactual = False

        if self.__counterfactualStep is not None:
            self.__counterfactualStep.done(1)
            self.__counterfactualStep = None

        QMessageBox.information(self.view, 'Counterfactual error',
                                'It was not possible to generate the counterfactual with those constraints', QMessageBox.Ok)
        self.view.enabledNextIteration(True)
        self.restorCursor()

    # this function generates the counterfactual given the current point
    def __generateCounterfactualAndNextIteration(self):
        self.__runningCounterfactual = True
        # running the counterfactual generation in another thread
        self.thread = QThread(self.view)
        self.worker = IterativeWorker(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        self.worker.counterfactualDataframe.connect(
            self.getCounterfactualExplanation)
        self.worker.counterfactualSteps.connect(
            self.handlerCounterfactualSteps)
        self.worker.counterfactualError.connect(
            self.handlerCounterfactualError)
        self.worker.finished.connect(self.restorCursor)
        # self.worker.finished.connect(self.handlerCounterfactualSteps)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def __handlerNextIteration(self):
        self.waitCursor()
        self.view.enabledNextIteration(False)

        self.__generateCounterfactualAndNextIteration()

    def __handlerFinishIteration(self):
        self.waitCursor()

        counterfactualComparison = []
        for i, feature in enumerate(self.model.features):
            if feature != 'Class':
                counterfactualComparison.append(
                    [feature, str(self.original.chosenDataPoint[i]),
                     str(self.updatedCurrentPoint[i])])

        finalIteration = FinalIterationController(self)
        finalIteration.updateComparisonLabel(
            'Comparison between the Scenario0 and the '+self.iterationName)
        finalIteration.setViewScenariosName('Scenario0', self.iterationName)
        finalIteration.updateClasses(
            self.predictedOriginalClass[0], self.updatedCurrentClass)
        finalIteration.updateCounterfactualTable(counterfactualComparison)
        self.original.addFinalIteration(finalIteration.view)

        self.restorCursor()

    def __errorPlotHandler(self, featureError):
        QMessageBox.information(
            self.view, 'Invalid feature value',
            'The graph could not be updated, because some constraint'
            ' is contradictory at feature '+featureError,
            QMessageBox.Ok)
