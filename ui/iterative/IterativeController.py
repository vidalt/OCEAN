import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
# Load UI functions
from .IterativeWorker import IterativeWorker
from .Iteration.IterationController import IterationController
from .IterativeViewer import IterativeViewer
from ui.interface.InterfaceModel import InterfaceModel
from ui.interface.InterfaceEnums import InterfaceEnums
from ui.interface.ComboboxList.ComboboxListController import ComboboxListController
from ui.interface.DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ui.interface.Slider3Ranges.Slider3RangesController import Slider3RangesController
from ui.engine.CounterfactualEngine import CounterfactualEngine
# Load OCEAN functions
from src.CounterFactualParameters import FeatureType


class IterativeController:

    def __init__(self):
        self.view = IterativeViewer()
        self.model = InterfaceModel()

        self.__initializeView()

        self.__chosenDataset = InterfaceEnums.SelectDataset.DEFAULT.value
        self.view.chosenDataset.connect(self.__handlerChosenDataset)
        self.view.randomPoint.connect(self.__handlerRandomPoint)
        self.view.calculateClass.connect(self.__handlerCalculateClass)
        self.view.nextIteration.connect(self.__handlerNextIteration)

        self.rfClassifier = None
        self.isolationForest = None

        self.chosenDataPoint = None
        self.__dataframeChosenDataPoint = None
        self.predictedOriginalClass = None

        self.__canvas = None

        self.transformedChosenDataPoint = None
        self.initPointFeatures = {}
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None

        self.__suggestedFeaturesToPlot = None

        self.__counterfactualStep = None

    # this function takes the dataframe names and send them to interface

    def __initializeView(self):
        datasets = self.model.getDatasetsName()

        datasetsName = [InterfaceEnums.SelectDataset.DEFAULT.value]
        for datasetName in datasets:
            auxDatasetName = datasetName.split('.')[0]
            datasetsName.append(auxDatasetName)

        self.view.initializeView(datasetsName)

    def __handlerChosenDataset(self):
        """
        Open the selected dataset, train a random forest and an isolation
        forest, and plot feature importance as a bar chart.
        """
        self.view.enableNext(False)

        # Get the name of the selected dataset
        self.__chosenDataset = self.view.getChosenDataset()
        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            # Clean the view
            self.view.clearView()
            self.initPointFeatures.clear()
            # Open the selected dataset
            self.model.openChosenDataset(self.__chosenDataset)
            xTrain, yTrain = self.model.getTrainData()

            # Training the random forest and isolation forest models
            if xTrain is not None and yTrain is not None:
                self.rfClassifier = CounterfactualEngine.trainRandomForestClassifier(
                    xTrain, yTrain)
                self.isolationForest = CounterfactualEngine.trainIsolationForest(
                    xTrain)

                # --- Plot the features importance ---
                featureImportance = self.rfClassifier.feature_importances_
                featureImportance = pd.DataFrame(data={
                    'features': self.model.transformedFeaturesOrdered,
                    'importance': featureImportance})
                featureImportance = featureImportance.sort_values(
                    by='importance', ascending=False)
                # Plot on widget
                self.__canvas = self.view.getCanvas()
                self.__canvas.updateFeatureImportanceGraph(
                    featureImportance, 'features', 'importance')

            # ------ Show features components and informations ------
            for feature in self.model.features:
                if feature != 'Class':
                    featureInformations = self.model.featuresInformations[feature]
                    featureType = featureInformations['featureType']
                    componentController = None
                    if featureType is FeatureType.Binary:
                        value0 = featureInformations['value0']
                        value1 = featureInformations['value1']
                        componentController = DoubleRadioButtonController(
                            self.view)
                        componentController.initializeView(
                            feature, str(value0), str(value1))
                    elif featureType is FeatureType.Discrete:
                        minValue = featureInformations['min']
                        maxValue = featureInformations['max']
                        componentController = Slider3RangesController(
                            self.view)
                        componentController.initializeView(
                            feature, minValue, maxValue, decimalPlaces=0)
                    elif featureType is FeatureType.Numeric:
                        minValue = featureInformations['min']
                        maxValue = featureInformations['max']
                        componentController = Slider3RangesController(
                            self.view)
                        componentController.initializeView(
                            feature, minValue, maxValue)
                    elif featureType is FeatureType.Categorical:
                        componentController = ComboboxListController(
                            self.view, featureInformations['possibleValues'])
                        componentController.initializeView(
                            feature, featureInformations['possibleValues'])

                    # Add the view to selectedPoint component
                    componentController.view.checkBoxActionability.hide()
                    self.view.addFeatureWidget(componentController.view)
                    # Save the controller to facilitate access to components
                    self.initPointFeatures[feature] = componentController
        else:
            # cleaning the view
            self.view.clearView()
            self.initPointFeatures.clear()

    # this function get a random datapoint from dataset
    def __handlerRandomPoint(self):
        self.view.clearClass()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            randomDataPoint = self.model.getRandomPoint(
                self.rfClassifier)
            randomDataPoint = ['4', 'male', '2',
                               'own', '1', '2', '5', '5', 'business']

            # showing the values in their respective component
            for index, feature in enumerate(self.model.features):
                if feature != 'Class':
                    self.initPointFeatures[feature].setSelectedValue(
                        randomDataPoint[index])

            self.view.enableNext(False)

    def __handlerCalculateClass(self):
        """
        Calculate the class of the selected data point.
        """
        self.view.clearClass()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            featureName = ''
            try:
                # Get the datapoint
                dataPoint = []
                for feature in self.model.features:
                    if feature != 'Class':
                        featureName = feature
                        content = self.initPointFeatures[feature].getContent()
                        dataPoint.append(content['value'])
                self.chosenDataPoint = np.array(dataPoint)

                # transforming the datapoint to predict its class
                self.transformedChosenDataPoint = self.model.transformDataPoint(self.chosenDataPoint)

                # predicting the datapoint class and showing its value
                self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(
                    self.rfClassifier, [self.transformedChosenDataPoint])
                self.view.showOriginalClass(self.predictedOriginalClass[0])
                self.view.enableNext(True)

            except:
                QMessageBox.information(
                    self.view, 'Missing value',
                    'Please verify the following feature '
                    + featureName, QMessageBox.Ok)

    def addNewIterationTab(self, nextIterationView):
        iterationName = self.view.addNewIterationTab(nextIterationView)

        return iterationName

    def addFinalIteration(self, finalIterationView):
        self.view.addFinalIteration(finalIterationView)

    def getCounterfactualExplanation(self, counterfactual):
        self.view.enableNext(True)

        dictNextFeaturesInformation = {}
        for i, feature in enumerate(self.model.features):
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                actionable = self.initPointFeatures[feature].getActionable(
                )
                content = self.initPointFeatures[feature].getContent(
                )
                currentValue = content['value']

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

        nextIteration = IterationController(original=self, parent=self, model=self.model,
                                            randomForestClassifier=self.rfClassifier, isolationForest=self.isolationForest)
        iterationName = self.addNewIterationTab(nextIteration.view)
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
        if self.__counterfactualStep is not None:
            self.__counterfactualStep.done(1)
            self.__counterfactualStep = None

        QMessageBox.information(self.view, 'Counterfactual error',
                                'It was not possible to generate the counterfactual with those constraints', QMessageBox.Ok)
        self.view.enableNext(True)
        self.restorCursor()

    # this function generates the counterfactual given the current point
    def __generateCounterfactualAndNextIteration(self):
        # running the counterfactual generation in another thread
        self.thread = QThread(self.view)
        self.worker = IterativeWorker(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.counterfactualDataframe.connect(
            self.getCounterfactualExplanation)
        self.worker.counterfactualSteps.connect(
            self.handlerCounterfactualSteps)
        self.worker.counterfactualError.connect(
            self.handlerCounterfactualError)
        # self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.restorCursor)
        # self.worker.finished.connect(self.handlerCounterfactualSteps)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def __handlerNextIteration(self):
        self.waitCursor()
        self.view.enableNext(False)

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            self.__generateCounterfactualAndNextIteration()

    # this function is used to change the default cursor to wait cursor
    def waitCursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)

    # this function is used to restor the default cursor
    def restorCursor(self):
        # updating cursor
        QApplication.restoreOverrideCursor()
