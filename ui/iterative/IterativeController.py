import pandas as pd
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox
# Load UI functions
from ui.interface.InterfaceController import InterfaceController
from .IterativeWorker import IterativeWorker
from .Iteration.IterationController import IterationController
from ui.iterative.IterativeViewer import IterativeViewer
from ui.interface.InterfaceEnums import InterfaceEnums
# Load OCEAN functions
from src.CounterFactualParameters import FeatureType


class IterativeController(InterfaceController):

    def __init__(self):
        super().__init__()
        self.view = IterativeViewer()
        self.initializeView()

        self.view.chosenDataset.connect(self.__handlerChosenDataset)
        self.view.randomPoint.connect(self.__handlerRandomPoint)
        self.view.calculateClass.connect(self.__handlerCalculateClass)

        self.view.nextIteration.connect(self.__handlerNextIteration)

        self.__dataframeChosenDataPoint = None
        self.__canvas = None
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None
        self.__suggestedFeaturesToPlot = None
        self.__counterfactualStep = None

        # Set each view on a tab
        self.interfaceViewer.tabWidget.addTab(
            self.view, 'Iterative Counterfactual')

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

            self.train_random_and_isolation_forests(self.__chosenDataset)
            self.show_feature_info_component()

            # --- Plot the features importance ---
            featureImportance = self.rfClassifier.feature_importances_
            featureImportance = pd.DataFrame(data={
                'features': self.model.processedFeaturesOrdered,
                'importance': featureImportance})
            featureImportance = featureImportance.sort_values(
                by='importance', ascending=False)
            # Plot on widget
            self.__canvas = self.view.getCanvas()
            self.__canvas.updateFeatureImportanceGraph(
                featureImportance, 'features', 'importance')

        else:
            # Clear the viewer
            self.view.clearView()
            self.initPointFeatures.clear()

    def __handlerRandomPoint(self):
        """
        Get a random datapoint from dataset.
        """
        self.view.clearClass()
        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            self.sample_random_point()
            self.view.enableNext(False)

    def __handlerCalculateClass(self):
        """
        Calculate the class of the selected data point.
        """
        self.view.clearClass()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            featureName = ''
            try:
                featureName = self.transform_and_predict_data_point()
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

        nextFeaturesInfo = {}
        for i, feature in enumerate(self.model.features):
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']
                actionable = self.initPointFeatures[feature].getActionable()
                content = self.initPointFeatures[feature].getContent()
                currentValue = content['value']

                if featureType is FeatureType.Binary:
                    value0 = content['value0']
                    value1 = content['value1']
                    nextFeaturesInfo[feature] = {'actionable': actionable,
                                                 'value0': value0,
                                                 'value1': value1,
                                                 'value': currentValue}

                elif featureType in [FeatureType.Discrete, FeatureType.Numeric]:
                    minimumValue = content['minimumValue']
                    maximumValue = content['maximumValue']
                    nextFeaturesInfo[feature] = {'actionable': actionable,
                                                 'minimumValue': minimumValue,
                                                 'maximumValue': maximumValue,
                                                 'value': currentValue}

                elif featureType is FeatureType.Categorical:
                    allowedValues = content['allowedValues']
                    notAllowedValues = content['notAllowedValues']
                    allPossibleValues = content['allPossibleValues']
                    nextFeaturesInfo[feature] = {
                        'actionable': actionable,
                        'allowedValues': allowedValues,
                        'notAllowedValues': notAllowedValues,
                        'allPossibleValues': allPossibleValues,
                        'value': currentValue}

        nextIteration = IterationController(
            original=self, parent=self, model=self.model,
            rfClassifier=self.rfClassifier, isolationForest=self.isolationForest)
        iterationName = self.addNewIterationTab(nextIteration.view)
        nextFeaturesInfo['iterationName'] = iterationName
        nextIteration.setFeaturesAndValues(nextFeaturesInfo)
        nextIteration.setCounterfactual(counterfactual)

        suggestedFeatures = []
        for ind, feature in enumerate(self.model.features):
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                if featureType in [FeatureType.Discrete, FeatureType.Numeric]:
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
                                'It was not possible to generate the '
                                'counterfactual with those constraints',
                                QMessageBox.Ok)
        self.view.enableNext(True)
        self.restorCursor()

    def __generateCounterfactualAndNextIteration(self):
        """
        Generate the counterfactual given the current initial point.
        """
        # Run the counterfactual generation algorithm in another thread
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
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.restorCursor)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def __handlerNextIteration(self):
        self.waitCursor()
        self.view.enableNext(False)
        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            self.__generateCounterfactualAndNextIteration()
