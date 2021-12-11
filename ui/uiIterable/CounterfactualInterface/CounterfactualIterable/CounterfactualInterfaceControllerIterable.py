import numpy as np
import pandas as pd

from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from ..CounterfactualInterfaceModel import CounterfactualInterfaceModel
from ..CounterfactualInterfaceEnums import CounterfactualInterfaceEnums
from .CounterfactualInterfaceViewIterable import CounterfactualInterfaceViewIterable

from CounterfactualEngine.CounterfactualEngine import CounterfactualEngine

from Canvas.CanvasController import CanvasController

from CounterFactualParameters import FeatureType

from ..ComboboxList.ComboboxListController import ComboboxListController
from ..DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ..Slider3Ranges.Slider3RangesController import Slider3RangesController

from .CounterfactualInferfaceWorkerIterable import CounterfactualInferfaceWorkerIterable

from .Iteration.IterationController import IterationController

class CounterfactualInterfaceControllerIterable:

    def __init__(self):
        self.view = CounterfactualInterfaceViewIterable()
        self.model = CounterfactualInterfaceModel()

        self.__initializeView()

        self.__chosenDataset = CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value
        self.view.chosenDataset.connect(self.__handlerChosenDataset)
        self.view.randomPoint.connect(self.__handlerRandomPoint)
        self.view.calculateClass.connect(self.__handlerCalculateClass)
        self.view.nextIteration.connect(self.__handlerNextIteration)

        self.randomForestClassifier = None
        self.isolationForest = None

        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None
        self.__dataframeChosenDataPoint = None
        self.predictedOriginalClass = None

        self.__canvas = None

        self.dictControllersSelectedPoint = {}
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None

        self.__suggestedFeaturesToPlot = None
        
        self.__counterfactualStep = None


    # this function takes the dataframe names and send them to interface
    def __initializeView(self):
        # self.__canvas.updateGraph()

        datasets = self.model.getDatasetsName()

        datasetsName = [CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value]
        for datasetName in datasets:
            auxDatasetName = datasetName.split('.')[0]
            datasetsName.append(auxDatasetName)

        self.view.initializeView(datasetsName)

    # this function opens the selected dataset
    # trains the random forest and the isolation forest,
    # and present the features components and its respectives informations
    def __handlerChosenDataset(self):
        self.view.enableNext(False)

        # getting the name of the desired dataset
        self.__chosenDataset = self.view.getChosenDataset()
        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            # cleaning the view
            self.view.clearView()
            self.dictControllersSelectedPoint.clear()

            # opening the desired dataset
            self.model.openChosenDataset(self.__chosenDataset)
            xTrain, yTrain = self.model.getTrainData()

            # training the random forest and isolation forest models
            if xTrain is not None and yTrain is not None: 
                self.randomForestClassifier = CounterfactualEngine.trainRandomForestClassifier(xTrain, yTrain)
                self.isolationForest = CounterfactualEngine.trainIsolationForest(xTrain)

                # plot the features importance
                importance = self.randomForestClassifier.feature_importances_
                importances = pd.DataFrame(data={
                    'features': self.model.transformedFeaturesOrdered,
                    'importance': importance
                })
                importances = importances.sort_values(by='importance', ascending=False)

                parameters = {'dataframe': importances, 'xVariable': 'features', 'yVariable': 'importance'}

                self.__canvas = self.view.getCanvas()
                self.__canvas.updateFeatureImportanceGraph(parameters)

                # featureImportance = self.model.invertTransformedFeatureImportance(importance)
                # tempSuggestedFeatureToPlot = []
                # for i in range(4):
                #     index = featureImportance.index(max(featureImportance))
                #     tempSuggestedFeatureToPlot.append(self.model.features[index])
                #     featureImportance[index] = -1

                # suggestedFeatureToPlot = []
                # for feature in self.model.features:
                #     if feature in tempSuggestedFeatureToPlot:
                #         suggestedFeatureToPlot.append(feature)

                # self.__suggestedFeaturesToPlot = suggestedFeatureToPlot

            # showing the features components and informations
            for feature in self.model.features:
                if feature != 'Class':
                    featureType = self.model.featuresInformations[feature]['featureType']
                    componentController = None
                    if featureType is FeatureType.Binary:
                        value0 = self.model.featuresInformations[feature]['value0']
                        value1 = self.model.featuresInformations[feature]['value1']

                        componentController = DoubleRadioButtonController(self.view)
                        componentController.initializeView(feature, str(value0), str(value1))

                    elif featureType is FeatureType.Discrete:
                        minValue = self.model.featuresInformations[feature]['min']
                        maxValue = self.model.featuresInformations[feature]['max']

                        # componentController = LineEditMinimumMaximumController(self.view)
                        componentController = Slider3RangesController(self.view)
                        componentController.initializeView(feature, minValue, maxValue, decimalPlaces=0)

                    elif featureType is FeatureType.Numeric:
                        minValue = self.model.featuresInformations[feature]['min']
                        maxValue = self.model.featuresInformations[feature]['max']

                        # componentController = LineEditMinimumMaximumController(self.view)
                        componentController = Slider3RangesController(self.view)
                        componentController.initializeView(feature, minValue, maxValue)
                        
                    elif featureType is FeatureType.Categorical:
                        componentController = ComboboxListController(self.view, self.model.featuresInformations[feature]['possibleValues'])
                        componentController.initializeView(feature, self.model.featuresInformations[feature]['possibleValues'])

                    # adding the view to selectedPoint component
                    componentController.view.checkBoxActionability.hide()
                    self.view.addFeatureWidget(componentController.view)
                    # saving the controller to facilitate the access to components
                    self.dictControllersSelectedPoint[feature] = componentController

        else:
            # cleaning the view
            self.view.clearView()
            self.dictControllersSelectedPoint.clear()

    # this function get a random datapoint from dataset 
    def __handlerRandomPoint(self):
        self.view.clearClass()

        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            randomDataPoint = self.model.getRandomPoint(self.randomForestClassifier)
            randomDataPoint = ['GP', 'M', '17', 'U', 'LE3', 'T', '4', '4', 'teacher', 
                               'health', 'reputation', 'mother', '1', '2', '0', 'no', 
                               'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', '3', '3', 
                               '3', '1', '2', '2', '0']

            # showing the values in their respective component
            for index, feature in enumerate(self.model.features):
                if feature != 'Class':
                    self.dictControllersSelectedPoint[feature].setSelectedValue(randomDataPoint[index])

            self.view.enableNext(False)

    # this function takes the selected data point and calculate the respective class
    def __handlerCalculateClass(self):
        self.view.clearClass()
        
        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            auxiliarFeatureName = ''
            try:
                # getting the datapoint
                auxiliarDataPoint = []
                for feature in self.model.features:
                    if feature != 'Class':
                        auxiliarFeatureName = feature
                        content = self.dictControllersSelectedPoint[feature].getContent()
                        auxiliarDataPoint.append(content['value'])
                        
                self.chosenDataPoint = np.array(auxiliarDataPoint)

                # transforming the datapoint to predict its class
                self.transformedChosenDataPoint = self.model.transformDataPoint(self.chosenDataPoint)
                
                # predicting the datapoint class and showing its value
                self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [self.transformedChosenDataPoint])
                self.view.showOriginalClass(self.predictedOriginalClass[0])   
                self.view.enableNext(True)
                
            except:
                QMessageBox.information(self.view, 'Missing value', 'Please verify the following feature '+auxiliarFeatureName, QMessageBox.Ok)

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

                actionable = self.dictControllersSelectedPoint[feature].getActionable()
                content = self.dictControllersSelectedPoint[feature].getContent()
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
        
        nextIteration = IterationController(original=self, parent=self, model=self.model, randomForestClassifier=self.randomForestClassifier, isolationForest=self.isolationForest)
        iterationName = self.addNewIterationTab(nextIteration.view)
        dictNextFeaturesInformation['iterationName'] = iterationName
        nextIteration.setFeaturesAndValues(dictNextFeaturesInformation)
        nextIteration.setCounterfactual(counterfactual)
        
        # print('#'*75)
        # print(len(self.chosenDataPoint), '---', self.chosenDataPoint)
        # print(len(counterfactual), '---', counterfactual)
        # print('#'*75)

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
        nextIteration.setSuggestedFeaturesToPlot(self.__suggestedFeaturesToPlot)
        self.restorCursor()
        
    def handlerCounterfactualSteps(self, step=None):
        if step is None and self.__counterfactualStep is not None:
            self.__counterfactualStep.done(1)
            self.__counterfactualStep = None
        
        elif self.__counterfactualStep is None:
            self.__counterfactualStep = QMessageBox(self.view)
            # self.__counterfactualStep.setWindowFlags(self.__counterfactualStep.windowFlags() | Qt.FramelessWindowHint)
            self.__counterfactualStep.setWindowTitle('Counterfactual information')
            self.__counterfactualStep.setStandardButtons(QMessageBox.Ok)
            self.__counterfactualStep.setText(step)
            result = self.__counterfactualStep.exec()
            
            print('RESULT:', result)
            if result == QMessageBox.Ok:
                self.__counterfactualStep = None
                
        else: 
            self.__counterfactualStep.setText(step)

    def handlerCounterfactualError(self):
        if self.__counterfactualStep is not None:
            self.__counterfactualStep.done(1)
            self.__counterfactualStep = None
            
        QMessageBox.information(self.view, 'Counterfactual error', 'It was not possible to generate the counterfactual with those constraints', QMessageBox.Ok)
        self.view.enableNext(True)
        self.restorCursor()

    # this function generates the counterfactual given the current point
    def __generateCounterfactualAndNextIteration(self):
        # running the counterfactual generation in another thread
        self.thread = QThread(self.view)
        self.worker = CounterfactualInferfaceWorkerIterable(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.counterfactualDataframe.connect(self.getCounterfactualExplanation)
        self.worker.counterfactualSteps.connect(self.handlerCounterfactualSteps)
        self.worker.counterfactualError.connect(self.handlerCounterfactualError)
        # self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.restorCursor)
        # self.worker.finished.connect(self.handlerCounterfactualSteps)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def __handlerNextIteration(self):
        self.waitCursor()
        self.view.enableNext(False)

        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            self.__generateCounterfactualAndNextIteration()

    # this function is used to change the default cursor to wait cursor
    def waitCursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
    
    # this function is used to restor the default cursor
    def restorCursor(self):
        # updating cursor
        QApplication.restoreOverrideCursor()