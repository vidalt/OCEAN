# Author: Moises Henrique Pereira
# this class handle the logic over the interface, interacting with model, view and worker
# also taking the selected dataset informations from model to send to counterfactual generator in worker class

from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from .IterationView import IterationView
from .IterationEnums import IterationEnums

from CounterFactualParameters import FeatureType

from CounterfactualEngine.CounterfactualEngine import CounterfactualEngine

# from Canvas.CanvasController import CanvasController

from ...ComboboxList.ComboboxListController import ComboboxListController
from ...DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ...Slider3Ranges.Slider3RangesController import Slider3RangesController

from ...CounterfactualInterfaceEnums import CounterfactualInterfaceEnums
from ..CounterfactualInferfaceWorkerIterable import CounterfactualInferfaceWorkerIterable

import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian

class IterationController():

    def __init__(self, parent, model, randomForestClassifier, isolationForest):
        self.parent = parent

        self.view = IterationView()
        self.model = model

        self.randomForestClassifier = randomForestClassifier
        self.isolationForest = isolationForest

        self.__initializeView()

        self.__dictControllersSelectedPoint = {}

        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None

        self.predictedCurrentClass = None
        self.predictedCurrentClassPercentage = None

        self.counterfactualToPlot = None

        self.__canvas = self.view.getCanvas()
        
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None
        self.transformedSamplesClassesPercentage = None

        self.view.selectedFeatures.connect(lambda: self.__updateGraph())


    # this function takes the dataframe names and send them to interface
    def __initializeView(self):
        self.view.initializeView()

    def setFeaturesAndValues(self, dictControllersSelectedPoint):
        for feature in self.model.features:
            if feature != 'Class':
                featureType = self.model.featuresInformations[feature]['featureType']

                content = dictControllersSelectedPoint[feature].getContent()

                if featureType is FeatureType.Binary:
                    value0 = self.model.featuresInformations[feature]['value0']
                    value1 = self.model.featuresInformations[feature]['value1']
                    value = content['value']

                    componentController = DoubleRadioButtonController(self.view)
                    componentController.initializeView(feature, str(value0), str(value1))
                    componentController.setSelectedValue(value)
                    # componentController.disableComponent()

                elif featureType is FeatureType.Discrete:
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    componentController = Slider3RangesController(self.view, smaller=True)
                    componentController.initializeView(feature, minValue, maxValue, decimalPlaces=0)
                    componentController.setSelectedValue(value)

                elif featureType is FeatureType.Numeric:
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    componentController = Slider3RangesController(self.view, smaller=True)
                    componentController.initializeView(feature, minValue, maxValue)
                    componentController.setSelectedValue(value)
                    
                elif featureType is FeatureType.Categorical:
                    value = content['value']
                    
                    componentController = ComboboxListController(self.view)
                    componentController.initializeView(feature, self.model.featuresInformations[feature]['possibleValues'])
                    componentController.setSelectedValue(value)
                    # componentController.disableComponent()

                # adding the view to selectedPoint component
                self.view.addFeatureWidget(componentController.view)
                # saving the controller to facilitate the access to components
                self.__dictControllersSelectedPoint[feature] = componentController

        self.view.addFeaturesOptions(list(self.model.features[:-1]))
        self.__calculateClass()

    # this function takes the selected data point and calculate the respective class
    def __calculateClass(self):
        self.view.clearClass()
        
        # getting the datapoint
        auxiliarDataPoint = []
        for feature in self.model.features:
            if feature != 'Class':
                content = self.__dictControllersSelectedPoint[feature].getContent()
                auxiliarDataPoint.append(content['value'])
                
        self.chosenDataPoint = np.array(auxiliarDataPoint)

        # transforming the datapoint to predict its class
        self.transformedChosenDataPoint = self.model.transformDataPoint(self.chosenDataPoint)
        
        # predicting the datapoint class and showing its value
        self.predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [self.transformedChosenDataPoint])
        self.predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [self.transformedChosenDataPoint])
        
        self.view.showCurrentClass(self.predictedCurrentClass[0])      

    def __updateGraph(self):
        if len(self.view.getSelectedFeatures()) != 0:
            self.__handlerCalculateDistances()

    def __buildDictParameters(self):
        # building the dict parameters to plot
        parameters = {}

        if self.__samplesToPlot is not None:
            # concatenating selected point with sample, and the counterfactual
            dataToPlot = pd.concat([self.__samplesToPlot, self.__dataframeChosenDataPoint, self.counterfactualToPlot])
            dataToPlot = dataToPlot.reset_index().drop(['index'], axis=1)

            parameters['dataframe'] = dataToPlot

        return parameters

    def __handlerCalculateDistances(self):
        self.waitCursor()
        
        # getting the datapoint
        auxiliarDataPoint = []
        for feature in self.model.features:
            if feature != 'Class':
                content = self.__dictControllersSelectedPoint[feature].getContent()
                auxiliarDataPoint.append(content['value'])
                
        self.chosenDataPoint = np.array(auxiliarDataPoint)

        # transforming the datapoint to predict its class
        self.transformedChosenDataPoint = self.model.transformDataPoint(self.chosenDataPoint)
        
        # predicting the datapoint class and showing its value
        self.predictedCurrentClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [self.transformedChosenDataPoint])
        self.predictedCurrentClassPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, [self.transformedChosenDataPoint])
        
        self.view.showCurrentClass(self.predictedCurrentClass[0])

        # getting a set of samples to plot
        # receive the initial and the current datapoint, keeping a historic
        parentDataPoint = self.parent.chosenDataPoint.copy()
        parentDataPoint = np.append(parentDataPoint, self.parent.predictedOriginalClass)
        self.__samplesToPlot = pd.DataFrame(data=[parentDataPoint], columns=self.model.features)

        transformedSamples = []
        for i in range(len(self.__samplesToPlot)):
            transformedSamples.append(self.model.transformDataPoint(self.__samplesToPlot.iloc[i][:-1]))
        
        self.transformedSamplesToPlot = transformedSamples
        self.transformedSamplesClasses = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, transformedSamples)
        self.transformedSamplesClassesPercentage = CounterfactualEngine.randomForestClassifierPredictProbabilities(self.randomForestClassifier, transformedSamples)

        # updating the samples classes by predictions
        self.__samplesToPlot['Class'] = self.transformedSamplesClasses
        # adding a column color to indicate the samples, the current datapoint and the counterfactual
        self.__samplesToPlot['color'] = 0 # samples/historic

        # building a dataframe with the selected point and the 'current' class 
        dataPoint = self.chosenDataPoint.copy()
        dataPoint = np.append(dataPoint, self.predictedCurrentClass)
        self.__dataframeChosenDataPoint = pd.DataFrame(data=[dataPoint], columns=self.model.features)
        # adding a column color to indicate the samples, the current datapoint and the counterfactual
        self.__dataframeChosenDataPoint['color'] = 1 # current

        # running the counterfactual generation in another thread
        self.thread = QThread()
        self.worker = CounterfactualInferfaceWorkerIterable(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.objectiveFunctionValues)
        self.worker.finished.connect(self.restorCursor)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.counterfactualDataframe.connect(self.createCounterfactualDataframe)
        self.thread.start()

    def objectiveFunctionValues(self, values):
        parameters = self.__buildDictParameters()
        self.__values = values
        # appending the counterfactual distance
        self.__values.append(0) 

        for i in range(len(self.__values)):
            self.__values[i] = round(self.__values[i], 2)

        parameters['dataframe']['distance'] = self.__values

        predictedProbabilities = list(self.transformedSamplesClassesPercentage.copy())
        predictedProbabilities.append(self.predictedCurrentClassPercentage[0])
        # appending the counterfactual probability
        predictedProbabilities.append(self.counterfactualProbability[0])

        prob1 = []
        for p in predictedProbabilities:
            prob1.append(p[1])
        parameters['dataframe']['predictedProbability1'] = prob1

        parameters['selectedFeatures'] = self.view.getSelectedFeatures()

        parameters['model'] = self.model

        self.__canvas.updateGraph(parameters)

    def createCounterfactualDataframe(self, counterfactualDataPoint, counterfactualProbability):
        self.counterfactualToPlot = pd.DataFrame(data=[counterfactualDataPoint], columns=self.model.features)
        # adding a column color to indicate the samples, the current datapoint and the counterfactual
        self.counterfactualToPlot['color'] = 2 # counterfactual

        self.counterfactualProbability = counterfactualProbability

    def __handlerNextIteration(self):
        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            nextIteration = IterationController(self.model, self.randomForestClassifier)
            nextIteration.setFeaturesAndValues(self.__dictControllersSelectedPoint)
            self.view.addNewIterationTab(nextIteration.view)

    # this function is used to change the default cursor to wait cursor
    def waitCursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
    
    # this function is used to restor the default cursor
    def restorCursor(self):
        # updating cursor
        QApplication.restoreOverrideCursor()