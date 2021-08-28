# Author: Moises Henrique Pereira
# this class handle the logic over the interface, interacting with model, view and worker
# also taking the selected dataset informations from model to send to counterfactual generator in worker class

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from .IterationView import IterationView

from CounterFactualParameters import FeatureType

from CounterfactualEngine.CounterfactualEngine import CounterfactualEngine

from Canvas.CanvasController import CanvasController

from ...ComboboxList.ComboboxListController import ComboboxListController
from ...DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ...Slider3Ranges.Slider3RangesController import Slider3RangesController

import numpy as np

class IterationController():

    def __init__(self, model, randomForestClassifier):
        self.view = IterationView()
        self.model = model

        self.randomForestClassifier = randomForestClassifier

        self.__initializeView()

        self.view.selectedAxisX.connect(self.__updateGraph)
        self.view.selectedAxisY.connect(self.__updateGraph)

        self.__updateGraph()

        self.__dictControllersSelectedPoint = {}

        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None

        self.predictedCurrentClass = None

        self.__canvas = self.view.getCanvas()


    # this function takes the dataframe names and send them to interface
    def __initializeView(self):
        self.view.initializeView()

    def setFeaturesAndValues(self, dictControllersSelectedPoint):
        import pprint as pp

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
                    componentController.disableComponent()

                elif featureType is FeatureType.Discrete:
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    # componentController = LineEditMinimumMaximumController(self.view)
                    componentController = Slider3RangesController(self.view, smaller=True)
                    componentController.initializeView(feature, minValue, maxValue, decimalPlaces=0)
                    componentController.setSelectedValue(value)

                elif featureType is FeatureType.Numeric:
                    minValue = content['minimumValue']
                    maxValue = content['maximumValue']
                    value = content['value']

                    # componentController = LineEditMinimumMaximumController(self.view)
                    componentController = Slider3RangesController(self.view, smaller=True)
                    componentController.initializeView(feature, minValue, maxValue)
                    componentController.setSelectedValue(value)
                    
                elif featureType is FeatureType.Categorical:
                    value = content['value']
                    
                    componentController = ComboboxListController(self.view)
                    componentController.initializeView(feature, self.model.featuresInformations[feature]['possibleValues'])
                    componentController.setSelectedValue(value)
                    componentController.disableComponent()

                # adding the view to selectedPoint component
                self.view.addFeatureWidget(componentController.view)
                # saving the controller to facilitate the access to components
                self.__dictControllersSelectedPoint[feature] = componentController

        self.view.addAxisOptions(list(self.model.features))
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
        self.view.showCurrentClass(self.predictedCurrentClass[0])      

    def __updateGraph(self):
        self.waitCursor()

        # parameters = self.__buildDictParameters()
        # parameters['dataframe']['distance'] = self.__values
        # self.__canvas.updateGraph(parameters)
        self.restorCursor()

     # this function is used to change the default cursor to wait cursor
    def waitCursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
    
    # this function is used to restor the default cursor
    def restorCursor(self):
        # updating cursor
        QApplication.restoreOverrideCursor()
