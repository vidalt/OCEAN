import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ..CounterfactualInterfaceModel import CounterfactualInterfaceModel
from ..CounterfactualInterfaceEnums import CounterfactualInterfaceEnums
from .CounterfactualInterfaceViewIterable import CounterfactualInterfaceViewIterable

from CounterfactualEngine.CounterfactualEngine import CounterfactualEngine

from Canvas.CanvasController import CanvasController

from CounterFactualParameters import FeatureType

from ..ComboboxList.ComboboxListController import ComboboxListController
from ..DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ..Slider3Ranges.Slider3RangesController import Slider3RangesController

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

        # self.view.calculateDistances.connect(self.__handlerCalculateDistances)

        # self.view.updateGraph.connect(self.__updateGraph)

        self.randomForestClassifier = None
        self.isolationForest = None

        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None
        self.__dataframeChosenDataPoint = None
        self.predictedOriginalClass = None

        self.__canvas = None

        self.__dictControllersSelectedPoint = {}
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None


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
        # getting the name of the desired dataset
        self.__chosenDataset = self.view.getChosenDataset()
        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            # cleaning the view
            self.view.clearView()
            self.__dictControllersSelectedPoint.clear()

            # opening the desired dataset
            self.model.openChosenDataset(self.__chosenDataset)
            xTrain, yTrain = self.model.getTrainData()

            # training the random forest and isolation forest models
            if xTrain is not None and yTrain is not None: 
                self.randomForestClassifier = CounterfactualEngine.trainRandomForestClassifier(xTrain, yTrain)
                self.isolationForest = CounterfactualEngine.trainIsolationForest(xTrain)

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
                        componentController = ComboboxListController(self.view)
                        componentController.initializeView(feature, self.model.featuresInformations[feature]['possibleValues'])

                    # adding the view to selectedPoint component
                    self.view.addFeatureWidget(componentController.view)
                    # saving the controller to facilitate the access to components
                    self.__dictControllersSelectedPoint[feature] = componentController

            # self.view.addAxisOptions(list(self.model.features))
            
        else:
            # cleaning the view
            self.view.clearView()
            self.__dictControllersSelectedPoint.clear()

    # this function get a random datapoint from dataset 
    def __handlerRandomPoint(self):
        self.view.clearClass()

        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            randomDataPoint = self.model.getRandomPoint(self.randomForestClassifier)

            # showing the values in their respective component
            for index, feature in enumerate(self.model.features):
                if feature != 'Class':
                    self.__dictControllersSelectedPoint[feature].setSelectedValue(randomDataPoint[index])

    # this function takes the selected data point and calculate the respective class
    def __handlerCalculateClass(self):
        self.view.clearClass()
        
        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
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
            self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, [self.transformedChosenDataPoint])
            self.view.showOriginalClass(self.predictedOriginalClass[0])      

    def __handlerNextIteration(self):
        if self.__chosenDataset != CounterfactualInterfaceEnums.SelectDataset.DEFAULT.value:
            nextIteration = IterationController(self.model, self.randomForestClassifier, self.isolationForest)
            nextIteration.setFeaturesAndValues(self.__dictControllersSelectedPoint)
            self.view.addNewIterationTab(nextIteration.view)

    # this function is used to change the default cursor to wait cursor
    def waitCursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
    
    # this function is used to restor the default cursor
    def restorCursor(self):
        # updating cursor
        QApplication.restoreOverrideCursor()