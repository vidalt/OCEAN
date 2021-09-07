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

    def __init__(self, model, randomForestClassifier, isolationForest):
        self.view = IterationView()
        self.model = model

        self.randomForestClassifier = randomForestClassifier
        self.isolationForest = isolationForest

        self.__initializeView()

        self.__dictControllersSelectedPoint = {}

        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None

        self.predictedCurrentClass = None

        self.__canvas = self.view.getCanvas()
        
        self.__samplesToPlot = None
        self.transformedSamplesToPlot = None
        self.transformedSamplesClasses = None

        self.view.selectedAxisX.connect(lambda: self.__updateGraph())
        self.view.selectedAxisY.connect(lambda: self.__updateGraph())


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
                    componentController.disableComponent()

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
        parameters = self.__buildDictParameters()

        if parameters['xVariable'] == '' or parameters['yVariable'] == '':
            return

        if parameters['xVariable'] != IterationEnums.DefaultAxes.DEFAULT_X.value and parameters['yVariable'] == IterationEnums.DefaultAxes.DEFAULT_Y.value:
            pass
        elif parameters['xVariable'] == IterationEnums.DefaultAxes.DEFAULT_X.value and parameters['yVariable'] != IterationEnums.DefaultAxes.DEFAULT_Y.value:
            pass
        elif parameters['xVariable'] != IterationEnums.DefaultAxes.DEFAULT_X.value and parameters['yVariable'] != IterationEnums.DefaultAxes.DEFAULT_Y.value:
            self.__handlerCalculateDistances()

    def __buildDictParameters(self):
        # building the dict parameters to plot
        xVariable, yVariable = self.view.getChosenAxis()
        parameters = {'xVariable':xVariable, 'yVariable':yVariable}

        if self.__samplesToPlot is not None:
            # concatenating selected point with sample
            dataToPlot = pd.concat([self.__samplesToPlot, self.__dataframeChosenDataPoint])
            dataToPlot = dataToPlot.reset_index().drop(['index'], axis=1)

            parameters['dataframe'] = dataToPlot

        return parameters

    def __getPossibities(self, variable):
        dictVariable = self.model.featuresInformations[variable]
        featureType = self.model.featuresInformations[variable]['featureType']

        possibilities = None
        if featureType is FeatureType.Binary:
            value0 = dictVariable['value0']
            value1 = dictVariable['value1']
            possibilities = [value0, value1]

        elif featureType is FeatureType.Discrete or featureType is FeatureType.Numeric:
            minValue = dictVariable['min']
            maxValue = dictVariable['max']

            content = self.__dictControllersSelectedPoint[variable].getContent()
            currentValue = content['value']

            adds = [-2, -1, 0, 1, 2]
            possibilities = []
            for add in adds:
                aux = currentValue + add
                if aux >= minValue and aux <= maxValue:
                    possibilities.append(aux)

        elif featureType is FeatureType.Categorical:
            possibilities = dictVariable['possibleValues']

        return possibilities

    def __buildSample(self):
        xVariable, yVariable = self.view.getChosenAxis()

        if xVariable != IterationEnums.DefaultAxes.DEFAULT_X.value and yVariable != IterationEnums.DefaultAxes.DEFAULT_Y.value:
            # print('!'*75)
            # print(self.model.featuresInformations[xVariable])  
            # print('!'*75)
            # print(self.model.featuresInformations[yVariable])  
            # print('!'*75)

            xPossibilities = self.__getPossibities(xVariable)
            yPossibilities = self.__getPossibities(yVariable)
            cartesianXY = None
            try:
                cartesianXY = cartesian((xPossibilities, yPossibilities))
            except:
                cartesianXY = []
                for xp in xPossibilities:
                    for yp in yPossibilities:
                        cartesianXY.append([xp, yp])

            # print(xPossibilities)
            # print('!'*75)
            # print(yPossibilities)
            # print('!'*75)
            # print(cartesianXY)
            # print('!'*75)
            
            samples = pd.DataFrame(data=None, columns=self.model.data.columns)
            for i, xy in enumerate(cartesianXY):
                try: 
                    xySample = self.model.data.loc[(self.model.data[xVariable] == xy[0]) & (self.model.data[yVariable] == xy[1])][:]
                    samples = samples.append(xySample.sample(n=1))
                except:
                    print('There is not sample with those values')

            if len(samples) == 0:
                samples = self.model.data.sample(n=5)

            return samples

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
        self.view.showCurrentClass(self.predictedCurrentClass[0])

        # getting a set of samples to plot
        self.__samplesToPlot = self.__buildSample()
        # self.__samplesToPlot = self.model.data.sample(n=5)

        transformedSamples = []
        for i in range(len(self.__samplesToPlot)):
            transformedSamples.append(self.model.transformDataPoint(self.__samplesToPlot.iloc[i][:-1]))
        
        self.transformedSamplesToPlot = transformedSamples
        self.transformedSamplesClasses = CounterfactualEngine.randomForestClassifierPredict(self.randomForestClassifier, transformedSamples)

        # building a dataframe with the selected point and the class 'current'
        dataPoint = self.chosenDataPoint.copy()
        dataPoint = np.append(dataPoint, 'current')
        self.__dataframeChosenDataPoint = pd.DataFrame(data=[dataPoint], columns=self.__samplesToPlot.columns)

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
        self.thread.start()

    def objectiveFunctionValues(self, values):
        parameters = self.__buildDictParameters()
        self.__values = values

        for i in range(len(self.__values)):
            self.__values[i] = round(self.__values[i], 2)

        if parameters['xVariable'] != IterationEnums.DefaultAxes.DEFAULT_X.value and parameters['yVariable'] == IterationEnums.DefaultAxes.DEFAULT_Y.value:
            pass
        elif parameters['xVariable'] == IterationEnums.DefaultAxes.DEFAULT_X.value and parameters['yVariable'] != IterationEnums.DefaultAxes.DEFAULT_Y.value:
            pass
        elif parameters['xVariable'] != IterationEnums.DefaultAxes.DEFAULT_X.value and parameters['yVariable'] != IterationEnums.DefaultAxes.DEFAULT_Y.value:
            parameters['dataframe']['distance'] = self.__values

            parameters['dataframe'].sort_values(by=[parameters['xVariable'], parameters['yVariable']], inplace=True)

            self.__canvas.updateGraph(parameters)

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