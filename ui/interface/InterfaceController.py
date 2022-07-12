# Author: Moises Henrique Pereira

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
# Import ui functions
from ui.interface.InterfaceViewer import InterfaceViewer
from ui.interface.InterfaceModel import InterfaceModel
from ui.interface.InterfaceEnums import InterfaceEnums
from ui.engine.CounterfactualEngine import CounterfactualEngine as engine
from .ComboboxList.ComboboxListController import ComboboxListController
from .DoubleRadioButton.DoubleRadioButtonController import (
    DoubleRadioButtonController)
from .Slider3Ranges.Slider3RangesController import Slider3RangesController
# Load OCEAN functions
from src.CounterFactualParameters import FeatureType


class InterfaceController():
    """ Handle the logic over the interface.

    Interact with model, viewer and worker.
    Take the selected dataset informations from model to send to counterfactual
    generator in worker class.
    """

    def __init__(self):
        self.interfaceViewer = InterfaceViewer()
        self.model = InterfaceModel()

        self.__chosenDataset = InterfaceEnums.SelectDataset.DEFAULT.value

        self.randomForestClassifier = None
        self.isolationForest = None

        self.initPointFeatures = {}
        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None
        self.predictedOriginalClass = None

    def initializeView(self):
        """ Send the dataframe names to interface. """
        datasets = self.model.getDatasetsName()

        datasetsName = [InterfaceEnums.SelectDataset.DEFAULT.value]
        for datasetName in datasets:
            auxDatasetName = datasetName.split('.')[0]
            datasetsName.append(auxDatasetName)

        self.view.initializeView(datasetsName)

    def train_random_and_isolation_forests(self, chosenDataset):
        # Open the selected dataset
        self.model.openChosenDataset(chosenDataset)
        xTrain, yTrain = self.model.getTrainData()

        # Train the random forest and isolation forest models
        assert(xTrain is not None and yTrain is not None)
        self.rfClassifier = engine.trainRandomForestClassifier(xTrain, yTrain)
        self.isolationForest = engine.trainIsolationForest(xTrain)

    def get_component_from_feature(self, feature):
        """
        Get the relevant component for each feature type,
        initialize it with its min/max values and return it.
        """
        featureInformations = self.model.featuresInformations[feature]
        featureType = featureInformations['featureType']
        component = None
        if featureType is FeatureType.Binary:
            value0 = featureInformations['value0']
            value1 = featureInformations['value1']
            component = DoubleRadioButtonController(self.view)
            component.initializeView(
                feature, str(value0), str(value1))
        elif featureType is FeatureType.Discrete:
            minValue = featureInformations['min']
            maxValue = featureInformations['max']
            component = Slider3RangesController(self.view)
            component.initializeView(
                feature, minValue, maxValue, decimalPlaces=0)
        elif featureType is FeatureType.Numeric:
            minValue = featureInformations['min']
            maxValue = featureInformations['max']
            component = Slider3RangesController(self.view)
            component.initializeView(
                feature, minValue, maxValue)
        elif featureType is FeatureType.Categorical:
            component = ComboboxListController(
                self.view, featureInformations['possibleValues'])
            component.initializeView(
                feature, featureInformations['possibleValues'])
        return component

    def sample_random_point(self):
        randomDataPoint = self.model.getRandomPoint(self.rfClassifier)
        # Show the values of the data point in components
        for index, feature in enumerate(self.model.features):
            if feature != 'Class':
                self.initPointFeatures[feature].setSelectedValue(
                    randomDataPoint[index])
        return randomDataPoint

    def transform_and_predict_data_point(self):
        # Get the datapoint
        dataPoint = []
        for feature in self.model.features:
            if feature != 'Class':
                featureName = feature
                content = self.initPointFeatures[feature].getContent()
                dataPoint.append(content['value'])
        self.chosenDataPoint = np.array(dataPoint)

        # Transform the datapoint to predict its class
        self.transformedChosenDataPoint = self.model.transformDataPoint(
            self.chosenDataPoint)

        # Predict the class
        self.predictedOriginalClass = engine.randomForestClassifierPredict(
            self.rfClassifier, [self.transformedChosenDataPoint])
        return featureName

    def waitCursor(self):
        """
        Change the cursor to a wait cursor.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def restorCursor(self):
        """
        Restore cursor to default.
        """
        QApplication.restoreOverrideCursor()
