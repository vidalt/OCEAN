# Author: Moises Henrique Pereira

import os
import random as rd
import pandas as pd
# Import OCEAN functions
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import getFeatureType
from src.CounterFactualParameters import getFeatureActionnability
from src.dataProcessing import DatasetReader


class InterfaceModel():
    """
    Handle access over dataframes, files and the icml code that
    prepare the selected dataset and gives the needed informations
    to be accessed by controller.
    """

    def __init__(self):
        self.datasetsPath = os.getcwd()
        self.datasetsPath = os.path.join(self.datasetsPath, 'datasets')

        self.currentDatasetReader = None

        self.data = None
        self.__transformedData = None

        self.features = None
        self.featuresType = None
        self.featuresActionability = None

        self.featuresInformations = None

        self.transformedFeatures = None
        self.transformedFeaturesOrdered = None
        self.transformedFeaturesType = None
        self.transformedFeaturesActionability = None
        self.transformedFeaturesPossibleValues = None

        self.featuresOneHotEncode = None

    def getDatasetsName(self):
        """ Access dataframes directory and return the name """
        _, _, datasetsName = next(os.walk(self.datasetsPath))

        return datasetsName

    # this function get the dataset system path and instantiate a tool dataset reader
    # the dataset reader opens and prepare the choosen dataset to training part
    def openChosenDataset(self, chosenDataset):
        assert isinstance(chosenDataset, str)
        assert chosenDataset != ''

        chosenDatasetName = chosenDataset + '.csv'
        chosenDatasetPath = os.path.join(self.datasetsPath, chosenDatasetName)

        self.data = pd.read_csv(chosenDatasetPath)
        self.features = self.data.columns
        self.featuresType = {feature:getFeatureType(self.data[feature][0]) for feature in self.data.columns if feature != 'Class'}
        self.data = self.data.drop(0)
        self.featuresActionability = {feature:getFeatureActionnability(self.data[feature][1]) for feature in self.data.columns if feature != 'Class'}
        self.data = self.data.drop(1)

        self.featuresInformations = {}
        for feature in self.features:
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    values = self.data[feature].unique()

                    self.featuresInformations[feature] = {'featureType':self.featuresType[feature],
                                                            'featureActionnability':self.featuresActionability[feature],
                                                            'value0':values[0],
                                                            'value1':values[1]}
                elif self.featuresType[feature] is FeatureType.Discrete or self.featuresType[feature] is FeatureType.Numeric:
                    self.featuresInformations[feature] = {'featureType':self.featuresType[feature],
                                                            'featureActionnability':self.featuresActionability[feature],
                                                            'min':min(self.data[feature].astype(float)),
                                                            'max':max(self.data[feature].astype(float))}
                elif self.featuresType[feature] is FeatureType.Categorical:
                    self.featuresInformations[feature] = {'featureType':self.featuresType[feature],
                                                            'featureActionnability':self.featuresActionability[feature],
                                                            'possibleValues':self.data[feature].value_counts().keys().tolist()}

        self.currentDatasetReader = DatasetReader(chosenDatasetPath)
        self.__transformedData = self.currentDatasetReader.data
        self.transformedFeatures = self.currentDatasetReader.data.columns

        self.transformedFeaturesOrdered = []
        for feature in self.features:
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    self.transformedFeaturesOrdered.append(feature)
                elif self.featuresType[feature] is FeatureType.Discrete or self.featuresType[feature] is FeatureType.Numeric:
                    self.transformedFeaturesOrdered.append(feature)
                elif self.featuresType[feature] is FeatureType.Categorical:
                    for value in self.featuresInformations[feature]['possibleValues']:
                            self.transformedFeaturesOrdered.append(feature+'_'+value)

        self.featuresOneHotEncode = self.currentDatasetReader.oneHotEncoding
        self.transformedFeaturesType = self.currentDatasetReader.featuresType
        self.transformedFeaturesActionability = self.currentDatasetReader.featuresActionnability
        self.transformedFeaturesPossibleValues = self.currentDatasetReader.featuresPossibleValues

    # this function returns the train and test data from chosen dataset and ordered columns
    def getTrainData(self):
        if self.currentDatasetReader is not None:
            return self.currentDatasetReader.X[self.transformedFeaturesOrdered], self.currentDatasetReader.y
        return None, None

    def getRandomPoint(self, randomForestClassifier):
        """
        Return a random point from training dataset.
        """
        if self.currentDatasetReader is not None:
            xTrain, yTrain = self.getTrainData()
            randomIndex = rd.randint(0, len(xTrain)-1)
            randomPoint = self.data.loc[randomIndex].to_numpy()
            return randomPoint
        else:
            return None

    # this function takes a selected point and create a dataframe that can be prepared in toolsDatasetReader
    # this prepared datapoint can be used in train and prediction functions
    def transformDataPoint(self, selectedDataPoint):
        assert selectedDataPoint is not None
        if 'Class' in self.features:
            assert len(selectedDataPoint) == len(self.features)-1
        else:
            assert len(selectedDataPoint) == len(self.features)

        transformedDataPoint = []
        for index, feature in enumerate(self.features):
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    if selectedDataPoint[index] == self.featuresInformations[feature]['value0']:
                        transformedDataPoint.append(0)
                    elif selectedDataPoint[index] == self.featuresInformations[feature]['value1']:
                        transformedDataPoint.append(1)

                elif self.featuresType[feature] is FeatureType.Discrete or self.featuresType[feature] is FeatureType.Numeric:
                    transformedValue = (float(selectedDataPoint[index])-self.featuresInformations[feature]['min'])/(self.featuresInformations[feature]['max']-self.featuresInformations[feature]['min'])
                    transformedDataPoint.append(transformedValue)

                elif self.featuresType[feature] is FeatureType.Categorical:
                    for value in self.featuresInformations[feature]['possibleValues']:
                        if value == selectedDataPoint[index]:
                            transformedDataPoint.append(1)
                        else:
                            transformedDataPoint.append(0)

        return transformedDataPoint

    # this function takes an encoded data point and returns a data point user readable in a dictionary
    def invertTransformedDataPoint(self, transformedDataPoint):
        assert transformedDataPoint is not None
        if 'Class' in self.transformedFeatures:
            assert len(transformedDataPoint) == len(self.transformedFeatures)-1
        else:
            assert len(transformedDataPoint) == len(self.transformedFeatures)

        dataPoint = []
        index = 0
        for feature in self.features:
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    if abs(transformedDataPoint[index]) == 0:
                        dataPoint.append(self.featuresInformations[feature]['value0'])
                    elif abs(transformedDataPoint[index]) == 1:
                        dataPoint.append(self.featuresInformations[feature]['value1'])
                    index += 1

                elif self.featuresType[feature] is FeatureType.Discrete or self.featuresType[feature] is FeatureType.Numeric:
                    value = self.featuresInformations[feature]['min']+(transformedDataPoint[index])*(self.featuresInformations[feature]['max']-self.featuresInformations[feature]['min'])
                    dataPoint.append(value)
                    index += 1

                elif self.featuresType[feature] is FeatureType.Categorical:
                    maxValue = 0
                    counterfactualValue = '-'
                    for value in self.featuresInformations[feature]['possibleValues']:
                        if transformedDataPoint[index] > maxValue:
                            maxValue = transformedDataPoint[index]
                            counterfactualValue = value

                        index += 1
                    dataPoint.append(counterfactualValue)

        return dataPoint

    # this function takes an encoded data point and returns a data point user readable in a dictionary
    def invertTransformedFeatureImportance(self, transformedFeatureImportance):
        assert transformedFeatureImportance is not None
        if 'Class' in self.transformedFeatures:
            assert len(transformedFeatureImportance) == len(self.transformedFeatures)-1
        else:
            assert len(transformedFeatureImportance) == len(self.transformedFeatures)

        featureImportance = [0 for i in range(len(self.features)-1)]
        index = 0
        for i, feature in enumerate(self.features):
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    featureImportance[i] = abs(transformedFeatureImportance[index])
                    index += 1

                elif self.featuresType[feature] is FeatureType.Discrete or self.featuresType[feature] is FeatureType.Numeric:
                    featureImportance[i] = abs(transformedFeatureImportance[index])
                    index += 1

                elif self.featuresType[feature] is FeatureType.Categorical:
                    maxValue = 0
                    for ind in self.featuresInformations[feature]['possibleValues']:
                        value = abs(transformedFeatureImportance[index])
                        if value > maxValue:
                            maxValue = value
                        index += 1

                    featureImportance[i] = maxValue

        return featureImportance

    # this function takes a feature name and value, and returns a min max scaled value
    def transformSingleNumericalValue(self, feature, value):
        assert isinstance(feature, str)
        assert value is not None

        transformedValue = (float(value)-self.featuresInformations[feature]['min'])/(self.featuresInformations[feature]['max']-self.featuresInformations[feature]['min'])

        return transformedValue
