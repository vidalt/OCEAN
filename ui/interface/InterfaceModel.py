"""InterfaceModel module."""
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

    def __init__(self) -> None:
        self.datasetsPath = os.getcwd()
        self.datasetsPath = os.path.join(self.datasetsPath, 'datasets')

    def getDatasetsName(self):
        """ Access dataframes directory and return the name """
        _, _, datasetsName = next(os.walk(self.datasetsPath))
        return datasetsName

    def openChosenDataset(self, chosenDataset: str) -> None:
        """
        Get the dataset system path and instantiate a tool dataset reader.
        The dataset reader opens and prepare the choosen dataset for training.
        """
        assert isinstance(chosenDataset, str)
        assert chosenDataset != ''
        # Find path to data set
        chosenDatasetName = chosenDataset + '.csv'
        chosenDatasetPath = os.path.join(self.datasetsPath, chosenDatasetName)
        # Read data from path
        self.data = pd.read_csv(chosenDatasetPath)
        self.features = self.data.columns
        self.featuresType = {
            feature: getFeatureType(self.data[feature][0])
            for feature in self.data.columns if feature != 'Class'}
        self.data = self.data.drop(0)
        self.featuresActionability = {
            feature: getFeatureActionnability(self.data[feature][1])
            for feature in self.data.columns if feature != 'Class'}
        self.data = self.data.drop(1)

        # Collect relevant feature informations
        self.featuresInformations = {}
        for feature in self.features:
            if feature != 'Class':
                actionability = self.featuresActionability[feature]
                featuresType = self.featuresType[feature]
                if self.featuresType[feature] is FeatureType.Binary:
                    values = self.data[feature].unique()
                    self.featuresInformations[feature] = {
                        'featureType': featuresType,
                        'featureActionnability': actionability,
                        'value0': values[0],
                        'value1': values[1]}

                elif self.featuresType[feature] in [FeatureType.Discrete,
                                                    FeatureType.Numeric]:
                    self.featuresInformations[feature] = {
                        'featureType': featuresType,
                        'featureActionnability': actionability,
                        'min': min(self.data[feature].astype(float)),
                        'max': max(self.data[feature].astype(float))}
                elif self.featuresType[feature] is FeatureType.Categorical:
                    uniqueValues = self.data[feature].value_counts().keys()
                    self.featuresInformations[feature] = {
                        'featureType': featuresType,
                        'featureActionnability': actionability,
                        'possibleValues': uniqueValues.tolist()}

        # Process data using datasetReader
        self.dataReader = DatasetReader(chosenDatasetPath)
        self.processedData = self.dataReader.data
        self.processedFeatures = self.dataReader.data.columns

        self.processedFeaturesOrdered = []
        for feature in self.features:
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    self.processedFeaturesOrdered.append(feature)
                elif self.featuresType[feature] in [FeatureType.Discrete,
                                                    FeatureType.Numeric]:
                    self.processedFeaturesOrdered.append(feature)
                elif self.featuresType[feature] is FeatureType.Categorical:
                    for value in self.featuresInformations[feature][
                            'possibleValues']:
                        self.processedFeaturesOrdered.append(
                            feature + '_' + value)

        self.featuresOneHotEncode = self.dataReader.oneHotEncoding
        self.processedFeaturesType = self.dataReader.featuresType
        self.processedFeaturesActionability = self.dataReader.featuresActionnability
        self.processedFeaturesPossibleValues = self.dataReader.featuresPossibleValues

    def getTrainData(self):
        """
        Returns the train and test data from chosen dataset
        """
        if self.dataReader is not None:
            return (self.dataReader.X[self.processedFeaturesOrdered],
                    self.dataReader.y)
        return None, None

    def getRandomPoint(self, randomForestClassifier):
        """
        Return a random point from training dataset.
        """
        if self.dataReader is not None:
            xTrain, yTrain = self.getTrainData()
            randomIndex = rd.randint(0, len(xTrain)-1)
            randomPoint = self.data.loc[randomIndex].to_numpy()
            return randomPoint
        else:
            return None

    def transformDataPoint(self, selectedDataPoint):
        """
        Transform the selected datapoint to have normalized features
        and one-hot-encoded categorical features.
        This follows the pre-processing of the datasetReader.
        """
        assert selectedDataPoint is not None
        if 'Class' in self.features:
            assert len(selectedDataPoint) == len(self.features)-1
        else:
            assert len(selectedDataPoint) == len(self.features)

        transformedDataPoint = []
        for index, feature in enumerate(self.features):
            if feature != 'Class':
                featureInfo = self.featuresInformations[feature]
                if self.featuresType[feature] is FeatureType.Binary:
                    if selectedDataPoint[index] == featureInfo['value0']:
                        transformedDataPoint.append(0)
                    elif selectedDataPoint[index] == featureInfo['value1']:
                        transformedDataPoint.append(1)

                elif self.featuresType[feature] in [FeatureType.Discrete,
                                                    FeatureType.Numeric]:
                    range = featureInfo['max']-featureInfo['min']
                    transformedValue = (float(
                        selectedDataPoint[index])-featureInfo['min'])/range
                    transformedDataPoint.append(transformedValue)

                elif self.featuresType[feature] is FeatureType.Categorical:
                    for value in featureInfo['possibleValues']:
                        if value == selectedDataPoint[index]:
                            transformedDataPoint.append(1)
                        else:
                            transformedDataPoint.append(0)
                else:
                    raise TypeError

        return transformedDataPoint

    def invertTransformedDataPoint(self, transformedDataPoint):
        """
        Convert an encoded data point into a
        data point user readable in a dictionary.
        """
        assert transformedDataPoint is not None
        if 'Class' in self.processedFeatures:
            assert len(transformedDataPoint) == len(self.processedFeatures)-1
        else:
            assert len(transformedDataPoint) == len(self.processedFeatures)

        dataPoint = []
        index = 0
        for feature in self.features:
            if feature != 'Class':
                featureInfo = self.featuresInformations[feature]
                if self.featuresType[feature] is FeatureType.Binary:
                    if abs(transformedDataPoint[index]) == 0:
                        dataPoint.append(featureInfo['value0'])
                    elif abs(transformedDataPoint[index]) == 1:
                        dataPoint.append(featureInfo['value1'])
                    index += 1

                elif self.featuresType[feature] in [FeatureType.Discrete,
                                                    FeatureType.Numeric]:
                    minVal = self.featuresInformations[feature]['min']
                    maxVal = self.featuresInformations[feature]['max']
                    value = minVal+(transformedDataPoint[index])*(
                        maxVal-minVal)
                    dataPoint.append(value)
                    index += 1

                elif self.featuresType[feature] is FeatureType.Categorical:
                    maxValue = 0
                    counterfactualValue = '-'
                    for value in featureInfo['possibleValues']:
                        if transformedDataPoint[index] > maxValue:
                            maxValue = transformedDataPoint[index]
                            counterfactualValue = value

                        index += 1
                    dataPoint.append(counterfactualValue)

        return dataPoint

    def invertTransformedFeatureImportance(self, transformedImportance):
        """
        Convert an encoded feature importance into a
        featureImportance user readable in a dictionary.
        """
        assert transformedImportance is not None
        if 'Class' in self.transformedFeatures:
            assert (len(transformedImportance)
                    == len(self.transformedFeatures)-1)
        else:
            assert len(transformedImportance) == len(self.transformedFeatures)

        featureImportance = [0 for i in range(len(self.features)-1)]
        index = 0
        for i, feature in enumerate(self.features):
            if feature != 'Class':
                if self.featuresType[feature] is FeatureType.Binary:
                    featureImportance[i] = abs(transformedImportance[index])
                    index += 1

                elif self.featuresType[feature] in [FeatureType.Discrete,
                                                    FeatureType.Numeric]:
                    featureImportance[i] = abs(transformedImportance[index])
                    index += 1

                elif self.featuresType[feature] is FeatureType.Categorical:
                    maxValue = 0
                    for ind in self.featuresInformations[feature]['possibleValues']:
                        value = abs(transformedImportance[index])
                        if value > maxValue:
                            maxValue = value
                        index += 1

                    featureImportance[i] = maxValue

        return featureImportance

    def transformSingleNumericalValue(self, feature, value):
        """
        Take a feature name and value, and returns a min max scaled value.
        """
        assert isinstance(feature, str)
        assert value is not None
        maxValue = self.featuresInformations[feature]['max']
        minValue = self.featuresInformations[feature]['min']
        range = maxValue - minValue
        transformedValue = (float(value)-minValue)/range
        return transformedValue
