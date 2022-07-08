import pandas as pd
from sklearn.model_selection import train_test_split
# Import OCEAN utility functions and custom classes
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import FeatureActionnability
from src.CounterFactualParameters import getFeatureType
from src.CounterFactualParameters import getFeatureActionnability
from src.CounterFactualParameters import isFeatureTypeScalable


class DatasetReader:
    def __init__(self, filename, extrapolateDicreteFeatureValues=False):
        self.filename = filename
        # -- Read raw data and get types --
        self.data = pd.read_csv(self.filename)
        # Get the datatypes and remove the corresponding row
        allFeatureColumns = [str(column)
                             for column in self.data.columns
                             if column != 'Class']
        columnFeatures = {column: getFeatureType(
            self.data[column][0]) for column in allFeatureColumns}
        self.data = self.data.drop(0)
        columnActionnability = {column: getFeatureActionnability(
            self.data[column][1]) for column in allFeatureColumns}
        self.data = self.data.drop(1)

        # -- Process raw data --
        self.__remove_columns_with_unique_value()
        self.__replace_binary_features_by_bynaries(columnFeatures)
        self.__one_hot_encoding_of_categorical_features(columnFeatures,
                                                        columnActionnability)
        # Reconstruct features type and one hot encoding
        self.featuresType = [columnFeatures[column]
                             for column in self.data.columns
                             if column != 'Class']
        self.featuresActionnability = [columnActionnability[column]
                                       for column in self.data.columns
                                       if column != 'Class']
        # Convert data to float
        for column in self.data.columns:
            if column == 'Class':
                self.data[column] = self.data[column].astype(float)
            elif columnFeatures[column] in [FeatureType.Discrete,
                                            FeatureType.Numeric]:
                self.data[column] = self.data[column].astype(float)
            elif columnFeatures[column] == FeatureType.Binary:
                self.data[column] = self.data[column].astype(float)
        # Normalize feature values
        self.__apply_min_max_scaling(columnFeatures)
        # Features possibles values
        self.__read_possible_feature_values(
            extrapolateDicreteFeatureValues)

        # -- Divide data in training, test, and counterfactual init --
        # Training set
        self.X = self.data.loc[:, self.data.columns != 'Class']
        self.y = self.data['Class']
        # Counterfactuals
        self.x0 = [self.data.iloc[0, self.data.columns != 'Class']]
        self.XwithGoodPoint = self.data.loc[self.data['Class']
                                            == 1, self.data.columns != 'Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=0)

    def __remove_columns_with_unique_value(self):
        for column in self.data.columns:
            if len(self.data[column].unique()) == 1:
                print("Drop column", column,
                      "because it contains a unique value")
                self.data.drop([column], axis=1, inplace=True)

    def __replace_binary_features_by_bynaries(self, columnFeatures):
        for column in self.data.columns:
            isNotClass = (column != 'Class')
            if isNotClass and columnFeatures[column] == FeatureType.Binary:
                values = self.data[column].unique()
                if len(values) != 2:
                    for val in values:
                        if val not in [0, 1, '0', '1']:
                            print("Error: more than two values in ", column,
                                  "which is indicated to be binary,"
                                  " treated as categorical")
                            columnFeatures[column] = FeatureType.Categorical
                if values[0] in ['0', '1'] and values[1] in ['0', '1']:
                    continue
                assert values[1] != '0'
                self.data[column] = self.data[column].str.replace(
                    values[0], '0')
                self.data[column] = self.data[column].str.replace(
                    values[1], '1')

    def __one_hot_encoding_of_categorical_features(self, features,
                                                   actionnability):
        oneHotEncoding = dict()
        categoricalColumns = [str(column) for column in self.data.columns
                              if column != 'Class'
                              and features[column] == FeatureType.Categorical]

        for column in categoricalColumns:
            if actionnability[column] == FeatureActionnability.Increasing:
                print("warning, categorical feature ", column,
                      "cannot be with increasing feasability, changed to free")
                actionnability[column] = FeatureActionnability.Free
            featureOneHot = pd.get_dummies(self.data[column], prefix=column)
            oneHotEncoding[column] = []
            for newCol in featureOneHot:
                features[newCol] = FeatureType.Binary
                actionnability[newCol] = actionnability[column]
                oneHotEncoding[column].append(newCol)
            self.data = pd.concat([self.data, featureOneHot], axis=1)
            self.data.drop([column], axis=1, inplace=True)

        self.oneHotEncoding = dict()
        for categoricalColumn in categoricalColumns:
            self.oneHotEncoding[categoricalColumn] = dict()
            c = 0
            for column in self.data.columns:
                if column.split("_") == categoricalColumn:
                    self.oneHotEncoding[categoricalColumn].append(c)
                c += 1
        self.data = self.data[[
            col for col in self.data if col != 'Class'] + ['Class']]

    def __apply_min_max_scaling(self, features):
        """ Normalize all features between 0 and 1."""
        self.lowerBounds = dict()
        self.upperBounds = dict()
        self.lowerBoundsList = []
        self.upperBoundsList = []
        f = 0
        for column in self.data.columns:
            if column != 'Class' and isFeatureTypeScalable(features[column]):
                # Set lower bound
                self.lowerBounds[column] = self.data[column].min()
                if self.featuresType[f] == FeatureType.Discrete:
                    self.lowerBounds[column] = 0
                else:
                    self.lowerBounds[column] = self.data[column].min()
                # Set upper bound
                self.upperBounds[column] = self.data[column].max()
                # Normalize all values in the column
                self.data[column] = (self.data[column] - self.data[column].min()) / \
                    (self.data[column].max() - self.data[column].min())
                # Add bounds to lists
                self.upperBoundsList.append(self.upperBounds[column])
                self.lowerBoundsList.append(self.lowerBounds[column])
            f += 1

    def __read_possible_feature_values(self, extrapolateDicreteFeatureValues):
        self.featuresPossibleValues = []
        c = 0
        for column in self.data:
            if column != 'Class':
                if self.featuresType[c] == FeatureType.Categorical:
                    self.featuresPossibleValues.append(
                        self.data[column].unique())
                elif self.featuresType[c] == FeatureType.Discrete:
                    if extrapolateDicreteFeatureValues:
                        # Discrete features can take any value between 0
                        # and maximum value observed
                        maxValue = self.upperBoundsList[c]
                        self.featuresPossibleValues.append(
                            [i/maxValue for i in range(0, int(maxValue+1))])
                    else:
                        # Discrete features can only take values seen in
                        # historical data
                        self.featuresPossibleValues.append(
                            self.data[column].unique())
                elif self.featuresType[c] in [FeatureType.Numeric,
                                              FeatureType.Binary]:
                    # Numeric features can take any value
                    self.featuresPossibleValues.append([])
                else:
                    print("Warning: wrong feature type:",
                          self.featuresType[c],
                          " for feature ", column)
            c += 1
