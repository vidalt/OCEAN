import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Import OCEAN utility functions and custom classes
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import FeatureActionnability
from src.CounterFactualParameters import getFeatureType
from src.CounterFactualParameters import getFeatureActionnability
from src.CounterFactualParameters import isFeatureTypeScalable


def is_categorical_feature(column, columnFeatures):
    if column != 'Class' and columnFeatures[column] == FeatureType.Categorical:
        return True
    return False


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
        splits = train_test_split(self.X, self.y,
                                  test_size=0.2, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = splits

    # -- Private methods --
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
        # Store original features names without the y class column
        self.featureNames = list(self.data.columns)
        self.featureNames.pop()

        # Change the order of features in list: categorical column at the end
        # to facilitate the one-hot encoding and decoding.
        count = 0
        for c in range(len(self.data.columns)):
            column = self.data.columns[c]
            if is_categorical_feature(column, features):
                self.featureNames.pop(c - count)
                count += 1

        categoricalColumns = [str(column) for column in self.data.columns
                              if is_categorical_feature(column, features)]
        # Duplicate each categorical column into as many dummy one-hot
        # columns as needed
        for column in categoricalColumns:
            if actionnability[column] == FeatureActionnability.Increasing:
                print("warning, categorical feature ", column,
                      "cannot be with increasing feasability, changed to free")
                actionnability[column] = FeatureActionnability.Free
            featureOneHot = pd.get_dummies(self.data[column], prefix=column)
            for newCol in featureOneHot:
                features[newCol] = FeatureType.Binary
                actionnability[newCol] = actionnability[column]
            # Add new columns to data
            self.data = pd.concat([self.data, featureOneHot], axis=1)
            # Remove the original columns
            self.data.drop([column], axis=1, inplace=True)

        self.data = self.data[[
            col for col in self.data if col != 'Class'] + ['Class']]

        # Store the position of the one-hot columns corresponding to each
        # categorical feature
        self.oneHotEncoding = dict()
        for categoricalColumn in categoricalColumns:
            self.oneHotEncoding[categoricalColumn] = []
            c = 0
            for column in self.data.columns:
                if categoricalColumn in column.split("_"):
                    self.oneHotEncoding[categoricalColumn].append(c)
                c += 1

        # Store the names of the categorical features
        for f in self.oneHotEncoding:
            self.featureNames.append(str(f))

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
                self.data[column] = (
                    (self.data[column] - self.lowerBounds[column])
                    / (self.upperBounds[column] - self.lowerBounds[column]))
                # Keep the bounds in memory to allow inverse operation
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

    def __decode_categorical_features(self, x):
        """
        Convert back the one-hot-encoded feature
        into a single categorical feature.
        """
        for f in self.oneHotEncoding:
            colIndices = self.oneHotEncoding[f]
            nbPossibleValues = len(colIndices)
            for i in range(nbPossibleValues):
                if x[colIndices[i]] == 1.0:
                    # Add a feature at the end
                    x = np.r_[x, i / (nbPossibleValues-1)]
        # Delete all columns corresponding to one-hot encodded features
        colsToDelete = list(self.oneHotEncoding.values())
        colsToDelete = [item for sublist in colsToDelete for item in sublist]
        x = np.delete(x, colsToDelete, axis=0)
        return x

    def __unscale_feature_values(self, x_exp):
        """ Convert the normalized features back to their full range. """
        x = np.zeros_like(x_exp)
        for f in range(len(x)):
            x[f] = (x_exp[f] * (self.upperBoundsList[f]
                        - self.lowerBoundsList[f])
                        + self.lowerBoundsList[f])
        return x

    # -- Public methods --
    def format_explanation(self, x_exp):
        """
        Return the explanation in the original format:
            - Decode the one-hot-encoded categorical features.
            - Inverse transform of the minmax scaling.
        """
        xList = []
        for i in range(len(x_exp)):
            x = self.__unscale_feature_values(x_exp[i])
            x = self.__decode_categorical_features(x)
            xList.append(x)
        return np.array(xList)
