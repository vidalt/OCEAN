import math
import pandas as pd
import numpy as np
import pickle
import copy

from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.CounterFactualParameters import *

# apply the min-max scaling in Pandas using the .min() and .max() methods


def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / \
                           (df_norm[column].max() - df_norm[column].min())
    return df_norm


def removeComma(x):
    if type(x) == str:
        x = x.replace('"', '')
        x = x.replace(',', '')
        assert not math.isnan(float(x))


def min_max_scaling_with_type(df):
    # copy the dataframe
    df_norm = df.copy()
    columnFeatures = {column: getFeatureType(
        df[column][0]) for column in df.columns}
    df_norm = df_norm.drop(0)
    df_norm = df_norm.drop(1)
    # Convert types
    for column in df_norm.columns:
        if columnFeatures[column] in [FeatureType.Discrete, FeatureType.Numeric]:
            # df_norm[column] = df_norm[column].map(removeComma)
            df_norm[column] = df_norm[column].astype(float)
        elif columnFeatures[column] == FeatureType.Binary:
            df_norm[column] = df_norm[column].astype(float)

    # apply min-max scaling
    for column in df_norm.columns:
        if isFeatureTypeScalable(columnFeatures[column]) and column != 'Class':
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / \
                               (df_norm[column].max() - df_norm[column].min())
    return df_norm


class DatasetReader:
    def __init__(self, filename, rangeFeasibilityForDiscreteFeatures=False):
        self.filename = filename
        self.data = pd.read_csv(self.filename)
        # Get the datatypes and remove the corresponding row
        columnFeatures = {column: getFeatureType(
            self.data[column][0]) for column in self.data.columns if column != 'Class'}
        self.data = self.data.drop(0)
        columnActionnability = {column: getFeatureActionnability(
            self.data[column][1]) for column in self.data.columns if column != 'Class'}
        self.data = self.data.drop(1)

        # Remove columns
        for column in self.data.columns:
            if len(self.data[column].unique()) == 1:
                print("Drop column", column,
                      "because it contains a unique value")
                self.data.drop([column], axis=1, inplace=True)

        # Replace binary categories by bynaries
        for column in self.data.columns:
            if column != 'Class' and columnFeatures[column] == FeatureType.Binary:
                values = self.data[column].unique()
                if len(values) != 2:
                    ok = True
                    for val in values:
                        if val not in [0, 1, '0', '1']:
                            ok = False
                        if not ok:
                            print("error, more than two values in ", column,
                                  "which is indicated to be binary, treated as categorical")
                            columnFeatures[column] = FeatureType.Categorical
                if values[0] in ['0', '1'] and values[1] in ['0', '1']:
                    continue
                assert values[1] != '0'
                self.data[column] = self.data[column].str.replace(
                    values[0], '0')
                self.data[column] = self.data[column].str.replace(
                    values[1], '1')

        # One hot encoding
        oneHotEncoding = dict()
        categoricalColumns = [str(column) for column in self.data.columns if column
                              != 'Class' and columnFeatures[column] == FeatureType.Categorical]

        for column in categoricalColumns:
            if columnActionnability[column] == FeatureActionnability.Increasing:
                print("warning, categorical feature ", column,
                      "cannot be with increasing feasability, changed to free")
                columnActionnability[column] = FeatureActionnability.Free
            featureOneHot = pd.get_dummies(self.data[column], prefix=column)
            oneHotEncoding[column] = []
            for newCol in featureOneHot:
                columnFeatures[newCol] = FeatureType.Binary
                columnActionnability[newCol] = columnActionnability[column]
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

        # Reconstruct features type and one hot encoding
        self.featuresType = [columnFeatures[column]
                             for column in self.data.columns if column != 'Class']
        self.featuresActionnability = [columnActionnability[column]
                                       for column in self.data.columns if column != 'Class']

        # Convert data type
        for column in self.data.columns:
            if column == 'Class':
                self.data[column] = self.data[column].astype(float)
            elif columnFeatures[column] in [FeatureType.Discrete, FeatureType.Numeric]:
                # self.data[column] = self.data[column].map(removeComma)
                # for st in self.data[column]:
                #     print(st)
                #     float(st)
                self.data[column] = self.data[column].astype(float)
            elif columnFeatures[column] == FeatureType.Binary:
                self.data[column] = self.data[column].astype(float)

        # apply min-max scaling
        self.lowerBounds = dict()
        self.upperBounds = dict()

        self.lowerBoundsList = []
        self.upperBoundsList = []

        f = 0
        for column in self.data.columns:
            if column != 'Class' and isFeatureTypeScalable(columnFeatures[column]):
                self.lowerBounds[column] = self.data[column].min()
                if self.featuresType[f] == FeatureType.Discrete:
                    self.lowerBounds[column] = 0
                self.lowerBoundsList.append(self.lowerBounds[column])
                self.upperBounds[column] = self.data[column].max()
                self.upperBoundsList.append(self.upperBounds[column])
                self.data[column] = (self.data[column] - self.data[column].min()) / \
                    (self.data[column].max() - self.data[column].min())
            f += 1

        # Features possibles values
        c = 0
        self.featuresPossibleValues = []
        for column in self.data:
            if column != 'Class':
                if self.featuresType[c] == FeatureType.Categorical:
                    self.featuresPossibleValues.append(
                        self.data[column].unique())
                elif self.featuresType[c] == FeatureType.Discrete:
                    if rangeFeasibilityForDiscreteFeatures:
                        self.featuresPossibleValues.append(
                            [i/self.upperBoundsList[c] for i in range(0, int(self.upperBoundsList[c]+1))])
                    else:
                        self.featuresPossibleValues.append(
                            self.data[column].unique())
                # if self.featuresType[c] in [FeatureType.Discrete, FeatureType.Categorical]:
                #     self.featuresPossibleValues.append(self.data[column].unique())
                else:
                    self.featuresPossibleValues.append([])
            c += 1
        # Training set
        self.X = self.data.loc[:, self.data.columns != 'Class']
        self.y = self.data['Class']
        # Counterfactuals
        self.x0 = [self.data.iloc[0, self.data.columns != 'Class']]

        self.XwithGoodPoint = self.data.loc[self.data['Class']
                                            == 1, self.data.columns != 'Class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def readRandomForestFromPickleAndApplyMinMaxScaling(self, serializedObject) -> RandomForestClassifier:
        for f in self.featuresType:
            if f == FeatureType.Categorical:
                print("Pickle version does not work with categorical features")
                raise "Categorical features not handled"

        clfRead: RandomForestClassifier = pickle.load(
            open(serializedObject, "rb"))
        assert clfRead.n_features_ == len(self.featuresType)
        assert type(clfRead) == RandomForestClassifier

        clfScaled = copy.deepcopy(clfRead)

        for est in clfScaled.estimators_:
            tree = est.tree_
            for v in range(tree.capacity):
                f = tree.feature[v]
                tree.threshold[v] = (tree.threshold[v] - self.lowerBoundsList[f]) / \
                    (self.upperBoundsList[f]-self.lowerBoundsList[f])
                assert 0 <= tree.threshold[v] <= 1

        return clfRead, clfScaled


# --------------------------------------------------------------------------------
# Old datasets without actionnability
# --------------------------------------------------------------------------------
def min_max_scaling_with_type_oldDatasetsWithoutActionnability(df):
    # copy the dataframe
    df_norm = df.copy()
    columnFeatures = {column: getFeatureType_oldDatasetsWithoutActionnability(
        df[column][0]) for column in df.columns}
    df_norm = df_norm.drop(0)
    # Convert types
    for column in df_norm.columns:
        if columnFeatures[column] in [FeatureType.Discrete, FeatureType.Numeric]:
            # df_norm[column] = df_norm[column].map(removeComma)
            df_norm[column] = df_norm[column].astype(float)
        elif columnFeatures[column] == FeatureType.Binary:
            df_norm[column] = df_norm[column].astype(float)

    # apply min-max scaling
    for column in df_norm.columns:
        if isFeatureTypeScalable(columnFeatures[column]) and column != 'Class':
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / \
                               (df_norm[column].max() - df_norm[column].min())
    return df_norm


class DatasetReader_oldDatasetsWithoutActionnability:
    def __init__(self, filename):
        self.filename = filename
        data = pd.read_csv(self.filename)
        # Get the datatypes and remove the corresponding row
        self.featuresType = [getFeatureType_oldDatasetsWithoutActionnability(
            data[column][0]) for column in data.columns if column != 'Class']
        self.featuresActionnability = [
            FeatureActionnability.Free for column in data.columns if column != 'Class']
        # Rescale data
        self.data = min_max_scaling_with_type_oldDatasetsWithoutActionnability(
            data)
        # Features possibles values
        c = 0
        self.featuresPossibleValues = []
        for column in self.data:
            if column != 'Class':
                if self.featuresType[c] in [FeatureType.Discrete, FeatureType.Categorical]:
                    self.featuresPossibleValues.append(
                        self.data[column].unique())
                else:
                    self.featuresPossibleValues.append([])
            c += 1
        # Training set
        self.X = self.data.loc[:, self.data.columns != 'Class']
        self.y = self.data['Class']
        # Counterfactuals
        self.x0 = [self.data.loc[1, self.data.columns != 'Class']]
