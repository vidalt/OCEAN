import pandas as pd
from sklearn.model_selection import train_test_split
# Import OCEAN utility functions
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import FeatureActionnability
from src.CounterFactualParameters import getFeatureType
from src.CounterFactualParameters import getFeatureActionnability
from src.CounterFactualParameters import isFeatureTypeScalable


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
