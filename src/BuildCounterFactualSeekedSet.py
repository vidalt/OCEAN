import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from pathlib import Path
# Import OCEAN functions
from src.dataProcessing import DatasetReader
from src.RfClassifierCounterFactual import RfClassifierCounterFactualMilp
from src.CounterFactualParameters import TreeConstraintsType
from src.CounterFactualParameters import BinaryDecisionVariables


def checkFeasibilityOfCounterFactuals(clf, ilf, reader,
                                      indices, desiredOutcome):
    allSolved = True
    count = 1
    for index in indices:
        print("Start checking", count, "out of", len(indices))
        count += 1
        x0 = [reader.data.loc[index, reader.data.columns != 'Class']]
        randomForestMilp = RfClassifierCounterFactualMilp(
            clf, x0, desiredOutcome,
            isolationForest=ilf,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=1, mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True, verbose=False,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            featuresActionnability=reader.featuresActionnability,
            featuresType=reader.featuresType,
            featuresPossibleValues=reader.featuresPossibleValues)
        randomForestMilp.buildModel()
        if not randomForestMilp.solveModel():
            print("Warning, index", index, "could not be solved")
            allSolved = False
    if allSolved:
        print("All models could be solved")
    else:
        print("Not all models could be solved")


def buildCounterFactualSeekedFile(datasetFile, desiredOutcome,
                                  nbCounterFactuals, checkFeasibility=False):
    print("Start treating", datasetFile)
    # Read data from file
    reader = DatasetReader(datasetFile)
    data = pd.DataFrame(reader.X_test)

    # Create and train a RandomForestClassifier
    clf = RandomForestClassifier(max_leaf_nodes=50, random_state=1,
                                 n_estimators=100)
    clf.fit(reader.X_train, reader.y_train)
    print("Random forest with", clf.n_estimators,
          "estimators with max depth", clf.max_depth,
          "and max leaf nodes", clf.max_leaf_nodes)
    nodes = [est.tree_.node_count for est in clf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")
    predictions = clf.predict(data)
    data['clf_result'] = predictions
    data['Class'] = reader.y_test

    # Sample initial samples for which to get counterfactuals
    dataWithoutDesiredResults = data.loc[(data['Class'] != desiredOutcome) & (
        data['clf_result'] != desiredOutcome)]
    data.drop(['clf_result'], axis=1, inplace=True)
    if len(dataWithoutDesiredResults) > nbCounterFactuals:
        dataWithoutDesiredResults = dataWithoutDesiredResults.sample(
            n=nbCounterFactuals)
    dataWithoutDesiredResults.drop(['clf_result'], axis=1, inplace=True)

    # Check feasibility of finding optimal counterfactuals
    if checkFeasibility:
        # Create and fit an IsolationForest
        ilf = IsolationForest(
            random_state=1, max_samples=100, n_estimators=100)
        ilf.fit(reader.X_train)
        print("Isolation forest with", ilf.n_estimators,
              "estimators with max samples", ilf.max_samples)
        nodes = [est.tree_.node_count for est in ilf.estimators_]
        print(sum(nodes)/len(nodes), "nodes on average")
        checkFeasibilityOfCounterFactuals(
            clf, ilf, reader, dataWithoutDesiredResults.index, desiredOutcome)

    # -- Output results to csv files --
    # Read paths to files and create folder
    outputFile, oneHotOutputFile = get_paths_to_counterfactuals_directory(
        datasetFile)
    # Write to file: Results in initial format
    result = pd.read_csv(datasetFile)
    result.drop(['Class'], axis=1, inplace=True)
    result['DesiredOutcome'] = desiredOutcome
    result = result.loc[dataWithoutDesiredResults.index, :]
    result.to_csv(outputFile, index=False)
    # Write to file: Results in oneHotEncodedFormat
    data.drop(['Class'], axis=1, inplace=True)
    data['DesiredOutcome'] = desiredOutcome
    data = data.loc[dataWithoutDesiredResults.index, :]
    data.to_csv(oneHotOutputFile, index=False)


def get_paths_to_counterfactuals_directory(datasetFile):
    """
    Read paths to files and create folder:
    path can be either a string or a Path object.
    """
    if isinstance(datasetFile, Path):
        datasetName = datasetFile.name
        pathToCounterfactual = datasetFile.parent / 'counterfactuals'
        if not os.path.exists(pathToCounterfactual):
            os.mkdir(pathToCounterfactual)
        outputFile = pathToCounterfactual / datasetName
        oneHotDatasetName = "OneHot_" + datasetName
        oneHotOutputFile = pathToCounterfactual / oneHotDatasetName
    else:
        words = datasetFile.split('/')
        path = ""
        for w in words[:-1]:
            path += w + "/"
        path += "counterfactuals/"
        if not os.path.exists(path):
            os.mkdir(path)
        outputFile = path + words[-1]
        oneHotOutputFile = path + "OneHot_" + words[-1]
    return outputFile, oneHotOutputFile
