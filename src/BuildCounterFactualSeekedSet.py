import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from pathlib import Path

from src.dataProcessing import *
from src.DecisionTreeCounterFactual import *
from src.RandomForestCounterFactual import *
from src.CuiRandomForestCounterFactual import *


def checkFeasibilityOfCounterFactuals(clf, ilf, reader, indices, desiredOutcome):
    allSolved = True
    count = 1
    for index in indices:
        print("Start cheking", count, "out of", len(indices))
        count += 1
        x0 = [reader.data.loc[index, reader.data.columns != 'Class']]
        randomForestMilp = RandomForestCounterFactualMilp(
            clf,
            x0,
            desiredOutcome,
            isolationForest=ilf,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=1, mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True, verbose=False,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            featuresActionnability=reader.featuresActionnability,
            featuresType=reader.featuresType,
            featuresPossibleValues=reader.featuresPossibleValues
        )
        randomForestMilp.buildModel()
        if not randomForestMilp.solveModel():
            print("Warning, index", index, "could not be solved")
            allSolved = False
    if allSolved:
        print("All models could be solved")
    else:
        print("Not all models could be solved")


def buildCounterFactualSeekedFile(datasetFile, desiredOutcome, nbCounterFactuals, checkFeasibility=False):
    print("Start treating", datasetFile)

    reader = DatasetReader(datasetFile)

    # Clf
    clf = RandomForestClassifier(
        max_leaf_nodes=50, random_state=1, n_estimators=100)
    clf.fit(reader.X_train, reader.y_train)
    print("Random forest with", clf.n_estimators, "estimators with max depth",
          clf.max_depth, "and max leaf nodes", clf.max_leaf_nodes)
    nodes = [est.tree_.node_count for est in clf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    # Ilf
    ilf = IsolationForest(random_state=1, max_samples=100, n_estimators=100)
    ilf.fit(reader.X_train)
    print("Isolation forest with", ilf.n_estimators,
          "estimators with max samples", ilf.max_samples)
    nodes = [est.tree_.node_count for est in ilf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    # Complete data
    data = pd.DataFrame(reader.X_test)
    X = data
    predictions = clf.predict(X)
    data["clf_result"] = predictions
    data['Class'] = reader.y_test
    dataWitoutDesiredResults = data.loc[(data['Class'] != desiredOutcome) & (
        data['clf_result'] != desiredOutcome)]
    data.drop(['clf_result'], axis=1, inplace=True)

    # Samples
    if len(dataWitoutDesiredResults) > nbCounterFactuals:
        dataWitoutDesiredResults = dataWitoutDesiredResults.sample(
            n=nbCounterFactuals)

    # Feasibility
    dataWitoutDesiredResults.drop(['clf_result'], axis=1, inplace=True)
    if checkFeasibility:
        checkFeasibilityOfCounterFactuals(
            clf, ilf, reader, dataWitoutDesiredResults.index, desiredOutcome)

    # Results in oneHotEncodedFormat
    data.drop(['Class'], axis=1, inplace=True)
    data['DesiredOutcome'] = desiredOutcome
    data = data.loc[dataWitoutDesiredResults.index, :]

    # Results in initial format
    result = pd.read_csv(datasetFile)
    result.drop(['Class'], axis=1, inplace=True)
    result['DesiredOutcome'] = desiredOutcome
    result = result.loc[dataWitoutDesiredResults.index, :]

    # Output
    if isinstance(datasetFile, Path):
        datasetName = datasetFile.name
        pathToCounterfactual = datasetFile.parent / 'counterfactuals'
        if not os.path.exists(pathToCounterfactual):
            os.mkdir(pathToCounterfactual)
        outputFile = pathToCounterfactual / datasetName
        result.to_csv(outputFile, index=False)
        oneHotDatasetName = "OneHot_" + datasetName
        outputFile = pathToCounterfactual / oneHotDatasetName
        data.to_csv(outputFile, index=False)
    else:
        words = datasetFile.split('/')
        path = ""
        for w in words[:-1]:
            path += w + "/"
        path += "counterfactuals/"
        if not os.path.exists(path):
            os.mkdir(path)
        outputFile = path + words[-1]
        result.to_csv(outputFile, index=False)

        outputFile = path + "OneHot_" + words[-1]
        data.to_csv(outputFile, index=False)
