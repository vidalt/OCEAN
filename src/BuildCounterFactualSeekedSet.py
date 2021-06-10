import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from dataProcessing import *
from DecisionTreeCounterFactual import *
from RandomForestCounterFactual import *
from CuiRandomForestCounterFactual import *


def checkFeasibilityOfCounterFactuals(clf, ilf, reader, indices, desiredOutcome):
    allSolved = True
    count = 1
    for index in indices:
        print("Start cheking", count, "out of", len(indices))
        count += 1
        x0 = [reader.data.loc[index,reader.data.columns != 'Class']]
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
    clf = RandomForestClassifier(max_leaf_nodes=50, random_state=1,n_estimators=100)
    # clf = RandomForestClassifier(random_state=1)
    clf.fit(reader.X_train, reader.y_train)
    print("Random forest with", clf.n_estimators, "estimators with max depth", clf.max_depth, "and max leaf nodes", clf.max_leaf_nodes)
    nodes = [est.tree_.node_count for est in clf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    # Ilf
    ilf = IsolationForest(random_state=1, max_samples=100, n_estimators=100)
    # ilf = IsolationForest(random_state=1)
    ilf.fit(reader.X_train)
    print("Isolation forest with", ilf.n_estimators, "estimators with max samples", ilf.max_samples)
    nodes = [est.tree_.node_count for est in ilf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    # Complete data
    # data = reader.data
    # X = data.loc[:, data.columns != 'Class']
    data = pd.DataFrame(reader.X_test)
    X = data
    predictions = clf.predict(X)
    data["clf_result"] = predictions
    data['Class'] = reader.y_test
    dataWitoutDesiredResults = data.loc[(data['Class'] != desiredOutcome) & (data['clf_result'] != desiredOutcome)]
    data.drop(['clf_result'], axis=1, inplace=True)

    # Samples
    if len(dataWitoutDesiredResults) > nbCounterFactuals:
        dataWitoutDesiredResults = dataWitoutDesiredResults.sample(n=nbCounterFactuals)

    # Feasibility
    dataWitoutDesiredResults.drop(['clf_result'], axis=1, inplace=True)
    if checkFeasibility:
        checkFeasibilityOfCounterFactuals(clf,ilf,reader,dataWitoutDesiredResults.index,desiredOutcome)

    # Results in oneHotEncodedFormat
    data.drop(['Class'], axis=1, inplace=True)
    data['DesiredOutcome'] = desiredOutcome
    data = data.loc[dataWitoutDesiredResults.index,:]

    # Results in initial format
    result = pd.read_csv(datasetFile)
    result.drop(['Class'], axis=1, inplace=True)
    result['DesiredOutcome'] = desiredOutcome
    result = result.loc[dataWitoutDesiredResults.index,:]


    # output
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

#----------------------------------------------------------------------------------------------
# Test
# buildCounterFactualSeekedFile('./datasets/test.csv', 1, 1, True)

datasetsWithDesiredOutcome = {
    # './datasets/test.csv':1,
    './datasets/Adult_processedMACE.csv':1,
    # './datasets/Adult.csv':1,
    # './datasets/COMPAS-ProPublica_processedMACE.csv':1,
    # './datasets/COMPAS-ProPublica.csv':1,
    # './datasets/Credit-Card-Default_processedMACE.csv':1,
    # './datasets/Credit-Card-Default.csv':1,
    # './datasets/FICO.csv':1,
    # './datasets/German-Credit.csv':1,
    # './datasets/Phishing.csv':1,
    # './datasets/Spambase.csv':1,
    # './datasets/Students-Performance-MAT.csv':1,
    # './datasets/Students-Performance-POR.csv':1,
    # './datasets/OnlineNewsPopularity.csv':1
}

for dataset in datasetsWithDesiredOutcome:
    buildCounterFactualSeekedFile(dataset, datasetsWithDesiredOutcome[dataset], 20, False)