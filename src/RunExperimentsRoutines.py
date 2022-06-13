import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
# Load OCEAN functions
from src.dataProcessing import DatasetReader
from src.RandomForestCounterFactual import RandomForestCounterFactualMilp
from src.CuiRandomForestCounterFactual import CuiRandomForestCounterFactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.writeResults import writeLegend


def trainModelAndSolveCounterFactuals(
        trainingSetFile, counterfactualsFile,
        rf_max_depth=7, rf_n_estimators=100,
        ilfActivated=True, ilf_max_samples=128, ilf_n_estimators=100,
        random_state=0, useCui=False,
        constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
        actionnabilityActivated=True, objectiveNorm=1,
        mutuallyExclusivePlanesCutsActivated=True,
        strictCounterFactual=True,
        binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
        randomCostsActivated=False,
        numericalResultsFileName="NumericalResults.csv",
        nbCounterFactualsComputed="all",
        writeColNames=False):
    if writeColNames:
        writeLegend(numericalResultsFileName)

    # - Load and read data from file -
    reader = DatasetReader(trainingSetFile)
    # - Train random and isolation forests -
    # Train random forest
    clf = RandomForestClassifier(
        max_depth=rf_max_depth, random_state=random_state, n_estimators=rf_n_estimators)
    clf.fit(reader.X_train.values, reader.y_train.values)
    train_score = clf.score(reader.X_train, reader.y_train)
    test_score = clf.score(reader.X_test, reader.y_test)
    print("Random forest with", clf.n_estimators, "estimators with max depth",
          clf.max_depth, "and max leaf nodes", clf.max_leaf_nodes)
    nodes = [est.tree_.node_count for est in clf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    # Train isolation forest
    ilf = IsolationForest(random_state=random_state, max_samples=ilf_max_samples,
                          n_estimators=ilf_n_estimators, contamination=0.1)
    ilf.fit(reader.XwithGoodPoint.values)
    print("Isolation forest with", ilf.n_estimators,
          "estimatorswith max samples", ilf.max_samples)
    nodes = [est.tree_.node_count for est in ilf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")
    ilfForMilp = None
    if ilfActivated:
        assert not useCui
        ilfForMilp = ilf

    # - Prepare the set of initial observations for counterfactuals -
    cfData = pd.read_csv(counterfactualsFile)
    count = 1
    for index in cfData.index:
        isFinished = (nbCounterFactualsComputed != 'all'
                      and type(nbCounterFactualsComputed) == int
                      and count == nbCounterFactualsComputed + 1)
        if isFinished:
            break
        print("Start launching", count, "out of", len(cfData.index))
        count += 1
        x0 = [cfData.iloc[index, cfData.columns != 'DesiredOutcome']]
        y0_desired = cfData['DesiredOutcome'][index]
        featuresActionnability = False
        if actionnabilityActivated:
            featuresActionnability = reader.featuresActionnability
        cuiMilp = None
        randomForestMilp = None
        # - Solve optimization model to find counterfactual -
        if useCui:
            cuiMilp = CuiRandomForestCounterFactualMilp(
                clf, x0, y0_desired,
                isolationForest=ilfForMilp,
                objectiveNorm=objectiveNorm,
                verbose=False,
                featuresType=reader.featuresType,
                featuresPossibleValues=reader.featuresPossibleValues,
                strictCounterFactual=strictCounterFactual,
                featuresActionnability=featuresActionnability)
            cuiMilp.buildModel()
            cuiMilp.solveModel()
        else:
            # Compute counterfactual
            randomForestMilp = RandomForestCounterFactualMilp(
                clf, x0, y0_desired,
                isolationForest=ilfForMilp,
                constraintsType=constraintsType,
                objectiveNorm=objectiveNorm,
                mutuallyExclusivePlanesCutsActivated=mutuallyExclusivePlanesCutsActivated,
                strictCounterFactual=strictCounterFactual,
                verbose=False,
                binaryDecisionVariables=binaryDecisionVariables,
                featuresActionnability=featuresActionnability,
                featuresType=reader.featuresType,
                featuresPossibleValues=reader.featuresPossibleValues,
                randomCostsActivated=randomCostsActivated)
            randomForestMilp.buildModel()
            randomForestMilp.solveModel()
        # - Write solution to csv file -
        write = open(numericalResultsFileName, "a")
        # Instance
        if isinstance(trainingSetFile, Path):
            trainingSetFile = trainingSetFile.name
        write.write(trainingSetFile)
        write.write(","+str(rf_max_depth)+','+str(rf_n_estimators)
                    + ','+str(ilfActivated)+','+str(ilf_max_samples)
                    + ','+str(ilf_n_estimators)
                    + ','+str(random_state)+','+str(train_score)
                    + ','+str(test_score))
        # Counterfactual
        if isinstance(counterfactualsFile, Path):
            counterfactualsFile = counterfactualsFile.name
        write.write("," + counterfactualsFile+"," + str(index))
        # Solver Parameters
        write.write(','+str(useCui)+','+str(constraintsType)
                    + ','+str(actionnabilityActivated)+','+str(objectiveNorm)
                    + ','+str(mutuallyExclusivePlanesCutsActivated)
                    + ','+str(strictCounterFactual)
                    + ','+str(binaryDecisionVariables))
        if useCui:
            isCuiCfValid = (cuiMilp.outputDesired
                            != clf.predict(cuiMilp.x_sol)[0])
            write.write(","+str(cuiMilp.model.status)
                        + ","+str(cuiMilp.model.Runtime)
                        + ","+str(cuiMilp.objValue)
                        + ","+str(isCuiCfValid)
                        + ','+str("cuiUndefined")
                        + ','+str("cuiUndefined")
                        + ","+str(ilf.predict(cuiMilp.x_sol)[0]))
            for x_i in cuiMilp.x_sol[0]:
                write.write(','+str(x_i))
        else:
            # Solver results
            isCfValid = (randomForestMilp.outputDesired
                         != clf.predict(randomForestMilp.x_sol)[0])
            write.write(","+str(randomForestMilp.model.status)
                        + ","+str(randomForestMilp.runTime)
                        + ","+str(randomForestMilp.objValue)
                        + ","+str(isCfValid)
                        + ','+str(randomForestMilp.maxSkLearnError)
                        + ','+str(randomForestMilp.maxMyMilpError)
                        + ","+str(ilf.predict(randomForestMilp.x_sol)[0]))
            # Solver Solution
            for x_i in randomForestMilp.x_sol[0]:
                write.write(','+str(x_i))
        # Finish
        write.write("\n")
        write.close()


def runNumericalExperiments(
    datasetsWithCounterfactualsDict,
    rf_max_depthList=[7],
    rf_n_estimatorsList=[100],
    ilfActivatedList=[True],
    ilf_max_samplesList=[128],
    ilf_n_estimatorsList=[100],
    random_stateList=[0],
    useCuiList=[False],
    constraintsTypeList=[TreeConstraintsType.LinearCombinationOfPlanes],
    actionnabilityActivatedList=[True],
    objectiveNormList=[1],
    mutuallyExclusivePlanesCutsActivatedList=[True],
    strictCounterFactualList=[True],
    binaryDecisionVariablesList=[BinaryDecisionVariables.PathFlow_y],
    randomCostsActivated=False,
    numericalResultsFileName="NumericalResults.csv"
):
    writeLegend(numericalResultsFileName)

    count = 0
    for trainingSetFile in datasetsWithCounterfactualsDict:
        counterfactualsFile = datasetsWithCounterfactualsDict[trainingSetFile]
        for rf_max_depth in rf_max_depthList:
            for rf_n_estimators in rf_n_estimatorsList:
                for ilfActivated in ilfActivatedList:
                    for ilf_max_samples in ilf_max_samplesList:
                        for ilf_n_estimators in ilf_n_estimatorsList:
                            for random_state in random_stateList:
                                for useCui in useCuiList:
                                    for constraintsType in constraintsTypeList:
                                        for actionnabilityActivated in actionnabilityActivatedList:
                                            for objectiveNorm in objectiveNormList:
                                                for mutuallyExclusivePlanesCutsActivated in mutuallyExclusivePlanesCutsActivatedList:
                                                    for strictCounterFactual in strictCounterFactualList:
                                                        for binaryDecisionVariables in binaryDecisionVariablesList:
                                                            count += 1

    total = count
    count = 0
    for trainingSetFile in datasetsWithCounterfactualsDict:
        counterfactualsFile = datasetsWithCounterfactualsDict[trainingSetFile]
        for rf_max_depth in rf_max_depthList:
            for rf_n_estimators in rf_n_estimatorsList:
                for ilfActivated in ilfActivatedList:
                    for ilf_max_samples in ilf_max_samplesList:
                        for ilf_n_estimators in ilf_n_estimatorsList:
                            for random_state in random_stateList:
                                for useCui in useCuiList:
                                    for constraintsType in constraintsTypeList:
                                        for actionnabilityActivated in actionnabilityActivatedList:
                                            for objectiveNorm in objectiveNormList:
                                                for mutuallyExclusivePlanesCutsActivated in mutuallyExclusivePlanesCutsActivatedList:
                                                    for strictCounterFactual in strictCounterFactualList:
                                                        for binaryDecisionVariables in binaryDecisionVariablesList:
                                                            print(
                                                                "\n\nLaunch Numerical experiments sequence", count, "out of", total)
                                                            print(
                                                                "trainingSetFile", trainingSetFile,)
                                                            print(
                                                                "counterfactualsFile", counterfactualsFile)
                                                            print(
                                                                "rf_max_depth", rf_max_depth)
                                                            print(
                                                                "rf_n_estimators", rf_n_estimators)
                                                            print(
                                                                "ilfActivated", ilfActivated)
                                                            print(
                                                                "ilf_max_samples", ilf_max_samples)
                                                            print(
                                                                "ilf_n_estimators", ilf_n_estimators)
                                                            print(
                                                                "random_state ", random_state)
                                                            print(
                                                                "useCui ", useCui)
                                                            print(
                                                                "constraintsType", constraintsType)
                                                            print(
                                                                "actionnabilityActivated", actionnabilityActivated)
                                                            print(
                                                                "objectiveNorm", objectiveNorm)
                                                            print(
                                                                "mutuallyExclusivePlanesCutsActivated", mutuallyExclusivePlanesCutsActivated)
                                                            print(
                                                                "strictCounterFactual", strictCounterFactual)
                                                            print(
                                                                "binaryDecisionVariables", binaryDecisionVariables)
                                                            print(
                                                                "numericalResultsFileName", numericalResultsFileName)
                                                            count += 1
                                                            trainModelAndSolveCounterFactuals(
                                                                trainingSetFile,
                                                                counterfactualsFile,
                                                                rf_max_depth=rf_max_depth,
                                                                rf_n_estimators=rf_n_estimators,
                                                                ilfActivated=ilfActivated,
                                                                ilf_max_samples=ilf_max_samples,
                                                                ilf_n_estimators=ilf_n_estimators,
                                                                random_state=random_state,
                                                                useCui=useCui,
                                                                constraintsType=constraintsType,
                                                                actionnabilityActivated=actionnabilityActivated,
                                                                objectiveNorm=objectiveNorm,
                                                                mutuallyExclusivePlanesCutsActivated=mutuallyExclusivePlanesCutsActivated,
                                                                strictCounterFactual=strictCounterFactual,
                                                                binaryDecisionVariables=binaryDecisionVariables,
                                                                numericalResultsFileName=numericalResultsFileName,
                                                                randomCostsActivated=randomCostsActivated
                                                            )


miniDatasetsWithCounterfactualsDict = {
    "./datasets/test.csv": "./datasets/counterfactuals/OneHot_test.csv"
}


cuiDatasetsWithCounterfactualsDict = {
    './datasets/Adult_processedMACE.csv': './datasets/counterfactuals/OneHot_Adult_processedMACE.csv',
    # './datasets/COMPAS-ProPublica_processedMACE.csv':'./datasets/counterfactuals/OneHot_COMPAS-ProPublica_processedMACE.csv',
    # './datasets/Credit-Card-Default_processedMACE.csv':'./datasets/counterfactuals/OneHot_Credit-Card-Default_processedMACE.csv',
    # './datasets/FICO.csv':'./datasets/counterfactuals/OneHot_FICO.csv',
    # './datasets/German-Credit.csv':'./datasets/counterfactuals/OneHot_German-Credit.csv',
    # './datasets/Phishing.csv':'./datasets/counterfactuals/OneHot_Phishing.csv',
    # './datasets/Spambase.csv':'./datasets/counterfactuals/OneHot_Spambase.csv',
    # './datasets/Students-Performance-MAT.csv':'./datasets/counterfactuals/OneHot_Students-Performance-MAT.csv',
    # './datasets/Students-Performance-POR.csv':'./datasets/counterfactuals/OneHot_Students-Performance-POR.csv',
    # './datasets/Adult.csv':'./datasets/counterfactuals/OneHot_Adult.csv',
    # './datasets/OnlineNewsPopularity.csv':'./datasets/counterfactuals/OneHot_OnlineNewsPopularity.csv',
    # './datasets/Credit-Card-Default.csv':'./datasets/counterfactuals/OneHot_Credit-Card-Default.csv',
    # './datasets/COMPAS-ProPublica.csv':'./datasets/counterfactuals/OneHot_COMPAS-ProPublica.csv',
}
