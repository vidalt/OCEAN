import os

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from src.dataProcessing import *
from src.DecisionTreeCounterFactual import *
from src.RandomForestCounterFactual import *
from src.CuiRandomForestCounterFactual import *


def writeLegend(numericalResultsFileName):
    write = open(numericalResultsFileName, "a")
    # Instance
    write.write("trainingSetFile")
    write.write(","+"rf_max_depth")
    write.write(','+"rf_n_estimators")
    write.write(','+"ilfActivated")
    write.write(','+"ilf_max_samples")
    write.write(','+"ilf_n_estimators")
    write.write(','+"random_state")
    write.write(','+"train_score")
    write.write(","+"test_score")
    # Counterfactual
    write.write("," + "counterfactualsFile")
    write.write("," + "counterfactual_index")
    # Solver Parameters
    write.write(','+"useCui")
    write.write(','+"constraintsType")
    write.write(','+"actionnabilityActivated")
    write.write(','+"objectiveNorm")
    write.write(','+"mutuallyExclusivePlanesCutsActivated")
    write.write(','+"strictCounterFactual")
    write.write(','+"binaryDecisionVariables")
    # Solver results
    write.write(","+"randomForestMilp.model.status")
    write.write(","+"randomForestMilp.runTime")
    write.write(","+"randomForestMilp.objValue")
    write.write(","+"notOuputDesired")
    write.write(","+"maxSkLearnError")
    write.write(","+"maxMyMilpError")
    write.write(","+"plausible")
    # Solver Solution
    write.write(','+"solution")
    # Finish
    write.write("\n")
    write.close()


def trainModelAndSolveCounterFactuals(
    trainingSetFile,
    counterfactualsFile,
    rf_max_depth=7,
    rf_n_estimators=100,
    ilfActivated=True,
    ilf_max_samples=128,
    ilf_n_estimators=100,
    random_state=0,
    useCui=False,
    constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
    actionnabilityActivated=True,
    objectiveNorm=1,
    mutuallyExclusivePlanesCutsActivated=True,
    strictCounterFactual=True,
    binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
    randomCostsActivated=False,
    numericalResultsFileName="NumericalResults.csv",
    nbCounterFactualsComputed="all",
    writeColNames=False
):
    if writeColNames:
        writeLegend(numericalResultsFileName)

    reader = DatasetReader(trainingSetFile)
    # Train random forest
    clf = RandomForestClassifier(
        max_depth=rf_max_depth, random_state=random_state, n_estimators=rf_n_estimators)
    clf.fit(reader.X_train, reader.y_train)

    train_score = clf.score(reader.X_train, reader.y_train)
    test_score = clf.score(reader.X_test, reader.y_test)

    print("Random forest with", clf.n_estimators, "estimators with max depth",
          clf.max_depth, "and max leaf nodes", clf.max_leaf_nodes)
    nodes = [est.tree_.node_count for est in clf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    ilf = IsolationForest(random_state=random_state, max_samples=ilf_max_samples,
                          n_estimators=ilf_n_estimators, contamination=0.1)
    ilf.fit(reader.XwithGoodPoint)
    print("Isolation forest with", ilf.n_estimators,
          "estimatorswith max samples", ilf.max_samples)
    nodes = [est.tree_.node_count for est in ilf.estimators_]
    print(sum(nodes)/len(nodes), "nodes on average")

    ilfForMilp = None
    if ilfActivated:
        assert not useCui
        ilfForMilp = ilf

    counterfactualsData = pd.read_csv(counterfactualsFile)
    count = 1
    for index in counterfactualsData.index:
        if nbCounterFactualsComputed != 'all' and type(nbCounterFactualsComputed) == int and count == nbCounterFactualsComputed + 1:
            break
        print("Start launching", count, "out of",
              len(counterfactualsData.index))
        count += 1
        x0 = [counterfactualsData.iloc[index,
                                       counterfactualsData.columns != 'DesiredOutcome']]
        y0_desired = counterfactualsData['DesiredOutcome'][index]

        featuresActionnability = False
        if actionnabilityActivated:
            featuresActionnability = reader.featuresActionnability

        cuiMilp = None
        randomForestMilp = None
        if useCui:
            cuiMilp = CuiRandomForestCounterFactualMilp(clf, x0, y0_desired,
                                                        isolationForest=ilfForMilp,
                                                        objectiveNorm=objectiveNorm,
                                                        verbose=False,
                                                        featuresType=reader.featuresType,
                                                        featuresPossibleValues=reader.featuresPossibleValues,
                                                        strictCounterFactual=strictCounterFactual,
                                                        featuresActionnability=featuresActionnability
                                                        )
            cuiMilp.buildModel()
            cuiMilp.solveModel()
        else:
            # Compute counterfactual
            randomForestMilp = RandomForestCounterFactualMilp(
                clf,
                x0,
                y0_desired,
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
                randomCostsActivated=randomCostsActivated
            )
            randomForestMilp.buildModel()
            randomForestMilp.solveModel()
        write = open(numericalResultsFileName, "a")
        # Instance
        if isinstance(trainingSetFile, Path):
            trainingSetFile = trainingSetFile.name
        write.write(trainingSetFile)
        write.write(","+str(rf_max_depth))
        write.write(','+str(rf_n_estimators))
        write.write(','+str(ilfActivated))
        write.write(','+str(ilf_max_samples))
        write.write(','+str(ilf_n_estimators))
        write.write(','+str(random_state))
        write.write(','+str(train_score))
        write.write(','+str(test_score))
        # write.write(',')
        # Counterfactual
        if isinstance(counterfactualsFile, Path):
            counterfactualsFile = counterfactualsFile.name
        write.write("," + counterfactualsFile)
        write.write("," + str(index))
        # write.write(',')
        # Solver Parameters
        write.write(','+str(useCui))
        write.write(','+str(constraintsType))
        write.write(','+str(actionnabilityActivated))
        write.write(','+str(objectiveNorm))
        write.write(','+str(mutuallyExclusivePlanesCutsActivated))
        write.write(','+str(strictCounterFactual))
        write.write(','+str(binaryDecisionVariables))
        # write.write(',')
        if useCui:
            write.write(","+str(cuiMilp.model.status))
            write.write(","+str(cuiMilp.model.Runtime))
            write.write(","+str(cuiMilp.objValue))
            write.write(","+str(cuiMilp.outputDesired
                        != clf.predict(cuiMilp.x_sol)[0]))
            write.write(','+str("cuiUndefined"))
            write.write(','+str("cuiUndefined"))
            write.write(","+str(ilf.predict(cuiMilp.x_sol)[0]))
            for x_i in cuiMilp.x_sol[0]:
                write.write(','+str(x_i))
        else:
            # Solver results
            write.write(","+str(randomForestMilp.model.status))
            write.write(","+str(randomForestMilp.runTime))
            write.write(","+str(randomForestMilp.objValue))
            write.write(","+str(randomForestMilp.outputDesired
                        != clf.predict(randomForestMilp.x_sol)[0]))
            write.write(','+str(randomForestMilp.maxSkLearnError))
            write.write(','+str(randomForestMilp.maxMyMilpError))
            write.write(","+str(ilf.predict(randomForestMilp.x_sol)[0]))
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

ourDatasetsWithCounterfactualsDict = {
    './datasets/Adult_processedMACE.csv': './datasets/counterfactuals/OneHot_Adult_processedMACE.csv',
    './datasets/COMPAS-ProPublica_processedMACE.csv': './datasets/counterfactuals/OneHot_COMPAS-ProPublica_processedMACE.csv',
    './datasets/German-Credit.csv': './datasets/counterfactuals/OneHot_German-Credit.csv',
    './datasets/Phishing.csv': './datasets/counterfactuals/OneHot_Phishing.csv',
    './datasets/Spambase.csv': './datasets/counterfactuals/OneHot_Spambase.csv',
    './datasets/Students-Performance-MAT.csv': './datasets/counterfactuals/OneHot_Students-Performance-MAT.csv',
    './datasets/Credit-Card-Default_processedMACE.csv': './datasets/counterfactuals/OneHot_Credit-Card-Default_processedMACE.csv',
    './datasets/OnlineNewsPopularity.csv': './datasets/counterfactuals/OneHot_OnlineNewsPopularity.csv',
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
