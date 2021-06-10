from dataProcessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import pandas as pd
import pickle
from RandomForestCounterFactual import *

def checkSamples(
        datasetFileName,
        unscaledFactualsFileName,
        unscaledCounterFactualsFileName,
        serializedClassifierFileName,
        roundMace = False
    ):

    reader = DatasetReader(datasetFileName,rangeFeasibilityForDiscreteFeatures=True)

    # Classifier
    clfRead, clfScaled = reader.readRandomForestFromPickleAndApplyMinMaxScaling(serializedClassifierFileName)

    # Factuals
    unscaledFactuals = pd.read_csv(unscaledFactualsFileName)
    scaledFactuals = pd.DataFrame()

    # Counterfactuals
    unscaledCounterFactuals = pd.read_csv(unscaledCounterFactualsFileName)
    scaledCounterFactuals = pd.DataFrame()

    if roundMace:
        for f in range(len(unscaledFactuals.columns)):
            if unscaledCounterFactuals.columns[f] != 'DesiredOutcome' and reader.featuresType[f] == FeatureType.Discrete:
                unscaledCounterFactuals[unscaledCounterFactuals.columns[f]] = round(unscaledCounterFactuals[unscaledFactuals.columns[f]])

    # min-max scaling
    for column in unscaledCounterFactuals.columns:
        if column != 'DesiredOutcome':
            scaledFactuals[column] = (unscaledFactuals[column] - reader.lowerBounds[column]) / (reader.upperBounds[column] - reader.lowerBounds[column])
            scaledCounterFactuals[column] = (unscaledCounterFactuals[column] - reader.lowerBounds[column]) / (reader.upperBounds[column] - reader.lowerBounds[column])

    for index in unscaledCounterFactuals.index:
        ufX = [unscaledFactuals.iloc[index, unscaledCounterFactuals.columns != 'DesiredOutcome']]
        sfX = [scaledFactuals.iloc[index]]
        ucX = [unscaledCounterFactuals.iloc[index, unscaledCounterFactuals.columns != 'DesiredOutcome']]
        scX = [scaledCounterFactuals.iloc[index]]
        y = unscaledCounterFactuals['DesiredOutcome'][index]
        ufy = clfRead.predict(ufX)
        sfy = clfScaled.predict(ufX)
        if ucX[0][1] == float('inf'):
            ucy = not y
            scy = not y
        else:
            ucy = clfRead.predict(ucX)
            scy = clfScaled.predict(scX)
        
        readOneNorm = 0
        for i in range(len(sfX[0])):
            readOneNorm += abs(sfX[0][i] - scX[0][i])

        randomForestMilp = RandomForestCounterFactualMilp(
            clfScaled,
            sfX,
            y,
            isolationForest=False, 
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=1,
            mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            randomCostsActivated=False,
            verbose=False,
            featuresActionnability=reader.featuresType,
            # featuresActionnability=False,
            featuresType=reader.featuresType, 
            featuresPossibleValues=reader.featuresPossibleValues,
        )
        randomForestMilp.buildModel()
        randomForestMilp.solveModel()
        # assert(randomForestMilp.objValue <= readOneNorm)

        for f in range(len(reader.featuresActionnability)):
            if not reader.featuresActionnability[f]:
                assert ufX[0][f] == ucX[0][f]
            if reader.featuresType[f] in [FeatureType.Discrete, FeatureType.Categorical] and roundMace:
                assert scX[0][f] in reader.featuresPossibleValues[f]

        if bool(ucy):
            gap = "{:.2%}".format((readOneNorm - randomForestMilp.objValue)/ randomForestMilp.objValue)
        else:
            gap = "MACE UNFEASIBLE"
        print(ufy, sfy, ucy, scy, y, readOneNorm, randomForestMilp.objValue, gap)

    pass    

maceExperimentsFolder = "../macePrivate/_experiments/2021.03.22_10.42.48__compass__forest__one_norm__MACE_eps_1e-5__pid2_md5_ne10/"

checkSamples("datasets/datasets/COMPAS-ProPublica_processedMACE.csv", 
    "datasets/datasets/counterfactuals/COMPAS-ProPublica_processedMACE.csv",  
    maceExperimentsFolder + "/samplesProduced.csv", 
    maceExperimentsFolder + "/_model_trained"
)
