# Important note: This file is given to reproduce the Benchmark of OCEAN paper. It is not part of the OCEAN package and will not be updated with new versions of OCEAN.

import pickle
import pandas as pd
import numpy as np
from enum import Enum
from CounterFactualParameters import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import copy

import multiprocessing as mp

import os
import sys
sys.path.append("/home/axel/Documents/code/counterfactuals/ocean/ocean")
import maceUpdatedForOceanBenchmark.newLoadData

import time

from RandomForestCounterFactual import *
from CuiRandomForestCounterFactual import *

class MaceDatasetReader:
    def __init__(self, maceFolder, rangeFeasibilityForDiscreteFeatures = False):
        maceDatasetSerializedFilename = maceFolder + "/_dataset_obj"
        maceSerializedClf = maceFolder + "/_model_trained"
        maceSerializedFactuals = maceFolder + "/_minimum_distances"
        
        self.initialized = True
        if not os.path.isfile(maceSerializedFactuals):
            self.initialized = False
            return

        # Initialize from Mace Dataset
        self.maceDataset : newLoadData.Dataset = pickle.load(open(maceDatasetSerializedFilename, "rb"))
        self.featuresActionnability = []
        featureTypeOfMaceFeature = {
            'numeric-int' : FeatureType.Discrete, 
            'numeric-real' : FeatureType.Numeric, 
            'binary' : FeatureType.Binary,
            'categorical' : FeatureType.CategoricalNonOneHot, 
            'ordinal' : FeatureType.Discrete,
        }
        self.featuresType = [featureTypeOfMaceFeature[column.attr_type] for kurzname, column in self.maceDataset.attributes_kurz.items() if kurzname != 'y']
        featureActionnabilityOfMaceActionnability = { 
            'none' : FeatureActionnability.Fixed,
            'any' : FeatureActionnability.Free,
            'same-or-increase' : FeatureActionnability.Increasing
        }
        self.featuresActionnability = [featureActionnabilityOfMaceActionnability[column.actionability] for kurzname, column in self.maceDataset.attributes_kurz.items() if kurzname != 'y']
        self.lowerBoundsList = [column.lower_bound for kurzname, column in self.maceDataset.attributes_kurz.items() if kurzname != 'y']
        self.upperBoundsList = [column.upper_bound for kurzname, column in self.maceDataset.attributes_kurz.items() if kurzname != 'y']

        self.featuresPossibleValues = []
        for kurzname, attr in self.maceDataset.attributes_kurz.items():
            if kurzname != 'y':
                if featureTypeOfMaceFeature[attr.attr_type] == FeatureType.Categorical:
                    self.featuresPossibleValues.append(self.maceDataset.data_frame_kurz[kurzname].unique())
                elif featureTypeOfMaceFeature[attr.attr_type] == FeatureType.Discrete:
                    if rangeFeasibilityForDiscreteFeatures:
                        self.featuresPossibleValues.append([(i - attr.lower_bound)/(attr.upper_bound - attr.lower_bound)  for i in range(int(attr.lower_bound), int(attr.upper_bound+1))])
                    else:
                        
                        self.featuresPossibleValues.append(sorted((self.maceDataset.data_frame_kurz[kurzname].unique() - attr.lower_bound) /(attr.upper_bound - attr.lower_bound)))
                elif featureTypeOfMaceFeature[attr.attr_type] == FeatureType.CategoricalNonOneHot:
                    self.featuresPossibleValues.append(sorted((self.maceDataset.data_frame_kurz[kurzname].unique() - attr.lower_bound) /(attr.upper_bound - attr.lower_bound)))    
                else:
                    self.featuresPossibleValues.append([])

        # initialize clfRead from Mace clf, and scale clf
        self.clfRead : RandomForestClassifier = pickle.load(open(maceSerializedClf, "rb"))
        assert self.clfRead.n_features_ == len(self.featuresType)
        assert type(self.clfRead) == RandomForestClassifier

        self.clfScaled = copy.deepcopy(self.clfRead)

        for est in self.clfScaled.estimators_:
            tree = est.tree_
            for v in range(tree.capacity):
                f = tree.feature[v]
                if f >= 0:
                    tree.threshold[v] = (tree.threshold[v] - self.lowerBoundsList[f])/(self.upperBoundsList[f]-self.lowerBoundsList[f])
                    assert 0 <= tree.threshold[v] <= 1

        # Load and scale factuals
        self.maceFactuals = pickle.load(open(maceSerializedFactuals, "rb"))

        self.factualSamples = []
        self.scaledFactualSamples = []
        self.desiredOutputs = []
        self.maceSolutionsValue = []
        facSampleName = 'fac_sample'
        for sampleName in self.maceFactuals:
            if facSampleName not in self.maceFactuals[sampleName]:
                facSampleName = 'factual_sample'
            self.factualSamples.append([column for kurzname, column in self.maceFactuals[sampleName][facSampleName].items() if kurzname != 'y'])

            scaledSample = []
            for kurzname, column in self.maceFactuals[sampleName][facSampleName].items():
                if kurzname != 'y':
                    attr = self.maceDataset.attributes_kurz[kurzname]
                    scaledValue = (column - attr.lower_bound)/(attr.upper_bound - attr.lower_bound)
                    scaledSample.append(scaledValue)

            self.scaledFactualSamples.append(scaledSample)
            self.desiredOutputs.append(int(not self.maceFactuals[sampleName][facSampleName]['y']))
            self.maceSolutionsValue.append(self.maceFactuals[sampleName]['cfe_distance'] * len(self.featuresPossibleValues))

        pass

def runExperimentsOnAllMaceSubfolders(maceFolder, objectiveNorm=1, runMilp=True):
    list_subfolders_with_paths = [f.path for f in os.scandir(maceFolder) if f.is_dir()]
    for subfolder in list_subfolders_with_paths:
        if "FT" in subfolder:
            runExperimentsOnMaceOutput(subfolder, objectiveNorm, runMilp=runMilp)
        else:
            runExperimentsOnMaceOutput(subfolder, objectiveNorm, runMilp=False)

def runExperimentsOnMaceOutput(maceFolder, objectiveNorm, runMilp=True):
    reader = MaceDatasetReader(maceFolder, rangeFeasibilityForDiscreteFeatures=False)
    if not reader.initialized:
        print("folder not initialized. Break")
        return
    outFolder = 'outMace'
    os.system('mkdir -p ' + outFolder)
    outFile = outFolder + "/" + maceFolder.split('/')[-1] + ".csv"
    result = open(outFile, 'w')

    count = 0

    result.write('sampleName,mace_cfe_found,mace_cfe_time,mace_obj,desiredOutput,mace_cfe_predicted,mace_norm_recomputed')
    if runMilp:
        result.write(',oaeRunTime,oaeObjValue,oceanRunTime,oceanObjValue')
    result.write('\n')

    for sampleName in reader.maceFactuals:

        sample = reader.maceFactuals[sampleName]
        result.write(sampleName + ',')
        result.write(str(sample['cfe_found']) + ',')
        result.write(str(sample['cfe_time']) + ',')
        result.write(str(reader.maceSolutionsValue[count]))

        sol = []
        scaledSol = []
        if sample['cfe_distance'] != float('inf'):
            facSampleName = 'fac_sample' 
            for name in ['fac_sample', 'factual_sample']:
                if name in sample:
                    facSampleName = name
                    break
            facSample = sample[facSampleName]
            cfeSample = sample['cfe_sample']
            f = 0
            cfeDist = 0.0
            for fName in facSample:
                if fName != 'y':
                    sol.append(cfeSample[fName])
                    attr = reader.maceDataset.attributes_kurz[fName]
                    scaledSol.append((cfeSample[fName] - attr.lower_bound)/(attr.upper_bound - attr.lower_bound))
                    if reader.featuresType[f] == FeatureType.CategoricalNonOneHot:
                        if abs(facSample[fName] - cfeSample[fName]) > 1e-5:
                            cfeDist += 1.0
                    elif reader.featuresType[f] == FeatureType.Binary:
                        if abs(int(round(cfeSample[fName]))- cfeSample[fName]) > 1e-3:
                            print('non integer binary')
                        if abs(facSample[fName] - cfeSample[fName]) > 1e-3:
                            cfeDist += 1.0                        
                    elif reader.featuresType[f] == FeatureType.Discrete:
                        if abs(int(round(cfeSample[fName]))- cfeSample[fName]) > 1e-3:
                            print('non integer discrete')
                        cfeSampleValue = round(cfeSample[fName])
                        cfeSampleDist = abs(facSample[fName] - cfeSampleValue)
                        if abs(facSample[fName] - cfeSample[fName]) > cfeSampleDist + 1e-3:
                            cfeSampleDist += 1
                        cfeDist += abs(cfeSampleDist)/(attr.upper_bound - attr.lower_bound)
                    else:
                        cfeDist += abs(facSample[fName] - cfeSample[fName])/(attr.upper_bound - attr.lower_bound)
                    f += 1
            result.write(',' + str(reader.desiredOutputs[count]) + ',' + str(reader.clfRead.predict([sol])) +','+str(cfeDist))
        else:
            result.write(',inf,inf,inf')

        if runMilp:
            cuiMilp = CuiRandomForestCounterFactualMilp(
                reader.clfScaled,
                [reader.scaledFactualSamples[count]],
                reader.desiredOutputs[count],
                isolationForest=None,
                objectiveNorm=objectiveNorm,
                verbose=False,
                featuresType=reader.featuresType,
                featuresActionnability=reader.featuresActionnability,
                featuresPossibleValues=reader.featuresPossibleValues
            )
            cuiMilp.buildModel()
            cuiMilp.solveModel()
            result.write(',' + str(cuiMilp.model.Runtime))
            result.write(',' + str(cuiMilp.objValue))                
            randomForestMilp = RandomForestCounterFactualMilp(
                reader.clfScaled,
                [reader.scaledFactualSamples[count]],
                reader.desiredOutputs[count],
                isolationForest=False, 
                constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                objectiveNorm=objectiveNorm,
                mutuallyExclusivePlanesCutsActivated=True,
                strictCounterFactual=True,
                binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
                randomCostsActivated=False,
                verbose=False,
                featuresActionnability=reader.featuresActionnability,
                featuresType=reader.featuresType, 
                featuresPossibleValues=reader.featuresPossibleValues,
            )
            randomForestMilp.buildModel()
            randomForestMilp.solveModel()
            result.write(',' + str(randomForestMilp.runTime))
            result.write(',' + str(randomForestMilp.objValue))
        result.write('\n')
        count += 1