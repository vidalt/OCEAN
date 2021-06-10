import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from ClassifierCounterFactual import *
from RandomAndIsolationForest import *
from sklearn.ensemble._iforest import _average_path_length

class CuiRandomForestCounterFactualMilp(ClassifierCounterFactualMilp):
    def __init__(
        self, classifier, sample, outputDesired,
        isolationForest=None,
        objectiveNorm=1, verbose=False,
        featuresType = False, 
        featuresPossibleValues = False,
        strictCounterFactual=False,
        featuresActionnability = False
    ):
        ClassifierCounterFactualMilp.__init__(self, classifier, sample, outputDesired, 
            objectiveNorm=objectiveNorm, verbose=verbose,
            featuresType=featuresType, featuresPossibleValues=featuresPossibleValues, 
        )
        self.isolationForest = isolationForest
        self.completeForest = RandomAndIsolationForest(self.clf, isolationForest)
        self.strictCounterFactual = strictCounterFactual
        assert self.objectiveNorm in [0,1]
        self.model.modelName="CuiForestCounterFactualMilp"
        if featuresActionnability:
            self.featuresActionnability = featuresActionnability
        else:
            self.featuresActionnability = [FeatureActionnability.Free for f in range(self.nFeatures)]
        
    def buildLevels(self):
        self.planes = dict()
        for f in range(self.nFeatures):
            self.planes[f] = dict()
        
        for t in range(self.completeForest.n_estimators):
            tree = self.completeForest.estimators_[t].tree_
            for v in range(tree.node_count):
                if not tree.children_left[v] == tree.children_right[v]:
                    f = tree.feature[v]
                    thres = tree.threshold[v]
                    newPlane = True
                    if self.planes[f]:
                        nearestThres = min(self.planes[f].keys(), key=lambda k: abs(k-thres))
                        if abs(thres - nearestThres) < 0.8*eps:
                            newPlane = False
                            self.planes[f][nearestThres].append((t,v))
                    if newPlane:
                        self.planes[f][thres] = [(t,v)]

        self.numberOfCellsOfFeature = dict()
        self.indexOfCellOnTheLeftOfTreeVertexThres = {t:dict() for t in range(self.completeForest.n_estimators)}
        self.cellLowerBound = dict()
        self.cellUpperBound = dict()
        for f in range(self.nFeatures):
            self.numberOfCellsOfFeature[f] = len(self.planes[f]) + 1
            self.cellLowerBound[f] = {cell:-eps for cell in range(self.numberOfCellsOfFeature[f])}
            self.cellUpperBound[f] = {cell:1.0 for cell in range(self.numberOfCellsOfFeature[f])}
            countCell = 0
            for thres in sorted(self.planes[f]):
                for t,v in self.planes[f][thres]:
                    self.indexOfCellOnTheLeftOfTreeVertexThres[t][v] = countCell
                self.cellUpperBound[f][countCell] = thres
                self.cellLowerBound[f][countCell + 1] = thres
                countCell += 1

    def buildCellVariablesAndConstraints(self):
        self.cellVar = dict()
        self.oneCellChosenConstr = dict()
        for f in range(self.nFeatures):
            self.cellVar[f] = dict()
            sumOfCell = gp.LinExpr(0)
            for c in range(self.numberOfCellsOfFeature[f]):
                self.cellVar[f][c] = self.model.addVar(vtype=GRB.BINARY, name="cellVar_f" + str(f) + "_c" + str(c))
                sumOfCell += self.cellVar[f][c]
            self.oneCellChosenConstr[f] = self.model.addConstr(sumOfCell == 1, "oneCellChosenConstr_f"+str(f))
            
    def buildLeafVariablesAndConstraints(self):
        self.leafVar = dict()
        self.oneLeafChosenConstr = dict()
        for t in range(self.completeForest.n_estimators):
            tree = self.completeForest.estimators_[t].tree_
            self.leafVar[t] = dict()
            sumOfLeaf = gp.LinExpr(0)
            for v in range(tree.node_count):
                if tree.children_left[v] == tree.children_right[v]:
                    self.leafVar[t][v] = self.model.addVar(vtype=GRB.BINARY, name="leafVar_t" + str(t) + "_v" + str(v))
                    sumOfLeaf += self.leafVar[t][v]
            self.oneLeafChosenConstr[t] = self.model.addConstr(sumOfLeaf == 1, "oneLeafChosenConstr_t" + str(t))

    def buildConsistencyConstraints(self):
        self.nodeLeftMostCellAlongFeature_tvf = dict()
        self.nodeRightMostCellAlongFeature_tvf = dict()
        self.nodeAncestorsAlongFeature_tvf = dict()
        # self.parent = dict()
        self.depth = dict()
        self.leafConsistencyConstraint = dict()
        for t in range(self.completeForest.n_estimators):
            tree = self.completeForest.estimators_[t].tree_
            self.nodeLeftMostCellAlongFeature_tvf[t]= dict()
            self.nodeRightMostCellAlongFeature_tvf[t] = dict()
            self.nodeAncestorsAlongFeature_tvf[t] = dict()
            # self.parent[t] = dict()
            self.depth[t] = dict()
            self.nodeLeftMostCellAlongFeature_tvf[t][0] = {f:0 for f in range(self.nFeatures)}
            self.nodeRightMostCellAlongFeature_tvf[t][0] = {f:(self.numberOfCellsOfFeature[f]-1) for f in range(self.nFeatures)}
            self.nodeAncestorsAlongFeature_tvf[t][0] = {f:[] for f in range(self.nFeatures)}
            self.depth[t][0] = 0

            stack = [0]
            while len(stack) > 0:
                v = stack.pop()
                cl = tree.children_left[v]
                cr = tree.children_right[v]
                if cr != cl:
                    # self.parent[cl] = v
                    self.depth[t][cl] = self.depth[t][v] + 1
                    self.nodeLeftMostCellAlongFeature_tvf[t][cl] = self.nodeLeftMostCellAlongFeature_tvf[t][v].copy()
                    self.nodeRightMostCellAlongFeature_tvf[t][cl] = self.nodeRightMostCellAlongFeature_tvf[t][v].copy()
                    self.nodeRightMostCellAlongFeature_tvf[t][cl][tree.feature[v]] = self.indexOfCellOnTheLeftOfTreeVertexThres[t][v]
                    self.nodeAncestorsAlongFeature_tvf[t][cl] = {f:self.nodeAncestorsAlongFeature_tvf[t][v][f].copy() for f in  self.nodeAncestorsAlongFeature_tvf[t][v]}
                    self.nodeAncestorsAlongFeature_tvf[t][cl][tree.feature[v]].append(cl)
                    stack.append(cl)
                    # self.parent[cr] = v
                    self.depth[t][cr] = self.depth[t][v] + 1
                    self.nodeLeftMostCellAlongFeature_tvf[t][cr] = self.nodeLeftMostCellAlongFeature_tvf[t][v].copy()
                    self.nodeRightMostCellAlongFeature_tvf[t][cr] = self.nodeRightMostCellAlongFeature_tvf[t][v].copy()
                    self.nodeLeftMostCellAlongFeature_tvf[t][cr][tree.feature[v]] = self.indexOfCellOnTheLeftOfTreeVertexThres[t][v] + 1
                    self.nodeAncestorsAlongFeature_tvf[t][cr] = {f:self.nodeAncestorsAlongFeature_tvf[t][v][f].copy() for f in  self.nodeAncestorsAlongFeature_tvf[t][v]}
                    self.nodeAncestorsAlongFeature_tvf[t][cr][tree.feature[v]].append(cr)
                    stack.append(cr)

            self.leafConsistencyConstraint[t] = dict()
            for leaf in self.leafVar[t]:
                upperBound = gp.LinExpr(0)
                for f in range(self.nFeatures):
                    for child in self.nodeAncestorsAlongFeature_tvf[t][leaf][f]:
                        for cell in range(self.nodeLeftMostCellAlongFeature_tvf[t][child][f], self.nodeRightMostCellAlongFeature_tvf[t][child][f] + 1):
                            upperBound += 1 / self.depth[t][leaf] * self.cellVar[f][cell]
                self.model.addConstr(self.leafVar[t][leaf] <= upperBound, "consistency_tree" + str(t) + "_leaf" + str(leaf))


    def buildObjective(self):
        obj = gp.LinExpr(0.0)
        self.nearestInCell = {f:dict() for f in self.cellVar}
        if self.objectiveNorm == 0:
            for f in self.cellVar:
                if self.featuresType[f] == FeatureType.Numeric:
                    for cell in self.cellVar[f]:
                        if self.cellUpperBound[f][cell] < self.x0[0][f]:
                            self.nearestInCell[f][cell] = self.cellUpperBound[f][cell] - eps
                            obj +=  self.cellVar[f][cell]
                        elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                            self.nearestInCell[f][cell] = self.cellLowerBound[f][cell] + eps
                            obj += self.cellVar[f][cell]
                        else:
                            self.nearestInCell[f][cell] =  self.x0[0][f]
                elif self.featuresType[f] in [FeatureType.Discrete, FeatureType.Binary, FeatureType.CategoricalNonOneHot]:
                    for cell in self.cellVar[f]:
                        cellValues = [x for x in self.featuresPossibleValues[f]
                            if x <= self.cellUpperBound[f][cell] and x > self.cellLowerBound[f][cell]]
                        if cellValues:
                            if self.cellUpperBound[f][cell] < self.x0[0][f]:
                                self.nearestInCell[f][cell] = max(cellValues)
                                obj += self.cellVar[f][cell]
                            elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                                self.nearestInCell[f][cell] = min(cellValues)                           
                                obj += self.cellVar[f][cell]
                            else:
                                self.nearestInCell[f][cell] =  self.x0[0][f]
                        else:
                            self.model.addConstr(self.cellVar[f][cell] == 0, "empty_cell_f" + str(f) + "_c" + str(cell))
                else:
                    print("Objective not implemented for this kind of variables")
        elif self.objectiveNorm == 1:
            for f in self.cellVar:
                if self.featuresType[f] == FeatureType.Numeric:
                    for cell in self.cellVar[f]:
                        if self.cellUpperBound[f][cell] < self.x0[0][f]:
                            self.nearestInCell[f][cell] = self.cellUpperBound[f][cell] - eps
                            obj += (self.x0[0][f] - self.nearestInCell[f][cell]) * self.cellVar[f][cell]
                        elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                            self.nearestInCell[f][cell] = self.cellLowerBound[f][cell] + eps
                            obj += (self.nearestInCell[f][cell] - self.x0[0][f]) * self.cellVar[f][cell]
                        else:
                            self.nearestInCell[f][cell] =  self.x0[0][f]
                elif self.featuresType[f] == FeatureType.Discrete or self.featuresType[f] == FeatureType.Binary:
                    for cell in self.cellVar[f]:
                        cellValues = [x for x in self.featuresPossibleValues[f]
                            if x <= self.cellUpperBound[f][cell] and x > self.cellLowerBound[f][cell]]
                        if cellValues:
                            if self.cellUpperBound[f][cell] < self.x0[0][f]:
                                self.nearestInCell[f][cell] = max(cellValues)
                                obj += (self.x0[0][f] - self.nearestInCell[f][cell]) * self.cellVar[f][cell]
                            elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                                self.nearestInCell[f][cell] = min(cellValues)                           
                                obj += (self.nearestInCell[f][cell] - self.x0[0][f]) * self.cellVar[f][cell]
                            else:
                                self.nearestInCell[f][cell] =  self.x0[0][f]
                        else:
                            self.model.addConstr(self.cellVar[f][cell] == 0, "empty_cell_f" + str(f) + "_c" + str(cell))
                            # obj += 1000 * self.cellVar[f][cell]
                elif self.featuresType[f] == FeatureType.CategoricalNonOneHot:
                     for cell in self.cellVar[f]:
                        cellValues = [x for x in self.featuresPossibleValues[f]
                            if x <= self.cellUpperBound[f][cell] and x > self.cellLowerBound[f][cell]]
                        if cellValues:
                            if self.cellUpperBound[f][cell] < self.x0[0][f]:
                                self.nearestInCell[f][cell] = max(cellValues)
                                obj += self.cellVar[f][cell]
                            elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                                self.nearestInCell[f][cell] = min(cellValues)                           
                                obj += self.cellVar[f][cell]
                            else:
                                self.nearestInCell[f][cell] =  self.x0[0][f]
                        else:
                            self.model.addConstr(self.cellVar[f][cell] == 0, "empty_cell_f" + str(f) + "_c" + str(cell))                   
                else:
                    print("Objective not implemented for this kind of variables")
        elif self.objectiveNorm == 2:
            for f in self.cellVar:
                if self.featuresType[f] == FeatureType.Numeric:
                    for cell in self.cellVar[f]:
                        if self.cellUpperBound[f][cell] < self.x0[0][f]:
                            self.nearestInCell[f][cell] = self.cellUpperBound[f][cell] - eps
                            obj += (self.x0[0][f] - self.nearestInCell[f][cell]) * (self.x0[0][f] - self.nearestInCell[f][cell]) * self.cellVar[f][cell]
                        elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                            self.nearestInCell[f][cell] = self.cellLowerBound[f][cell] + eps
                            obj += (self.nearestInCell[f][cell] - self.x0[0][f]) * (self.nearestInCell[f][cell] - self.x0[0][f]) * self.cellVar[f][cell]
                        else:
                            self.nearestInCell[f][cell] =  self.x0[0][f]
                elif self.featuresType[f] == FeatureType.Discrete or self.featuresType[f] == FeatureType.Binary:
                    for cell in self.cellVar[f]:
                        cellValues = [x for x in self.featuresPossibleValues[f]
                            if x <= self.cellUpperBound[f][cell] and x > self.cellLowerBound[f][cell]]
                        if cellValues:
                            if self.cellUpperBound[f][cell] < self.x0[0][f]:
                                self.nearestInCell[f][cell] = max(cellValues)
                                obj += (self.x0[0][f] - self.nearestInCell[f][cell]) * (self.x0[0][f] - self.nearestInCell[f][cell]) * self.cellVar[f][cell]
                            elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                                self.nearestInCell[f][cell] = min(cellValues)                           
                                obj += (self.nearestInCell[f][cell] - self.x0[0][f]) * (self.nearestInCell[f][cell] - self.x0[0][f]) * self.cellVar[f][cell]
                            else:
                                self.nearestInCell[f][cell] =  self.x0[0][f]
                        else:
                            self.model.addConstr(self.cellVar[f][cell] == 0, "empty_cell_f" + str(f) + "_c" + str(cell))
                            # obj += 1000 * self.cellVar[f][cell]
                elif self.featuresType[f] == FeatureType.CategoricalNonOneHot:
                     for cell in self.cellVar[f]:
                        cellValues = [x for x in self.featuresPossibleValues[f]
                            if x <= self.cellUpperBound[f][cell] and x > self.cellLowerBound[f][cell]]
                        if cellValues:
                            if self.cellUpperBound[f][cell] < self.x0[0][f]:
                                self.nearestInCell[f][cell] = max(cellValues)
                                obj += self.cellVar[f][cell]
                            elif self.cellLowerBound[f][cell] > self.x0[0][f]:
                                self.nearestInCell[f][cell] = min(cellValues)                           
                                obj += self.cellVar[f][cell]
                            else:
                                self.nearestInCell[f][cell] =  self.x0[0][f]
                        else:
                            self.model.addConstr(self.cellVar[f][cell] == 0, "empty_cell_f" + str(f) + "_c" + str(cell))                   
                else:
                    print("Objective not implemented for this kind of variables")
        else:
            print("Unknown objective norm, objective not implemented in the lp")
        self.model.setObjective(obj, GRB.MINIMIZE)

    def addMajorityVoteConstraint(self):
        majorityVoteExpr = {cl : gp.LinExpr(0.0) for cl in self.clf.classes_ if cl != self.outputDesired}
        for t in self.completeForest.randomForestEstimatorsIndices:
            for leaf in self.leafVar[t]:
                leaf_val = self.completeForest.estimators_[t].tree_.value[leaf][0]
                tot = sum(leaf_val)
                for output in range(len(leaf_val)):
                    if output == self.outputDesired:
                        for cl in majorityVoteExpr:
                            majorityVoteExpr[cl] += self.leafVar[t][leaf] * (leaf_val[output])/(tot * self.completeForest.n_estimators)
                    else:
                        majorityVoteExpr[output] -= self.leafVar[t][leaf]  * (leaf_val[output])/(tot * self.completeForest.n_estimators)                
        self.majorityVoteConstr = dict()
        for cl in majorityVoteExpr:
            if self.strictCounterFactual:
                majorityVoteExpr[cl] -= 1e-4
            self.majorityVoteConstr[cl] = self.model.addConstr(majorityVoteExpr[cl] >= 0, "majorityVoteConstr_cl" + str(cl))

    def addIsolationForestPlausibilityConstraint(self):
        expr = gp.LinExpr(0.0)
        for t in self.completeForest.isolationForestEstimatorsIndices:
            tree = self.completeForest.estimators_[t]
            expr -= _average_path_length([self.isolationForest.max_samples_])[0]
            for leaf in self.leafVar[t]:
                leafDepth = self.depth[t][leaf] + _average_path_length([tree.tree_.n_node_samples[leaf]])[0]
                expr += leafDepth * self.leafVar[t][leaf]
        self.model.addConstr(expr >= 0, "isolationForestInlierConstraint")

    def addActionnabilityConstraints(self):
        self.actionnabilityConstraints = dict()
        for f in range(self.nFeatures):
            if self.featuresActionnability[f] == FeatureActionnability.Increasing:
                if self.featuresType[f] not in [FeatureType.Numeric, FeatureType.Discrete]:
                    print("Increasing actionnability is avaialable only for numeric and discrete features")
                else:
                    expr = gp.LinExpr(0.0)
                    for cell in range(self.numberOfCellsOfFeature[f]):
                        if self.cellUpperBound[f][cell] >= self.x0[0][f]:
                            expr += self.cellVar[f][cell]
                    self.model.addConstr(expr == 1, "actionnability_f"+str(f))
            elif self.featuresActionnability[f] == FeatureActionnability.Fixed:
                for cell in range(self.numberOfCellsOfFeature[f]):
                    if self.cellLowerBound[f][cell] < self.x0[0][f] and self.cellUpperBound[f][cell] >= self.x0[0][f]:
                        self.model.addConstr(self.cellVar[f][cell] == 1, "actionnability_f"+str(f))

    def buildModel(self):
        self.buildLevels()
        self.buildCellVariablesAndConstraints()
        self.buildLeafVariablesAndConstraints()
        self.buildConsistencyConstraints()
        self.addMajorityVoteConstraint()
        if self.isolationForest:
            self.addIsolationForestPlausibilityConstraint()
        self.addActionnabilityConstraints()
        self.buildObjective()

    def solveModel(self):
        self.model.write("cui.lp")
        self.model.setParam(GRB.Param.ImpliedCuts,2)
        self.model.setParam(GRB.Param.Threads,4)
        self.model.setParam(GRB.Param.TimeLimit,900)
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            self.objValue = "inf"
            self.x_sol=self.x0
            return False

        self.objValue = self.model.ObjVal
        self.x_sol=[[]] 
        for f in self.cellVar:
            value = 0.0
            for cell in self.cellVar[f]:
                if self.cellVar[f][cell].getAttr(GRB.Attr.X):
                    value = self.nearestInCell[f][cell]
            self.x_sol[0].append(value) 
        
        print("Solution built \n", self.x_sol, " with prediction ", self.clf.predict(self.x_sol))

