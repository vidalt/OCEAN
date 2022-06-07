import random

from sklearn.ensemble._iforest import _average_path_length

from src.TreeMilpManager import *
from src.ClassifierCounterFactual import *
from src.RandomAndIsolationForest import *


class RandomForestCounterFactualMilp(ClassifierCounterFactualMilp):
    def __init__(self,
                 classifier,
                 sample,
                 outputDesired,
                 isolationForest=None,
                 constraintsType=TreeConstraintsType.ExtendedFormulation,
                 objectiveNorm=2,
                 interTreeCutsActivated=False,
                 mutuallyExclusivePlanesCutsActivated=False,
                 strictCounterFactual=False,
                 verbose=False,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda,
                 randomCostsActivated=False
                 ):
        ClassifierCounterFactualMilp.__init__(
            self,
            classifier,
            sample,
            outputDesired,
            constraintsType,
            objectiveNorm,
            verbose,
            featuresType,
            featuresPossibleValues,
            featuresActionnability,
            oneHotEncoding,
            binaryDecisionVariables
        )
        self.isolationForest = isolationForest
        self.completeForest = RandomAndIsolationForest(
            self.clf, isolationForest)
        self.interTreeCutsActivated = interTreeCutsActivated
        self.mutuallyExclusivePlanesCutsActivated = mutuallyExclusivePlanesCutsActivated
        self.strictCounterFactual = strictCounterFactual
        self.model.modelName = "RandomForestCounterFactualMilp"
        self.randomCostsActivated = randomCostsActivated
        if randomCostsActivated:
            random.seed(0)
            self.greaterCosts = [random.uniform(
                0.5, 2) for i in range(self.nFeatures)]
            self.smallerCosts = [random.uniform(
                0.5, 2) for i in range(self.nFeatures)]

    def addMajorityVoteConstraint(self):
        majorityVoteExpr = {cl: gp.LinExpr(
            0.0) for cl in self.clf.classes_ if cl != self.outputDesired}
        for t in self.completeForest.randomForestEstimatorsIndices:
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    leaf_val = tm.tree.value[v][0]
                    tot = sum(leaf_val)
                    for output in range(len(leaf_val)):
                        if output == self.outputDesired:
                            for cl in majorityVoteExpr:
                                majorityVoteExpr[cl] += tm.y_var[v] * (leaf_val[output])/(
                                    tot * self.completeForest.n_estimators)
                        else:
                            majorityVoteExpr[output] -= tm.y_var[v] * (
                                leaf_val[output])/(tot * self.completeForest.n_estimators)
        self.majorityVoteConstr = dict()
        for cl in majorityVoteExpr:
            if self.strictCounterFactual:
                majorityVoteExpr[cl] -= 1e-4
            self.majorityVoteConstr[cl] = self.model.addConstr(
                majorityVoteExpr[cl] >= 0, "majorityVoteConstr_cl" + str(cl))

    def addIsolationForestPlausibilityConstraint(self):
        expr = gp.LinExpr(0.0)
        for t in self.completeForest.isolationForestEstimatorsIndices:
            tm = self.treeManagers[t]
            tree = self.completeForest.estimators_[t]
            expr -= _average_path_length(
                [self.isolationForest.max_samples_])[0]
            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    leafDepth = tm.node_depth[v] + _average_path_length(
                        [tree.tree_.n_node_samples[v]])[0]
                    expr += leafDepth * tm.y_var[v]
        self.model.addConstr(expr >= 0, "isolationForestInlierConstraint")

    def addPlaneConsistencyConstraints(self):
        self.planes = dict()
        for f in self.continuousFeatures:
            self.planes[f] = dict()
            # Add the initial value as a plane
            self.planes[f][self.x0[0][f]] = []

        for t in range(self.completeForest.n_estimators):
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    if self.featuresType[f] == FeatureType.Numeric:
                        thres = tm.tree.threshold[v]
                        newPlane = True
                        if self.planes[f]:
                            nearestThres = min(
                                self.planes[f].keys(), key=lambda k: abs(k-thres))
                            if abs(thres - nearestThres) < 0.8*eps:
                                newPlane = False
                                self.planes[f][nearestThres].append((t, v))
                        if newPlane:
                            self.planes[f][thres] = [(t, v)]

        self.rightMutuallyExclusivePlanesVar = dict()
        self.rightMutuallyExclusivePlanesConstr = dict()
        self.rightPlanesDominateRightFlowConstr = dict()
        self.rightPlanesOrderConstr = dict()
        self.linearCombinationOfPlanesConstr = dict()
        for f in self.continuousFeatures:
            self.rightMutuallyExclusivePlanesVar[f] = dict()
            self.rightMutuallyExclusivePlanesConstr[f] = dict()
            self.rightPlanesDominateRightFlowConstr[f] = dict()
            self.rightPlanesOrderConstr[f] = dict()
            previousThres = 0
            linearCombination = gp.LinExpr(0.0)
            self.rightMutuallyExclusivePlanesVar[f][previousThres] = self.model.addVar(lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                                                                                       name="rightMutuallyExclusivePlanesVar_f" + str(f) + "_th" + str(previousThres))
            for thres in sorted(self.planes[f]):
                self.rightMutuallyExclusivePlanesVar[f][thres] = self.model.addVar(lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                                                                                   name="rightMutuallyExclusivePlanesVar_f" + str(f) + "_th" + str(thres))
                self.rightMutuallyExclusivePlanesConstr[f][thres] = []
                self.rightPlanesDominateRightFlowConstr[f][thres] = []

                for t, v in self.planes[f][thres]:
                    tm = self.treeManagers[t]
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(self.model.addConstr(
                        tm.y_var[tm.tree.children_left[v]]
                        + self.rightMutuallyExclusivePlanesVar[f][thres] <= 1,
                        "rightMutuallyExclusivePlanesVar_f"
                        + str(f) + "_t" + str(t) + "_v" + str(v)
                    ))
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(self.model.addConstr(
                        tm.y_var[tm.tree.children_right[v]
                                 ] <= self.rightMutuallyExclusivePlanesVar[f][previousThres],
                        "rightPlanesDominatesLeftFlowConstr_t"
                        + str(t)+"_v"+str(v)
                    ))
                    # Avoid numerical precision errors
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(self.model.addConstr(
                        (thres - previousThres) * self.rightMutuallyExclusivePlanesVar[f][previousThres] <= (
                            thres - previousThres) - min(thres - previousThres, eps) * tm.y_var[tm.tree.children_left[v]],
                        # self.rightMutuallyExclusivePlanesVar[f][previousThres] <= 1 - eps * tm.y_var[tm.tree.children_left[v]] ,
                        "rightMutuallyExclusivePlanesVar_eps_f" + \
                        str(f) + "_t" + str(t) + "_v" + str(v)
                    ))
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(self.model.addConstr(
                        eps * tm.y_var[tm.tree.children_right[v]] <= self.rightMutuallyExclusivePlanesVar[f][thres] * max(
                            eps, (thres - previousThres)),
                        # eps * tm.y_var[tm.tree.children_right[v]] <= self.rightMutuallyExclusivePlanesVar[f][thres],
                        "rightPlanesDominatesLeftFlowConstr_eps_t" + \
                        str(t)+"_v"+str(v)
                    ))

                self.rightPlanesOrderConstr[f][thres] = self.model.addConstr(
                    self.rightMutuallyExclusivePlanesVar[f][
                        previousThres] >= self.rightMutuallyExclusivePlanesVar[f][thres],
                    "rightPlanesOrderConstr_f"+str(f)+"_th"+str(thres)
                )

                linearCombination += self.rightMutuallyExclusivePlanesVar[f][previousThres] * (
                    thres - previousThres)

                previousThres = thres
            linearCombination += self.rightMutuallyExclusivePlanesVar[f][previousThres] * (
                1.0 - previousThres)
            self.linearCombinationOfPlanesConstr[f] = self.model.addConstr(
                self.x_var_sol[f] == linearCombination, "x_as_linear_combination_of_planes_f")

    def addInterTreeCuts(self):
        self.interTreeCuts = dict()
        for s in range(self.completeForest.n_estimators):
            self.interTreeCuts[s] = dict()
            tms = self.treeManagers[s]
            for t in range(s+1, self.completeForest.n_estimators):
                self.interTreeCuts[s][t] = dict()
                tmt = self.treeManagers[t]
                for u in range(tms.n_nodes):
                    if not tms.is_leaves[u] and self.featuresType[tms.tree.feature[u]] == FeatureType.Numeric:
                        self.interTreeCuts[s][t][u] = dict()
                        for v in range(tmt.n_nodes):
                            if not tmt.is_leaves[v] and self.featuresType[tmt.tree.feature[v]] == FeatureType.Numeric:
                                if tms.tree.feature[u] == tmt.tree.feature[v]:
                                    if tms.tree.threshold[u] <= tmt.tree.threshold[v]:
                                        self.interTreeCuts[s][t][u][v] = self.model.addConstr(
                                            tms.y_var[tms.tree.children_left[u]]
                                            + tmt.y_var[tmt.tree.children_right[v]] <= 1,
                                            "interTreeCuts_s"+str(s)+"_t"+str(t)+"_u"+str(u)+"_v"+str(v))
                                    if tms.tree.threshold[u] >= tmt.tree.threshold[v]:
                                        self.interTreeCuts[s][t][u][v] = self.model.addConstr(
                                            tms.y_var[tms.tree.children_right[u]]
                                            + tmt.y_var[tmt.tree.children_left[v]] <= 1,
                                            "interTreeCuts_s"+str(s)+"_t"+str(t)+"_u"+str(u)+"_v"+str(v))

    def addMutuallyExclusivePlanesCuts(self):
        self.planes = dict()
        for f in range(self.nFeatures):
            self.planes[f] = dict()

        for t in range(self.completeForest.n_estimators):
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    if self.featuresType[f] == FeatureType.Numeric:
                        thres = tm.tree.threshold[v]
                        newPlane = True
                        if self.planes[f]:
                            nearestThres = min(
                                self.planes[f].keys(), key=lambda k: abs(k-thres))
                            if abs(thres - nearestThres) < 0.8*eps:
                                newPlane = False
                                self.planes[f][nearestThres].append((t, v))
                        if newPlane:
                            self.planes[f][thres] = [(t, v)]

                        # if thres not in self.planes[f]:
                        #     self.planes[f][thres] = []
                        # self.planes[f][thres].append((t,v))

        self.rightMutuallyExclusivePlanesVar = dict()
        self.rightMutuallyExclusivePlanesConstr = dict()
        self.rightPlanesDominateRightFlowConstr = dict()
        self.rightPlanesOrderConstr = dict()
        for f in self.continuousFeatures:
            self.rightMutuallyExclusivePlanesVar[f] = dict()
            self.rightMutuallyExclusivePlanesConstr[f] = dict()
            self.rightPlanesDominateRightFlowConstr[f] = dict()
            self.rightPlanesOrderConstr[f] = dict()
            previousThres = -1
            for thres in sorted(self.planes[f]):
                self.rightMutuallyExclusivePlanesVar[f][thres] = self.model.addVar(lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                                                                                   name="rightMutuallyExclusivePlanesVar_f" + str(f) + "_th" + str(thres))
                self.rightMutuallyExclusivePlanesConstr[f][thres] = []
                self.rightPlanesDominateRightFlowConstr[f][thres] = []

                for t, v in self.planes[f][thres]:
                    tm = self.treeManagers[t]
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(self.model.addConstr(
                        tm.y_var[tm.tree.children_left[v]]
                        + self.rightMutuallyExclusivePlanesVar[f][thres] <= 1,
                        "rightMutuallyExclusivePlanesVar_f"
                        + str(f) + "_t" + str(t) + "_v" + str(v)
                    ))
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(self.model.addConstr(
                        tm.y_var[tm.tree.children_right[v]
                                 ] <= self.rightMutuallyExclusivePlanesVar[f][thres],
                        "rightPlanesDominatesLeftFlowConstr_t"
                        + str(t)+"_v"+str(v)
                    ))

                if previousThres != -1:
                    self.rightPlanesOrderConstr[f][thres] = self.model.addConstr(
                        self.rightMutuallyExclusivePlanesVar[f][
                            previousThres] >= self.rightMutuallyExclusivePlanesVar[f][thres],
                        "rightPlanesOrderConstr_f"+str(f)+"_th"+str(thres)
                    )

                previousThres = thres

    def buildTrees(self):
        self.treeManagers = dict()
        for t in range(self.completeForest.n_estimators):
            self.treeManagers[t] = TreeInMilpManager(self.completeForest.estimators_[t].tree_, self.model, self.x_var_sol,
                                                     self.outputDesired, self.featuresType, self.constraintsType, self.binaryDecisionVariables)
            self.treeManagers[t].addTreeVariablesAndConstraintsToMilp()

    def addContinuousVariablesConsistencyConstraints(self):
        if self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes:
            self.addPlaneConsistencyConstraints()

    def addDiscreteVariablesConsistencyConstraints(self):
        self.leftDiscreteVariablesConsistencyConstraints = dict()
        self.rightDiscreteVariablesConsistencyConstraints = dict()
        for t in range(self.completeForest.n_estimators):
            tm = self.treeManagers[t]
            self.leftDiscreteVariablesConsistencyConstraints[t] = dict()
            self.rightDiscreteVariablesConsistencyConstraints[t] = dict()
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    if self.featuresType[f] == FeatureType.Discrete or self.featuresType[f] == FeatureType.CategoricalNonOneHot:
                        thres = tm.tree.threshold[v]
                        levels = list(self.featuresPossibleValues[f])
                        levels.append(1.0)
                        v_level = -1
                        for l in range(len(levels)):
                            if levels[l] > thres:
                                v_level = l
                                break
                        self.leftDiscreteVariablesConsistencyConstraints[t][v] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][v_level]
                            + tm.y_var[tm.tree.children_left[v]] <= 1,
                            "leftDiscreteVariablesConsistencyConstraints_t"
                            + str(t) + "_v" + str(v)
                        )
                        self.rightDiscreteVariablesConsistencyConstraints[t][v] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][v_level] >= tm.y_var[tm.tree.children_right[v]],
                            "rightDiscreteVariablesConsistencyConstraints_t"
                            + str(t) + "_v" + str(v)
                        )

    def addInterTreeConstraints(self):
        self.addContinuousVariablesConsistencyConstraints()
        self.addDiscreteVariablesConsistencyConstraints()
        self.addMajorityVoteConstraint()

    def addCuts(self):
        if self.interTreeCutsActivated:
            self.addInterTreeCuts()
        if self.mutuallyExclusivePlanesCutsActivated and not self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes:
            self.addMutuallyExclusivePlanesCuts()

    # def initLinearCombinationOfPlane_L0_ObjOfFeature(self,f):
    #     self.absoluteValueVar[f] = self.model.addVar(vtype=GRB.BINARY,name="modified_f"+str(f))
    #     thresholds = list(self.rightMutuallyExclusivePlanesVar[f].keys())
    #     assert 1.0 not in thresholds
    #     thresholds.append(1.0)
    #     # for thres in self.rightMutuallyExclusivePlanesVar[f]:
    #     for t in range(len(self.rightMutuallyExclusivePlanesVar[f])):
    #         thres = thresholds[t]
    #         cellVar = self.rightMutuallyExclusivePlanesVar[f][thres]
    #         cellLb = thres
    #         cellUb = thresholds[t+1]
    #         if cellUb < self.x0[0][f]:
    #             self.model.addConstr(self.absoluteValueVar[f] >= (1-cellVar), "modified_f" + str(f) + "_t" + str(thres))
    #         elif self.x0[0][f] < cellLb:
    #             self.model.addConstr(self.absoluteValueVar[f] >= cellVar, "modified_f" + str(f) + "_t" + str(thres))
    #         else:
    #             x_val = gp.LinExpr(cellLb) + cellVar * (cellUb - cellLb)
    #             self.absoluteValueVar[f] = self.model.addVar(vtype=GRB.BINARY,name="abs_value_f"+str(f))
    #             self.absoluteValueLeftConstr[f] = self.model.addConstr(self.absoluteValueVar[f] >= x_val - self.x0[0][f], "absoluteValueLeftConstr_f" + str(f))
    #             self.absoluteValueRightConstr[f] = self.model.addConstr(self.absoluteValueVar[f] >= self.x0[0][f] - x_val, "absoluteValueRightConstr_f" + str(f))
    #     self.obj += self.absoluteValueVar[f]

    # def initLinearCombinationOfPlane_L1_ObjOfFeature(self,f):
    #     thresholds = list(self.rightMutuallyExclusivePlanesVar[f].keys())
    #     assert 1.0 not in thresholds
    #     thresholds.append(1.0)
    #     # for thres in self.rightMutuallyExclusivePlanesVar[f]:
    #     for t in range(len(self.rightMutuallyExclusivePlanesVar[f])):
    #         thres = thresholds[t]
    #         cellVar = self.rightMutuallyExclusivePlanesVar[f][thres]
    #         cellLb = thres
    #         cellUb = thresholds[t+1]
    #         if cellUb < self.x0[0][f]:
    #             self.obj += (1-cellVar) * (cellUb - cellLb)
    #         elif self.x0[0][f] < cellLb:
    #             self.obj += cellVar * (cellUb - cellLb)
    #         else:
    #             x_val = gp.LinExpr(cellLb) + cellVar * (cellUb - cellLb)
    #             self.absoluteValueVar[f] = self.model.addVar(lb=0.0,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,name="abs_value_f"+str(f))
    #             self.absoluteValueLeftConstr[f] = self.model.addConstr(self.absoluteValueVar[f] >= x_val - self.x0[0][f], "absoluteValueLeftConstr_f" + str(f))
    #             self.absoluteValueRightConstr[f] = self.model.addConstr(self.absoluteValueVar[f] >= self.x0[0][f] - x_val, "absoluteValueRightConstr_f" + str(f))
    #             self.obj += self.absoluteValueVar[f]

    # def initLinearCombinationOfPlane_L2_ObjOfFeature(self,f):
    #     thresholds = list(self.rightMutuallyExclusivePlanesVar[f].keys())
    #     assert 1.0 not in thresholds
    #     thresholds.append(1.0)
    #     # for thres in self.rightMutuallyExclusivePlanesVar[f]:
    #     for t in range(len(self.rightMutuallyExclusivePlanesVar[f])):
    #         thres = thresholds[t]
    #         cellVar = self.rightMutuallyExclusivePlanesVar[f][thres]
    #         cellLb = thres
    #         cellUb = thresholds[t+1]
    #         if cellUb < self.x0[0][f]:
    #             self.obj += (1-cellVar) * (cellUb - cellLb)
    #         elif self.x0[0][f] < cellLb:
    #             self.obj += cellVar * (cellUb - cellLb)
    #         else:
    #             x_val = gp.LinExpr(cellLb) + cellVar * (cellUb - cellLb)
    #             self.absoluteValueVar[f] = self.model.addVar(lb=0.0,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,name="abs_value_f"+str(f))
    #             self.absoluteValueLeftConstr[f] = self.model.addConstr(self.absoluteValueVar[f] >= x_val - self.x0[0][f], "absoluteValueLeftConstr_f" + str(f))
    #             self.absoluteValueRightConstr[f] = self.model.addConstr(self.absoluteValueVar[f] >= self.x0[0][f] - x_val, "absoluteValueRightConstr_f" + str(f))
    #             self.obj += self.absoluteValueVar[f]

    def lossFunctionValue(self, f, xf):
        if self.randomCostsActivated:
            if self.featuresType[f] == FeatureType.CategoricalNonOneHot:
                assert False
            elif self.objectiveNorm == 0:
                if xf > self.x0[0][f] + eps:
                    return self.greaterCosts[f]
                elif xf < self.x0[0][f] - eps:
                    return self.smallerCosts[f]
                else:
                    return 0.0
            elif self.objectiveNorm == 1:
                if xf > self.x0[0][f]:
                    return abs(xf - self.x0[0][f]) * self.greaterCosts[f]
                else:
                    return abs(xf - self.x0[0][f]) * self.smallerCosts[f]
            elif self.objectiveNorm == 2:
                if xf > self.x0[0][f]:
                    return ((xf - self.x0[0][f]) ** 2) * self.greaterCosts[f]
                else:
                    return ((xf - self.x0[0][f]) ** 2) * self.smallerCosts[f]
            else:
                print("unsupported norm")
                return float("inf")
        else:
            if self.featuresType[f] == FeatureType.CategoricalNonOneHot:
                if abs(xf - self.x0[0][f]) > eps:
                    return 1.0
                else:
                    return 0.0
            elif self.objectiveNorm == 0:
                if abs(xf - self.x0[0][f]) > eps:
                    return 1.0
                else:
                    return 0.0
            elif self.objectiveNorm == 1:
                return abs(xf - self.x0[0][f])
            elif self.objectiveNorm == 2:
                return (xf - self.x0[0][f]) ** 2
            else:
                print("unsupported norm")
                return float("inf")

    def initLinearCombinationOfPlaneObjOfFeature(self, f):
        thresholds = list(self.rightMutuallyExclusivePlanesVar[f].keys())
        assert 0.0 in thresholds
        if 1.0 not in thresholds:
            thresholds.append(1.0)
        else:
            thresholds.append(1.0 + eps)
        self.obj += self.lossFunctionValue(f, 0.0)
        for t in range(len(self.rightMutuallyExclusivePlanesVar[f])):
            thres = thresholds[t]
            cellVar = self.rightMutuallyExclusivePlanesVar[f][thres]
            cellLb = thres
            cellUb = thresholds[t+1]
            self.obj += cellVar * \
                (self.lossFunctionValue(f, cellUb)
                 - self.lossFunctionValue(f, cellLb))

    # def initDiscreteFeature_L0_Obj(self, f):
    #     self.absoluteValueVar[f] = self.model.addVar(vtype=GRB.BINARY,name="modified_f"+str(f))
    #     for l in range(1,len(self.featuresPossibleValues[f])):
    #         cellUb = self.featuresPossibleValues[f][l]
    #         cellVar = self.discreteFeaturesLevel_var[f][l]
    #         if cellUb > self.x0[0][f]:
    #             self.model.addConstr(self.absoluteValueVar[f] >= cellVar, "modified_f" + str(f) + "_l" + str(l))
    #         else:
    #             self.model.addConstr(self.absoluteValueVar[f] >= (1-cellVar), "modified_f" + str(f) + "_l" + str(l))
    #     self.obj += self.absoluteValueVar[f]

    def initDiscreteFeatureObj(self, f):
        self.obj += self.lossFunctionValue(f,
                                           self.featuresPossibleValues[f][0])
        for l in range(1, len(self.featuresPossibleValues[f])):
            cellLb = self.featuresPossibleValues[f][l-1]
            cellUb = self.featuresPossibleValues[f][l]
            cellVar = self.discreteFeaturesLevel_var[f][l]
            self.obj += cellVar * \
                (self.lossFunctionValue(f, cellUb)
                 - self.lossFunctionValue(f, cellLb))

    def initBinaryFeatureObj(self, f):
        self.obj += self.lossFunctionValue(f, 0) + self.x_var_sol[f] * (
            self.lossFunctionValue(f, 1) - self.lossFunctionValue(f, 0))

    def initObjectiveStructures(self):
        assert self.objectiveNorm in [0, 1, 2]
        self.obj = gp.LinExpr(0.0)

    def initLinearCombinationOfPlanesObj(self):
        self.initObjectiveStructures()
        for f in self.continuousFeatures:
            self.initLinearCombinationOfPlaneObjOfFeature(f)
        for f in self.discreteFeatures:
            self.initDiscreteFeatureObj(f)
        for f in self.categoricalNonOneHotFeatures:
            self.initDiscreteFeatureObj(f)
        for f in self.binaryFeatures:
            self.initBinaryFeatureObj(f)
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def initObjective(self):
        if self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes:
            self.initLinearCombinationOfPlanesObj()
        else:
            ClassifierCounterFactualMilp.initObjectiveStructures(self)
            ClassifierCounterFactualMilp.initObjective(self)

    def buildModel(self):
        self.initSolution()
        self.buildTrees()
        self.addInterTreeConstraints()
        self.addCuts()
        if self.isolationForest:
            self.addIsolationForestPlausibilityConstraint()
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()
        self.initObjective()

    def checkPredictionResult(self):
        badPrediction = True
        x_sol = np.array(self.x_sol, dtype=np.float32)
        if self.strictCounterFactual:
            badPrediction = (self.outputDesired
                             != self.clf.predict(self.x_sol))
            if not badPrediction and self.verbose:
                print("The desired counterfactual", self.outputDesired,
                      " is the class predicted by sklearn", self.clf.predict(self.x_sol))
        else:
            badPrediction = (self.outputDesired not in np.argwhere(
                max(self.clf.predict_proba(x_sol))))
            if not badPrediction and self.verbose:
                print("The desired counterfactual", self.outputDesired,
                      "is one of the argmax of the prediction proba", self.clf.predict_proba(x_sol))
        # if self.outputDesired != self.clf.predict(self.x_sol):
        if badPrediction:
            print("Error, the desired class is not the predicted one.")
        # if True:
        # Check that the trees output are the one desired
        if self.verbose:
            print("Proba predicted by sklearn", self.clf.predict_proba(x_sol))
        myProba = [0 for i in self.clf.classes_]
        for t in range(self.clf.n_estimators):
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    leaf_val = tm.tree.value[v][0]
                    tot = sum(leaf_val)
                    for output in range(len(leaf_val)):
                        myProba[output] += tm.y_var[v].getAttr(
                            GRB.Attr.X) * (leaf_val[output])/tot
        for p in range(len(myProba)):
            myProba[p] /= self.clf.n_estimators

        if self.verbose:
            print("My proba: ", myProba)
            print("Desired output:", self.outputDesired)
            print("Initial solution:\n", [self.x0[0][i] for i in range(
                len(self.x0[0]))], " with prediction ", self.clf.predict(self.x0))

        self.maxSkLearnError = 0.0
        self.maxMyMilpError = 0.0
        # Check decision path
        myMilpErrors = False
        skLearnErrors = False
        for t in range(self.clf.n_estimators):
            estimator = self.clf.estimators_[t]
            predictionPath = estimator.decision_path(self.x_sol)
            predictionPathList = list(
                [tuple(row) for row in np.transpose(predictionPath.nonzero())])
            verticesInPath = [v for d, v in predictionPathList]
            tm = self.treeManagers[t]
            solutionPathList = [u for u in range(
                tm.n_nodes) if tm.y_var[u].getAttr(GRB.attr.X) >= 0.1]
            if verticesInPath != solutionPathList:
                lastCommonVertex = max(
                    set(verticesInPath).intersection(set(solutionPathList)))
                f = tm.tree.feature[lastCommonVertex]
                if self.verbose:
                    print("Sklearn decision path ", verticesInPath,
                          " and my MILP decision path ", solutionPathList)
                if self.verbose:
                    print("Wrong decision vertex", lastCommonVertex, "Feature", f, "threshold",
                          tm.tree.threshold[lastCommonVertex], "solution feature value x_sol[f]=", self.x_sol[0][f])
                nextVertex = -1
                if (self.x_sol[0][f] <= tm.tree.threshold[lastCommonVertex]):
                    if self.verbose:
                        print("x_sol[f] <= threshold, next vertex in decision path should be:",
                              tm.tree.children_left[lastCommonVertex])
                    nextVertex = tm.tree.children_left[lastCommonVertex]
                else:
                    if self.verbose:
                        print("x_sol[f] > threshold,next vertex in decision path should be:",
                              tm.tree.children_right[lastCommonVertex])
                    nextVertex = tm.tree.children_right[lastCommonVertex]
                if nextVertex not in verticesInPath:
                    skLearnErrors = True
                    self.maxSkLearnError = max(self.maxSkLearnError, abs(
                        self.x_sol[0][f]-tm.tree.threshold[lastCommonVertex]))
                    if self.verbose:
                        print("sklearn is wrong")
                if nextVertex not in solutionPathList:
                    print("MY MILP IS WRONG")
                    myMilpErrors = True
                    self.maxMyMilpError = max(self.maxMyMilpError, abs(
                        self.x_sol[0][f]-tm.tree.threshold[lastCommonVertex]))
        if skLearnErrors and not myMilpErrors:
            print("Only sklearn numerical precision errors")
            # for v in range(tm.n_nodes):
            #     if tm.y_var[v].getAttr(GRB.Attr.X) != float(v in verticesInPath):
            #         print("In tree", t, "for vertex", v, "sklearn say path", verticesInPath, "while y_var[", v, "] = ", tm.y_var[v].getAttr(GRB.Attr.X), "and my solution path is", solutionPathList)
            #         prev = -1
            #         for u in verticesInPath:
            #             if u == v and u != 0:
            #                 f = tm.tree.feature[prev]
            #                 print("Parent", prev, "Feature", f, "threshold", tm.tree.threshold[prev], "solution feature value x_sol[f]=", self.x_sol[0][f])
            #                 if (self.x_sol[0][f] <= tm.tree.threshold[prev]):
            #                     print("x_sol[f] <= threshold, next vertex in decision path should be:", tm.tree.children_left[prev], "and it is", u)
            #                 else:
            #                     print("x_sol[f] > threshold,next vertex in decision path should be:", tm.tree.children_right[prev], "and it is", u)

            #                 for thres in self.rightMutuallyExclusivePlanesVar[f]:
            #                     print("Thres", thres, " var ", self.rightMutuallyExclusivePlanesVar[f][thres].getAttr(GRB.Attr.X))
            #             prev = u

            # print("counter")
            # print(estimator.decision_path(self.x_sol))
            # print(estimator.predict(self.x_sol))
            # print("initial")
            # print(estimator.decision_path(self.x0))
            # print(estimator.predict(self.x_sol))
        # Print variables
        # for v in self.model.getVars():
        #     print('%s %g' % (v.varName, v.x))

    def checkResultPlausibility(self):
        x_sol = np.array(self.x_sol, dtype=np.float32)
        if self.isolationForest.predict(x_sol)[0] == 1:
            if self.verbose:
                print("Result is an inlier")
        else:
            assert self.isolationForest.predict(x_sol)[0] == -1
            print("Result is an outlier")

    def checkSolution(self, solution):
        for f in range(len(solution)):
            self.x_var_sol[f].setAttr(GRB.Attr.LB, solution[f] - eps)
            self.x_var_sol[f].setAttr(GRB.Attr.UB, solution[f] + eps)

        self.model.write("rf_check.lp")
        self.model.setParam(GRB.Param.ImpliedCuts, 2)
        # self.model.setParam(GRB.Param.Method,3)
        self.model.setParam(GRB.Param.Threads, 4)
        self.model.setParam(GRB.Param.TimeLimit, 300)

        self.model.optimize()
        self.solutionFeasibility = self.model.status == GRB.OPTIMAL

    def solveModel(self):
        self.model.write("rf.lp")
        self.model.setParam(GRB.Param.ImpliedCuts, 2)
        # self.model.setParam(GRB.Param.Method,3)
        self.model.setParam(GRB.Param.Threads, 4)
        self.model.setParam(GRB.Param.TimeLimit, 900)

        self.model.optimize()

        self.runTime = self.model.Runtime

        if self.model.status != GRB.OPTIMAL:
            self.objValue = "inf"
            self.maxSkLearnError = "inf"
            self.maxMyMilpError = "inf"
            self.x_sol = self.x0
            return False

        self.objValue = self.model.ObjVal
        self.x_sol = [[]]
        for f in range(self.nFeatures):
            self.x_sol[0].append(self.x_var_sol[f].getAttr(GRB.Attr.X))

        self.checkPredictionResult()
        if self.isolationForest:
            self.checkResultPlausibility()

        if self.verbose:
            print("Solution built \n", self.x_sol,
                  " with prediction ", self.clf.predict(self.x_sol))

        return True
