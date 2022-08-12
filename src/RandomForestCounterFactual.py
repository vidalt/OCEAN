import gurobipy as gp
from gurobipy import GRB
from sklearn.ensemble._iforest import _average_path_length
# Import OCEAN functions and classes
from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables as BinDec
from src.CounterFactualParameters import TreeConstraintsType
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import eps
from src.TreeMilpManager import TreeInMilpManager


class RandomForestCounterfactualMilp():
    def __init__(self, mutuallyExclusivePlanesCutsActivated=False,
                 constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                 binaryDecisionVariables=BinDec.LeftRight_lambda):
        self.useExclusivePlanesCuts = mutuallyExclusivePlanesCutsActivated
        # Specify formulation parameters of forest MILP
        self.constraintsType = constraintsType
        self.binaryDecisionVariables = binaryDecisionVariables

    # ---------------------- Private methods ------------------------
    # -- Initialize RandomForestCounterFactualMilp --
    # -- Build optimization model --
    def __buildTrees(self):
        """ Build a TreeMilpManager for each tree in the completeForest."""
        self.treeManagers = dict()
        for t in range(self.completeForest.n_estimators):
            self.treeManagers[t] = TreeInMilpManager(
                self.completeForest.estimators_[t].tree_,
                self.model, self.x_var_sol,
                self.outputDesired, self.featuresType, self.constraintsType)
            self.treeManagers[t].addTreeVariablesAndConstraintsToMilp()

    def __addInterTreeConstraints(self):
        useLinCombPlanes = (self.constraintsType
                            == TreeConstraintsType.LinearCombinationOfPlanes)
        if useLinCombPlanes:
            self.__addPlaneConsistencyConstraints()
        self.__addDiscreteVariablesConsistencyConstraints()

    def __addPlaneConsistencyConstraints(self):
        """
        Express the counterfactual sample flow through
        the completeForest as a linear combination of planes.
        """
        # Add a plane for each continuous feature
        self.planes = dict()
        for f in self.continuousFeatures:
            self.planes[f] = dict()
            # Add the initial value as a plane
            self.planes[f][self.x0[0][f]] = []

        # Read the split threshold of all trees for all continuous features
        for t in range(self.completeForest.n_estimators):
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    if self.featuresType[f] == FeatureType.Numeric:
                        thres = tm.tree.threshold[v]
                        newPlane = True
                        if self.planes[f]:
                            nearestThres = min(self.planes[f].keys(),
                                               key=lambda k: abs(k-thres))
                            # Check that two thres are sufficiently distinct
                            if abs(thres - nearestThres) < 0.8*eps:
                                newPlane = False
                                self.planes[f][nearestThres].append((t, v))
                        if newPlane:
                            self.planes[f][thres] = [(t, v)]

        # Add the constraints
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
            self.rightMutuallyExclusivePlanesVar[f][previousThres] = self.model.addVar(
                lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                name="rightMutuallyExclusivePlanesVar_f" + str(f)
                + "_th" + str(previousThres))
            for thres in sorted(self.planes[f]):
                self.rightMutuallyExclusivePlanesVar[f][thres] = self.model.addVar(
                    lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                    name="rightMutuallyExclusivePlanesVar_f" + str(f)
                    + "_th" + str(thres))
                self.rightMutuallyExclusivePlanesConstr[f][thres] = []
                self.rightPlanesDominateRightFlowConstr[f][thres] = []

                for t, v in self.planes[f][thres]:
                    tm = self.treeManagers[t]
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(
                        self.model.addConstr(
                            tm.y_var[tm.tree.children_left[v]]
                            + self.rightMutuallyExclusivePlanesVar[f][thres]
                            <= 1, "rightMutuallyExclusivePlanesVar_f"
                            + str(f) + "_t" + str(t) + "_v" + str(v)))
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(
                        self.model.addConstr(
                            tm.y_var[tm.tree.children_right[v]]
                            <= self.rightMutuallyExclusivePlanesVar[f][previousThres],
                            "rightPlanesDominatesLeftFlowConstr_t"
                            + str(t)+"_v"+str(v)))
                    # Avoid numerical precision errors
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(
                        self.model.addConstr(
                            (thres - previousThres)
                            * self.rightMutuallyExclusivePlanesVar[f][previousThres]
                            <= (thres - previousThres)
                            - min(thres - previousThres, eps)
                            * tm.y_var[tm.tree.children_left[v]],
                            "rightMutuallyExclusivePlanesVar_eps_f"
                            + str(f) + "_t" + str(t) + "_v" + str(v)))
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(
                        self.model.addConstr(
                            eps * tm.y_var[tm.tree.children_right[v]]
                            <= self.rightMutuallyExclusivePlanesVar[f][thres]
                            * max(eps, (thres - previousThres)),
                            "rightPlanesDominatesLeftFlowConstr_eps_t"
                            + str(t)+"_v"+str(v)))

                self.rightPlanesOrderConstr[f][thres] = self.model.addConstr(
                    self.rightMutuallyExclusivePlanesVar[f][previousThres]
                    >= self.rightMutuallyExclusivePlanesVar[f][thres],
                    "rightPlanesOrderConstr_f"+str(f)+"_th"+str(thres))

                linearCombination += self.rightMutuallyExclusivePlanesVar[f][previousThres] * (
                    thres - previousThres)

                previousThres = thres
            linearCombination += self.rightMutuallyExclusivePlanesVar[f][previousThres] * (
                1.0 - previousThres)
            self.linearCombinationOfPlanesConstr[f] = self.model.addConstr(
                self.x_var_sol[f] == linearCombination, "x_as_linear_combination_of_planes_f")

    def __addDiscreteVariablesConsistencyConstraints(self):
        self.leftDiscreteVariablesConsistencyConstraints = dict()
        self.rightDiscreteVariablesConsistencyConstraints = dict()
        for t in range(self.completeForest.n_estimators):
            tm = self.treeManagers[t]
            self.leftDiscreteVariablesConsistencyConstraints[t] = dict()
            self.rightDiscreteVariablesConsistencyConstraints[t] = dict()
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    isFeatDiscrete = (self.featuresType[f]
                                      == FeatureType.Discrete)
                    isFeatCategoricalNonOneHot = (
                        self.featuresType[f] == FeatureType.CategoricalNonOneHot)
                    if isFeatDiscrete or isFeatCategoricalNonOneHot:
                        thres = tm.tree.threshold[v]
                        levels = list(self.featuresPossibleValues[f])
                        levels.append(1.0)
                        v_level = -1
                        for levelIndex in range(len(levels)):
                            if levels[levelIndex] > thres:
                                v_level = levelIndex
                                break
                        self.leftDiscreteVariablesConsistencyConstraints[t][v] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][v_level]
                            + tm.y_var[tm.tree.children_left[v]] <= 1,
                            "leftDiscreteVariablesConsistencyConstraints_t"
                            + str(t) + "_v" + str(v))
                        self.rightDiscreteVariablesConsistencyConstraints[t][v] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][v_level]
                            >= tm.y_var[tm.tree.children_right[v]],
                            "rightDiscreteVariablesConsistencyConstraints_t"
                            + str(t) + "_v" + str(v))

    def __addMutuallyExclusivePlanesCuts(self):
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
                            nearestThres = min(self.planes[f].keys(),
                                               key=lambda k: abs(k-thres))
                            if abs(thres - nearestThres) < 0.8*eps:
                                newPlane = False
                                self.planes[f][nearestThres].append((t, v))
                        if newPlane:
                            self.planes[f][thres] = [(t, v)]

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
                self.rightMutuallyExclusivePlanesVar[f][thres] = self.model.addVar(
                    lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                    name="rightMutuallyExclusivePlanesVar_f" + str(f)
                    + "_th" + str(thres))

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

    def __addIsolationForestPlausibilityConstraint(self):
        """
        The counterfactual should not be classified
        as an outlier by the input trained isolationForest.
        """
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

    # -- Implement objective function --
    def __initModelObjective(self):
        useLinCombPlanes = (self.constraintsType
                            == TreeConstraintsType.LinearCombinationOfPlanes)
        if useLinCombPlanes:
            self.__initLinearCombinationOfPlanesObj()
        else:
            ClassifierCounterFactualMilp.initObjectiveStructures(self)
            ClassifierCounterFactualMilp.initObjective(self)

    def __initLinearCombinationOfPlanesObj(self):
        if self.objectiveNorm not in [0, 1, 2]:
            raise ValueError("Unknown objective norm")
        self.obj = gp.LinExpr(0.0)
        for f in self.continuousFeatures:
            self.__initLinearCombinationOfPlaneObjOfFeature(f)
        for f in self.discreteFeatures:
            self.__initDiscreteFeatureObj(f)
        for f in self.categoricalNonOneHotFeatures:
            self.__initDiscreteFeatureObj(f)
        for f in self.binaryFeatures:
            self.__initBinaryFeatureObj(f)
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def __lossFunctionValue(self, f, xf):
        """
        Returns the cost of a counterfactual change on feature f
        according to the norm chosen (default=2)
        and whether random costs are used.
        """
        # Scale differently the positive and negative
        # changes if random costs are given
        if self.randomCostsActivated:
            posCostScale = self.greaterCosts[f]
            negCostScale = self.smallerCosts[f]
        else:
            posCostScale = 1
            negCostScale = 1
        # Measure the loss according to specified norm
        if self.featuresType[f] == FeatureType.CategoricalNonOneHot:
            # CategoricalNonOneHot feature is a special case
            if self.randomCostsActivated:
                raise ValueError("Cannot have CategoricalNonOneHot"
                                 " with random costs.")
            if abs(xf - self.x0[0][f]) > eps:
                loss = 1.0
            else:
                loss = 0.0
        elif self.objectiveNorm == 0:
            if xf > self.x0[0][f] + eps:
                loss = posCostScale
            elif xf < self.x0[0][f] - eps:
                loss = negCostScale
            else:
                loss = 0.0
        elif self.objectiveNorm == 1:
            absDiff = abs(xf - self.x0[0][f])
            if xf > self.x0[0][f]:
                loss = absDiff * posCostScale
            else:
                loss = absDiff * negCostScale
        elif self.objectiveNorm == 2:
            squaredDiff = ((xf - self.x0[0][f]) ** 2)
            if xf > self.x0[0][f]:
                loss = squaredDiff * posCostScale
            else:
                loss = squaredDiff * negCostScale
        else:
            raise ValueError("Unsupported norm")
        return loss

    def __initLinearCombinationOfPlaneObjOfFeature(self, f):
        thresholds = list(self.rightMutuallyExclusivePlanesVar[f].keys())
        assert 0.0 in thresholds
        if 1.0 not in thresholds:
            thresholds.append(1.0)
        else:
            thresholds.append(1.0 + eps)
        self.obj += self.__lossFunctionValue(f, 0.0)
        for t in range(len(self.rightMutuallyExclusivePlanesVar[f])):
            thres = thresholds[t]
            cellVar = self.rightMutuallyExclusivePlanesVar[f][thres]
            cellLb = thres
            cellUb = thresholds[t+1]
            self.obj += cellVar * \
                (self.__lossFunctionValue(f, cellUb)
                 - self.__lossFunctionValue(f, cellLb))

    def __initDiscreteFeatureObj(self, f):
        self.obj += self.__lossFunctionValue(f,
                                             self.featuresPossibleValues[f][0])
        for valIndex in range(1, len(self.featuresPossibleValues[f])):
            cellLb = self.featuresPossibleValues[f][valIndex-1]
            cellUb = self.featuresPossibleValues[f][valIndex]
            cellVar = self.discreteFeaturesLevel_var[f][valIndex]
            self.obj += cellVar * \
                (self.__lossFunctionValue(f, cellUb)
                 - self.__lossFunctionValue(f, cellLb))

    def __initBinaryFeatureObj(self, f):
        self.obj += self.__lossFunctionValue(f, 0) \
            + self.x_var_sol[f] * (self.__lossFunctionValue(f, 1)
                                   - self.__lossFunctionValue(f, 0))

    # ---------------------- Public methods ------------------------
    def buildForest(self):
        self.__buildTrees()
        self.__addInterTreeConstraints()
        useMutuallyExcPlanesCuts = self.useExclusivePlanesCuts
        useLinCombOfPlanes = (self.constraintsType
                              == TreeConstraintsType.LinearCombinationOfPlanes)
        if useMutuallyExcPlanesCuts and not useLinCombOfPlanes:
            self.__addMutuallyExclusivePlanesCuts()
        if self.isolationForest:
            self.__addIsolationForestPlausibilityConstraint()
        self.__initModelObjective()
