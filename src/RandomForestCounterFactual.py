import random
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.ensemble._iforest import _average_path_length
# Import OCEAN functions and classes
from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.RandomAndIsolationForest import RandomAndIsolationForest
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import eps
from src.TreeMilpManager import TreeInMilpManager


class RandomForestCounterFactualMilp(ClassifierCounterFactualMilp):
    def __init__(
            self, classifier, sample, outputDesired,
            objectiveNorm=2, isolationForest=None, verbose=False,
            mutuallyExclusivePlanesCutsActivated=False,
            strictCounterFactual=False,
            featuresType=False, featuresPossibleValues=False,
            featuresActionnability=False, oneHotEncoding=False,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda,
            randomCostsActivated=False):
        # Instantiate the ClassifierMilp: implements actionability constraints
        # and feature consistency according to featuresType
        ClassifierCounterFactualMilp.__init__(
            self, classifier, sample, outputDesired, constraintsType,
            objectiveNorm, verbose, featuresType, featuresPossibleValues,
            featuresActionnability, oneHotEncoding, binaryDecisionVariables)
        self.model.modelName = "RandomForestCounterFactualMilp"
        # Combine random forest and isolation forest into a completeForest
        self.isolationForest = isolationForest
        self.completeForest = RandomAndIsolationForest(self.clf,
                                                       isolationForest)
        # - Read and set formulation parameters -
        self.strictCounterFactual = strictCounterFactual
        self.mutuallyExclusivePlanesCutsActivated = mutuallyExclusivePlanesCutsActivated
        self.randomCostsActivated = randomCostsActivated
        if randomCostsActivated:
            self.__generate_random_costs()

    # ---------------------- Private methods ------------------------
    # -- Initialize RandomForestCounterFactualMilp --
    def __generate_random_costs(self):
        """ Sample cost parameters for features from uniform distribution."""
        random.seed(0)
        self.greaterCosts = [random.uniform(0.5, 2)
                             for i in range(self.nFeatures)]
        self.smallerCosts = [random.uniform(0.5, 2)
                             for i in range(self.nFeatures)]

    # -- Build optimization model --
    def __buildTrees(self):
        """ Build a TreeMilpManager for each tree in the completeForest."""
        self.treeManagers = dict()
        for t in range(self.completeForest.n_estimators):
            self.treeManagers[t] = TreeInMilpManager(
                self.completeForest.estimators_[t].tree_,
                self.model, self.x_var_sol,
                self.outputDesired, self.featuresType,
                self.constraintsType, self.binaryDecisionVariables)
            self.treeManagers[t].addTreeVariablesAndConstraintsToMilp()

    def __addInterTreeConstraints(self):
        useLinCombPlanes = (self.constraintsType
                            == TreeConstraintsType.LinearCombinationOfPlanes)
        if useLinCombPlanes:
            self.__addPlaneConsistencyConstraints()
        self.__addDiscreteVariablesConsistencyConstraints()
        self.__addMajorityVoteConstraint()

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

    def __addMajorityVoteConstraint(self):
        """
        Ensures that the random forest predicts the target class
        for the counterfactual explanation.
        At least 50% of the trees should vote for the target class.
        """
        # Measure the classification score:
        #   the average of the tree class predictions
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
        # Add (strict) constraint on target score
        self.majorityVoteConstr = dict()
        for cl in majorityVoteExpr:
            if self.strictCounterFactual:
                majorityVoteExpr[cl] -= 1e-4
            self.majorityVoteConstr[cl] = self.model.addConstr(
                majorityVoteExpr[cl] >= 0, "majorityVoteConstr_cl" + str(cl))

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

    # -- Check model status and solution --
    def __checkIfBadPrediction(self, x_sol):
        badPrediction = (self.outputDesired != self.clf.predict(self.x_sol))
        if badPrediction:
            print("Error, the desired class is not the predicted one.")
            if self.verbose:
                # Print a detailed error statement
                skLearnPrediction = self.clf.predict(self.x_sol)
                skLearnScore = self.clf.predict_proba(x_sol)
                if self.strictCounterFactual:
                    print("The desired counterfactual", self.outputDesired,
                          " is the class predicted by sklearn",
                          skLearnPrediction)
                else:
                    badScore = (self.outputDesired not in np.argwhere(
                        max(skLearnScore)))
                    if not badScore:
                        print("The desired counterfactual", self.outputDesired,
                              "is one of the argmax of the prediction proba",
                              skLearnScore)

    def __checkClassificationScore(self, x_sol):
        if self.verbose:
            print("Score predicted by sklearn", self.clf.predict_proba(x_sol))
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
            print("Initial solution:\n",
                  [self.x0[0][i] for i in range(len(self.x0[0]))],
                  " with prediction ", self.clf.predict(self.x0))

    def __checkDecisionPath(self, x_sol):
        """
        Compare the counterfactual sample flow in sklearn
        and in the MILP implementation: they should be identical.
        """
        self.maxSkLearnError = 0.0
        self.maxMyMilpError = 0.0
        myMilpErrors = False
        skLearnErrors = False
        for t in range(self.clf.n_estimators):
            estimator = self.clf.estimators_[t]
            predictionPath = estimator.decision_path(x_sol)
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
                    print("Wrong decision vertex", lastCommonVertex,
                          "Feature", f,
                          "threshold", tm.tree.threshold[lastCommonVertex],
                          "solution feature value x_sol[f]=", x_sol[0][f])
                nextVertex = -1
                if (x_sol[0][f] <= tm.tree.threshold[lastCommonVertex]):
                    if self.verbose:
                        print("x_sol[f] <= threshold,"
                              " next vertex in decision path should be:",
                              tm.tree.children_left[lastCommonVertex])
                    nextVertex = tm.tree.children_left[lastCommonVertex]
                else:
                    if self.verbose:
                        print("x_sol[f] > threshold,"
                              " next vertex in decision path should be:",
                              tm.tree.children_right[lastCommonVertex])
                    nextVertex = tm.tree.children_right[lastCommonVertex]
                if nextVertex not in verticesInPath:
                    skLearnErrors = True
                    self.maxSkLearnError = max(self.maxSkLearnError, abs(
                        x_sol[0][f]-tm.tree.threshold[lastCommonVertex]))
                    if self.verbose:
                        print("sklearn is wrong")
                if nextVertex not in solutionPathList:
                    print("MY MILP IS WRONG")
                    myMilpErrors = True
                    self.maxMyMilpError = max(self.maxMyMilpError, abs(
                        x_sol[0][f]-tm.tree.threshold[lastCommonVertex]))
        if skLearnErrors and not myMilpErrors:
            print("Only sklearn numerical precision errors")

    def __checkResultPlausibility(self):
        x_sol = np.array(self.x_sol, dtype=np.float32)
        if self.isolationForest.predict(x_sol)[0] == 1:
            if self.verbose:
                print("Result is an inlier")
        else:
            assert self.isolationForest.predict(x_sol)[0] == -1
            print("Result is an outlier")

    # ---------------------- Public methods ------------------------
    def buildModel(self):
        self.initSolution()
        self.__buildTrees()
        self.__addInterTreeConstraints()
        useMutuallyExcPlanesCuts = self.mutuallyExclusivePlanesCutsActivated
        useLinCombOfPlanes = (self.constraintsType
                              == TreeConstraintsType.LinearCombinationOfPlanes)
        if useMutuallyExcPlanesCuts and not useLinCombOfPlanes:
            self.__addMutuallyExclusivePlanesCuts()
        if self.isolationForest:
            self.__addIsolationForestPlausibilityConstraint()
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()
        self.__initModelObjective()

    def solveModel(self):
        self.model.write("rf.lp")
        self.model.setParam(GRB.Param.ImpliedCuts, 2)
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
        # Get model solution
        self.objValue = self.model.ObjVal
        self.x_sol = [[]]
        for f in range(self.nFeatures):
            self.x_sol[0].append(self.x_var_sol[f].getAttr(GRB.Attr.X))
        if self.verbose:
            print("Solution built \n", self.x_sol,
                  " with prediction ", self.clf.predict(self.x_sol))
        # Check results consistency
        self.__checkIfBadPrediction(self.x_sol)
        self.__checkClassificationScore(self.x_sol)
        self.__checkDecisionPath(self.x_sol)
        if self.isolationForest:
            self.__checkResultPlausibility()
        return True
