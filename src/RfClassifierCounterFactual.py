import random
import gurobipy as gp
from gurobipy import GRB
import numpy as np
# Import OCEAN functions and classes
from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.RandomForestCounterfactual import RandomForestCounterfactualMilp
from src.RandomAndIsolationForest import RandomAndIsolationForest
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType


class RfClassifierCounterFactualMilp(ClassifierCounterFactualMilp,
                                     RandomForestCounterfactualMilp):
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
            self, classifier, sample, outputDesired,
            objectiveNorm, verbose, featuresType, featuresPossibleValues,
            featuresActionnability, oneHotEncoding)
        # Instantiate the RandomForestCounterfactualMilp object
        RandomForestCounterfactualMilp.__init__(
            self, mutuallyExclusivePlanesCutsActivated,
            constraintsType, binaryDecisionVariables)
        assert len(self.clf.feature_importances_) == self.nFeatures
        self.model.modelName = "RandomForestCounterFactualMilp"
        # Combine random forest and isolation forest into a completeForest
        self.isolationForest = isolationForest
        self.completeForest = RandomAndIsolationForest(self.clf,
                                                       isolationForest)
        # - Read and set formulation parameters -
        self.strictCounterFactual = strictCounterFactual
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
        self.buildForest()
        self.__addMajorityVoteConstraint()
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()

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
