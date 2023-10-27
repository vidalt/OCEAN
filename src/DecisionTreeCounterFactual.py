from gurobipy import GRB
import numpy as np

from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.TreeMilpManager import TreeInMilpManager


class DecisionTreeCounterFactualMilp(ClassifierCounterFactualMilp):
    def __init__(
            self, classifier, sample, outputDesired,
            objectiveNorm=1, verbose=False,
            featuresType=False,
            featuresPossibleValues=False,
            featuresActionnability=False,
            oneHotEncoding=False,
            constraintsType=TreeConstraintsType.ExtendedFormulation,
            binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda):
        ClassifierCounterFactualMilp.__init__(
            self, classifier, sample, int(outputDesired),
            objectiveNorm=objectiveNorm,
            verbose=verbose,
            featuresType=featuresType,
            featuresPossibleValues=featuresPossibleValues,
            featuresActionnability=featuresActionnability,
            oneHotEncoding=oneHotEncoding)
        self.model.modelName = "DecisionTreeCounterFactualMilp"
        # Specify formulation parameters of forest MILP
        self.constraintsType = constraintsType
        self.binaryDecisionVariables = binaryDecisionVariables
        self.strictCounterFactual = False

    # - Private methods -
    def _sum_all_leaves_class(self, treeMng):
        """Get tree prediction as a linear expression."""
        treePredictLinExpr = 0.0
        for node in range(treeMng.n_nodes):
            if treeMng.is_leaves[node]:  # Leaf node
                leafSamples = treeMng.tree.value[node][0]
                majorityClass = np.argmax(leafSamples)
                treePredictLinExpr += treeMng.y_var[node] * majorityClass
        return treePredictLinExpr

    def _add_target_class_constraints(self):
        # Initialize variable for the tree's prediction
        self.milp_class = self.model.addVar(vtype=GRB.BINARY,
                                            name="milp_class")
        # Identify tree's prediction from y variables
        self.model.addConstr(
            self.milp_class == self._sum_all_leaves_class(
                self.treeManager))
        # Add constraint that predicted class is target class
        self.model.addConstr(self.milp_class == self.outputDesired)

    # -- Check model status and solution --
    def _checkIfBadPrediction(self, x_sol):
        # Compare MILP class and target class
        milpPredictedClass = self.milp_class.getAttr(GRB.attr.X)
        if not milpPredictedClass == self.outputDesired:
            print("MILP does not find target class!")
        # Compare sklearn class and MILP class
        skLearnPredictedClass = self.clf.predict(x_sol)[0]
        badPrediction = (self.outputDesired != skLearnPredictedClass)
        if badPrediction:
            print("Error: the desired class is not the predicted one.")

    def _checkClassificationScore(self, x_sol):
        if self.verbose:
            skLearnScore = self.clf.predict_proba(x_sol)[0]
            print("Score predicted by sklearn", skLearnScore)
            badScore = not self.outputDesired == np.argmax(skLearnScore)
            if badScore:
                print("The desired counterfactual with class",
                      self.outputDesired,
                      "is NOT the argmax of sklearn's score",
                      skLearnScore)
            else:
                print("The desired counterfactual with class",
                      self.outputDesired,
                      "is the argmax of sklearn's score", skLearnScore)

    def _checkDecisionPath(self, x_sol):
        """Check path of sample in tree.

        Compare the counterfactual sample flow in sklearn
        and in the MILP implementation: they should be identical.
        """
        self.maxSkLearnError = 0.0
        self.maxMyMilpError = 0.0
        myMilpErrors = False
        skLearnErrors = False
        predictionPath = self.clf.decision_path(x_sol)
        predictionPathList = list(
            [tuple(row) for row in np.transpose(predictionPath.nonzero())])
        verticesInPath = [v for _, v in predictionPathList]
        tm = self.treeManager
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

    # - Public methods -
    def buildModel(self):
        self.initSolution()
        # Create tree manager: track the path of explanation in tree
        self.treeManager = TreeInMilpManager(
            self.clf.tree_, self.model, self.x_var_sol,
            self.outputDesired, self.featuresType,
            constraintsType=self.constraintsType,
            binaryDecisionVariables=self.binaryDecisionVariables)
        self.treeManager.addTreeVariablesAndConstraintsToMilp()
        # Add constraints on features type and actionability
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()
        # Add constraint on target class
        self._add_target_class_constraints()
        self.initObjective()

    def solveModel(self):
        self.model.write("tree.lp")
        self.model.optimize()
        self.runTime = self.model.Runtime
        if self.model.status != GRB.OPTIMAL:
            self.objValue = "inf"
            self.maxSkLearnError = "inf"
            self.maxMyMilpError = "inf"
            self.x_sol = self.x0
            return False
        # Extract solution explanation and print details
        self.x_sol = [[]]
        for f in range(self.nFeatures):
            self.x_sol[0].append(self.x_var_sol[f].getAttr(GRB.Attr.X))
        print("Desired output:", self.outputDesired)
        print("Solution built \n", self.x_sol,
              " with prediction ", self.clf.predict(self.x_sol))
        print("Initial solution:\n", self.x0,
              " with prediction ", self.clf.predict(self.x0))
        # Check results consistency
        self._checkIfBadPrediction(self.x_sol)
        self._checkClassificationScore(self.x_sol)
        self._checkDecisionPath(self.x_sol)
        return True
