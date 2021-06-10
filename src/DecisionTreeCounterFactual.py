from TreeMilpManager import * 
from ClassifierCounterFactual import *         

class DecisionTreeCounterFactualMilp(ClassifierCounterFactualMilp):
    def __init__(
        self, classifier, sample, outputDesired, 
        constraintsType=TreeConstraintsType.ExtendedFormulation, objectiveNorm=2,
        verbose=False, featuresType=False, featuresPossibleValues = False,
        binaryDecisionVariables = BinaryDecisionVariables.LeftRight_lambda
    ):
        ClassifierCounterFactualMilp.__init__(self, classifier, sample, outputDesired, 
            constraintsType, objectiveNorm, verbose, featuresType, featuresPossibleValues,
            binaryDecisionVariables
        )
        self.model.modelName="DecisionTreeCounterFactualMilp"

    def buildModel(self):
        self.initSolution()
        self.treeManager = TreeInMilpManager(self.clf.tree_,self.model,self.x_var_sol, 
            self.outputDesired, self.featuresType, self.constraintsType, self.binaryDecisionVariables
        )
        self.treeManager.addTreeVariablesAndConstraintsToMilp()
        self.treeManager.addTreeOuputConstraints()
        self.initObjective()

    def solveModel(self):
        self.model.write("tree.lp")
        self.model.optimize()

        self.x_sol=[[]] 
        for f in range(self.nFeatures):
            self.x_sol[0].append(self.x_var_sol[f].getAttr(GRB.Attr.X) )
        print("Desired output:", self.outputDesired)
        print("Solution built \n", self.x_sol, " with prediction ", self.clf.predict(self.x_sol))
        print("Solution built decision path:\n", self.clf.decision_path(self.x_sol))
        print("Initial solution:\n", self.x0, " with prediction ", self.clf.predict(self.x0))
        print("Initial solution decision path:\n", self.clf.decision_path(self.x0))

