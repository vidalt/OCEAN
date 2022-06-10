from gurobipy import GRB
import numpy as np
# Import OCEAN utility functions and types
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import eps


class TreeInMilpManager:
    treeCount = 0

    def __init__(self, tree, model, x_var_sol, outputDesired,
                 featuresType,
                 constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                 binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda
                 ):
        self.id = TreeInMilpManager.treeCount
        TreeInMilpManager.treeCount += 1
        self.model = model
        self.tree = tree
        self.x_var_sol = x_var_sol
        self.nFeatures = len(self.x_var_sol)
        self.outputDesired = outputDesired
        self.constraintsType = constraintsType
        self.binaryDecisionVariables = binaryDecisionVariables
        assert featuresType
        self.featuresType = featuresType
        self.initTreeInfo()

    def initTreeInfo(self):
        self.n_nodes = self.tree.node_count
        self.is_leaves = dict()
        self.node_depth = dict()
        self.continuousFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Numeric]
        self.binaryFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Binary]
        stack = [(0, 0)]
        while len(stack) > 0:
            node_id, depth = stack.pop()
            self.node_depth[node_id] = depth
            is_split_node = self.tree.children_left[node_id] != self.tree.children_right[node_id]
            if is_split_node:
                stack.append((self.tree.children_left[node_id], depth + 1))
                stack.append((self.tree.children_right[node_id], depth + 1))
                self.is_leaves[node_id] = False
            else:
                self.is_leaves[node_id] = True

    def addTreeOuputConstraints(self):
        self.leaves_constr = dict()
        for v in range(self.n_nodes):
            if self.is_leaves[v] and np.argmax(self.tree.value[v]) != self.outputDesired:
                self.leaves_constr[v] = self.model.addConstr(
                    self.y_var[v] <= 0, "leaf_v" + str(v)+"_t"+str(self.id))

    def addTreeExtendedFormulationVariablesAndConstraints(self):
        # Tree node disjunction variables
        self.x_var_nodes = dict()
        for f in self.continuousFeatures:
            self.x_var_nodes[f] = dict()
            for v in range(self.n_nodes):
                self.x_var_nodes[f][v] = self.model.addVar(
                    lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x_f"+str(f)+"_"+str(v)+"_t"+str(self.id))

        # Disjunction polytope constraints
        root_constr_x = dict()
        for f in self.continuousFeatures:
            root_constr_x[f] = self.model.addConstr(
                self.x_var_sol[f] == self.x_var_nodes[f][0], "root_constr_x_f"+str(f)+"_t"+str(self.id))
        disjunction_flow_constr = dict()
        disjunction_left_constr = dict()
        disjunction_right_constr = dict()
        for f in self.continuousFeatures:
            disjunction_flow_constr[f] = dict()
            for v in range(self.n_nodes):
                if not self.is_leaves[v]:
                    disjunction_flow_constr[f][v] = self.model.addConstr(self.x_var_nodes[f][v] == self.x_var_nodes[f][self.tree.children_left[v]]
                                                                         + self.x_var_nodes[f][self.tree.children_right[v]], "disjunction_flow_constr_f" + str(f) + "_v" + str(v)+"_t"+str(self.id))
                    if self.tree.feature[v] == f:
                        disjunction_left_constr[v] = self.model.addConstr(self.x_var_nodes[f][self.tree.children_left[v]] <= (
                            self.tree.threshold[v]) * self.y_var[self.tree.children_left[v]], "disjunction_left_const_v" + str(v)+"_t"+str(self.id))
                        disjunction_right_constr[v] = self.model.addConstr(self.x_var_nodes[f][self.tree.children_right[v]] >= (
                            self.tree.threshold[v] + eps) * self.y_var[self.tree.children_right[v]], "disjunction_right_const_v" + str(v)+"_t"+str(self.id))

        # Linking constraints between the disjonction polytope and the decision path
        link_constr = dict()
        for f in range(self.nFeatures):
            link_constr[f] = dict()
            for v in range(self.n_nodes):
                link_constr[f][v] = self.model.addConstr(
                    self.x_var_nodes[f][v] <= self.y_var[v], "link_constr_f" + str(f) + "_v" + str(v)+"_t"+str(self.id))

    def addTreeBigMConstraints(self):
        self.bigMleftConstr = dict()
        self.bigMrightConstr = dict()
        for v in range(self.n_nodes):
            if not self.is_leaves[v] and self.featuresType[self.tree.feature[v]] == FeatureType.Numeric:
                bigMleft = 1 - self.tree.threshold[v] + eps
                self.bigMleftConstr[v] = self.model.addConstr(self.x_var_sol[self.tree.feature[v]] <= self.tree.threshold[v] - eps + bigMleft * (
                    1-self.y_var[self.tree.children_left[v]]), "bigMleftConstr_v"+str(v)+"_t"+str(self.id))
                bigMright = self.tree.threshold[v] + eps
                self.bigMrightConstr[v] = self.model.addConstr(self.x_var_sol[self.tree.feature[v]] >= (
                    self.tree.threshold[v] + eps) - bigMright * (1-self.y_var[self.tree.children_right[v]]), "bigMrightConstr_v"+str(v)+"_t"+str(self.id))

    def addBranchingAndDecisionPathVariablesAndConstraints(self):
        self.y_var = dict()
        if self.binaryDecisionVariables == BinaryDecisionVariables.LeftRight_lambda:
            for v in range(self.n_nodes):
                self.y_var[v] = self.model.addVar(
                    lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y"+str(v)+"_t"+str(self.id))

            # Tree branching integer variables
            self.tree_branching_vars = dict()
            for depth in range(self.tree.max_depth):
                self.tree_branching_vars[depth] = self.model.addVar(
                    vtype=GRB.BINARY, name="lambda"+str(depth)+"_t"+str(self.id))

            # Path and branching constraints
            self.root_constr_y = self.model.addConstr(
                self.y_var[0] == 1, "root_constr_y"+"_t"+str(self.id))
            self.flow_constr = dict()
            self.branch_constr_left = dict()
            self.branch_constr_right = dict()
            for v in range(self.n_nodes):
                if not self.is_leaves[v]:
                    self.flow_constr[v] = self.model.addConstr(
                        self.y_var[v] == self.y_var[self.tree.children_left[v]] + self.y_var[self.tree.children_right[v]], "flow_" + str(v)+"_t"+str(self.id))
                    self.branch_constr_left[v] = self.model.addConstr(
                        self.y_var[self.tree.children_left[v]] <= self.tree_branching_vars[self.node_depth[v]], "branch_left_v" + str(v)+"_t"+str(self.id))
                    self.branch_constr_right[v] = self.model.addConstr(
                        self.y_var[self.tree.children_right[v]] <= 1 - self.tree_branching_vars[self.node_depth[v]], "branch_right_v" + str(v)+"_t"+str(self.id))
        elif self.binaryDecisionVariables == BinaryDecisionVariables.PathFlow_y:
            for v in range(self.n_nodes):
                self.y_var[v] = self.model.addVar(
                    vtype=GRB.BINARY, name="y"+str(v)+"_t"+str(self.id))
            self.root_constr_y = self.model.addConstr(
                self.y_var[0] == 1, "root_constr_y"+"_t"+str(self.id))
            self.flow_constr = dict()
            for v in range(self.n_nodes):
                if not self.is_leaves[v]:
                    self.flow_constr[v] = self.model.addConstr(
                        self.y_var[v] == self.y_var[self.tree.children_left[v]] + self.y_var[self.tree.children_right[v]], "flow_" + str(v)+"_t"+str(self.id))
        else:
            print("Error, unknown binary decision variables")

    def addContinuousVariablesConsistencyConstraints(self):
        if self.constraintsType == TreeConstraintsType.ExtendedFormulation:
            self.addTreeExtendedFormulationVariablesAndConstraints()
        elif self.constraintsType == TreeConstraintsType.BigM:
            self.addTreeBigMConstraints()
        elif self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes:
            pass
        else:
            print("unknown constraints type")

    def addBinaryVariablesConsistencyConstraints(self):
        self.leftBinaryVariablesConsistencyConstraints = dict()
        self.rightBinaryVariablesConsistencyConstraints = dict()
        for v in range(self.n_nodes):
            if not self.is_leaves[v]:
                f = self.tree.feature[v]
                if self.featuresType[f] == FeatureType.Binary:
                    assert self.tree.threshold[v] > 0
                    assert self.tree.threshold[v] < 1
                    self.leftBinaryVariablesConsistencyConstraints[v] = self.model.addConstr(
                        self.x_var_sol[f]
                        + self.y_var[self.tree.children_left[v]] <= 1,
                        "leftBinaryVariablesConsistencyConstraints_t"
                        + str(self.id) + "_v" + str(v)
                    )
                    self.rightBinaryVariablesConsistencyConstraints[v] = self.model.addConstr(
                        self.x_var_sol[f] >= self.y_var[self.tree.children_right[v]],
                        "rightBinaryVariablesConsistencyConstraints_t"
                        + str(self.id) + "_v" + str(v)
                    )

    def addTreeVariablesAndConstraintsToMilp(self):
        self.addBranchingAndDecisionPathVariablesAndConstraints()
        self.addContinuousVariablesConsistencyConstraints()
        self.addBinaryVariablesConsistencyConstraints()
