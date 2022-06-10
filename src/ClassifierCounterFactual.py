import gurobipy as gp
from gurobipy import GRB
# Import OCEAN utility functions and types
from src.CounterFactualParameters import BinaryDecisionVariables
from src.CounterFactualParameters import TreeConstraintsType
from src.CounterFactualParameters import FeatureActionnability
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import eps


class ClassifierCounterFactualMilp:
    def __init__(self, classifier, sample, outputDesired,
                 constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                 objectiveNorm=2,
                 verbose=False,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda
                 ):
        self.verbose = verbose
        self.clf = classifier
        self.x0 = sample
        self.outputDesired = outputDesired
        self.objectiveNorm = objectiveNorm
        if self.clf.predict(self.x0)[0] == outputDesired:
            print("Warning, ouput of sampled is already", outputDesired)
            print("It does not make sense to seek a counterfactual")
        self.constraintsType = constraintsType
        self.nFeatures = len(self.clf.feature_importances_)
        assert len(self.x0[0]) == self.nFeatures
        # - Read and format feature information -
        # If there is no featuresPossibleValues provided,
        # the counterfactual is not restricted to discrete values.
        if featuresPossibleValues:
            self.featuresPossibleValues = featuresPossibleValues
        else:
            self.featuresPossibleValues = [None for i in range(self.nFeatures)]
        # Read feature types
        self.__read_feature_types(featuresType)
        assert self.nFeatures == len(self.featuresType)
        # Actionability
        if featuresActionnability:
            self.featuresActionnability = featuresActionnability
        else:
            self.featuresActionnability = [
                FeatureActionnability.Free for f in range(self.nFeatures)]
        # One-hot encoding of categorical features
        if oneHotEncoding:
            self.oneHotEncoding = oneHotEncoding
            assert(len(self.categoricalNonOneHotFeatures) == 0)
        else:
            self.oneHotEncoding = dict()

        self.binaryDecisionVariables = binaryDecisionVariables
        # Create Gurobi environment and specify parameters
        env = gp.Env()
        if not self.verbose:
            env.setParam('OutputFlag', 0)
            env.start()
        self.model = gp.Model("ClassifierCounterFactualMilp", env=env)

    def __read_feature_types(self, featuresType):
        """
        Read feature types:
         - Default value (if input is False): FeatureType.Numeric.
        """
        featuresRange = range(self.nFeatures)
        if not featuresType:
            self.featuresType = [
                FeatureType.Numeric for f in featuresRange]
        else:
            # Read types of features
            self.featuresType = featuresType
            self.binaryFeatures = [
                f for f in featuresRange if self.featuresType[f] == FeatureType.Binary]
            # Restrict possible values for binary features
            for f in self.binaryFeatures:
                self.featuresPossibleValues[f] = [0, 1]
            self.continuousFeatures = [
                f for f in featuresRange if self.featuresType[f] == FeatureType.Numeric]
            self.discreteFeatures = [
                f for f in featuresRange if self.featuresType[f] == FeatureType.Discrete]
            self.categoricalNonOneHotFeatures = [
                f for f in featuresRange if self.featuresType[f] == FeatureType.CategoricalNonOneHot]
            # - discreteAndCategoricalNonOneHotFeatures -
            self.discreteAndCategoricalNonOneHotFeatures = [f for f in featuresRange if self.featuresType[f] in [
                FeatureType.CategoricalNonOneHot, FeatureType.Discrete]]
            # Specify possible values
            for f in self.discreteAndCategoricalNonOneHotFeatures:
                assert len(self.featuresPossibleValues[f]) > 1
                self.featuresPossibleValues[f] = sorted(
                    self.featuresPossibleValues[f])
                # Check that all possible values are sufficiently different
                for v in range(len(self.featuresPossibleValues[f]) - 1):
                    margin = self.featuresPossibleValues[f][v] + eps
                    assert self.featuresPossibleValues[f][v + 1] >= margin

    def initSolution(self):
        """
        Initialize decision variables for the counterfactual solution.
        Specify the acceptable values for all features:
        - Numerica: any value in [0, 1],
        - Binary: only 0 or 1,
        - Discrete: values specified in self.featuresPossibleValues.
        """
        self.x_var_sol = dict()
        self.discreteFeaturesLevel_var = dict()
        self.discreteFeaturesLevelLinearOrderConstr = dict()
        self.discreteFeaturesLevelLinearCombinationConstr = dict()
        for f in range(self.nFeatures):
            if self.featuresType[f] == FeatureType.Binary:
                self.x_var_sol[f] = self.model.addVar(
                    vtype=GRB.BINARY, name="x_f"+str(f))
            elif self.featuresType[f] == FeatureType.Numeric:
                self.x_var_sol[f] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x_f"+str(f))
            elif self.featuresType[f] in [FeatureType.Discrete, FeatureType.CategoricalNonOneHot]:
                self.x_var_sol[f] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x_f"+str(f))
                self.discreteFeaturesLevel_var[f] = dict()
                self.discreteFeaturesLevelLinearOrderConstr[f] = dict()
                linearCombination = gp.LinExpr(
                    self.featuresPossibleValues[f][0])
                for l in range(1, len(self.featuresPossibleValues[f])):
                    self.discreteFeaturesLevel_var[f][l] = self.model.addVar(
                        lb=0, ub=1, vtype=GRB.BINARY, name="level_f"+str(f)+"_l"+str(l))
                    if l > 1:
                        self.discreteFeaturesLevelLinearOrderConstr[f][l] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][l]
                            <= self.discreteFeaturesLevel_var[f][l-1],
                            "discreteFeaturesLevelLinearOrderConstr_f"
                            + str(f) + "_l" + str(l))
                    linearCombination += self.discreteFeaturesLevel_var[f][l] * (
                        self.featuresPossibleValues[f][l] - self.featuresPossibleValues[f][l-1])
                self.discreteFeaturesLevelLinearCombinationConstr[f] = self.model.addConstr(
                    self.x_var_sol[f] == linearCombination,
                    name="x_f"+str(f) + "_discreteLinearCombination")
            else:
                raise ValueError(
                    'Incorrect feature type in solution initialization.')

    def addActionnabilityConstraints(self):
        self.actionnabilityConstraints = dict()
        for f in range(self.nFeatures):
            if self.featuresActionnability[f] == FeatureActionnability.Increasing:
                if self.featuresType[f] not in [FeatureType.Numeric, FeatureType.Discrete]:
                    print("Increasing actionnability is avaialable only for"
                          " numeric and discrete features")
                else:
                    self.actionnabilityConstraints[f] = self.model.addConstr(
                        self.x_var_sol[f] >= self.x0[0][f],
                        "ActionnabilityIncreasing_f" + str(f))
            elif self.featuresActionnability[f] == FeatureActionnability.Fixed:
                self.actionnabilityConstraints[f] = self.model.addConstr(
                    self.x_var_sol[f] == self.x0[0][f],
                    "ActionnabilityFixed_f" + str(f))

    def addOneHotEncodingConstraints(self):
        self.oneHotEncodingConstraints = dict()
        for featureName in self.oneHotEncoding:
            expr = gp.LinExpr(0.0)
            for f in self.oneHotEncoding[featureName]:
                expr += self.x_var_sol[f]
            self.model.addConstr(expr == 1,
                                 "oneHotEncodingOf_" + featureName)

    def initObjectiveStructures(self):
        if self.objectiveNorm == 0:
            self.obj = gp.LinExpr(0.0)
        elif self.objectiveNorm == 1:
            self.obj = gp.LinExpr(0.0)
            self.absoluteValueVar = dict()
            self.absoluteValueLeftConstr = dict()
            self.absoluteValueRightConstr = dict()
        elif self.objectiveNorm == 2:
            self.obj = gp.QuadExpr(0.0)
        else:
            raise ValueError('Object norm should be in [0, 1, 2].')

    def initObjectiveOfFeature(self, f):
        if self.featuresType[f] == FeatureType.CategoricalNonOneHot:
            self.absoluteValueVar[f] = self.model.addVar(
                lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY,
                name="modified_CategoricalNonOneHot_f"+str(f))
            self.absoluteValueLeftConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] * (max(self.featuresPossibleValues[f])
                                            - self.x0[0][f])
                >= self.x_var_sol[f] - self.x0[0][f],
                "absoluteValueLeftConstr_f"+str(f))
            self.absoluteValueRightConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] * (min(self.featuresPossibleValues[f])
                                            + self.x0[0][f])
                >= - self.x_var_sol[f] + self.x0[0][f],
                "absoluteValueRightConstr_f"+str(f))
            self.obj += self.absoluteValueVar[f]
        elif self.objectiveNorm == 0 or self.objectiveNorm == 1:
            if self.objectiveNorm == 0:
                self.absoluteValueVar[f] = self.model.addVar(
                    lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY,
                    name="modified_f"+str(f))
            else:
                self.absoluteValueVar[f] = self.model.addVar(
                    lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                    name="abs_value_f"+str(f))
            self.absoluteValueLeftConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] >= self.x_var_sol[f]
                - self.x0[0][f], "absoluteValueLeftConstr_f"+str(f))
            self.absoluteValueRightConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] >= - self.x_var_sol[f]
                + self.x0[0][f], "absoluteValueRightConstr_f"+str(f))
            self.obj += self.absoluteValueVar[f]
        elif self.objectiveNorm == 2:
            self.obj += (self.x_var_sol[f] - gp.LinExpr(self.x0[0][f])) * \
                         (self.x_var_sol[f] - gp.LinExpr(self.x0[0][f]))
        else:
            raise ValueError('Object norm should be in [0, 1, 2].')

    def initObjective(self):
        self.initObjectiveStructures()
        for f in range(self.nFeatures):
            self.initObjectiveOfFeature(f)
        self.model.setObjective(self.obj, GRB.MINIMIZE)
