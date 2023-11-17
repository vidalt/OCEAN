import gurobipy as gp
from gurobipy import GRB
# Import OCEAN utility functions and types
from src.CounterFactualParameters import FeatureActionnability
from src.CounterFactualParameters import FeatureType
from src.CounterFactualParameters import eps


class CounterfactualMilp:
    """ CounterfactualMilp:
    Main Class for optimization-based counterfactual explanations.
    Its child classes are adapted for explaining classification
    or decision.

    Provide methods related to input sample x_0 and feature
    types and actionability constraints.
    """

    def __init__(self, sample, objectiveNorm=2, verbose=False,
                 featuresType=False, featuresPossibleValues=False,
                 featuresActionnability=False, oneHotEncoding=False):
        self.verbose = verbose
        self.x0 = sample
        self.objectiveNorm = objectiveNorm
        self.nFeatures = len(self.x0[0])
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
            if featuresType:
                assert(len(self.categoricalNonOneHotFeatures) == 0)
        else:
            self.oneHotEncoding = dict()
        # Create Gurobi environment and specify parameters
        env = gp.Env()
        if not self.verbose:
            env.setParam('OutputFlag', 0)
            env.start()
        # Initialize gurobi model
        self.model = gp.Model("CounterFactualMilp", env=env)

    # ----------- Private methods -----------
    def __read_feature_types(self, featuresType):
        """
        Read feature types:
         - Default value (if input is False): FeatureType.Numeric.
        """
        features = range(self.nFeatures)
        if not featuresType:
            self.featuresType = [
                FeatureType.Numeric for f in features]
        else:
            # Read types of features
            self.featuresType = featuresType
            self.binaryFeatures = [
                f for f in features if featuresType[f] == FeatureType.Binary]
            # Restrict possible values for binary features
            for f in self.binaryFeatures:
                self.featuresPossibleValues[f] = [0, 1]
            self.continuousFeatures = [
                f for f in features if featuresType[f] == FeatureType.Numeric]
            self.discreteFeatures = [
                f for f in features if featuresType[f] == FeatureType.Discrete]
            self.categoricalNonOneHotFeatures = [
                f for f in features
                if featuresType[f] == FeatureType.CategoricalNonOneHot]
            # - discreteAndCategoricalNonOneHotFeatures -
            self.discreteAndCategoricalNonOneHotFeatures = [
                f for f in features
                if featuresType[f] in [FeatureType.CategoricalNonOneHot,
                                       FeatureType.Discrete]]
            # Specify possible values
            for f in self.discreteAndCategoricalNonOneHotFeatures:
                assert len(self.featuresPossibleValues[f]) > 1
                self.featuresPossibleValues[f] = sorted(
                    self.featuresPossibleValues[f])
                # Check that all possible values are sufficiently different
                for v in range(len(self.featuresPossibleValues[f]) - 1):
                    margin = self.featuresPossibleValues[f][v] + eps
                    assert self.featuresPossibleValues[f][v + 1] >= margin

    def __initObjectiveOfFeature(self, f):
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
            elif self.featuresType[f] in [FeatureType.Discrete,
                                          FeatureType.CategoricalNonOneHot]:
                self.x_var_sol[f] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x_f"+str(f))
                self.discreteFeaturesLevel_var[f] = dict()
                self.discreteFeaturesLevelLinearOrderConstr[f] = dict()
                linearCombination = gp.LinExpr(
                    self.featuresPossibleValues[f][0])
                for i in range(1, len(self.featuresPossibleValues[f])):
                    self.discreteFeaturesLevel_var[f][i] = self.model.addVar(
                        lb=0, ub=1, vtype=GRB.BINARY,
                        name="level_f"+str(f)+"_l"+str(i))
                    if i > 1:
                        self.discreteFeaturesLevelLinearOrderConstr[f][i] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][i]
                            <= self.discreteFeaturesLevel_var[f][i-1],
                            "discreteFeaturesLevelLinearOrderConstr_f"
                            + str(f) + "_l" + str(i))
                    linearCombination += self.discreteFeaturesLevel_var[f][i] * (
                        self.featuresPossibleValues[f][i] - self.featuresPossibleValues[f][i-1])
                self.discreteFeaturesLevelLinearCombinationConstr[f] = self.model.addConstr(
                    self.x_var_sol[f] == linearCombination,
                    name="x_f"+str(f) + "_discreteLinearCombination")
            else:
                raise ValueError(
                    'Incorrect feature type in solution initialization.')

    # ----------- Public methods -----------
    def addActionnabilityConstraints(self):
        self.actionnabilityConstraints = dict()
        for f in range(self.nFeatures):
            isOnlyIncreasing = (self.featuresActionnability[f]
                                == FeatureActionnability.Increasing)
            if isOnlyIncreasing:
                if self.featuresType[f] not in [FeatureType.Numeric,
                                                FeatureType.Discrete]:
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
    
    def addBoundingBoxConstraints(self, boundingBox):
        """
        Add constraints to restrict the counterfactual solution to
        the bounding box specified by the input parameter.
        
        :param boundingBox: list of tuples (min, max) for each feature.
        if min or max is None, the corresponding constraint is not added.
        """
        self.BoundingBoxConstraints = dict()
        if boundingBox is not None:
            for f in range(self.nFeatures):
                if self.featuresType[f] == FeatureType.Numeric:
                    if boundingBox[f][0] is not None:    
                        self.BoundingBoxConstraints[f] = self.model.addConstr(
                        self.x_var_sol[f] >= boundingBox[f][0],
                        "boundingBoxMin_f" + str(f))
                    if boundingBox[f][1] is not None:
                        self.BoundingBoxConstraints[f] = self.model.addConstr(
                            self.x_var_sol[f] <= boundingBox[f][1],
                            "boundingBoxMax_f" + str(f))

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
            print('Objective norm is: ', self.objectiveNorm)
            raise ValueError('Object norm should be in [0, 1, 2].')

    def initObjective(self):
        self.initObjectiveStructures()
        for f in range(self.nFeatures):
            self.__initObjectiveOfFeature(f)
        self.model.setObjective(self.obj, GRB.MINIMIZE)
