from src.CounterFactualParameters import *


class ClassifierCounterFactualMilp:
    def __init__(
        self,
        classifier,
        sample,
        outputDesired,
        constraintsType=TreeConstraintsType.ExtendedFormulation,
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

        if featuresPossibleValues:
            self.featuresPossibleValues = featuresPossibleValues
        else:
            self.featuresPossibleValues = [None for i in range(self.nFeatures)]
        if not featuresType:
            self.featuresType = [
                FeatureType.Numeric for f in range(self.nFeatures)]
        else:
            self.featuresType = featuresType
        assert self.nFeatures == len(self.featuresType)
        self.binaryFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Binary]
        for f in self.binaryFeatures:
            self.featuresPossibleValues[f] = [0, 1]
        self.continuousFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Numeric]
        self.discreteFeatures = [i for i in range(
            self.nFeatures) if self.featuresType[i] == FeatureType.Discrete]
        self.categoricalNonOneHotFeatures = [i for i in range(
            self.nFeatures) if self.featuresType[i] == FeatureType.CategoricalNonOneHot]
        self.discreteAndCategoricalNonOneHotFeatures = [i for i in range(
            self.nFeatures) if self.featuresType[i] in [FeatureType.CategoricalNonOneHot, FeatureType.Discrete]]
        for f in self.discreteAndCategoricalNonOneHotFeatures:
            assert len(self.featuresPossibleValues[f]) > 1
            # for v in self.featuresPossibleValues[f]:
            #     assert type(v) in [int, float]
            self.featuresPossibleValues[f] = sorted(
                self.featuresPossibleValues[f])
            for v in range(len(self.featuresPossibleValues[f]) - 1):
                assert self.featuresPossibleValues[f][v
                                                      + 1] >= self.featuresPossibleValues[f][v] + eps

        if featuresActionnability:
            self.featuresActionnability = featuresActionnability
        else:
            self.featuresActionnability = [
                FeatureActionnability.Free for f in range(self.nFeatures)]

        if oneHotEncoding:
            self.oneHotEncoding = oneHotEncoding
        else:
            self.oneHotEncoding = dict()

        self.binaryDecisionVariables = binaryDecisionVariables
        env = gp.Env()
        if not self.verbose:
            env.setParam('OutputFlag', 0)
            env.start()
        self.model = gp.Model("ClassifierCounterFactualMilp", env=env)

    def initSolution(self):
        # Solution
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
                    # self.discreteFeaturesLevel_var[f][l] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,name="level_f"+str(f)+"_l"+str(l))
                    if l > 1:
                        self.discreteFeaturesLevelLinearOrderConstr[f][l] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][l] <= self.discreteFeaturesLevel_var[f][l-1],
                            "discreteFeaturesLevelLinearOrderConstr_f"
                            + str(f) + "_l" + str(l)
                        )
                    linearCombination += self.discreteFeaturesLevel_var[f][l] * (
                        self.featuresPossibleValues[f][l] - self.featuresPossibleValues[f][l-1])
                self.discreteFeaturesLevelLinearCombinationConstr[f] = self.model.addConstr(
                    self.x_var_sol[f] == linearCombination, name="x_f"+str(f) + "_discreteLinearCombination"
                )

    def addActionnabilityConstraints(self):
        self.actionnabilityConstraints = dict()
        for f in range(self.nFeatures):
            if self.featuresActionnability[f] == FeatureActionnability.Increasing:
                if self.featuresType[f] not in [FeatureType.Numeric, FeatureType.Discrete]:
                    print(
                        "Increasing actionnability is avaialable only for numeric and discrete features")
                else:
                    self.actionnabilityConstraints[f] = self.model.addConstr(
                        self.x_var_sol[f] >= self.x0[0][f],
                        "ActionnabilityIncreasing_f" + str(f)
                    )
            elif self.featuresActionnability[f] == FeatureActionnability.Fixed:
                self.actionnabilityConstraints[f] = self.model.addConstr(
                    self.x_var_sol[f] == self.x0[0][f],
                    "ActionnabilityFixed_f" + str(f)
                )

    def addOneHotEncodingConstraints(self):
        self.oneHotEncodingConstraints = dict()
        for featureName in self.oneHotEncoding:
            expr = gp.LinExpr(0.0)
            for f in self.oneHotEncoding[featureName]:
                expr += self.x_var_sol[f]
            self.model.addConstr(
                expr == 1,
                "oneHotEncodingOf_" + featureName
            )

    def initObjectiveStructures(self):
        assert self.objectiveNorm in [0, 1, 2]
        self.absoluteValueVar = dict()
        self.absoluteValueLeftConstr = dict()
        self.absoluteValueRightConstr = dict()
        if self.objectiveNorm == 0 or self.objectiveNorm == 1:
            self.obj = gp.LinExpr(0.0)
        elif self.objectiveNorm == 2:
            self.obj = gp.QuadExpr(0.0)

    def initObjectiveOfFeature(self, f):
        if self.featuresType[f] == FeatureType.CategoricalNonOneHot:
            self.absoluteValueVar[f] = self.model.addVar(
                lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, name="modified_CategoricalNonOneHot_f"+str(f))
            self.absoluteValueLeftConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] * (max(self.featuresPossibleValues[f]) - self.x0[0]
                                            [f]) >= self.x_var_sol[f] - self.x0[0][f], "absoluteValueLeftConstr_f"+str(f)
            )
            self.absoluteValueRightConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] * (min(self.featuresPossibleValues[f]) + self.x0[0][f]) >=
                - self.x_var_sol[f]
                + self.x0[0][f], "absoluteValueRightConstr_f"+str(f)
            )
            self.obj += self.absoluteValueVar[f]
        elif self.objectiveNorm == 0 or self.objectiveNorm == 1:
            if self.objectiveNorm == 0:
                self.absoluteValueVar[f] = self.model.addVar(
                    lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, name="modified_f"+str(f))
            else:
                self.absoluteValueVar[f] = self.model.addVar(
                    lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_value_f"+str(f))
            self.absoluteValueLeftConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] >= self.x_var_sol[f]
                - self.x0[0][f], "absoluteValueLeftConstr_f"+str(f)
            )
            self.absoluteValueRightConstr[f] = self.model.addConstr(
                self.absoluteValueVar[f] >= - self.x_var_sol[f]
                + self.x0[0][f], "absoluteValueRightConstr_f"+str(f)
            )
            self.obj += self.absoluteValueVar[f]
        elif self.objectiveNorm == 2:
            self.obj += (self.x_var_sol[f] - gp.LinExpr(self.x0[0][f])) * \
                         (self.x_var_sol[f] - gp.LinExpr(self.x0[0][f]))
        else:
            print("unknown objective norm:", self.objectiveNorm)

    def initObjective(self):
        self.initObjectiveStructures()
        for f in range(self.nFeatures):
            self.initObjectiveOfFeature(f)
        self.model.setObjective(self.obj, GRB.MINIMIZE)
