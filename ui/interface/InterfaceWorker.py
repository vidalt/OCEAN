"""Parent class of static and iterative workers."""

import numpy as np
from PyQt5.QtCore import QObject
# Load OCEAN functions
from src.CounterFactualParameters import FeatureType
from src.RandomForestCounterFactual import RandomForestCounterFactualMilp


class InterfaceWorker(QObject):
    """Parent class of static and iterative workers."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def buildMilpModel(self, model, controller):
        """Instantiate the optimization model."""
        actionability = model.processedFeaturesActionability
        possibleValues = model.processedFeaturesPossibleValues
        featuresType = model.processedFeaturesType
        # Build OCEAN model
        oceanMilp = RandomForestCounterFactualMilp(
            controller.rfClassifier,
            [controller.transformedChosenDataPoint],
            1-controller.predictedOriginalClass[0],
            isolationForest=controller.isolationForest,
            strictCounterFactual=True,
            verbose=True,
            featuresActionnability=actionability,
            featuresType=featuresType,
            featuresPossibleValues=possibleValues)
        oceanMilp.buildModel()

        return oceanMilp

    def isFeasible(self, cfExplanation, initialPoint):
        return (np.array(cfExplanation)
                == np.array([initialPoint])).all()

    def read_counterfactual_and_class(self, controller, cfExplanation):
        cfExplanationClass = controller.rfClassifier.predict(
            cfExplanation)
        assert (cfExplanationClass == 1
                - controller.predictedOriginalClass[0])
        result = controller.model.invertTransformedDataPoint(
            cfExplanation[0])
        return cfExplanationClass, result

    def add_user_constraints(self, oceanMilp, controller):
        """Add the user constraints to the optimization model."""
        constraintIndex = 0
        model = controller.model
        for feature in controller.model.features:
            if feature != 'Class':
                featureInformations = model.featuresInformations[feature]
                featureConstraints = controller.featuresConstraints[feature]
                featureType = model.featuresType[feature]
                if featureType is FeatureType.Binary:
                    disallowed = featureConstraints['notAllowedValue']
                    if disallowed == featureInformations['value0']:
                        oceanMilp.model.addConstr(
                            oceanMilp.x_var_sol[constraintIndex] == 1,
                            disallowed + ' not allowed')
                    elif disallowed == featureInformations['value1']:
                        oceanMilp.model.addConstr(
                            oceanMilp.x_var_sol[constraintIndex] == 0,
                            disallowed + ' not allowed')

                    constraintIndex += 1

                elif featureType in [FeatureType.Discrete,
                                     FeatureType.Numeric]:
                    minVal = featureConstraints['selectedMinimum']
                    maxVal = featureConstraints['selectedMaximum']
                    oceanMilp.model.addConstr(
                        oceanMilp.x_var_sol[constraintIndex] >= minVal,
                        feature + ' minimum constraint')
                    oceanMilp.model.addConstr(
                        oceanMilp.x_var_sol[constraintIndex] <= maxVal,
                        feature + ' maximum constraint')
                    constraintIndex += 1

                elif featureType is FeatureType.Categorical:
                    notAllowedValues = featureConstraints['notAllowedValues']
                    for value in featureInformations['possibleValues']:
                        if value in notAllowedValues:
                            oceanMilp.model.addConstr(
                                oceanMilp.x_var_sol[constraintIndex] == 0,
                                feature + '_' + value + ' not allowed')
                        constraintIndex += 1
                else:
                    print('Error: unknown feature type:'
                          ' feature ', feature,
                          ' of type ', featureType,
                          ' is unsupported.')
                    raise TypeError
        return oceanMilp
