# Import OCEAN utility functions and types
from src.CounterfactualMilp import CounterfactualMilp


class ClassifierCounterFactualMilp(CounterfactualMilp):
    def __init__(self, classifier, sample, outputDesired,
                 objectiveNorm=2,
                 verbose=False,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 ):
        CounterfactualMilp.__init__(self, sample,
                                    objectiveNorm, verbose, featuresType,
                                    featuresPossibleValues,
                                    featuresActionnability, oneHotEncoding)
        # Store classification specific objects
        self.outputDesired = outputDesired
        self.clf = classifier
        # Check that initial observation does not already have the target class
        if self.clf.predict(self.x0)[0] == outputDesired:
            print("Warning, ouput of sampled is already", outputDesired)
            print("It does not make sense to seek a counterfactual")
