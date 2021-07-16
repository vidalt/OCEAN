# Author: Moises Henrique Pereira
# this class is used to agroup some values that are repeated along the code

from enum import Enum

class CounterfactualEngineEnums:

    class RandomForestClassifierParameters(Enum):
        MAX_LEAF_NODES = 32
        RANDOM_STATE = 1
        N_ESTIMATORS = 20

    class IsolationForestParameters(Enum):
        RANDOM_STATE = 1 
        MAX_SAMPLES = 20
        N_ESTIMATORS = 20