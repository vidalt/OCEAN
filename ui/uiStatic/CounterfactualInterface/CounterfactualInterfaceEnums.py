# Author: Moises Henrique Pereira
# this class is used to agroup some values that are repeated along the code

from enum import Enum

class CounterfactualInterfaceEnums:

    class SelectDataset(Enum):
        DEFAULT = ''

    class Status(Enum):
        STEP1 = 'Collecting data point...\n'
        STEP2 = 'Calculating original class...\n'
        STEP3 = 'Generating counterfactual...\n'
        STEP4 = 'Calculating counterfactual class...\n'
        STEP5 = 'Showing the counterfactual values'

        ERROR_MSG = 'It was not possible to find a counterfactual'
        ERROR_CUSTOM_MSG = 'With that(those) constraint(s) over the feature {} was not possible to find a counterfactual'
