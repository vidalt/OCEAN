# Author: Moises Henrique Pereira

import numpy as np
from PyQt5.QtCore import pyqtSignal
# Import ui functions
from ui.interface.InterfaceWorker import InterfaceWorker


class IterativeWorker(InterfaceWorker):
    """Handle the counterfactual generation."""
    # Initialize pyqt signals
    finished = pyqtSignal()
    counterfactualDataframe = pyqtSignal(object)
    counterfactualSteps = pyqtSignal(str)
    counterfactualError = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.__controller = controller

    def run(self):
        # Build OCEAN model
        oceanMilp = self.buildMilpModel(self.__controller.model,
                                        self.__controller)
        # oceanMilp = self.add_user_constraints(oceanMilp, self.__controller)
        # ---- Generate counterfactual explanation ----
        oceanMilp.solveModel()
        # Get the counterfactual explanation of the current datapoint
        cfExplanation = oceanMilp.x_sol

        # Check results
        counterfactualNotFound = self.isFeasible(
            cfExplanation, self.__controller.transformedChosenDataPoint)
        if counterfactualNotFound:
            print('!'*75)
            print('ERROR: Could not find a counterfactual explanations.')
            print('The model is infeasible.')
            print('!'*75)
            self.counterfactualError.emit()

        elif cfExplanation is not None:
            # Predict class of counterfactual
            cfExplanationClass, result = self.read_counterfactual_and_class(
                self.__controller, cfExplanation)
            result = np.append(result, cfExplanationClass[0])
            # Success: send the counterfactual
            self.counterfactualDataframe.emit(result)

        self.finished.emit()
