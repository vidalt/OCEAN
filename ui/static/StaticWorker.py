# Author: Moises Henrique Pereira

from PyQt5.QtCore import pyqtSignal
# Import ui functions
from ui.interface.InterfaceWorker import InterfaceWorker
from ui.interface.InterfaceEnums import InterfaceEnums


class StaticWorker(InterfaceWorker):
    """ Run the counterfactual generation."""
    # Initialize pyqt signals
    progress = pyqtSignal(str)
    couterfactualClass = pyqtSignal(str)
    tableCounterfactualValues = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.__controller = controller

    def run(self):
        # Show the progress step
        self.progress.emit(InterfaceEnums.Status.STEP3.value)
        # Build OCEAN model
        oceanMilp = self.buildMilpModel(self.__controller.model,
                                        self.__controller)
        oceanMilp = self.add_user_constraints(oceanMilp, self.__controller)
        oceanMilp.solveModel()
        cfExplanation = oceanMilp.x_sol

        counterfactualNotFound = self.isFeasible(
            cfExplanation, self.__controller.transformedChosenDataPoint)
        if counterfactualNotFound:
            self.progress.emit('Model is infeasible')

        elif cfExplanation is not None:
            cfExplanationClass, result = self.read_counterfactual_and_class(
                self.__controller, cfExplanation)
            counterfactualComparison = []
            for index, feature in enumerate(self.__controller.model.features):
                if feature != 'Class':
                    item1 = self.__controller.chosenDataPoint[index]
                    item2 = result[index]
                    if isinstance(item2, float):
                        item1 = float(item1)
                    counterfactualComparison.append(
                        [feature, str(item1), str(item2)])

            # showing the steps
            self.progress.emit(InterfaceEnums.Status.STEP4.value)
            # showing the counterfactual class
            self.couterfactualClass.emit(str(cfExplanationClass[0]))
            # showing the steps
            self.progress.emit(InterfaceEnums.Status.STEP5.value)
            # showing the comparisson between the selected and the counterfactual values
            self.tableCounterfactualValues.emit(counterfactualComparison)

        else:
            # showing the steps
            self.progress.emit(InterfaceEnums.Status.ERROR_MSG.value)

        self.finished.emit()
