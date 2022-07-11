# Author: Moises Henrique Pereira

from PyQt5 import QtWidgets
# Import ui functions
from interface.InterfaceController import InterfaceController


class MainApplicationWindow(QtWidgets.QMainWindow):
    """
    Instantiate the controller responsible for the counterfactual interface
    """

    def __init__(self):
        super(MainApplicationWindow, self).__init__()

        # Setting the minimum size
        width, height = 720, 380
        self.setMinimumSize(width, height)

        self.setWindowTitle('OCEAN: Optimal Counterfactual Explanations')

        self.__counterfactualInterfaceController = InterfaceController()
        self.__counterfactualInterfaceController.view.show()
        self.setCentralWidget(self.__counterfactualInterfaceController.view)

        self.showMaximized()
