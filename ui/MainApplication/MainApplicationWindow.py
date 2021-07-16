# Author: Moises Henrique Pereira
# this class instantiates the controller responsible by the entire counterfactual interface

from PyQt5 import QtWidgets

from CounterfactualInterface.CounterfactualInterfaceController import CounterfactualInterfaceController
# from Dash.DashView import DashView

class MainApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApplicationWindow, self).__init__()

        # setting the minimum size
        width, height = 720, 380
        self.setMinimumSize(width, height)

        self.setWindowTitle('OceanUI')

        self.__counterfactualInterfaceController = CounterfactualInterfaceController()
        self.__counterfactualInterfaceController.view.show()
        self.setCentralWidget(self.__counterfactualInterfaceController.view)

        self.showMaximized()

        # dashView = DashView()
        # self.setCentralWidget(dashView) 