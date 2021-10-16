# Author: Moises Henrique Pereira
# this class instantiates the controller responsible by the entire counterfactual interface

import requests

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


    # this function event is used to kill the flask server
    def closeEvent(self, event):
        requests.post('http://127.0.0.1:8050/shutdown')
        event.accept()
