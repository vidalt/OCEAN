# Author: Moises Henrique Pereira

import os
import requests
from PyQt5 import QtWidgets, Qt
# Import UI functions
from ui.interface.InterfaceController import InterfaceController


class MainApplicationWindow(QtWidgets.QMainWindow):
    """
    Instantiate the controller responsible for the counterfactual interface.
    """

    def __init__(self):
        super(MainApplicationWindow, self).__init__()

        # Setting the minimum size
        WIDTH, HEIGHT = 720, 380
        self.setMinimumSize(WIDTH, HEIGHT)

        self.setWindowTitle('OCEAN: Optimal Counterfactual Explanations')

        self.__interfaceController = InterfaceController(
            interfaceType='iterative')
        self.__interfaceController.view.show()
        self.setCentralWidget(self.__interfaceController.view)

        helpAction = QtWidgets.QAction('&About', self)
        # helpAction.setShortcut('Ctrl+Q')
        helpAction.setStatusTip('About')
        helpAction.triggered.connect(self.menuAction)

        menubar = self.menuBar()
        helpMenu = menubar.addMenu('&Help')
        helpMenu.addAction(helpAction)

        self.showMaximized()

    def menuAction(self):
        currentPath = os.getcwd()
        tutorialPath = os.path.join(currentPath, 'tutorial', 'index.html')
        # the browser needs / instead of \
        tutorialPath = tutorialPath.replace('\\', '/')
        url = Qt.QUrl(tutorialPath)
        Qt.QDesktopServices.openUrl(url)

    # this function event is used to kill the flask server
    def closeEvent(self, event):
        requests.post('http://127.0.0.1:8050/shutdown')
        event.accept()
