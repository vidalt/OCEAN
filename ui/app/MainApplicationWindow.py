# Author: Moises Henrique Pereira

import os
import requests
from PyQt5 import QtWidgets, Qt
# Import UI functions
from ui.static.StaticController import StaticController
from ui.iterative.IterativeController import IterativeController


class MainApplicationWindow(QtWidgets.QMainWindow):
    """
    Instantiate the controller responsible for the counterfactual interface.
    """

    def __init__(self, interfaceType):
        super(MainApplicationWindow, self).__init__()

        self.setWindowTitle('OCEAN: Optimal Counterfactual Explanations')
        # Set the minimum size
        WIDTH, HEIGHT = 720, 380
        self.setMinimumSize(WIDTH, HEIGHT)

        # Instantiate interface controller
        if interfaceType == 'static':
            self.__interfaceController = StaticController()
        elif interfaceType == 'iterative':
            self.__interfaceController = IterativeController()

        self.__interfaceController.interfaceViewer.show()
        self.setCentralWidget(self.__interfaceController.interfaceViewer)

        if interfaceType == 'iterative':
            self.__setupMenuBar()

        self.showMaximized()

    def __setupMenuBar(self):
        """
        Create a Menu bar and add an 'About/Help' option.
        """
        helpAction = QtWidgets.QAction('&About', self)
        helpAction.setStatusTip('About')
        helpAction.triggered.connect(self.__openTutorial)

        menubar = self.menuBar()
        helpMenu = menubar.addMenu('&Help')
        helpMenu.addAction(helpAction)

    def __openTutorial(self):
        """
        Open tutorial file.
        """
        currentPath = os.getcwd()
        tutorialPath = os.path.join(currentPath,
                                    'ui', 'tutorial', 'index.html')
        tutorialPath = tutorialPath.replace('\\', '/')
        url = Qt.QUrl(tutorialPath)
        Qt.QDesktopServices.openUrl(url)

    def closeEvent(self, event):
        """
        Kill the flask server.
        """
        requests.post('http://127.0.0.1:8050/shutdown')
        event.accept()
