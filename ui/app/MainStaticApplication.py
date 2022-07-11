# Author: Moises Henrique Pereira

import sys
from PyQt5 import QtWidgets
# Import ui functions
from ui.app.MainStaticApplicationWindow import MainApplicationWindow


class MainApplication():
    """
    Handle the interface and the dash application (if needed).
    """

    def __init__(self):
        self.MainApplicationWindow = None

    def run(self):
        """
        Instantiates the QApplication in order to instantiate
        the MainWindow widget.
        """
        app = QtWidgets.QApplication(sys.argv)
        self.mainApplicationWindow = MainApplicationWindow()
        self.mainApplicationWindow.show()
        sys.exit(app.exec_())
