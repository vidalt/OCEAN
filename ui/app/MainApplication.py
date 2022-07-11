# Author: Moises Henrique Pereira

import sys
from PyQt5 import QtWidgets
# Import UI functions
from ui.app.MainApplicationWindow import MainApplicationWindow


class MainApplication():
    """
    Handle the interface.
    """

    def __init__(self, interfaceType):
        self.interfaceType = interfaceType

    def run(self):
        """
        Instantiate the QApplication to instantiate the MainWindow widget.
        """
        app = QtWidgets.QApplication(sys.argv)
        self.mainApplicationWindow = MainApplicationWindow(self.interfaceType)
        self.mainApplicationWindow.show()
        sys.exit(app.exec_())
