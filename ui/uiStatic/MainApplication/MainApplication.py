# Author: Moises Henrique Pereira
# this class handle the interface and the dash application (if needed)

import sys
from PyQt5 import QtWidgets

from .MainApplicationWindow import MainApplicationWindow

class MainApplication():
    
    def __init__(self):
        self.MainApplicationWindow = None

    # this function instantiates the QApplication to be possible to instantiates the MainWindow widget
    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        self.mainApplicationWindow = MainApplicationWindow()
        self.mainApplicationWindow.show()
        sys.exit(app.exec_())
