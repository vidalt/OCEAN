# Author: Moises Henrique Pereira
# this class handle the interface and the dash application (if needed)

import sys
from PyQt5 import QtWidgets

from Dash.DashApp import dashApp
import threading

from .MainApplicationWindow import MainApplicationWindow

class MainApplication():
    
    def __init__(self):
        self.MainApplicationWindow = None

        # this line enable to run dash in another thread
        threading.Thread(target=runDash, args=(False, False), daemon=False).start()

    # this function instantiates the QApplication to be possible to instantiates the MainWindow widget
    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        self.mainApplicationWindow = MainApplicationWindow()
        self.mainApplicationWindow.show()
        sys.exit(app.exec_())


# this function run the dash server
def runDash(debug, use_reloader):
    dashApp.run_server(debug=False) 
