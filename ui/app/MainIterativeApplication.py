# Author: Moises Henrique Pereira

import sys
import threading
from PyQt5 import QtWidgets
import dash
# Import UI functions
from ui.app.MainIterativeApplicationWindow import MainApplicationWindow


class MainApplication():
    """
    Handle the interface and the dash application
    """

    def __init__(self):
        self.MainApplicationWindow = None

        # Run dash in another thread
        threading.Thread(target=runDash, args=(False, False),
                         daemon=False).start()

    def run(self):
        """
        Instantiate the QApplication to instantiate the MainWindow widget.
        """
        app = QtWidgets.QApplication(sys.argv)
        self.mainApplicationWindow = MainApplicationWindow()
        self.mainApplicationWindow.show()
        sys.exit(app.exec_())


def runDash(debug, use_reloader):
    """
    Run the dash server
    """
    app = dash.Dash(__name__)
    app.run_server(debug=False)
