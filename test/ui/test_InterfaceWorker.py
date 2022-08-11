# Author: Moises Henrique Pereira
# this class handle the worker's functions tests

import sys
from PyQt5 import QtWidgets
from ui.interface.InterfaceWorker import InterfaceWorker


# the init worker expect a CounterfactualInterfaceController as parameter
# send it would not arrise an assertionError
def test_CIV_init_right_type_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    _ = InterfaceWorker()
