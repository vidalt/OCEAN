# Author: Moises Henrique Pereira
# this class handle the worker's functions tests  

import pytest
import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects
from ui.CounterfactualInterface.CounterfactualInferfaceWorker import CounterfactualInferfaceWorker

# the init worker expect a CounterfactualInterfaceController as parameter
# send another type would arrise an assertionError
@pytest.mark.parametrize('controller', [1, 2.9, False, ('t1', 't2'), [], None])
def test_CIV_init_wrong_type_parameter(controller):
    with pytest.raises(AssertionError):
        counterfactualInferfaceWorker = CounterfactualInferfaceWorker(controller)

# the init worker expect a CounterfactualInterfaceController as parameter
# send it would not arrise an assertionError
def test_CIV_init_right_type_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceController = StaticObjects.staticCounterfactualInterfaceController()
    counterfactualInferfaceWorker = CounterfactualInferfaceWorker(counterfactualInterfaceController)
