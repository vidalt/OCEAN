# Author: Moises Henrique Pereira
# this class handle the functions tests of controller of the component of the numerical features  
 
import pytest

import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CILEMMC_initializeView_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceLineEditMinimumMaximumController = StaticObjects.staticCounterfactualInterfaceLineEditMinimumMaximumController()
        counterfactualInterfaceLineEditMinimumMaximumController.initializeView(featureName, 0, 1)

def test_CILEMMC_initializeView_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceLineEditMinimumMaximumController = StaticObjects.staticCounterfactualInterfaceLineEditMinimumMaximumController()
        counterfactualInterfaceLineEditMinimumMaximumController.initializeView('featureName', None, 1)

def test_CILEMMC_initializeView_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceLineEditMinimumMaximumController = StaticObjects.staticCounterfactualInterfaceLineEditMinimumMaximumController()
        counterfactualInterfaceLineEditMinimumMaximumController.initializeView('featureName', 0, None)

def test_CILEMMC_initializeView_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceLineEditMinimumMaximumController = StaticObjects.staticCounterfactualInterfaceLineEditMinimumMaximumController()
    counterfactualInterfaceLineEditMinimumMaximumController.initializeView('featureName', 0, 1)

def test_CILEMMC_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceLineEditMinimumMaximumController = StaticObjects.staticCounterfactualInterfaceLineEditMinimumMaximumController()
        counterfactualInterfaceLineEditMinimumMaximumController.setSelectedValue(None)

def test_CILEMMC_setSelectedValue_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceLineEditMinimumMaximumController = StaticObjects.staticCounterfactualInterfaceLineEditMinimumMaximumController()
    counterfactualInterfaceLineEditMinimumMaximumController.setSelectedValue(0.5)