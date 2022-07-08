# Author: Moises Henrique Pereira
# this class handle the functions tests of controller of the component of the numerical features  
 
import pytest

import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CIS3RC_initializeView_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
        counterfactualInterfaceSlider3RangesControllerController.initializeView(featureName, 0, 1)

@pytest.mark.parametrize('minValue', ['str', [False], ('t1', 't2'), None])
def test_CIS3RC_initializeView_none_min_parameter(minValue):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
        counterfactualInterfaceSlider3RangesControllerController.initializeView('featureName', minValue, 1)

@pytest.mark.parametrize('maxValue', ['str', [False], ('t1', 't2'), None])
def test_CIS3RC_initializeView_none_max_parameter(maxValue):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
        counterfactualInterfaceSlider3RangesControllerController.initializeView('featureName', 0, maxValue)

def test_CIS3RC_initializeView_min_greater_than_max():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
        counterfactualInterfaceSlider3RangesControllerController.initializeView('featureName', 1, 0)

def test_CIS3RC_initializeView_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
    counterfactualInterfaceSlider3RangesControllerController.initializeView('featureName', 0, 1)

def test_CIS3RC_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
        counterfactualInterfaceSlider3RangesControllerController.setSelectedValue(None)

def test_CIS3RC_setSelectedValue_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesControllerController = StaticObjects.staticCounterfactualInterfaceSlider3RangesController()
    counterfactualInterfaceSlider3RangesControllerController.initializeView('featureName', 0, 1)
    counterfactualInterfaceSlider3RangesControllerController.setSelectedValue(0.5)