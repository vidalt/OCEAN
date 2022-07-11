# Author: Moises Henrique Pereira
# this class handle the functions tests of controller of the component of the numerical features  

import pytest

import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

def test_CIS_initializeSlider_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(None, 1, 1)

def test_CIS_initializeSlider_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, None, 1)

def test_CIS_initializeSlider_none_value_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, None)

def test_CIS_initializeSlider_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
    counterfactualInterfaceSlider.initializeSlider(0, 1, 1)

def test_CIS_updateSlider_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.updateSlider(None, 1)

def test_CIS_updateSlider_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.updateSlider(0, None)

def test_CIS_updateSlider_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
    counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
    counterfactualInterfaceSlider.updateSlider(0, 1)

def test_CIS_getValueAtXPosition_none_x_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.getValueAtXPosition(None)

def test_CIS_getValueAtXPosition_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
    counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
    value = counterfactualInterfaceSlider.getValueAtXPosition(50)
    assert isinstance(value, int)

def test_CIS_getPositionFromValue_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.getPositionFromValue(None, 1, 0.5)

def test_CIS_getPositionFromValue_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.getPositionFromValue(0, None, 0.5)

def test_CIS_getPositionFromValue_none_value_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.getPositionFromValue(0, 1, None)

def test_CIS_getPositionFromValue_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
    counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
    value = counterfactualInterfaceSlider.getPositionFromValue(0, 1, 0.5)
    assert isinstance(value, int) or isinstance(value, float)

def test_CIS_getValueFromPos_none_pos_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.getValueFromPos(None)

def test_CIS_getValueFromPos_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
    counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
    value = counterfactualInterfaceSlider.getValueFromPos(50)
    assert isinstance(value, int) or isinstance(value, float)

def test_CIS_setValue_none_pos_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
        counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
        counterfactualInterfaceSlider.setValue(None)

def test_CIS_setValue_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider = StaticObjects.staticCounterfactualInterfaceSlider()
    counterfactualInterfaceSlider.initializeSlider(0, 1, 1)
    counterfactualInterfaceSlider.setValue(0.5)