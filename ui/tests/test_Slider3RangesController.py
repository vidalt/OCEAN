# Author: Moises Henrique Pereira
# this class handle the functions tests
# of controller of the component of the numerical features

import pytest
import sys
from PyQt5 import QtWidgets
from ui.interface.Slider3Ranges.Slider3RangesController import (
    Slider3RangesController)


@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CIS3RC_initializeView_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesController = Slider3RangesController()
        slider3RangesController.initializeView(featureName, 0, 1)


@pytest.mark.parametrize('minValue', ['str', [False], ('t1', 't2'), None])
def test_CIS3RC_initializeView_none_min_parameter(minValue):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesController = Slider3RangesController()
        slider3RangesController.initializeView('featureName', minValue, 1)


@pytest.mark.parametrize('maxValue', ['str', [False], ('t1', 't2'), None])
def test_CIS3RC_initializeView_none_max_parameter(maxValue):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesController = Slider3RangesController()
        slider3RangesController.initializeView('featureName', 0, maxValue)


def test_CIS3RC_initializeView_min_greater_than_max():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesController = Slider3RangesController()
        slider3RangesController.initializeView('featureName', 1, 0)


def test_CIS3RC_initializeView_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesController = Slider3RangesController()
    slider3RangesController.initializeView('featureName', 0, 1)


def test_CIS3RC_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesController = Slider3RangesController()
        slider3RangesController.setSelectedValue(None)


def test_CIS3RC_setSelectedValue_right_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesController = Slider3RangesController()
    slider3RangesController.initializeView('featureName', 0, 1)
    slider3RangesController.setSelectedValue(0.5)
