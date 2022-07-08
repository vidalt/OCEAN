# Author: Moises Henrique Pereira
# this class handle the functions tests of controller of the component of the numerical features  

import pytest

import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

@pytest.mark.parametrize('slider', [1, 2.9, False, ('t1', 't2'), None])
def test_CIR_setSlider_wrong_parameter(slider):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider)

def test_CIR_setSlider_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
    rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)

def test_CIR_initializeRange_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(None, 1, 0.5, 15)

def test_CIR_initializeRange_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, None, 0.5, 15)

def test_CIR_initializeRange_none_value_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, None, 15)

def test_CIR_initializeRange_none_space_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, None)

def test_CIR_initializeRange_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
    rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
    rangeMin.initializeRange(0, 1, 0.5, 15)

def test_CIR_updateRange_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.updateRange(None, 1, 0.5)

def test_CIR_updateRange_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.updateRange(0, None, 0.5)

def test_CIR_updateRange_none_value_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.updateRange(0, 1, None)

def test_CIR_updateRange_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
    rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
    rangeMin.initializeRange(0, 1, 0.5, 15)
    rangeMin.updateRange(0, 1, 0.3)

def test_CIR_setValue_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
        rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.setValue(None)

def test_CIR_setValue_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = counterfactualInterfaceSlider3RangesView.labelRangeMinimum
    rangeMin.setSlider(counterfactualInterfaceSlider3RangesView.labelSlider)
    rangeMin.initializeRange(0, 1, 0.5, 15)
    rangeMin.setValue(0.3)