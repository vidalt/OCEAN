# Author: Moises Henrique Pereira
# this class handle the functions tests
# of controller of the component of the numerical features

import pytest
import sys
from PyQt5 import QtWidgets
from ui.interface.Slider3Ranges.Slider3RangesView import Slider3RangesView


@pytest.mark.parametrize('slider', [1, 2.9, False, ('t1', 't2'), None])
def test_CIR_setSlider_wrong_parameter(slider):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider)


def test_CIR_setSlider_right_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = slider3RangesView.labelRangeMinimum
    rangeMin.setSlider(slider3RangesView.labelSlider)


def test_CIR_initializeRange_none_min_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(None, 1, 0.5, 15)


def test_CIR_initializeRange_none_max_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, None, 0.5, 15)


def test_CIR_initializeRange_none_value_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, None, 15)


def test_CIR_initializeRange_none_space_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, None)


def test_CIR_initializeRange_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = slider3RangesView.labelRangeMinimum
    rangeMin.setSlider(slider3RangesView.labelSlider)
    rangeMin.initializeRange(0, 1, 0.5, 15)


def test_CIR_updateRange_none_min_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.updateRange(None, 1, 0.5)


def test_CIR_updateRange_none_max_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.updateRange(0, None, 0.5)


def test_CIR_updateRange_none_value_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.updateRange(0, 1, None)


def test_CIR_updateRange_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = slider3RangesView.labelRangeMinimum
    rangeMin.setSlider(slider3RangesView.labelSlider)
    rangeMin.initializeRange(0, 1, 0.5, 15)
    rangeMin.updateRange(0, 1, 0.3)


def test_CIR_setValue_none_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
        rangeMin = slider3RangesView.labelRangeMinimum
        rangeMin.setSlider(slider3RangesView.labelSlider)
        rangeMin.initializeRange(0, 1, 0.5, 15)
        rangeMin.setValue(None)


def test_CIR_setValue_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.labelSlider.initializeSlider(0, 1, 1)
    rangeMin = slider3RangesView.labelRangeMinimum
    rangeMin.setSlider(slider3RangesView.labelSlider)
    rangeMin.initializeRange(0, 1, 0.5, 15)
    rangeMin.setValue(0.3)


@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CIS3RV_setContent_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.setContent(featureName, 0, 1, 0.5, 1)


def test_CIS3RV_setContent_none_min_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.setContent('featureName', None, 1, 0.5, 1)


def test_CIS3RV_setContent_none_max_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.setContent('featureName', 0, None, 0.5, 1)


def test_CIS3RV_setContent_none_value_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.setContent('featureName', 0, 1, None, 1)


def test_CIS3RV_setContent_none_decimalPlaces_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.setContent('featureName', 0, 1, 0.5, None)


def test_CIS3RV_setContent_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.setContent('featureName', 0, 1, 0.5, 1)


def test_CIS3RV_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        slider3RangesView = Slider3RangesView()
        slider3RangesView.setSelectedValue(None)


def test_CIS3RV_setSelectedValue_right_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.setContent('featureName', 0, 1, 0.5, 1)
    slider3RangesView.setSelectedValue(0.5)


def test_CIS3RV_getContent():
    _ = QtWidgets.QApplication(sys.argv)
    slider3RangesView = Slider3RangesView()
    slider3RangesView.setContent('featureName', 0, 1, 0.5, 1)
    slider3RangesView.setSelectedValue(0.5)
    content = slider3RangesView.getContent()
    assert len(content.keys()) == 3
    assert content['value'] == 0.5
    assert content['minimumValue'] == 0
    assert content['maximumValue'] == 1
