# Author: Moises Henrique Pereira
# this class handle the functions tests of view
# of the component of the binary features

import pytest
import sys
from PyQt5 import QtWidgets
from ui.interface.DoubleRadioButton.DoubleRadioButtonView import (
    DoubleRadioButtonView)


@pytest.mark.parametrize('featureName',
                         [1, 2.9, False, ('t1', 't2'), None])
def test_DRBV_setContent_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        doubleRadioButtonViewer = DoubleRadioButtonView()
        doubleRadioButtonViewer.setContent(featureName, 0, 1)


@pytest.mark.parametrize('value0',
                         [1, 2.9, False, ('t1', 't2'), None])
def test_DRBV_setContent_wrong_type_value0_parameter(value0):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        doubleRadioButtonViewer = DoubleRadioButtonView()
        doubleRadioButtonViewer.setContent('featureName', value0, 'value1')


@pytest.mark.parametrize('value1',
                         [1, 2.9, False, ('t1', 't2'), None])
def test_DRBV_setContent_wrong_type_value1_parameter(value1):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        doubleRadioButtonViewer = DoubleRadioButtonView()
        doubleRadioButtonViewer.setContent('featureName', 'value0', value1)


def test_DRBV_setContent_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    doubleRadioButtonViewer = DoubleRadioButtonView()
    doubleRadioButtonViewer.setContent('featureName', 'value0', 'value1')


@pytest.mark.parametrize('value',
                         [1, 2.9, False, ('t1', 't2'), None])
def test_DRBV_setSelectedValue_wrong_type_value_parameter(value):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        doubleRadioButtonViewer = DoubleRadioButtonView()
        doubleRadioButtonViewer.setSelectedValue(value)


def test_DRBV_setSelectedValue_not_allowed_value_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        doubleRadioButtonViewer = DoubleRadioButtonView()
        doubleRadioButtonViewer.setContent('featureName', 'value0', 'value1')
        doubleRadioButtonViewer.setSelectedValue('value2')


def test_DRBV_setSelectedValue_right_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    doubleRadioButtonViewer = DoubleRadioButtonView()
    doubleRadioButtonViewer.setContent('featureName', 'value0', 'value1')
    doubleRadioButtonViewer.setSelectedValue('value0')


def test_DRBV_getContent():
    _ = QtWidgets.QApplication(sys.argv)
    doubleRadioButtonViewer = DoubleRadioButtonView()
    doubleRadioButtonViewer.setContent('featureName', 'value0', 'value1')
    doubleRadioButtonViewer.setSelectedValue('value0')
    _ = doubleRadioButtonViewer.getContent()
