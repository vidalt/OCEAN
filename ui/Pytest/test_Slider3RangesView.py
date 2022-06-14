# Author: Moises Henrique Pereira
# this class handle the functions tests of view of the component of the numerical features
import pytest
import sys
from PyQt5 import QtWidgets
from ui.mainTest import StaticObjects


@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CIS3RV_setContent_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.setContent(
            featureName, 0, 1, 0.5, 1)


def test_CIS3RV_setContent_none_min_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.setContent(
            'featureName', None, 1, 0.5, 1)


def test_CIS3RV_setContent_none_max_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.setContent(
            'featureName', 0, None, 0.5, 1)


def test_CIS3RV_setContent_none_value_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.setContent(
            'featureName', 0, 1, None, 1)


def test_CIS3RV_setContent_none_decimalPlaces_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.setContent(
            'featureName', 0, 1, 0.5, None)


def test_CIS3RV_setContent_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.setContent(
        'featureName', 0, 1, 0.5, 1)


def test_CIS3RV_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
        counterfactualInterfaceSlider3RangesView.setSelectedValue(None)


def test_CIS3RV_setSelectedValue_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.setContent(
        'featureName', 0, 1, 0.5, 1)
    counterfactualInterfaceSlider3RangesView.setSelectedValue(0.5)


def test_CIS3RV_getContent():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceSlider3RangesView = StaticObjects.staticCounterfactualInterfaceSlider3RangesView()
    counterfactualInterfaceSlider3RangesView.setContent(
        'featureName', 0, 1, 0.5, 1)
    counterfactualInterfaceSlider3RangesView.setSelectedValue(0.5)
    content = counterfactualInterfaceSlider3RangesView.getContent()

    assert len(content.keys()) == 3
    assert content['value'] == 0.5
    assert content['minimumValue'] == 0
    assert content['maximumValue'] == 1
