# Author: Moises Henrique Pereira
# this class handle the interface's functions tests  

# command to run pytest showing print: pytest -s 

import pytest
import os
import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

# the function initializeView expect a list type
# send another type as parameter would arrise an assertionError
@pytest.mark.parametrize('datasets', [1, 2.9, 'str', False, ('t1', 't2'), None])
def test_CIV_initializeView_wrong_type_parameter(datasets):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.initializeView(datasets)

# the function initializeView expect a list type fill with string
# send another content inside the list would arrise an assertionError
@pytest.mark.parametrize('datasets', [[1], [2.9], [False], [('t1', 't2')], [None]])
def test_CIV_initializeView_wrong_array_type_parameters(datasets):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.initializeView(datasets)

# the function initializeView expect a list type fill with string
# a valid string inside a list would not arrise assertionError
def test_CIV_initializeView_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    datasetsPath = os.getcwd()
    datasetsPath = os.path.join(datasetsPath, '..', 'datasets')
    _, _, datasetsName = next(os.walk(datasetsPath))
    counterfactualInterfaceView.initializeView(datasetsName)

# the function getChosenDataset returns the selected dataset
# to do this test lets consider that the first dataset is the selected dataset
def test_CIV_getChosenDataset_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    datasetsPath = os.getcwd()
    datasetsPath = os.path.join(datasetsPath, '..', 'datasets')
    _, _, datasetsName = next(os.walk(datasetsPath))
    counterfactualInterfaceView.initializeView(datasetsName)
    chosenDataset = counterfactualInterfaceView.getChosenDataset()
    assert chosenDataset == datasetsName[0]

# the function showOriginalClass expect a not none parameter
# send none would arrise an assertionError
def test_CIV_showOriginalClass_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.showOriginalClass(None)

# the function showOriginalClass expect a not none parameter
# send a different content would not arrise an assertionError
@pytest.mark.parametrize('originalClass', [0, [0], (0), [(0)]])
def test_CIV_showOriginalClass_right_parameter(originalClass):
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    counterfactualInterfaceView.showOriginalClass(originalClass)

# the function showCounterfactualClass expect a not none parameter
# send none would arrise an assertionError
def test_CIV_showCounterfactualClass_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.showCounterfactualClass(None)

# the function showCounterfactualClass expect a not none parameter
# send a different content would not arrise an assertionError
@pytest.mark.parametrize('originalClass', [1, [1], (1), [(1)]])
def test_CIV_showCounterfactualClass_right_parameter(originalClass):
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    counterfactualInterfaceView.showCounterfactualClass(originalClass)

# the function addFeatureWidget expect a qWidget as parameter
# send another type would arrise an assertionError
@pytest.mark.parametrize('featureWidget', [1, 2.9, 'str', False, ('t1', 't2'), None])
def test_CIV_addFeatureWidget_wrong_parameter(featureWidget):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.addFeatureWidget(featureWidget)

# the function addFeatureWidget expect a qWidget as parameter
# send it would not arrise an assertionError
def test_CIV_addFeatureWidget_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    widget = QtWidgets.QWidget()
    counterfactualInterfaceView.addFeatureWidget(widget)

# the function showCounterfactualStatus expect a string as parameter
# send another type would not arrise an assertionError
@pytest.mark.parametrize('status', [1, 2.9, False, ('t1', 't2'), None])
def test_CIV_showCounterfactualStatus_wrong_parameter(status):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.showCounterfactualStatus(status)

# the function showCounterfactualStatus expect a string as parameter
# send it would not arrise an assertionError
def test_CIV_showCounterfactualStatus_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    counterfactualInterfaceView.showCounterfactualStatus('Status: OK')

# the function showCounterfactualComparison expect a 2d list as parameter
# where each lists inside it need to has 3 elements
# send another type as parameter would arrise an assertionError
@pytest.mark.parametrize('counterfactualComparison', [1, 2.9, False, ('t1', 't2'), None])
def test_CIV_showCounterfactualComparison_wrong_type_parameter(counterfactualComparison):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.showCounterfactualComparison(counterfactualComparison)

# the function showCounterfactualComparison expect a 2d list as parameter
# where each lists inside it need to has 3 elements
# send a list where each list inside it has not 3 items would arrise an assertionError
@pytest.mark.parametrize('counterfactualComparison', [[[1]], [[1, 2.9]], [[1, 2.9, False, ('t1', 't2')]]])
def test_CIV_showCounterfactualComparison_wrong_length_parameter(counterfactualComparison):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
        counterfactualInterfaceView.showCounterfactualComparison(counterfactualComparison)

# the function showCounterfactualComparison expect a 2d list as parameter
# where each lists inside it need to has 3 elements
# send it would not arrise an assertionError
@pytest.mark.parametrize('counterfactualComparison', [[['feature', '0', '1']], [['feature', '0', 1]], [['feature', 0, '1']]])
def test_CIV_showCounterfactualComparison_right_parameter(counterfactualComparison):
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceView = StaticObjects.staticCounterfactualInterfaceView()
    counterfactualInterfaceView.showCounterfactualComparison(counterfactualComparison)
