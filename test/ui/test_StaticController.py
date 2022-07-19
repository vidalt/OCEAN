# Author: Moises Henrique Pereira
# this class handle the controller's functions tests

import pytest
import sys
from PyQt5 import QtWidgets
from ui.static.StaticController import StaticController


# the function reportProgress expect a string as parameter
# send another type would arrise an assertionError
@pytest.mark.parametrize('progress', [1, 2.9, False, ('t1', 't2'), [], None])
def test_reportProgress_wrong_type_parameter(progress):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        staticController = StaticController()
        staticController.reportProgress(progress)


# the function reportProgress expect a string as parameter
# send it would not arrise assertionError
def test_reportProgress_right_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    staticController = StaticController()
    staticController.reportProgress('progress: OK')


# the function updateCounterfactualClass expect a not none parameter
# send none would arrise an assertionError
def test_updateCounterfactualClass_none_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        staticController = StaticController()
        staticController.updateCounterfactualClass(None)


# the function updateCounterfactualClass expect a not none parameter
# send a different content would not arrise an assertionError
@pytest.mark.parametrize('counterfactualClass',
                         [1, [1], (1), [(1)]])
def test_updateCounterfactualClass_right_parameter(counterfactualClass):
    _ = QtWidgets.QApplication(sys.argv)
    staticController = StaticController()
    staticController.updateCounterfactualClass(counterfactualClass)


# the function updateCounterfactualTable expect a 2d list as parameter
# where each lists inside it need to has 3 elements
# send another type as parameter would arrise an assertionError
@pytest.mark.parametrize('counterfactualTable',
                         [1, 2.9, False, ('t1', 't2'), None])
def test_updateCounterfactualTable_wrong_type_parameter(counterfactualTable):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        staticController = StaticController()
        staticController.updateCounterfactualTable(counterfactualTable)


# the function updateCounterfactualTable expect a 2d list as parameter
# where each lists inside it need to has 3 elements
# send a list where each list inside it has not 3 items
# would arrise an assertionError
@pytest.mark.parametrize('counterfactualTable',
                         [[[1]], [[1, 2.9]],
                          [[1, 2.9, False, ('t1', 't2')]]])
def test_updateCounterfactualTable_wrong_length_parameter(counterfactualTable):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        staticController = StaticController()
        staticController.updateCounterfactualTable(counterfactualTable)


# the function updateCounterfactualTable expect a 2d list as parameter
# where each lists inside it need to has 3 elements
# send it would not arrise an assertionError
@pytest.mark.parametrize('counterfactualTable',
                         [[['feature', '0', '1']],
                          [['feature', '0', 1]],
                          [['feature', 0, '1']]])
def test_updateCounterfactualTable_right_parameter(counterfactualTable):
    _ = QtWidgets.QApplication(sys.argv)
    staticController = StaticController()
    staticController.updateCounterfactualTable(counterfactualTable)
