# Author: Moises Henrique Pereira
# this class handle the functions tests of controller of the component of the categorical features  
 
import pytest

import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CICLC_initializeView_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceComboboxListController = StaticObjects.staticCounterfactualInterfaceComboboxListController()
        counterfactualInterfaceComboboxListController.initializeView(featureName, ['value1', 'value2', 'value3'])

@pytest.mark.parametrize('content', [1, 2.9, 'str', False, ('t1', 't2'), None, [1]])
def test_CICLC_initializeView_wrong_type_content_parameter(content):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceComboboxListController = StaticObjects.staticCounterfactualInterfaceComboboxListController()
        counterfactualInterfaceComboboxListController.initializeView('featureName', content)

def test_CICLC_initializeView_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceComboboxListController = StaticObjects.staticCounterfactualInterfaceComboboxListController()
    counterfactualInterfaceComboboxListController.initializeView('featureName', ['value1', 'value2', 'value3'])

def test_CICLC_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceComboboxListController = StaticObjects.staticCounterfactualInterfaceComboboxListController()
        counterfactualInterfaceComboboxListController.setSelectedValue(None)

def test_CICLC_setSelectedValue_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceComboboxListController = StaticObjects.staticCounterfactualInterfaceComboboxListController()
    counterfactualInterfaceComboboxListController.initializeView('featureName', ['value1', 'value2', 'value3'])
    counterfactualInterfaceComboboxListController.setSelectedValue('value1')
