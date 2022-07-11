# Author: Moises Henrique Pereira
# this class handle the functions tests of view of the component of the categorical features  
 
import pytest

import sys

from PyQt5 import QtWidgets

from ui.mainTest import StaticObjects

@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CICLV_setContent_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceComboboxListView = StaticObjects.staticCounterfactualInterfaceComboboxListView()
        counterfactualInterfaceComboboxListView.setContent(featureName, ['value1', 'value2', 'value3'])

@pytest.mark.parametrize('content', [1, 2.9, 'str', False, ('t1', 't2'), None, [1]])
def test_CICLV_setContent_wrong_type_featureName_parameter(content):
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceComboboxListView = StaticObjects.staticCounterfactualInterfaceComboboxListView()
        counterfactualInterfaceComboboxListView.setContent('featureName', content)

def test_CICLV_setContent_right_parameters():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceComboboxListView = StaticObjects.staticCounterfactualInterfaceComboboxListView()
    counterfactualInterfaceComboboxListView.setContent('featureName', ['value1', 'value2', 'value3'])

def test_CICLV_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        app = QtWidgets.QApplication(sys.argv)
        counterfactualInterfaceComboboxListView = StaticObjects.staticCounterfactualInterfaceComboboxListView()
        counterfactualInterfaceComboboxListView.setSelectedValue(None)

def test_CICLV_setSelectedValue_right_parameter():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceComboboxListView = StaticObjects.staticCounterfactualInterfaceComboboxListView()
    counterfactualInterfaceComboboxListView.setContent('featureName', ['value1', 'value2', 'value3'])
    counterfactualInterfaceComboboxListView.setSelectedValue('value1')

def test_CICLV_getContent():
    app = QtWidgets.QApplication(sys.argv)
    counterfactualInterfaceComboboxListView = StaticObjects.staticCounterfactualInterfaceComboboxListView()
    counterfactualInterfaceComboboxListView.setContent('featureName', ['value1', 'value2', 'value3'])
    counterfactualInterfaceComboboxListView.setSelectedValue('value1')
    content = counterfactualInterfaceComboboxListView.getContent()
    
    assert len(content.keys()) == 3
    assert content['value'] == 'value1'
    assert content['allowedValues'] == ['value1', 'value2', 'value3']
    assert content['notAllowedValues'] == []
