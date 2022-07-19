# Author: Moises Henrique Pereira
# this class handle the functions tests of view
# of the component of the categorical features

import pytest
import sys
from PyQt5 import QtWidgets
from ui.interface.ComboboxList.ComboboxListView import (
    ComboboxListView)


@pytest.mark.parametrize('featureName', [1, 2.9, False, ('t1', 't2'), None])
def test_CICLV_setContent_wrong_type_featureName_parameter(featureName):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        comboboxListView = ComboboxListView()
        comboboxListView.setContent(featureName,
                                    ['value1', 'value2', 'value3'])


@pytest.mark.parametrize('content',
                         [1, 2.9, 'str', False, ('t1', 't2'), None, [1]])
def test_CICLV_setContent_wrong_content(content):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        comboboxListView = ComboboxListView()
        comboboxListView.setContent('featureName', content)


def test_CICLV_setContent_right_parameters():
    _ = QtWidgets.QApplication(sys.argv)
    comboboxListView = ComboboxListView()
    comboboxListView.setContent('featureName', ['value1', 'value2', 'value3'])


def test_CICLV_setSelectedValue_none_parameter():
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        comboboxListView = ComboboxListView()
        comboboxListView.setSelectedValue(None)


def test_CICLV_setSelectedValue_right_parameter():
    _ = QtWidgets.QApplication(sys.argv)
    comboboxListView = ComboboxListView()
    comboboxListView.setContent('featureName', ['value1', 'value2', 'value3'])
    comboboxListView.setSelectedValue('value1')


def test_CICLV_getContent():
    _ = QtWidgets.QApplication(sys.argv)
    comboboxListView = ComboboxListView()
    comboboxListView.setContent('featureName', ['value1', 'value2', 'value3'])
    comboboxListView.setSelectedValue('value1')
    content = comboboxListView.getContent()
    assert len(content.keys()) == 3
    assert content['value'] == 'value1'
    assert content['allowedValues'] == ['value1', 'value2', 'value3']
    assert content['notAllowedValues'] == []
