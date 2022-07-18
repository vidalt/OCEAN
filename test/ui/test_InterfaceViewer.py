# Author: Moises Henrique Pereira
# this class handle the interface's functions tests

import pytest
import os
import sys
from PyQt5 import QtWidgets
from ui.interface.InterfaceViewer import InterfaceViewer


# the function initializeView expect a list type
# send another type as parameter would arrise an assertionError
@pytest.mark.parametrize('datasets',
                         [1, 2.9, 'str', False, ('t1', 't2'), None])
def test_CIV_initializeView_wrong_type_parameter(datasets):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        interfaceViewer = InterfaceViewer()
        interfaceViewer.initializeView(datasets)


# the function initializeView expect a list type fill with string
# send another content inside the list would arrise an assertionError
@pytest.mark.parametrize('datasets',
                         [[1], [2.9], [False],
                          [('t1', 't2')],
                          [None]])
def test_CIV_initializeView_wrong_array_type_parameters(datasets):
    with pytest.raises(AssertionError):
        _ = QtWidgets.QApplication(sys.argv)
        interfaceViewer = InterfaceViewer()
        interfaceViewer.initializeView(datasets)
