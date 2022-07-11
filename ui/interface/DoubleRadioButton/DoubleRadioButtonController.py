# Author: Moises Henrique Pereira

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
# Load UI functions
from .DoubleRadioButtonView import DoubleRadioButtonView


class DoubleRadioButtonController(QWidget):
    """
    Handles the auxiliar component ComboboxDoubleListView.
    Has a combobox with completion to help the user
    and has to lists to show to user the allowed values
    and the not allowes values given a feature
    """

    outdatedGraph = pyqtSignal()

    def __init__(self, parent=None):
        super(DoubleRadioButtonController, self).__init__()
        self.__view = DoubleRadioButtonView(parent)

    @property
    def view(self):
        return self.__view

    # this function set the initial values to the component
    def initializeView(self, featureName, value0, value1):
        assert isinstance(featureName, str)
        assert isinstance(value0, str)
        assert isinstance(value1, str)

        self.__view.setContent(featureName, value0, value1)
        self.__view.show()

    # this function blocks the user from changing the value
    def disableComponent(self):
        self.__view.disableComponent()

    # this function enables the user from changind the value
    def enableComponent(self):
        self.__view.enableComponent()

    # this function returns the actionability
    def getActionable(self):
        return self.view.getActionable()

    # this function sets the actionability
    def setActionable(self, actionable):
        self.view.setActionable(actionable)

    # this function set a specific value to corresponding widget
    def setSelectedValue(self, selectedValue):
        # assert isinstance(selectedValue, str)
        assert selectedValue is not None

        self.__view.setSelectedValue(str(selectedValue))

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()
