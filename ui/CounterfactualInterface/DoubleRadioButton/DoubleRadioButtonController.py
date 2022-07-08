# Author: Moises Henrique Pereira
# this class handles the auxiliar component ComboboxDoubleListView
# this component has a combobox with completion to help the user
# and has to lists to show to user the allowed values and the not allowes values given a feature

from numpy.lib.arraysetops import isin
from .DoubleRadioButtonView import DoubleRadioButtonView

class DoubleRadioButtonController:

    def __init__(self, parent=None):
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

    # this function set a specific value to corresponding widget
    def setSelectedValue(self, selectedValue):
        # assert isinstance(selectedValue, str) 
        assert selectedValue is not None

        self.__view.setSelectedValue(str(selectedValue))

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()