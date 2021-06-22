# Author: Moises Henrique Pereira
# this class handles the auxiliar component Slyder3RangesView
# this component handles to get the user value
# and to get the minimum e maximum values given a feature
 
from .Slider3RangesView import Slider3RangesView

class Slider3RangesController:

    def __init__(self, parent=None):
        self.__view = Slider3RangesView(parent)

    @property
    def view(self):
        return self.__view

    def initializeView(self, featureName, minValue, maxValue, value=None, decimalPlaces=1):
        assert isinstance(featureName, str)
        assert isinstance(minValue, int) or isinstance(minValue, float)
        assert isinstance(maxValue, int) or isinstance(maxValue, float)
        assert minValue < maxValue

        if value == None:
            value = minValue + (maxValue-minValue)/2

        self.__view.setContent(featureName, minValue, maxValue, value, decimalPlaces)
        self.__view.show()

    # this function set a specific value to corresponding widget
    def setSelectedValue(self, selectedValue):
        assert selectedValue is not None

        self.__view.setSelectedValue(selectedValue)

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()