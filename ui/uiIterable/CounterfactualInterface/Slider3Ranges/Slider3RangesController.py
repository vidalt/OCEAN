# Author: Moises Henrique Pereira
# this class handles the auxiliar component Slyder3RangesView
# this component handles to get the user value
# and to get the minimum e maximum values given a feature
 
from .Slider3RangesView import Slider3RangesView
from .Slider3RangesViewSmaller import Slider3RangesViewSmaller

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

class Slider3RangesController(QWidget):

    outdatedGraph = pyqtSignal()

    def __init__(self, parent=None, smaller=False):
        super(Slider3RangesController, self).__init__()
        self.__view = None

        if smaller:
            self.__view = Slider3RangesViewSmaller(parent)
        else:
            self.__view = Slider3RangesView(parent)

        self.__view.outdatedGraph.connect(lambda: self.outdatedGraph.emit())
            

    @property
    def view(self):
        return self.__view

    def initializeView(self, featureName, minValue, maxValue, value=None, decimalPlaces=1):
        assert isinstance(featureName, str)
        assert isinstance(minValue, int) or isinstance(minValue, float)
        assert isinstance(maxValue, int) or isinstance(maxValue, float)
        assert minValue <= maxValue

        if value == None:
            value = minValue + (maxValue-minValue)/2

        self.__view.setContent(featureName, minValue, maxValue, value, decimalPlaces)
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
        assert selectedValue is not None

        self.__view.setSelectedValue(selectedValue)

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()