# Author: Moises Henrique Pereira
# this class handles the auxiliar component LineEditMinimumMaximumView
# this component handles to get the user value
# and to get the minimum e maximum values given a feature

from .LineEditMinimumMaximumView import LineEditMinimumMaximumView

class LineEditMinimumMaximumController:

    def __init__(self, parent=None):
        self.__view = LineEditMinimumMaximumView(parent)
        

    @property
    def view(self):
        return self.__view

    def initializeView(self, featureName, minimumValue, maximumValue):
        assert isinstance(featureName, str)
        assert minimumValue is not None
        assert maximumValue is not None

        self.__view.setContent(featureName, minimumValue, maximumValue)
        self.__view.show()

    # this function set a specific value to corresponding widget
    def setSelectedValue(self, selectedValue):
        assert selectedValue is not None

        self.__view.setSelectedValue(selectedValue)

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()
