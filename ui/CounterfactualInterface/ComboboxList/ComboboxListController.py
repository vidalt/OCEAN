# Author: Moises Henrique Pereira
# this class handles the auxiliar component ComboboxDoubleListView
# this component has a combobox with completion to help the user
# and has to lists to show to user the allowed values and the not allowes values given a feature

from .ComboboxListView import ComboboxListView

class ComboboxListController:

    def __init__(self, parent=None):
        self.__view = ComboboxListView(parent)

    @property
    def view(self):
        return self.__view

    # this function set the initial values to the component
    def initializeView(self, featureName, content):
        assert isinstance(featureName, str)
        assert isinstance(content, list)
        for item in content:
            assert isinstance(item, str)

        self.__view.setContent(featureName, content)
        self.__view.show()

    # this function blocks the user from changing the value
    def disableComponent(self):
        self.__view.disableComponent()

    # this function enables the user from changind the value
    def enableComponent(self):
        self.__view.enableComponent()

    # this function set a specific value to corresponding widget
    def setSelectedValue(self, selectedValue):
        assert selectedValue is not None

        self.__view.setSelectedValue(selectedValue)

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()