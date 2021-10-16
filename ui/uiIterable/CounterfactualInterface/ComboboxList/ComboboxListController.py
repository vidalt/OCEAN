# Author: Moises Henrique Pereira
# this class handles the auxiliar component ComboboxDoubleListView
# this component has a combobox with completion to help the user
# and has to lists to show to user the allowed values and the not allowes values given a feature

from .ComboboxListView import ComboboxListView

class ComboboxListController:

    def __init__(self, parent=None, allPossibleValues=None):
        self.__view = ComboboxListView(parent)

        self.__allPossibleValues = allPossibleValues

        self.__view.resetOptions.connect(self.__resetOptionsHandler)

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

    # this function updates the possible values
    def __updatePossibleValues(self, content):
        assert isinstance(content, list)
        for item in content:
            assert isinstance(item, str)

        self.view.updatePossibleValues(content)

    # this function reset the options values using the allOptions attribute
    def __resetOptionsHandler(self):
        self.__updatePossibleValues(self.__allPossibleValues)

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        return self.__view.getContent()