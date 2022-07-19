# Author: Moises Henrique Pereira

from PyQt5.QtWidgets import QWidget, QTableWidgetItem
from PyQt5.QtGui import QColor

from .Ui_FinalIteration import Ui_FinalIteration


class FinalIterationView(QWidget, Ui_FinalIteration):
    """
    Import the UI file to access and interact
    with the interface components.
    """

    def __init__(self):
        super(FinalIterationView, self).__init__()
        self.setupUi(self)

        self.__originalScenarioName = None
        self.__counterfactualScenarioName = None

        self.tableWidgetCounterfactualComparison.resizeColumnsToContents()

    # this function sets the initial configurations to the view

    def initializeView(self):
        pass

    # this function sets the scenarios names to the view
    def setViewScenariosName(self, originalScenarioName, counterfactualScenarioName):
        self.__originalScenarioName = originalScenarioName
        self.__counterfactualScenarioName = counterfactualScenarioName

    # this function is used to clean the entire view
    def clearView(self):
        self.labelOriginalClass.setText(self.__originalScenarioName+' Class:')
        self.labelCounterfactualClass.setText(
            self.__counterfactualScenarioName+' Class:')

        self.__clearTableWidgetCounterfactualComparison()

    # this function is used to update the label to comparison
    def updateComparisonLabel(self, text):
        assert text is not None

        self.labelCounterfactualComparison.setText(text)

    # this function is used to update the class component
    def showOriginalClass(self, classValue):
        assert classValue is not None

        self.labelOriginalClass.setText(
            self.__originalScenarioName+' Class: '+str(classValue))

    # this function is used to update the counterfactual class component
    def showCounterfactualClass(self, classValue):
        assert classValue is not None

        self.labelCounterfactualClass.setText(
            self.__counterfactualScenarioName+' Class: '+str(classValue))

    # this function is used to clean just the comparison table
    def __clearTableWidgetCounterfactualComparison(self):
        rowCount = self.tableWidgetCounterfactualComparison.rowCount()
        for index in reversed(range(rowCount)):
            self.tableWidgetCounterfactualComparison.removeRow(index)

    # this function is used to update the counterfactual comparison component
    def showCounterfactualComparison(self, counterfactualComparison):
        assert isinstance(counterfactualComparison, list)

        self.__clearTableWidgetCounterfactualComparison()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderLabels(
            ['Feature', self.__originalScenarioName, self.__counterfactualScenarioName])

        for item in counterfactualComparison:
            assert isinstance(item, list)
            assert len(item) == 3

            rowPosition = self.tableWidgetCounterfactualComparison.rowCount()
            self.tableWidgetCounterfactualComparison.insertRow(rowPosition)

            item0 = QTableWidgetItem(str(item[0]))
            item1 = QTableWidgetItem(str(item[1]))
            item2 = QTableWidgetItem(str(item[2]))

            if item[1] != item[2]:
                item0.setBackground(QColor(4, 157, 217))
                item1.setBackground(QColor(4, 157, 217))
                item2.setBackground(QColor(4, 157, 217))

            self.tableWidgetCounterfactualComparison.setItem(
                rowPosition, 0, item0)
            self.tableWidgetCounterfactualComparison.setItem(
                rowPosition, 1, item1)
            self.tableWidgetCounterfactualComparison.setItem(
                rowPosition, 2, item2)

        self.tableWidgetCounterfactualComparison.resizeColumnsToContents()
        self.tableWidgetCounterfactualComparison.show()
