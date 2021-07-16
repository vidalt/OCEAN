# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access and interact with the interface components

from PyQt5.QtWidgets import QWidget, QListWidgetItem, QTableWidgetItem
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor

from .Ui_CounterfactualInterfaceStatic import Ui_CounterfactualInterfaceStatic

class CounterfactualInterfaceViewStatic(QWidget, Ui_CounterfactualInterfaceStatic):

    chosenDataset = pyqtSignal()
    randomPoint = pyqtSignal()
    calculateClass = pyqtSignal()
    generateCounterfactual = pyqtSignal()

    def __init__(self):
        super(CounterfactualInterfaceViewStatic, self).__init__()
        self.setupUi(self)

        self.comboBoxSelectDataset.currentTextChanged.connect(lambda: self.chosenDataset.emit())
        
        self.pushButtonRandomPoint.clicked.connect(lambda: self.randomPoint.emit())

        self.pushButtonCalculateClass.clicked.connect(lambda: self.calculateClass.emit())
        self.pushButtonGenerateCounterfactual.clicked.connect(lambda: self.generateCounterfactual.emit())

        self.tableWidgetCounterfactualComparison.resizeColumnsToContents()


    # this function fill the combobox
    # first cleaning the combobox,
    # and adding the datasets name
    def initializeView(self, datasets):
        assert isinstance(datasets, list)
        for dataset in datasets:
            assert isinstance(dataset, str)
    
        if datasets is not None:
            self.comboBoxSelectDataset.clear()
            self.comboBoxSelectDataset.addItems(datasets)

    # this function is used to clean the entire view
    def clearView(self):
        self.labelOriginalClass.setText('Original Class:')
        self.labelCounterfactualClass.setText('Counterfactual Class:')

        self.listWidgetSelectedPoint.clear()
        self.__clearTableWidgetCounterfactualComparison()
        self.plainTextEditCounterfactualStatus.clear()

    # this function is used to clean just the counterfactual components
    def clearCounterfactual(self):
        self.labelOriginalClass.setText('Original Class:')
        self.labelCounterfactualClass.setText('Counterfactual Class:')

        self.__clearTableWidgetCounterfactualComparison()
        self.plainTextEditCounterfactualStatus.clear()

    # this function is used to get the selected dataset name
    def getChosenDataset(self):
        return self.comboBoxSelectDataset.currentText()

    # this function is used to update the class component
    def showOriginalClass(self, classValue):
        assert classValue is not None

        self.labelOriginalClass.setText('Original Class: '+str(classValue))

    # this function is used to update the counterfactual class component
    def showCounterfactualClass(self, classValue):
        assert classValue is not None

        self.labelCounterfactualClass.setText('Counterfactual Class: '+str(classValue))

    # this function is used to add the features components inside the main view
    def addFeatureWidget(self, feature:'QWidget'):
        assert isinstance(feature, QWidget)

        item = QListWidgetItem(self.listWidgetSelectedPoint)
        item.setSizeHint(feature.size())
        self.listWidgetSelectedPoint.addItem(item)
        self.listWidgetSelectedPoint.setItemWidget(item, feature)

    # this function is used to update the counterfactual status component
    def showCounterfactualStatus(self, status):
        assert isinstance(status, str)

        self.plainTextEditCounterfactualStatus.insertPlainText(status)

    # this function is used to clean just the comparison table 
    def __clearTableWidgetCounterfactualComparison(self):
        rowCount = self.tableWidgetCounterfactualComparison.rowCount()
        for index in reversed(range(rowCount)):
            self.tableWidgetCounterfactualComparison.removeRow(index)

    # this function is used to update the counterfactual comparison component
    def showCounterfactualComparison(self, counterfactualComparison):
        assert isinstance(counterfactualComparison, list)

        self.__clearTableWidgetCounterfactualComparison()
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

            self.tableWidgetCounterfactualComparison.setItem(rowPosition , 0, item0)
            self.tableWidgetCounterfactualComparison.setItem(rowPosition , 1, item1)
            self.tableWidgetCounterfactualComparison.setItem(rowPosition , 2, item2)

        self.tableWidgetCounterfactualComparison.resizeColumnsToContents()
