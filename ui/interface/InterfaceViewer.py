# Author: Moises Henrique Pereira
# Import the UI file to access and interact with interface components

from PyQt5.QtWidgets import QWidget, QListWidgetItem
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore, QtWidgets


class InterfaceViewer(QWidget):
    """
    Import the UI file to access and
    interact with the interface components
    """
    # Initialize pyqt signals
    chosenDataset = pyqtSignal()
    randomPoint = pyqtSignal()
    calculateClass = pyqtSignal()

    def __init__(self):
        super(InterfaceViewer, self).__init__()

    def getCanvas(self):
        return self.widgetContainerCanvas

    def initializeView(self, datasets):
        """ Fill the combobox:
        Clean the combobox, and add the datasets name.
        """
        assert isinstance(datasets, list)
        for dataset in datasets:
            assert isinstance(dataset, str)

        if datasets is not None:
            self.comboBoxSelectDataset.clear()
            self.comboBoxSelectDataset.addItems(datasets)

    def getChosenDataset(self):
        """Get the selected dataset name."""
        return self.comboBoxSelectDataset.currentText()

    def showOriginalClass(self, classValue):
        """Show class of initial point."""
        assert classValue is not None
        self.labelOriginalClass.setText('Original Class: ' + str(classValue))

    def addFeatureWidget(self, feature: 'QWidget'):
        """
        Add the features components inside the main view.
        """
        assert isinstance(feature, QWidget)
        item = QListWidgetItem(self.listWidgetSelectedPoint)
        item.setSizeHint(feature.size())
        self.listWidgetSelectedPoint.addItem(item)
        self.listWidgetSelectedPoint.setItemWidget(item, feature)

    def setupUi(self, CounterfactualInterface):
        CounterfactualInterface.setObjectName("CounterfactualInterface")
        CounterfactualInterface.resize(880, 479)
        CounterfactualInterface.setStyleSheet(
            "QWidget [objectName*=\"counterfactualInterface\"]{\n"
            "    background-color: rgb(203, 203, 203);\n}")
        self.horizontalLayout = QtWidgets.QHBoxLayout(CounterfactualInterface)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(CounterfactualInterface)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setObjectName("tabWidget")
        self.horizontalLayout.addWidget(self.tabWidget)

        self.retranslateUi(CounterfactualInterface)
        self.tabWidget.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(CounterfactualInterface)

    def retranslateUi(self, CounterfactualInterface):
        _translate = QtCore.QCoreApplication.translate
        CounterfactualInterface.setWindowTitle(
            _translate("CounterfactualInterface", "Form"))
