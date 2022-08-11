# Author: Moises Henrique Pereira

from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
# Import UI functions
from ui.interface.InterfaceViewer import InterfaceViewer


class StaticViewer(InterfaceViewer):
    """
    Import the UI file to access and
    interact with the interface components.
    """
    generateCounterfactual = pyqtSignal()

    def __init__(self):
        super(StaticViewer, self).__init__()
        self.setupStaticUi(self)
        # Connect pyqt signals
        self.comboBoxSelectDataset.currentTextChanged.connect(
            lambda: self.chosenDataset.emit())
        self.pushButtonRandomPoint.clicked.connect(
            lambda: self.randomPoint.emit())
        self.pushButtonCalculateClass.clicked.connect(
            lambda: self.calculateClass.emit())
        self.pushButtonGenerateCounterfactual.clicked.connect(
            lambda: self.generateCounterfactual.emit())

        self.tableWidgetCounterfactualComparison.resizeColumnsToContents()

    def clearView(self):
        """Clean the entire view."""
        self.labelOriginalClass.setText('Original Class:')
        self.labelCounterfactualClass.setText('Counterfactual Class:')

        self.listWidgetSelectedPoint.clear()
        self.__clearTableWidgetCounterfactualComparison()
        self.plainTextEditCounterfactualStatus.clear()

    def clearCounterfactual(self):
        """Clean the counterfactual components."""
        self.labelOriginalClass.setText('Original Class:')
        self.labelCounterfactualClass.setText('Counterfactual Class:')
        self.__clearTableWidgetCounterfactualComparison()
        self.plainTextEditCounterfactualStatus.clear()

    def showCounterfactualClass(self, classValue):
        """Update the counterfactual class component."""
        assert classValue is not None
        self.labelCounterfactualClass.setText(
            'Counterfactual Class: '+str(classValue))

    def showCounterfactualStatus(self, status):
        """Update the counterfactual status component."""
        assert isinstance(status, str)
        self.plainTextEditCounterfactualStatus.insertPlainText(status)

    def __clearTableWidgetCounterfactualComparison(self):
        """Clean the comparison table. """
        rowCount = self.tableWidgetCounterfactualComparison.rowCount()
        for index in reversed(range(rowCount)):
            self.tableWidgetCounterfactualComparison.removeRow(index)

    def showCounterfactualComparison(self, counterfactualComparison):
        """Update the counterfactual comparison component."""
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

            self.tableWidgetCounterfactualComparison.setItem(
                rowPosition, 0, item0)
            self.tableWidgetCounterfactualComparison.setItem(
                rowPosition, 1, item1)
            self.tableWidgetCounterfactualComparison.setItem(
                rowPosition, 2, item2)

        self.tableWidgetCounterfactualComparison.resizeColumnsToContents()

    def setupStaticUi(self, staticInterface):
        staticInterface.setObjectName("staticInterface")
        staticInterface.resize(586, 246)
        self.horizontalLayout = QtWidgets.QHBoxLayout(staticInterface)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widgetContainerDataset = QtWidgets.QWidget(staticInterface)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetContainerDataset.sizePolicy().hasHeightForWidth())
        self.widgetContainerDataset.setSizePolicy(sizePolicy)
        self.widgetContainerDataset.setMinimumSize(QtCore.QSize(200, 0))
        self.widgetContainerDataset.setStyleSheet(
            "QWidget [objectName*=\"widgetContainerDataset\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n"
            "}")
        self.widgetContainerDataset.setObjectName("widgetContainerDataset")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widgetContainerDataset)
        self.verticalLayout.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelSelectDataset = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelSelectDataset.setMinimumSize(QtCore.QSize(0, 25))
        self.labelSelectDataset.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelSelectDataset.setObjectName("labelSelectDataset")
        self.verticalLayout.addWidget(self.labelSelectDataset)
        self.comboBoxSelectDataset = QtWidgets.QComboBox(
            self.widgetContainerDataset)
        self.comboBoxSelectDataset.setMinimumSize(QtCore.QSize(0, 25))
        self.comboBoxSelectDataset.setObjectName("comboBoxSelectDataset")
        self.verticalLayout.addWidget(self.comboBoxSelectDataset)
        self.pushButtonRandomPoint = QtWidgets.QPushButton(
            self.widgetContainerDataset)
        self.pushButtonRandomPoint.setObjectName("pushButtonRandomPoint")
        self.verticalLayout.addWidget(self.pushButtonRandomPoint)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.labelOriginalClass = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelOriginalClass.setMinimumSize(QtCore.QSize(0, 25))
        self.labelOriginalClass.setObjectName("labelOriginalClass")
        self.verticalLayout.addWidget(self.labelOriginalClass)
        self.labelCounterfactualClass = QtWidgets.QLabel(
            self.widgetContainerDataset)
        self.labelCounterfactualClass.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualClass.setObjectName("labelCounterfactualClass")
        self.verticalLayout.addWidget(self.labelCounterfactualClass)
        self.horizontalLayout.addWidget(self.widgetContainerDataset)
        self.widgetContainerSelectedPoint = QtWidgets.QWidget(
            staticInterface)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetContainerSelectedPoint.sizePolicy().hasHeightForWidth())
        self.widgetContainerSelectedPoint.setSizePolicy(sizePolicy)
        self.widgetContainerSelectedPoint.setStyleSheet(
            "QWidget [objectName*=\"widgetContainerSelectedPoint\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n"
            "}")
        self.widgetContainerSelectedPoint.setObjectName(
            "widgetContainerSelectedPoint")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.widgetContainerSelectedPoint)
        self.verticalLayout_2.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widgetContainerLabels = QtWidgets.QWidget(
            self.widgetContainerSelectedPoint)
        self.widgetContainerLabels.setObjectName("widgetContainerLabels")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(
            self.widgetContainerLabels)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.labelSelectedPoint = QtWidgets.QLabel(self.widgetContainerLabels)
        self.labelSelectedPoint.setMinimumSize(QtCore.QSize(0, 25))
        self.labelSelectedPoint.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelSelectedPoint.setObjectName("labelSelectedPoint")
        self.horizontalLayout_3.addWidget(self.labelSelectedPoint)
        self.verticalLayout_2.addWidget(self.widgetContainerLabels)
        self.listWidgetSelectedPoint = QtWidgets.QListWidget(
            self.widgetContainerSelectedPoint)
        self.listWidgetSelectedPoint.setObjectName("listWidgetSelectedPoint")
        self.verticalLayout_2.addWidget(self.listWidgetSelectedPoint)
        self.widgetContainerButtons = QtWidgets.QWidget(
            self.widgetContainerSelectedPoint)
        self.widgetContainerButtons.setObjectName("widgetContainerButtons")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(
            self.widgetContainerButtons)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonCalculateClass = QtWidgets.QPushButton(
            self.widgetContainerButtons)
        self.pushButtonCalculateClass.setObjectName("pushButtonCalculateClass")
        self.horizontalLayout_2.addWidget(self.pushButtonCalculateClass)
        self.pushButtonGenerateCounterfactual = QtWidgets.QPushButton(
            self.widgetContainerButtons)
        self.pushButtonGenerateCounterfactual.setObjectName(
            "pushButtonGenerateCounterfactual")
        self.horizontalLayout_2.addWidget(
            self.pushButtonGenerateCounterfactual)
        self.verticalLayout_2.addWidget(self.widgetContainerButtons)
        self.horizontalLayout.addWidget(self.widgetContainerSelectedPoint)
        self.widgetContainerCounterfactual = QtWidgets.QWidget(staticInterface)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetContainerCounterfactual.sizePolicy().hasHeightForWidth()
            )
        self.widgetContainerCounterfactual.setSizePolicy(sizePolicy)
        self.widgetContainerCounterfactual.setStyleSheet(
            "QWidget [objectName*=\"widgetContainerCounterfactual\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n"
            "}")
        self.widgetContainerCounterfactual.setObjectName(
            "widgetContainerCounterfactual")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(
            self.widgetContainerCounterfactual)
        self.verticalLayout_3.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.labelCounterfactualStatus = QtWidgets.QLabel(
            self.widgetContainerCounterfactual)
        self.labelCounterfactualStatus.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualStatus.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelCounterfactualStatus.setObjectName(
            "labelCounterfactualStatus")
        self.verticalLayout_3.addWidget(self.labelCounterfactualStatus)
        self.plainTextEditCounterfactualStatus = QtWidgets.QPlainTextEdit(
            self.widgetContainerCounterfactual)
        self.plainTextEditCounterfactualStatus.setEnabled(False)
        self.plainTextEditCounterfactualStatus.setMaximumSize(
            QtCore.QSize(16777215, 150))
        self.plainTextEditCounterfactualStatus.setObjectName(
            "plainTextEditCounterfactualStatus")
        self.verticalLayout_3.addWidget(self.plainTextEditCounterfactualStatus)
        self.labelCounterfactualComparison = QtWidgets.QLabel(
            self.widgetContainerCounterfactual)
        self.labelCounterfactualComparison.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualComparison.setMaximumSize(
            QtCore.QSize(16777215, 25))
        self.labelCounterfactualComparison.setObjectName(
            "labelCounterfactualComparison")
        self.verticalLayout_3.addWidget(self.labelCounterfactualComparison)
        self.tableWidgetCounterfactualComparison = QtWidgets.QTableWidget(
            self.widgetContainerCounterfactual)
        self.tableWidgetCounterfactualComparison.setObjectName(
            "tableWidgetCounterfactualComparison")
        self.tableWidgetCounterfactualComparison.setColumnCount(3)
        self.tableWidgetCounterfactualComparison.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderItem(
            0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderItem(
            1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderItem(
            2, item)
        self.tableWidgetCounterfactualComparison.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidgetCounterfactualComparison.horizontalHeader().setMinimumSectionSize(39)
        self.tableWidgetCounterfactualComparison.verticalHeader().setVisible(False)
        self.verticalLayout_3.addWidget(self.tableWidgetCounterfactualComparison)
        self.horizontalLayout.addWidget(self.widgetContainerCounterfactual)

        self.retranslateUi(staticInterface)
        QtCore.QMetaObject.connectSlotsByName(staticInterface)

    def retranslateUi(self, staticInterface):
        _translate = QtCore.QCoreApplication.translate
        staticInterface.setWindowTitle(
            _translate("staticInterface", "Form"))
        self.labelSelectDataset.setText(
            _translate("staticInterface", "Select Dataset"))
        self.pushButtonRandomPoint.setText(
            _translate("staticInterface", "Random Point"))
        self.labelOriginalClass.setText(
            _translate("staticInterface", "Original Class:"))
        self.labelCounterfactualClass.setText(
             _translate("staticInterface", "Counterfactual Class:"))
        self.labelSelectedPoint.setText(
            _translate("staticInterface", "Selected Point"))
        self.pushButtonCalculateClass.setText(
             _translate("staticInterface", "Calculate Class"))
        self.pushButtonGenerateCounterfactual.setText(
            _translate("staticInterface", "Generate Counterfactual"))
        self.labelCounterfactualStatus.setText(
            _translate("staticInterface", "Counterfactual Status"))
        self.labelCounterfactualComparison.setText(
            _translate("staticInterface", "Counterfactual"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(0)
        item.setText(_translate("staticInterface", "Feature"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(1)
        item.setText(_translate("staticInterface", "Selected Value"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(2)
        item.setText(_translate("staticInterface", "Counterfactual Value"))
