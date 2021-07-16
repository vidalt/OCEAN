# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Moises\Documents\GitHub\OCEAN\ui\CounterfactualInterface\CounterfactualInterfaceStatic.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CounterfactualInterfaceStatic(object):
    def setupUi(self, CounterfactualInterfaceStatic):
        CounterfactualInterfaceStatic.setObjectName("CounterfactualInterfaceStatic")
        CounterfactualInterfaceStatic.resize(586, 246)
        self.horizontalLayout = QtWidgets.QHBoxLayout(CounterfactualInterfaceStatic)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widgetContainerDataset = QtWidgets.QWidget(CounterfactualInterfaceStatic)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetContainerDataset.sizePolicy().hasHeightForWidth())
        self.widgetContainerDataset.setSizePolicy(sizePolicy)
        self.widgetContainerDataset.setMinimumSize(QtCore.QSize(200, 0))
        self.widgetContainerDataset.setStyleSheet("QWidget [objectName*=\"widgetContainerDataset\"]{\n"
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
        self.comboBoxSelectDataset = QtWidgets.QComboBox(self.widgetContainerDataset)
        self.comboBoxSelectDataset.setMinimumSize(QtCore.QSize(0, 25))
        self.comboBoxSelectDataset.setObjectName("comboBoxSelectDataset")
        self.verticalLayout.addWidget(self.comboBoxSelectDataset)
        self.pushButtonRandomPoint = QtWidgets.QPushButton(self.widgetContainerDataset)
        self.pushButtonRandomPoint.setObjectName("pushButtonRandomPoint")
        self.verticalLayout.addWidget(self.pushButtonRandomPoint)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.labelOriginalClass = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelOriginalClass.setMinimumSize(QtCore.QSize(0, 25))
        self.labelOriginalClass.setObjectName("labelOriginalClass")
        self.verticalLayout.addWidget(self.labelOriginalClass)
        self.labelCounterfactualClass = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelCounterfactualClass.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualClass.setObjectName("labelCounterfactualClass")
        self.verticalLayout.addWidget(self.labelCounterfactualClass)
        self.horizontalLayout.addWidget(self.widgetContainerDataset)
        self.widgetContainerSelectedPoint = QtWidgets.QWidget(CounterfactualInterfaceStatic)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetContainerSelectedPoint.sizePolicy().hasHeightForWidth())
        self.widgetContainerSelectedPoint.setSizePolicy(sizePolicy)
        self.widgetContainerSelectedPoint.setStyleSheet("QWidget [objectName*=\"widgetContainerSelectedPoint\"]{\n"
"    background-color: rgb(255, 255, 255);\n"
"\n"
"}")
        self.widgetContainerSelectedPoint.setObjectName("widgetContainerSelectedPoint")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widgetContainerSelectedPoint)
        self.verticalLayout_2.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widgetContainerLabels = QtWidgets.QWidget(self.widgetContainerSelectedPoint)
        self.widgetContainerLabels.setObjectName("widgetContainerLabels")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widgetContainerLabels)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.labelSelectedPoint = QtWidgets.QLabel(self.widgetContainerLabels)
        self.labelSelectedPoint.setMinimumSize(QtCore.QSize(0, 25))
        self.labelSelectedPoint.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelSelectedPoint.setObjectName("labelSelectedPoint")
        self.horizontalLayout_3.addWidget(self.labelSelectedPoint)
        self.verticalLayout_2.addWidget(self.widgetContainerLabels)
        self.listWidgetSelectedPoint = QtWidgets.QListWidget(self.widgetContainerSelectedPoint)
        self.listWidgetSelectedPoint.setObjectName("listWidgetSelectedPoint")
        self.verticalLayout_2.addWidget(self.listWidgetSelectedPoint)
        self.widgetContainerButtons = QtWidgets.QWidget(self.widgetContainerSelectedPoint)
        self.widgetContainerButtons.setObjectName("widgetContainerButtons")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widgetContainerButtons)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonCalculateClass = QtWidgets.QPushButton(self.widgetContainerButtons)
        self.pushButtonCalculateClass.setObjectName("pushButtonCalculateClass")
        self.horizontalLayout_2.addWidget(self.pushButtonCalculateClass)
        self.pushButtonGenerateCounterfactual = QtWidgets.QPushButton(self.widgetContainerButtons)
        self.pushButtonGenerateCounterfactual.setObjectName("pushButtonGenerateCounterfactual")
        self.horizontalLayout_2.addWidget(self.pushButtonGenerateCounterfactual)
        self.verticalLayout_2.addWidget(self.widgetContainerButtons)
        self.horizontalLayout.addWidget(self.widgetContainerSelectedPoint)
        self.widgetContainerCounterfactual = QtWidgets.QWidget(CounterfactualInterfaceStatic)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetContainerCounterfactual.sizePolicy().hasHeightForWidth())
        self.widgetContainerCounterfactual.setSizePolicy(sizePolicy)
        self.widgetContainerCounterfactual.setStyleSheet("QWidget [objectName*=\"widgetContainerCounterfactual\"]{\n"
"    background-color: rgb(255, 255, 255);\n"
"\n"
"}")
        self.widgetContainerCounterfactual.setObjectName("widgetContainerCounterfactual")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widgetContainerCounterfactual)
        self.verticalLayout_3.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.labelCounterfactualStatus = QtWidgets.QLabel(self.widgetContainerCounterfactual)
        self.labelCounterfactualStatus.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualStatus.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelCounterfactualStatus.setObjectName("labelCounterfactualStatus")
        self.verticalLayout_3.addWidget(self.labelCounterfactualStatus)
        self.plainTextEditCounterfactualStatus = QtWidgets.QPlainTextEdit(self.widgetContainerCounterfactual)
        self.plainTextEditCounterfactualStatus.setEnabled(False)
        self.plainTextEditCounterfactualStatus.setMaximumSize(QtCore.QSize(16777215, 150))
        self.plainTextEditCounterfactualStatus.setObjectName("plainTextEditCounterfactualStatus")
        self.verticalLayout_3.addWidget(self.plainTextEditCounterfactualStatus)
        self.labelCounterfactualComparison = QtWidgets.QLabel(self.widgetContainerCounterfactual)
        self.labelCounterfactualComparison.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualComparison.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelCounterfactualComparison.setObjectName("labelCounterfactualComparison")
        self.verticalLayout_3.addWidget(self.labelCounterfactualComparison)
        self.tableWidgetCounterfactualComparison = QtWidgets.QTableWidget(self.widgetContainerCounterfactual)
        self.tableWidgetCounterfactualComparison.setObjectName("tableWidgetCounterfactualComparison")
        self.tableWidgetCounterfactualComparison.setColumnCount(3)
        self.tableWidgetCounterfactualComparison.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetCounterfactualComparison.setHorizontalHeaderItem(2, item)
        self.tableWidgetCounterfactualComparison.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidgetCounterfactualComparison.horizontalHeader().setMinimumSectionSize(39)
        self.tableWidgetCounterfactualComparison.verticalHeader().setVisible(False)
        self.verticalLayout_3.addWidget(self.tableWidgetCounterfactualComparison)
        self.horizontalLayout.addWidget(self.widgetContainerCounterfactual)

        self.retranslateUi(CounterfactualInterfaceStatic)
        QtCore.QMetaObject.connectSlotsByName(CounterfactualInterfaceStatic)

    def retranslateUi(self, CounterfactualInterfaceStatic):
        _translate = QtCore.QCoreApplication.translate
        CounterfactualInterfaceStatic.setWindowTitle(_translate("CounterfactualInterfaceStatic", "Form"))
        self.labelSelectDataset.setText(_translate("CounterfactualInterfaceStatic", "Select Dataset"))
        self.pushButtonRandomPoint.setText(_translate("CounterfactualInterfaceStatic", "Random Point"))
        self.labelOriginalClass.setText(_translate("CounterfactualInterfaceStatic", "Original Class:"))
        self.labelCounterfactualClass.setText(_translate("CounterfactualInterfaceStatic", "Counterfactual Class:"))
        self.labelSelectedPoint.setText(_translate("CounterfactualInterfaceStatic", "Selected Point"))
        self.pushButtonCalculateClass.setText(_translate("CounterfactualInterfaceStatic", "Calculate Class"))
        self.pushButtonGenerateCounterfactual.setText(_translate("CounterfactualInterfaceStatic", "Generate Counterfactual"))
        self.labelCounterfactualStatus.setText(_translate("CounterfactualInterfaceStatic", "Counterfactual Status"))
        self.labelCounterfactualComparison.setText(_translate("CounterfactualInterfaceStatic", "Counterfactual"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(0)
        item.setText(_translate("CounterfactualInterfaceStatic", "Feature"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(1)
        item.setText(_translate("CounterfactualInterfaceStatic", "Selected Value"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(2)
        item.setText(_translate("CounterfactualInterfaceStatic", "Counterfactual Value"))
