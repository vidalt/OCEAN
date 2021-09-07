# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Moises\Documents\GitHub\OCEAN\ui\CounterfactualInterface\CounterfactualIterable\Iteration\Iteration.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Iteration(object):
    def setupUi(self, Iteration):
        Iteration.setObjectName("Iteration")
        Iteration.resize(904, 299)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Iteration)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widgetContainerDataset = QtWidgets.QWidget(Iteration)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetContainerDataset.sizePolicy().hasHeightForWidth())
        self.widgetContainerDataset.setSizePolicy(sizePolicy)
        self.widgetContainerDataset.setMinimumSize(QtCore.QSize(340, 0))
        self.widgetContainerDataset.setMaximumSize(QtCore.QSize(340, 16777215))
        self.widgetContainerDataset.setStyleSheet("QWidget [objectName*=\"widgetContainerDataset\"]{\n"
"    background-color: rgb(255, 255, 255);\n"
"\n"
"}")
        self.widgetContainerDataset.setObjectName("widgetContainerDataset")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widgetContainerDataset)
        self.verticalLayout_3.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.labelAdjustConstraints = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelAdjustConstraints.setMinimumSize(QtCore.QSize(0, 25))
        self.labelAdjustConstraints.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelAdjustConstraints.setObjectName("labelAdjustConstraints")
        self.verticalLayout_3.addWidget(self.labelAdjustConstraints)
        self.widgetContainerSelectData = QtWidgets.QWidget(self.widgetContainerDataset)
        self.widgetContainerSelectData.setObjectName("widgetContainerSelectData")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widgetContainerSelectData)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_3.addWidget(self.widgetContainerSelectData)
        self.listWidgetSelectedPoint = QtWidgets.QListWidget(self.widgetContainerDataset)
        self.listWidgetSelectedPoint.setObjectName("listWidgetSelectedPoint")
        self.verticalLayout_3.addWidget(self.listWidgetSelectedPoint)
        self.widgetContainerClass = QtWidgets.QWidget(self.widgetContainerDataset)
        self.widgetContainerClass.setObjectName("widgetContainerClass")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widgetContainerClass)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.labelCurrentClass = QtWidgets.QLabel(self.widgetContainerClass)
        self.labelCurrentClass.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCurrentClass.setObjectName("labelCurrentClass")
        self.horizontalLayout_6.addWidget(self.labelCurrentClass)
        self.verticalLayout_3.addWidget(self.widgetContainerClass)
        self.horizontalLayout.addWidget(self.widgetContainerDataset)
        self.widgetContainerGraph = QtWidgets.QWidget(Iteration)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetContainerGraph.sizePolicy().hasHeightForWidth())
        self.widgetContainerGraph.setSizePolicy(sizePolicy)
        self.widgetContainerGraph.setObjectName("widgetContainerGraph")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widgetContainerGraph)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.labelSelectAxes = QtWidgets.QLabel(self.widgetContainerGraph)
        self.labelSelectAxes.setMinimumSize(QtCore.QSize(0, 25))
        self.labelSelectAxes.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelSelectAxes.setObjectName("labelSelectAxes")
        self.verticalLayout_2.addWidget(self.labelSelectAxes)
        self.widgetContainerAxes = QtWidgets.QWidget(self.widgetContainerGraph)
        self.widgetContainerAxes.setMaximumSize(QtCore.QSize(16777215, 25))
        self.widgetContainerAxes.setObjectName("widgetContainerAxes")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widgetContainerAxes)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.comboBoxAxisX = QtWidgets.QComboBox(self.widgetContainerAxes)
        self.comboBoxAxisX.setMinimumSize(QtCore.QSize(0, 25))
        self.comboBoxAxisX.setObjectName("comboBoxAxisX")
        self.comboBoxAxisX.addItem("")
        self.comboBoxAxisX.addItem("")
        self.comboBoxAxisX.addItem("")
        self.comboBoxAxisX.addItem("")
        self.horizontalLayout_9.addWidget(self.comboBoxAxisX)
        self.comboBoxAxisY = QtWidgets.QComboBox(self.widgetContainerAxes)
        self.comboBoxAxisY.setMinimumSize(QtCore.QSize(0, 25))
        self.comboBoxAxisY.setObjectName("comboBoxAxisY")
        self.comboBoxAxisY.addItem("")
        self.comboBoxAxisY.addItem("")
        self.comboBoxAxisY.addItem("")
        self.comboBoxAxisY.addItem("")
        self.horizontalLayout_9.addWidget(self.comboBoxAxisY)
        self.verticalLayout_2.addWidget(self.widgetContainerAxes)
        self.widgetContainerCanvas = DashView(self.widgetContainerGraph)
        self.widgetContainerCanvas.setObjectName("widgetContainerCanvas")
        self.verticalLayout_2.addWidget(self.widgetContainerCanvas)
        self.horizontalLayout.addWidget(self.widgetContainerGraph)
        self.widgetContainerCounterfactual = QtWidgets.QWidget(Iteration)
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
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widgetContainerCounterfactual)
        self.verticalLayout_4.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.labelCounterfactualComparison = QtWidgets.QLabel(self.widgetContainerCounterfactual)
        self.labelCounterfactualComparison.setMinimumSize(QtCore.QSize(0, 25))
        self.labelCounterfactualComparison.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelCounterfactualComparison.setObjectName("labelCounterfactualComparison")
        self.verticalLayout_4.addWidget(self.labelCounterfactualComparison)
        self.tableWidgetCounterfactualComparison = QtWidgets.QTableWidget(self.widgetContainerCounterfactual)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidgetCounterfactualComparison.sizePolicy().hasHeightForWidth())
        self.tableWidgetCounterfactualComparison.setSizePolicy(sizePolicy)
        self.tableWidgetCounterfactualComparison.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
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
        self.verticalLayout_4.addWidget(self.tableWidgetCounterfactualComparison)
        self.widgetNextFinish = QtWidgets.QWidget(self.widgetContainerCounterfactual)
        self.widgetNextFinish.setObjectName("widgetNextFinish")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widgetNextFinish)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonNext = QtWidgets.QPushButton(self.widgetNextFinish)
        self.pushButtonNext.setMinimumSize(QtCore.QSize(0, 25))
        self.pushButtonNext.setObjectName("pushButtonNext")
        self.horizontalLayout_2.addWidget(self.pushButtonNext)
        self.pushButtonFinish = QtWidgets.QPushButton(self.widgetNextFinish)
        self.pushButtonFinish.setObjectName("pushButtonFinish")
        self.horizontalLayout_2.addWidget(self.pushButtonFinish)
        self.verticalLayout_4.addWidget(self.widgetNextFinish)
        self.horizontalLayout.addWidget(self.widgetContainerCounterfactual)

        self.retranslateUi(Iteration)
        QtCore.QMetaObject.connectSlotsByName(Iteration)

    def retranslateUi(self, Iteration):
        _translate = QtCore.QCoreApplication.translate
        Iteration.setWindowTitle(_translate("Iteration", "Form"))
        self.labelAdjustConstraints.setText(_translate("Iteration", "Adjust Constraints"))
        self.labelCurrentClass.setText(_translate("Iteration", "Current Class:"))
        self.labelSelectAxes.setText(_translate("Iteration", "Select Axes"))
        self.comboBoxAxisX.setItemText(0, _translate("Iteration", "axisX"))
        self.comboBoxAxisX.setItemText(1, _translate("Iteration", "feature1"))
        self.comboBoxAxisX.setItemText(2, _translate("Iteration", "feature2"))
        self.comboBoxAxisX.setItemText(3, _translate("Iteration", "feature3"))
        self.comboBoxAxisY.setItemText(0, _translate("Iteration", "axisY"))
        self.comboBoxAxisY.setItemText(1, _translate("Iteration", "feature1"))
        self.comboBoxAxisY.setItemText(2, _translate("Iteration", "feature2"))
        self.comboBoxAxisY.setItemText(3, _translate("Iteration", "feature3"))
        self.labelCounterfactualComparison.setText(_translate("Iteration", "Inspect"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(0)
        item.setText(_translate("Iteration", "Feature"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(1)
        item.setText(_translate("Iteration", "Selected Value"))
        item = self.tableWidgetCounterfactualComparison.horizontalHeaderItem(2)
        item.setText(_translate("Iteration", "Counterfactual Value"))
        self.pushButtonNext.setText(_translate("Iteration", "Next"))
        self.pushButtonFinish.setText(_translate("Iteration", "Finish"))
from Dash.DashView import DashView
