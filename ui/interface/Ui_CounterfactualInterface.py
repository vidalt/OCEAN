# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Moises\Documents\GitHub\OCEAN\ui\CounterfactualInterface\CounterfactualInterface.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_CounterfactualInterface(object):
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
