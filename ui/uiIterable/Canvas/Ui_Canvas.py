# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Moises\Documents\GitHub\OCEAN\ui\Canvas\Canvas.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Canvas(object):
    def setupUi(self, Canvas):
        Canvas.setObjectName("Canvas")
        Canvas.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Canvas)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widgetCanvas = MatplotLibCanvas(Canvas)
        self.widgetCanvas.setObjectName("widgetCanvas")
        self.verticalLayout.addWidget(self.widgetCanvas)

        self.retranslateUi(Canvas)
        QtCore.QMetaObject.connectSlotsByName(Canvas)

    def retranslateUi(self, Canvas):
        _translate = QtCore.QCoreApplication.translate
        Canvas.setWindowTitle(_translate("Canvas", "Form"))
from Canvas.MatplotLibCanvas import MatplotLibCanvas