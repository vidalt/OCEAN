# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Moises\Documents\GitHub\OCEAN\ui\CounterfactualInterface\Slider3Ranges\Slider3RangesSmaller.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Slider3RangesSmaller(object):
    def setupUi(self, Slider3RangesSmaller):
        Slider3RangesSmaller.setObjectName("Slider3RangesSmaller")
        Slider3RangesSmaller.resize(300, 130)
        Slider3RangesSmaller.setMinimumSize(QtCore.QSize(300, 130))
        Slider3RangesSmaller.setMaximumSize(QtCore.QSize(300, 16777215))
        self.gridLayout = QtWidgets.QGridLayout(Slider3RangesSmaller)
        self.gridLayout.setObjectName("gridLayout")
        self.labelFeatureName = QtWidgets.QLabel(Slider3RangesSmaller)
        self.labelFeatureName.setMinimumSize(QtCore.QSize(0, 25))
        self.labelFeatureName.setStyleSheet("QLabel {\n"
"    font-weight: bold;\n"
"}")
        self.labelFeatureName.setText("")
        self.labelFeatureName.setObjectName("labelFeatureName")
        self.gridLayout.addWidget(self.labelFeatureName, 0, 0, 1, 2)
        self.checkBoxActionability = QtWidgets.QCheckBox(Slider3RangesSmaller)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBoxActionability.sizePolicy().hasHeightForWidth())
        self.checkBoxActionability.setSizePolicy(sizePolicy)
        self.checkBoxActionability.setMinimumSize(QtCore.QSize(0, 25))
        self.checkBoxActionability.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBoxActionability.setChecked(True)
        self.checkBoxActionability.setObjectName("checkBoxActionability")
        self.gridLayout.addWidget(self.checkBoxActionability, 0, 2, 1, 2)
        self.widget = QtWidgets.QWidget(Slider3RangesSmaller)
        self.widget.setMinimumSize(QtCore.QSize(280, 50))
        self.widget.setObjectName("widget")
        self.labelRangeMaximumValue = RangeValue(self.widget)
        self.labelRangeMaximumValue.setGeometry(QtCore.QRect(240, 30, 21, 25))
        self.labelRangeMaximumValue.setMinimumSize(QtCore.QSize(0, 25))
        self.labelRangeMaximumValue.setText("")
        self.labelRangeMaximumValue.setObjectName("labelRangeMaximumValue")
        self.labelRangeMinimumValue = RangeValue(self.widget)
        self.labelRangeMinimumValue.setGeometry(QtCore.QRect(0, 30, 21, 25))
        self.labelRangeMinimumValue.setMinimumSize(QtCore.QSize(0, 25))
        self.labelRangeMinimumValue.setText("")
        self.labelRangeMinimumValue.setObjectName("labelRangeMinimumValue")
        self.labelRangeValue = Range(self.widget)
        self.labelRangeValue.setGeometry(QtCore.QRect(120, 0, 10, 25))
        self.labelRangeValue.setMinimumSize(QtCore.QSize(10, 25))
        self.labelRangeValue.setMaximumSize(QtCore.QSize(10, 25))
        self.labelRangeValue.setStyleSheet("QLabel {\n"
"    background-color: #049DD9;\n"
"    border-style:solid;\n"
"    border-width:1px;\n"
"    border-color: #049DD9;\n"
"}")
        self.labelRangeValue.setText("")
        self.labelRangeValue.setObjectName("labelRangeValue")
        self.labelRangeMaximum = Range(self.widget)
        self.labelRangeMaximum.setGeometry(QtCore.QRect(270, 0, 10, 15))
        self.labelRangeMaximum.setMinimumSize(QtCore.QSize(10, 15))
        self.labelRangeMaximum.setMaximumSize(QtCore.QSize(10, 15))
        self.labelRangeMaximum.setStyleSheet("QLabel {\n"
"    background-color: #376b80;\n"
"    border-style:solid;\n"
"    border-width:1px;\n"
"    border-color: #376b80;\n"
"}")
        self.labelRangeMaximum.setText("")
        self.labelRangeMaximum.setObjectName("labelRangeMaximum")
        self.labelRangeValueValue = RangeValue(self.widget)
        self.labelRangeValueValue.setGeometry(QtCore.QRect(110, 30, 21, 25))
        self.labelRangeValueValue.setMinimumSize(QtCore.QSize(0, 25))
        self.labelRangeValueValue.setText("")
        self.labelRangeValueValue.setObjectName("labelRangeValueValue")
        self.labelSlider = Slider(self.widget)
        self.labelSlider.setGeometry(QtCore.QRect(0, 10, 280, 5))
        self.labelSlider.setMinimumSize(QtCore.QSize(280, 5))
        self.labelSlider.setMaximumSize(QtCore.QSize(290, 5))
        self.labelSlider.setStyleSheet("QLabel {\n"
"    border-style:solid;\n"
"    border-width:1px;\n"
"    border-color: #049DD9;\n"
"}")
        self.labelSlider.setText("")
        self.labelSlider.setObjectName("labelSlider")
        self.labelRangeMinimum = Range(self.widget)
        self.labelRangeMinimum.setGeometry(QtCore.QRect(0, 10, 10, 15))
        self.labelRangeMinimum.setMinimumSize(QtCore.QSize(10, 15))
        self.labelRangeMinimum.setMaximumSize(QtCore.QSize(10, 15))
        self.labelRangeMinimum.setStyleSheet("QLabel {\n"
"    background-color: #376b80;\n"
"    border-style:solid;\n"
"    border-width:1px;\n"
"    border-color: #376b80;\n"
"}")
        self.labelRangeMinimum.setText("")
        self.labelRangeMinimum.setObjectName("labelRangeMinimum")
        self.labelRangeMaximumValue.raise_()
        self.labelRangeMinimumValue.raise_()
        self.labelRangeValueValue.raise_()
        self.labelSlider.raise_()
        self.labelRangeMinimum.raise_()
        self.labelRangeMaximum.raise_()
        self.labelRangeValue.raise_()
        self.gridLayout.addWidget(self.widget, 1, 0, 1, 4)
        self.doubleSpinBoxMinimum = QtWidgets.QDoubleSpinBox(Slider3RangesSmaller)
        self.doubleSpinBoxMinimum.setMinimumSize(QtCore.QSize(50, 25))
        self.doubleSpinBoxMinimum.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.doubleSpinBoxMinimum.setDecimals(1)
        self.doubleSpinBoxMinimum.setMaximum(1000000.0)
        self.doubleSpinBoxMinimum.setSingleStep(0.5)
        self.doubleSpinBoxMinimum.setObjectName("doubleSpinBoxMinimum")
        self.gridLayout.addWidget(self.doubleSpinBoxMinimum, 2, 0, 1, 1)
        self.labelAdjustPosition1 = QtWidgets.QLabel(Slider3RangesSmaller)
        self.labelAdjustPosition1.setMinimumSize(QtCore.QSize(0, 25))
        self.labelAdjustPosition1.setStyleSheet("QLabel {\n"
"    font-weight: bold;\n"
"}")
        self.labelAdjustPosition1.setText("")
        self.labelAdjustPosition1.setObjectName("labelAdjustPosition1")
        self.gridLayout.addWidget(self.labelAdjustPosition1, 2, 1, 1, 2)
        self.doubleSpinBoxMaximum = QtWidgets.QDoubleSpinBox(Slider3RangesSmaller)
        self.doubleSpinBoxMaximum.setMinimumSize(QtCore.QSize(50, 25))
        self.doubleSpinBoxMaximum.setDecimals(1)
        self.doubleSpinBoxMaximum.setMaximum(1000000.0)
        self.doubleSpinBoxMaximum.setSingleStep(0.5)
        self.doubleSpinBoxMaximum.setObjectName("doubleSpinBoxMaximum")
        self.gridLayout.addWidget(self.doubleSpinBoxMaximum, 2, 3, 1, 1)

        self.retranslateUi(Slider3RangesSmaller)
        QtCore.QMetaObject.connectSlotsByName(Slider3RangesSmaller)

    def retranslateUi(self, Slider3RangesSmaller):
        _translate = QtCore.QCoreApplication.translate
        Slider3RangesSmaller.setWindowTitle(_translate("Slider3RangesSmaller", "Form"))
        self.checkBoxActionability.setText(_translate("Slider3RangesSmaller", "actionable"))
from .Range import Range
from .RangeValue import RangeValue
from .Slider import Slider
