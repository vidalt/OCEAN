# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access the interface components

from PyQt5.QtWidgets import QWidget, QCompleter, QListWidgetItem
from PyQt5.QtCore import Qt, pyqtSignal

from .Ui_ComboboxList import Ui_ComboboxList

class ComboboxListView(QWidget, Ui_ComboboxList):

    resetOptions = pyqtSignal()
    outdatedGraph = pyqtSignal()
    
    def __init__(self, parent=None):
        super(ComboboxListView, self).__init__(parent)
        self.setupUi(self)

        self.comboBoxValues.setEditable(True)

        # self.comboBoxValues.currentTextChanged.connect(lambda: self.__updateAllowedNotActionable())
        self.checkBoxActionability.stateChanged.connect(lambda: self.__actionabilityOptionHandler())
        self.pushButtonResetOptions.clicked.connect(lambda: self.__resetOptionsHandler())
        self.pushButtonResetOptions.clicked.connect(lambda: self.resetOptions.emit())
        self.pushButtonCheckAll.clicked.connect(lambda: self.__checkAllHandler())
        self.pushButtonUncheckAll.clicked.connect(lambda: self.__uncheckAllHandler())

        self.listWidgetAllowedValues.model().dataChanged.connect(lambda: self.outdatedGraph.emit())


    # this function blocks the user from changing the value
    def disableComponent(self):
        self.comboBoxValues.setEnabled(False)

    # this function enables the user from changind the value
    def enableComponent(self):
        self.comboBoxValues.setEnabled(True)

    # this function ensures that when the feature is not actionable, 
    # the allowed value will be the value inside the combobox
    def __updateAllowedNotActionable(self):
        if not self.checkBoxActionability.isChecked():
            for i in range(self.listWidgetAllowedValues.count()):
                if i != 0:
                    if self.listWidgetAllowedValues.item(i).text() != self.comboBoxValues.currentText():
                        self.listWidgetAllowedValues.item(i).setCheckState(Qt.Unchecked)
                    else:
                        self.listWidgetAllowedValues.item(i).setCheckState(Qt.Checked)
    
    # this function disables the component interactions
    def __actionabilityOptionHandler(self):
        if self.checkBoxActionability.isChecked():
            # self.checkBoxActionability.setText('actionable')
            self.listWidgetAllowedValues.setEnabled(True)
            self.pushButtonCheckAll.setEnabled(True)
            self.pushButtonUncheckAll.setEnabled(True)
            # self.__checkAllHandler()
        else:
            # self.checkBoxActionability.setText('not actionable')
            self.listWidgetAllowedValues.setEnabled(False)
            # for i in range(self.listWidgetAllowedValues.count()):
            #     if i != 0:
            #         if self.listWidgetAllowedValues.item(i).text() != self.comboBoxValues.currentText():
            #             self.listWidgetAllowedValues.item(i).setCheckState(Qt.Unchecked)
            #         else:
            #             self.listWidgetAllowedValues.item(i).setCheckState(Qt.Checked)
            self.pushButtonCheckAll.setEnabled(False)
            self.pushButtonUncheckAll.setEnabled(False)

    # this function returns the actionability
    def getActionable(self):
        return True if self.checkBoxActionability.isChecked() else False

    # this function sets the actionability
    def setActionable(self, actionable):
        self.checkBoxActionability.setChecked(actionable)

    def __resetOptionsHandler(self):
        self.resetOptions.emit()
        self.outdatedGraph.emit()

    # this function set all the options as checked
    def __checkAllHandler(self):
        for i in range(self.listWidgetAllowedValues.count()):
            if i != 0:
                self.listWidgetAllowedValues.item(i).setCheckState(Qt.Checked)
        # self.outdatedGraph.emit()

    # this function set all the options as unchecked
    def __uncheckAllHandler(self):
        for i in range(self.listWidgetAllowedValues.count()):
            if i != 0:
                self.listWidgetAllowedValues.item(i).setCheckState(Qt.Unchecked)
        # self.outdatedGraph.emit()

    # this function updates the possible values
    def updatePossibleValues(self, content):
        assert isinstance(content, list)
        for item in content:
            assert isinstance(item, str)

        # save the current value 
        selectedValue = self.comboBoxValues.currentText()

        self.comboBoxValues.clear()
        self.comboBoxValues.addItems(content)
        completer = QCompleter(content)
        completer.setModelSorting(QCompleter.UnsortedModel)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(False)
        self.comboBoxValues.setCompleter(completer)

        self.listWidgetAllowedValues.clear()
        item = QListWidgetItem()
        item.setText('Allowed Values')
        self.listWidgetAllowedValues.addItem(item)
        for value in content:
            item = QListWidgetItem()
            item.setText(value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.listWidgetAllowedValues.addItem(item)

        index = self.comboBoxValues.findText(str(selectedValue), Qt.MatchFixedString)
        if index >= 0:
            self.comboBoxValues.setCurrentIndex(index)

    # this function set the initial values to the component
    def setContent(self, featureName, content):
        assert isinstance(featureName, str)
        assert isinstance(content, list)
        for item in content:
            assert isinstance(item, str)

        self.labelFeatureName.setText(featureName)
        
        self.comboBoxValues.clear()
        self.comboBoxValues.addItems(content)
        completer = QCompleter(content)
        completer.setModelSorting(QCompleter.UnsortedModel)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(False)
        self.comboBoxValues.setCompleter(completer)

        self.listWidgetAllowedValues.clear()
        item = QListWidgetItem()
        item.setText('Allowed Values')
        self.listWidgetAllowedValues.addItem(item)
        for value in content:
            item = QListWidgetItem()
            item.setText(value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.listWidgetAllowedValues.addItem(item)

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        content = {'value':self.__getSelectedValue(), 
                   'allowedValues':self.__getAllowedValues(), 
                   'notAllowedValues':self.__getNotAllowedValues()}

        return content

    def __getSelectedValue(self):
        return self.comboBoxValues.currentText()

    def setSelectedValue(self, selectedValue):
        assert selectedValue is not None

        index = self.comboBoxValues.findText(str(selectedValue), Qt.MatchFixedString)
        if index >= 0:
            self.comboBoxValues.setCurrentIndex(index)

    def __getAllowedValues(self):
        if self.checkBoxActionability.isChecked():
            allowedValues = []
            for i in range(self.listWidgetAllowedValues.count()):
                if i != 0:
                    if self.listWidgetAllowedValues.item(i).checkState() == Qt.Checked:
                        allowedValues.append(self.listWidgetAllowedValues.item(i).text())

            return allowedValues

        else:
            return self.__getSelectedValue()

    def __getNotAllowedValues(self):
        if self.checkBoxActionability.isChecked():
            notAllowedValues = []
            for i in range(self.listWidgetAllowedValues.count()):
                if i != 0:
                    if not self.listWidgetAllowedValues.item(i).checkState() == Qt.Checked:
                        notAllowedValues.append(self.listWidgetAllowedValues.item(i).text())
            
            return notAllowedValues

        else:
            return self.__getSelectedValue()
