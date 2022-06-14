# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access the interface components

from PyQt5.QtWidgets import QWidget, QCompleter

from ui.interface.LineEditMinimumMaximum.Ui_LineEditMinimumMaximum import Ui_LineEditMinimumMaximum

class LineEditMinimumMaximumView(QWidget, Ui_LineEditMinimumMaximum):

    def __init__(self, parent=None):
        super(LineEditMinimumMaximumView, self).__init__(parent)
        self.setupUi(self)

        self.lineEditUserValue.textChanged.connect(lambda: self.__updateMinMaxNotActionable())
        self.checkBoxActionability.stateChanged.connect(lambda: self.__actionabilityOptionHandler())

    # this function ensures that when the feature is not actionable,
    # the value inside the min e max will be the user value
    def __updateMinMaxNotActionable(self):
        if not self.checkBoxActionability.isChecked():
            self.lineEditMinimumInputValue.setText(self.lineEditUserValue.text())
            self.lineEditMaximumInputValue.setText(self.lineEditUserValue.text())

    # this function disables the component interactions
    def __actionabilityOptionHandler(self):
        if self.checkBoxActionability.isChecked():
            # self.checkBoxActionability.setText('actionable')
            self.lineEditMinimumInputValue.setEnabled(True)
            self.lineEditMaximumInputValue.setEnabled(True)
        else:
            # self.checkBoxActionability.setText('not actionable')
            self.lineEditMinimumInputValue.setEnabled(False)
            self.lineEditMinimumInputValue.setText(self.lineEditUserValue.text())
            self.lineEditMaximumInputValue.setEnabled(False)
            self.lineEditMaximumInputValue.setText(self.lineEditUserValue.text())

    # this function set the initial values to the component
    def setContent(self, featureName, minimumValue, maximumValue):
        assert isinstance(featureName, str)
        assert minimumValue is not None
        assert maximumValue is not None

        self.labelFeatureName.setText(featureName)
        self.lineEditMinimumValue.setText(str(minimumValue))
        self.lineEditMinimumInputValue.setText(str(minimumValue))
        self.lineEditMaximumValue.setText(str(maximumValue))
        self.lineEditMaximumInputValue.setText(str(maximumValue))

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        content = {'value':self.__getSelectedValue(),
                   'minimumValue':self.__getMinimumInputValue(),
                   'maximumValue':self.__getMaximumInputValue()}

        return content

    def setSelectedValue(self, selectedValue):
        assert selectedValue is not None

        self.lineEditUserValue.setText(str(selectedValue))

    def __getSelectedValue(self):
        return self.lineEditUserValue.text()

    def __getMinimumInputValue(self):
        return self.lineEditMinimumInputValue.text()

    def __getMaximumInputValue(self):
        return self.lineEditMaximumInputValue.text()
