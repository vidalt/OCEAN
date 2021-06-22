# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access the interface components

from PyQt5.QtWidgets import QWidget

from .Slider3RangesEnums import Slider3RangesEnums
from .Ui_Slider3Ranges import Ui_Slider3Ranges

class Slider3RangesView(QWidget, Ui_Slider3Ranges):

    def __init__(self, parent=None):
        super(Slider3RangesView, self).__init__(parent)
        self.setupUi(self)

        self.__minValue = 0 
        self.__maxValue = 1 

        self.doubleSpinBoxMinimum.valueChanged.connect(lambda: self.__onUpdateMinimumValue())
        self.doubleSpinBoxMaximum.valueChanged.connect(lambda: self.__onUpdateMaximumValue())

        self.checkBoxActionability.stateChanged.connect(lambda: self.__actionabilityOptionHandler())


    # this function disables the component interactions
    def __actionabilityOptionHandler(self):
        if self.checkBoxActionability.isChecked():
            # self.checkBoxActionability.setText('actionable')
            self.doubleSpinBoxMinimum.setEnabled(True)
            self.doubleSpinBoxMaximum.setEnabled(True)
        else:
            # self.checkBoxActionability.setText('not actionable')
            self.doubleSpinBoxMinimum.setEnabled(False)
            self.doubleSpinBoxMaximum.setEnabled(False)

    def setContent(self, featureName, minValue, maxValue, value, decimalPlaces):
        assert isinstance(featureName, str)
        assert minValue is not None
        assert maxValue is not None
        assert value is not None
        assert decimalPlaces is not None

        self.__minValue = minValue
        self.__maxValue = maxValue

        self.labelFeatureName.setText(featureName)

        self.labelSlider.initializeSlider(minValue, maxValue, decimalPlaces)

        self.labelRangeMinimum.setSlider(self.labelSlider)
        self.labelRangeMinimum.initializeRange(minValue, maxValue, minValue, Slider3RangesEnums.Space.MIN_MAX.value)

        self.labelRangeValue.setSlider(self.labelSlider)
        self.labelRangeValue.initializeRange(minValue, maxValue, value, Slider3RangesEnums.Space.VALUE.value)

        self.labelRangeMaximum.setSlider(self.labelSlider)
        self.labelRangeMaximum.initializeRange(minValue, maxValue, maxValue, Slider3RangesEnums.Space.MIN_MAX.value)

        self.doubleSpinBoxMinimum.setValue(minValue)

        self.doubleSpinBoxMaximum.setValue(maxValue)
    
    def __updateView(self, minValue, maxValue, value):
        assert minValue is not None
        assert maxValue is not None
        assert value is not None

        self.__minValue = minValue
        self.__maxValue = maxValue

        self.labelSlider.updateSlider(minValue, maxValue)
        
        self.labelRangeMinimum.updateRange(minValue, maxValue, minValue)
        self.labelRangeValue.updateRange(minValue, maxValue, value)
        self.labelRangeMaximum.updateRange(minValue, maxValue, maxValue)

    def __getRangeMinimumValue(self):
        return self.labelRangeMinimum.value

    def __getRangeMaximumValue(self):
        return self.labelRangeMaximum.value

    def __getRangeCurrentValue(self):
        return self.labelRangeValue.value

    # this function update the bounds of the slider
    def __onUpdateMinimumValue(self):
        self.__updateView(self.doubleSpinBoxMinimum.value(), self.__maxValue, self.__getRangeCurrentValue())

    # this function update the bounds of the slider
    def __onUpdateMaximumValue(self):
        self.__updateView(self.__minValue, self.doubleSpinBoxMaximum.value(), self.__getRangeCurrentValue())

    # this function is used to set the value to the main range
    def setSelectedValue(self, selectedValue):
        assert selectedValue is not None

        self.labelRangeValue.updateRange(self.__minValue, self.__maxValue, float(selectedValue))

    # this function returns a dictionary with the value of the widgets
    def getContent(self):
        content = None

        if self.checkBoxActionability.isChecked():
            content = {'value':self.__getRangeCurrentValue(), 
                       'minimumValue':self.__getRangeMinimumValue(), 
                       'maximumValue':self.__getRangeMaximumValue()}
        
        else:
            content = {'value':self.__getRangeCurrentValue(), 
                       'minimumValue':self.__getRangeCurrentValue(), 
                       'maximumValue':self.__getRangeCurrentValue()}

        return content
