# Author: Moises Henrique Pereira

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QElapsedTimer
from PyQt5.QtWidgets import QLabel
# Import UI functions
from .RangeValue import RangeValue
from .Slider import Slider


class Range(QLabel):
    """
    Handle the range, range value that are shown,
    and where the value needs to be shown
    """

    outdatedGraph = pyqtSignal()

    def __init__(self, parent=None):
        super(Range, self).__init__(parent)

        self.__signals = RangeSignals()

        self.__isPressed = False

        self.__minX = 0
        self.__maxX = self.parent().width()-self.width()
        self.__minValue = None
        self.__maxValue = None

        self.__rangeValue = RangeValue(self.parent())
        self.__rangeValue.Range = self
        self.__rangeValue.raise_()
        self.__rangeValue.adjustSize()
        self.__rangeValue.show()

        self.__slider = None
        self.__value = None
        self.__space = None

        self.__clickTimer = QElapsedTimer()

        self.setScaledContents(True)
        self.__updateRangeValue()

        self.__enabled = True

    def enableComponent(self):
        """
        Allow the user to change the value.
        """
        self.__enabled = True
        self.setStyleSheet('''QLabel {background-color: #049DD9;
                                      border-style: solid;
                                      border-width: 1px;
                                      border-color: #049DD9;}''')

    def disableComponent(self):
        """
        Prohibit the user from changing the value.
        """
        self.__enabled = False
        self.setStyleSheet('''QLabel {background-color: grey;
                                      border-style:solid;
                                      border-width:1px;
                                      border-color: #049DD9;}''')

    @property
    def minValue(self):
        return self.__minValue

    @property
    def maxValue(self):
        return self.__maxValue

    @property
    def value(self):
        return self.__value

    def setSlider(self, slider):
        assert isinstance(slider, Slider)

        self.__slider = slider
        self.__minX = slider.geometry().topLeft().x()
        self.__maxX = slider.geometry().topRight().x()
        self.__minValue = slider.minValue
        self.__maxValue = slider.maxValue

    def initializeRange(self, minValue, maxValue, value, space):
        assert minValue is not None
        assert maxValue is not None
        assert value is not None
        assert space is not None

        self.__minValue = minValue
        self.__maxValue = maxValue
        self.__value = value
        self.__space = space

        positionFromValue = self.__slider.getPositionFromValue(
            self.__minX, self.__maxX, value)
        self.__updatePos(positionFromValue)
        self.__updateRangeValue()

    # this function is used to update the ranges
    def updateRange(self, minValue, maxValue, value):
        assert minValue is not None
        assert maxValue is not None
        assert value is not None

        self.__minValue = minValue
        self.__maxValue = maxValue
        self.__value = value

        positionFromValue = self.__slider.getPositionFromValue(
            self.__minX, self.__maxX, value)
        self.__updatePos(positionFromValue)
        self.__updateRangeValue()

    def mousePressEvent(self, event):
        self.__clickTimer.start()
        self.__isPressed = True

    def mouseMoveEvent(self, event):
        if self.__isPressed:
            posX = self.mapToParent(event.pos()).x()
            self.__updatePos(posX)

    @pyqtSlot(int)
    def setValue(self, newValue):
        assert newValue is not None

        if (newValue >= self.__slider.minValue
                and newValue <= self.__slider.maxValue):
            self.__value = newValue

            intervalLength = self.__slider.maxValue-self.__slider.minValue
            if intervalLength == 0:
                t = (newValue-self.__slider.minValue)
            else:
                t = (newValue-self.__slider.minValue)/intervalLength

            newPos = ((1-t)*self.__minX)+(t*self.__maxX)

            self.__updateValueFromPosition(newPos)
            self.__slider.setValue(newValue)

    def __updatePos(self, posX):
        assert posX is not None

        if self.__enabled:
            value = self.__slider.getValueFromPos(posX)
            self.setValue(value)

    def __updateRangeValue(self):
        if self.__slider is not None:
            self.__rangeValue.setText(str(self.__value))
            self.__rangeValue.adjustSize()
            self.__rangeValue.raise_()
            rangeWidth = self.__rangeValue.width()

            labelPosX = self.pos().x()+(self.width()/2-rangeWidth/2)
            if labelPosX < self.__minX+rangeWidth:
                labelPosX += rangeWidth/2
            elif labelPosX > self.__maxX-rangeWidth:
                labelPosX -= rangeWidth/2
            labelPosY = self.__slider.pos().y()+self.__space

            self.__rangeValue.setGeometry(labelPosX, labelPosY,
                                          rangeWidth,
                                          self.__rangeValue.height())

    def mouseReleaseEvent(self, event):
        self.__isPressed = False

    def __updateValueFromPosition(self, posX):
        assert posX is not None

        self.setGeometry(posX-self.width()/2,
                         self.geometry().y(), self.width(), self.height())
        self.__updateRangeValue()

        self.__signals.updateValueSignal.emit(self)


class RangeSignals(QObject):

    startEditingSignal = pyqtSignal(Range)
    updateValueSignal = pyqtSignal(Range)
    stopEditingSignal = pyqtSignal(Range)
