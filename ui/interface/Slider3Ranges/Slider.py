# Author: Moises Henrique Pereira

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel


class Slider(QLabel):
    """
    Know where are the bounds,
    and what is the scale to show the correct values
    """

    slyderPressedSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super(Slider, self).__init__(parent)

        self.value = 0

        self.__minValue = None
        self.__maxValue = None

        self.__decimalPlaces = None

        self.setMouseTracking(True)

    @property
    def minValue(self):
        return self.__minValue

    @property
    def maxValue(self):
        return self.__maxValue

    def initializeSlider(self, minValue, maxValue, decimalPlaces):
        assert minValue is not None
        assert maxValue is not None
        assert decimalPlaces is not None

        self.__minValue = minValue
        self.__maxValue = maxValue

        self.__decimalPlaces = decimalPlaces

    def updateSlider(self, minValue, maxValue):
        assert minValue is not None
        assert maxValue is not None

        self.__minValue = minValue
        self.__maxValue = maxValue

    def getValueAtXPosition(self, x):
        assert x is not None

        minX = self.geometry().left()
        maxX = self.geometry().right()

        value = self.minValue + \
            ((self.maxValue-self.minValue)*((x-minX)/(maxX-minX)))

        return int(value)

    def getPositionFromValue(self, minX, maxX, value):
        assert minX is not None
        assert maxX is not None
        assert value is not None

        try:
            posX = minX+((maxX-minX)*((value-self.minValue)
                         / (self.maxValue-self.minValue)))
        except:
            posX = minX+((maxX-minX)*((value-self.minValue)/1))

        return posX

    def getValueFromPos(self, pos):
        assert pos is not None

        minX = self.geometry().left()
        maxX = self.geometry().right()

        if self.__decimalPlaces == 0:
            value = int(round(self.minValue+pos*(self.maxValue
                        - self.minValue)/(maxX-minX), self.__decimalPlaces))
        else:
            value = round(self.minValue+pos*(self.maxValue
                          - self.minValue)/(maxX-minX), self.__decimalPlaces)

        return value

    def mousePressEvent(self, event):
        posX = event.pos().x()
        self.slyderPressedSignal.emit(posX)

    # this function is used to set a given value in the correct position
    def setValue(self, value):
        assert value is not None

        if (value >= self.minValue and value <= self.maxValue):
            self.value = value

            try:
                valueNorm = (self.value-self.minValue) / \
                             (self.maxValue-self.minValue)
            except:
                valueNorm = (self.value-self.minValue)/1

            if valueNorm >= 0 and valueNorm <= 1:
                if valueNorm >= 0 and valueNorm <= 0.01:
                    valueNorm += 0.01
