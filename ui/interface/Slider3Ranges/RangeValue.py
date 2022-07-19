# Author: Moises Henrique Pereira

from PyQt5.QtWidgets import QLabel


class RangeValue(QLabel):
    """
    show the values of the ranges and to help the controller to manage it
    """

    def __init__(self, parent=None):
        super(RangeValue, self).__init__(parent)
        self.__range = None

    def mousePressEvent(self, event):
        if self.__range is not None:
            self.__range.isPressed = True

    def mouseMoveEvent(self, event):
        if self.__range is not None:
            posX = self.mapToParent(event.pos()).x()
            self.__range.updatePos(posX)

    def mouseReleaseEvent(self, event):
        if self.__range is not None:
            self.__range.isPressed = False
