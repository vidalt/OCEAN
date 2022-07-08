from PyQt5.QtWidgets import QWidget

from .Ui_Canvas import Ui_Canvas

class CanvasView(QWidget, Ui_Canvas):

    def __init__(self):
        super(CanvasView, self).__init__()
        self.setupUi(self)


    def getCanvas(self):
        return self.widgetCanvas

    def resizeEvent(self, event):
        width = self.getCanvas().size().width()
        height = self.getCanvas().size().height()
        self.getCanvas().resizeCanvas(width, height)
