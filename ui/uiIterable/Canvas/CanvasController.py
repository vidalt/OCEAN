from .CanvasView import CanvasView

class CanvasController:

    def __init__(self):
        self.view = CanvasView()


    def updateGraph(self, parameters):
        self.view.getCanvas().updateGraph(parameters)