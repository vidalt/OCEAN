# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access and interact with the interface components

from PyQt5.QtWidgets import QWidget

from .Ui_CounterfactualInterface import Ui_CounterfactualInterface

class CounterfactualInterfaceView(QWidget, Ui_CounterfactualInterface):

    def __init__(self):
        super(CounterfactualInterfaceView, self).__init__()
        self.setupUi(self)


    def getCanvas(self):
        return self.iterableCounterfactual
