# Author: Moises Henrique Pereira
# Import the UI file to access and interact with interface components

from PyQt5.QtWidgets import QWidget
from .Ui_CounterfactualInterface import Ui_CounterfactualInterface


class InterfaceViewer(QWidget, Ui_CounterfactualInterface):

    def __init__(self):
        super(InterfaceViewer, self).__init__()
        self.setupUi(self)

    def getCanvas(self):
        return self.iterableCounterfactual
