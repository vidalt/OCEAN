# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access and interact with the interface components

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QListWidgetItem

from .Ui_Iteration import Ui_Iteration

from .IterationEnums import IterationEnums

class IterationView(QWidget, Ui_Iteration):

    selectedFeatures = pyqtSignal()
    nextIteration = pyqtSignal()
    finishIteration = pyqtSignal()

    def __init__(self):
        super(IterationView, self).__init__()
        self.setupUi(self)

        self.pushButtonUpdateGraph.clicked.connect(lambda: self.selectedFeatures.emit())
        self.pushButtonNext.clicked.connect(lambda: self.nextIteration.emit())
        self.pushButtonFinish.clicked.connect(lambda: self.finishIteration.emit())


    def initializeView(self):
        pass

    def getCanvas(self):
        return self.widgetContainerCanvas

    # this function is used to add the features components inside the main view
    def addFeatureWidget(self, feature:'QWidget'):
        assert isinstance(feature, QWidget)

        item = QListWidgetItem(self.listWidgetSelectedPoint)
        item.setSizeHint(feature.size())
        self.listWidgetSelectedPoint.addItem(item)
        self.listWidgetSelectedPoint.setItemWidget(item, feature)

    def selectFeatures(self, options):
        self.comboBoxCheckable.selectItems(options)
    
    def addFeaturesOptions(self, options):
        assert isinstance(options, list)
        for features in options:
            assert isinstance(features, str)

        self.comboBoxCheckable.addItems(options)

    def getSelectedFeatures(self):
        return self.comboBoxCheckable.currentData()

    # this function is used to clean the calculated class
    def clearClass(self):
        self.labelCurrentClass.setText('Current Class: ')

    # this function is used to update the class component
    def showCurrentClass(self, classValue):
        assert classValue is not None

        self.labelCurrentClass.setText('Current Class: '+str(classValue))
