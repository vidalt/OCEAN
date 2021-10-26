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

        self.__dictItems = {}

        self.comboBoxCheckable.itemsChanged.connect(lambda: self.__onItemsChanged())


    def initializeView(self):
        pass

    def getCanvas(self):
        return self.widgetCanvas

    def getCanvasDistribution(self):
        return self.widgetCanvasDistribution

    def selectFeatures(self, options):
        self.comboBoxCheckable.selectItems(options)

    # this function is used to add the features components inside the main view
    def addFeatureWidget(self, featureName:str, feature:'QWidget'):
        assert isinstance(featureName, str)
        assert isinstance(feature, QWidget)

        item = QListWidgetItem(self.listWidgetFeatureInformations)
        item.setSizeHint(feature.size())
        item.setHidden(True)
        self.listWidgetFeatureInformations.addItem(item)
        self.listWidgetFeatureInformations.setItemWidget(item, feature)

        self.__dictItems[featureName] = item
    
    def addFeaturesOptions(self, options):
        assert isinstance(options, list)
        for features in options:
            assert isinstance(features, str)

        self.comboBoxCheckable.addItems(options)

    def getSelectedFeatures(self):
        return self.comboBoxCheckable.currentData()

    def showItemByFeature(self, feature):
        assert isinstance(feature, str)

        for i in range(self.listWidgetFeatureInformations.count()):
            self.listWidgetFeatureInformations.item(i).setHidden(True)
        
        self.__dictItems[feature].setHidden(False)

        height = self.__dictItems[feature].sizeHint().height()
        self.listWidgetFeatureInformations.setMinimumHeight(height)

    def __onItemsChanged(self):
        print('#'*75)
        print('GRÁFICO DESATUALIZADO')
        print('#'*75)
