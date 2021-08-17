# Author: Moises Henrique Pereira
# this class imports the UI file to be possible to access and interact with the interface components

from PyQt5.QtWidgets import QWidget, QListWidgetItem

from .Ui_Iteration import Ui_Iteration

from .IterationEnums import IterationEnums

class IterationView(QWidget, Ui_Iteration):

    def __init__(self):
        super(IterationView, self).__init__()
        self.setupUi(self)


    def initializeView(self):
        pass

    # this function is used to add the features components inside the main view
    def addFeatureWidget(self, feature:'QWidget'):
        assert isinstance(feature, QWidget)

        item = QListWidgetItem(self.listWidgetSelectedPoint)
        item.setSizeHint(feature.size())
        self.listWidgetSelectedPoint.addItem(item)
        self.listWidgetSelectedPoint.setItemWidget(item, feature)

    def addAxisOptions(self, options):
        assert isinstance(options, list)
        for features in options:
            assert isinstance(features, str)

        optionsX = options.copy()
        optionsX.insert(0, IterationEnums.DefaultAxes.DEFAULT_X.value)
        optionsY = options.copy()
        optionsY.insert(0, IterationEnums.DefaultAxes.DEFAULT_Y.value)
    
        if options is not None:
            self.comboBoxAxisX.clear()
            self.comboBoxAxisY.clear()
            self.comboBoxAxisX.addItems(optionsX)
            self.comboBoxAxisY.addItems(optionsY)

    # this function is used to clean the calculated class
    def clearClass(self):
        self.labelOriginalClass.setText('Original Class: ')

    # this function is used to update the class component
    def showOriginalClass(self, classValue):
        assert classValue is not None

        self.labelOriginalClass.setText('Original Class: '+str(classValue))
