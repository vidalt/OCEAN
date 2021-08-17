from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QListWidgetItem

from .Ui_CounterfactualInterfaceIterable import Ui_CounterfactualInterfaceIterable

class CounterfactualInterfaceViewIterable(QWidget, Ui_CounterfactualInterfaceIterable):

    chosenDataset = pyqtSignal()
    randomPoint = pyqtSignal()
    calculateClass = pyqtSignal()
    nextIteration = pyqtSignal()
    # calculateDistances = pyqtSignal()
    # updateGraph = pyqtSignal()

    def __init__(self):
        super(CounterfactualInterfaceViewIterable, self).__init__()
        self.setupUi(self)

        self.comboBoxSelectDataset.currentTextChanged.connect(lambda: self.chosenDataset.emit())
        self.pushButtonRandomPoint.clicked.connect(lambda: self.randomPoint.emit())
        self.pushButtonCalculateClass.clicked.connect(lambda: self.calculateClass.emit())
        self.pushButtonNext.clicked.connect(lambda: self.nextIteration.emit())
        # self.pushButtonCalculateDistances.clicked.connect(lambda: self.calculateDistances.emit())
        # self.pushButtonUpdateGraph.clicked.connect(lambda: self.updateGraph.emit())

        self.__iterationNumber = 1

        self.tabWidget.tabBar().setTabButton(0, self.tabWidget.tabBar().RightSide, None)
        self.tabWidget.tabCloseRequested.connect(lambda index: self.tabWidget.removeTab(index))


    # this function fill the combobox
    # first cleaning the combobox,
    # and adding the datasets name
    def initializeView(self, datasets):
        assert isinstance(datasets, list)
        for dataset in datasets:
            assert isinstance(dataset, str)
    
        if datasets is not None:
            self.comboBoxSelectDataset.clear()
            self.comboBoxSelectDataset.addItems(datasets)

    # this function is used to clean the entire view
    def clearView(self):
        self.listWidgetSelectedPoint.clear()
        # self.comboBoxAxisX.clear()
        # self.comboBoxAxisY.clear()
        self.labelOriginalClass.clear()
        self.labelOriginalClass.setText('Original Class: ')

    # this function is used to clean the calculated class
    def clearClass(self):
        self.labelOriginalClass.setText('Original Class: ')

    # this function is used to get the selected dataset name
    def getChosenDataset(self):
        return self.comboBoxSelectDataset.currentText()

    # this function is used to add the features components inside the main view
    def addFeatureWidget(self, feature:'QWidget'):
        assert isinstance(feature, QWidget)

        item = QListWidgetItem(self.listWidgetSelectedPoint)
        item.setSizeHint(feature.size())
        self.listWidgetSelectedPoint.addItem(item)
        self.listWidgetSelectedPoint.setItemWidget(item, feature)

    def addAxisOptions(self, options):
        assert isinstance(options, list)
        for dataset in options:
            assert isinstance(dataset, str)
    
        if options is not None:
            self.comboBoxAxisX.clear()
            self.comboBoxAxisY.clear()
            self.comboBoxAxisX.addItems(options)
            self.comboBoxAxisY.addItems(options)

    def addGraphTab(self, canvasView):
        self.tabWidgetContainerCanvas.addTab(canvasView, 'Iteration'+str(self.__iterationNumber))
        self.__iterationNumber += 1

    def addNewIterationTab(self, iterationView):
        self.tabWidget.addTab(iterationView, 'Step'+str(self.__iterationNumber))
        self.__iterationNumber += 1

        self.tabWidget.setCurrentIndex(self.tabWidget.count()-1)

    def getChosenAxis(self):
        return self.comboBoxAxisX.currentText(), self.comboBoxAxisY.currentText()

    # this function is used to update the class component
    def showOriginalClass(self, classValue):
        assert classValue is not None

        self.labelOriginalClass.setText('Original Class: '+str(classValue))