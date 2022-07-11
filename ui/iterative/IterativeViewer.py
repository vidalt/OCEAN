from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QListWidgetItem
# Import UI functions
from .UI_Iterative import UI_Iterative


class IterativeViewer(QWidget, UI_Iterative):
    # Initialize pyqt signals
    chosenDataset = pyqtSignal()
    randomPoint = pyqtSignal()
    calculateClass = pyqtSignal()
    nextIteration = pyqtSignal()

    def __init__(self):
        super(IterativeViewer, self).__init__()
        self.setupUi(self)
        # Connect pyqt signals
        self.comboBoxSelectDataset.currentTextChanged.connect(
            lambda: self.chosenDataset.emit())
        self.pushButtonRandomPoint.clicked.connect(
            lambda: self.randomPoint.emit())
        self.pushButtonCalculateClass.clicked.connect(
            lambda: self.calculateClass.emit())
        self.pushButtonNext.clicked.connect(
            lambda: self.nextIteration.emit())

        self.__iterationNumber = 1
        # Define tab functions
        self.tabWidget.tabBar().setTabButton(
            0, self.tabWidget.tabBar().RightSide, None)
        self.tabWidget.tabCloseRequested.connect(
            lambda index: self.tabWidget.removeTab(index))

        self.pushButtonNext.setEnabled(False)

    def getCanvas(self):
        return self.widgetContainerCanvas

    def initializeView(self, datasets):
        """ Fill the combobox.

        First clean the combobox, then add the datasets name.
        """
        assert isinstance(datasets, list)
        for dataset in datasets:
            assert isinstance(dataset, str)

        if datasets is not None:
            self.comboBoxSelectDataset.clear()
            self.comboBoxSelectDataset.addItems(datasets)

    def clearView(self):
        """ Clean the entire view.
        """
        self.listWidgetSelectedPoint.clear()
        self.labelOriginalClass.clear()
        self.clearClass()

    def clearClass(self):
        """
        Clean the calculated class.
        """
        self.labelOriginalClass.setText('Original Class: ')

    def getChosenDataset(self):
        """
        Get the selected dataset name.
        """
        return self.comboBoxSelectDataset.currentText()

    def addFeatureWidget(self, feature: 'QWidget'):
        """
        Add the features components inside the main view.
        """
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

    def addNewIterationTab(self, iterationView):
        iterationName = 'Scenario '+str(self.__iterationNumber)
        self.tabWidget.addTab(iterationView, iterationName)
        self.__iterationNumber += 1
        self.tabWidget.setCurrentIndex(self.tabWidget.count()-1)

        return iterationName

    def addFinalIteration(self, finalIterationView):
        iterationName = 'Final Scenario'
        self.tabWidget.addTab(finalIterationView, iterationName)
        self.tabWidget.setCurrentIndex(self.tabWidget.count()-1)

    def getChosenAxis(self):
        xText = self.comboBoxAxisX.currentText()
        yText = self.comboBoxAxisY.currentText()
        return xText, yText

    # this function is used to update the class component
    def showOriginalClass(self, classValue):
        assert classValue is not None
        self.labelOriginalClass.setText('Original Class: '+str(classValue))

    def enableNext(self, enabled):
        self.pushButtonNext.setEnabled(enabled)
