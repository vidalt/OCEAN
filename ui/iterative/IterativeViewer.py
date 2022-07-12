from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore, QtWidgets
# Import UI functions
from ui.interface.InterfaceViewer import InterfaceViewer
from .Dash.DashView import DashView


class IterativeViewer(InterfaceViewer):
    """
    Import the UI file to access and
    interact with the interface components.
    """
    nextIteration = pyqtSignal()

    def __init__(self):
        super(IterativeViewer, self).__init__()
        self.setupIterativeUi(self)
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

    def clearView(self):
        """Clean the entire view."""
        self.listWidgetSelectedPoint.clear()
        self.labelOriginalClass.clear()
        self.clearClass()

    def clearClass(self):
        """Clean the calculated class."""
        self.labelOriginalClass.setText('Original Class: ')

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

    def enableNext(self, enabled):
        self.pushButtonNext.setEnabled(enabled)

    def setupIterativeUi(self, IterativeInterface):
        IterativeInterface.setObjectName("IterativeInterface")
        IterativeInterface.resize(951, 512)
        self.horizontalLayout = QtWidgets.QHBoxLayout(IterativeInterface)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(IterativeInterface)
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tabScenario0 = QtWidgets.QWidget()
        self.tabScenario0.setObjectName("tabScenario0")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tabScenario0)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widgetContainerDataset = QtWidgets.QWidget(self.tabScenario0)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetContainerDataset.sizePolicy().hasHeightForWidth())
        self.widgetContainerDataset.setSizePolicy(sizePolicy)
        self.widgetContainerDataset.setMinimumSize(QtCore.QSize(540, 0))
        self.widgetContainerDataset.setMaximumSize(QtCore.QSize(540, 16777215))
        self.widgetContainerDataset.setStyleSheet(
            "QWidget [objectName*=\"widgetContainerDataset\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n"
            "}")
        self.widgetContainerDataset.setObjectName("widgetContainerDataset")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.widgetContainerDataset)
        self.verticalLayout_2.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.labelSelectDatasetAndDatapoint = QtWidgets.QLabel(
            self.widgetContainerDataset)
        self.labelSelectDatasetAndDatapoint.setMinimumSize(QtCore.QSize(0, 25))
        self.labelSelectDatasetAndDatapoint.setMaximumSize(
            QtCore.QSize(16777215, 16777215))
        self.labelSelectDatasetAndDatapoint.setObjectName(
            "labelSelectDatasetAndDatapoint")
        self.verticalLayout_2.addWidget(self.labelSelectDatasetAndDatapoint)
        self.widgetContainerSelectData = QtWidgets.QWidget(
            self.widgetContainerDataset)
        self.widgetContainerSelectData.setObjectName(
            "widgetContainerSelectData")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(
            self.widgetContainerSelectData)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.comboBoxSelectDataset = QtWidgets.QComboBox(
            self.widgetContainerSelectData)
        self.comboBoxSelectDataset.setMinimumSize(QtCore.QSize(0, 25))
        self.comboBoxSelectDataset.setObjectName("comboBoxSelectDataset")
        self.horizontalLayout_2.addWidget(self.comboBoxSelectDataset)
        self.pushButtonRandomPoint = QtWidgets.QPushButton(
            self.widgetContainerSelectData)
        self.pushButtonRandomPoint.setMinimumSize(QtCore.QSize(0, 25))
        self.pushButtonRandomPoint.setObjectName("pushButtonRandomPoint")
        self.horizontalLayout_2.addWidget(self.pushButtonRandomPoint)
        self.verticalLayout_2.addWidget(self.widgetContainerSelectData)
        self.listWidgetSelectedPoint = QtWidgets.QListWidget(
            self.widgetContainerDataset)
        self.listWidgetSelectedPoint.setObjectName("listWidgetSelectedPoint")
        self.verticalLayout_2.addWidget(self.listWidgetSelectedPoint)
        self.labelClassInfo = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelClassInfo.setMinimumSize(QtCore.QSize(0, 25))
        self.labelClassInfo.setObjectName("labelClassInfo")
        self.verticalLayout_2.addWidget(self.labelClassInfo)
        self.widget = QtWidgets.QWidget(self.widgetContainerDataset)
        self.widget.setObjectName("widget")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButtonCalculateClass = QtWidgets.QPushButton(self.widget)
        self.pushButtonCalculateClass.setMinimumSize(QtCore.QSize(0, 25))
        self.pushButtonCalculateClass.setObjectName("pushButtonCalculateClass")
        self.horizontalLayout_9.addWidget(self.pushButtonCalculateClass)
        self.labelOriginalClass = QtWidgets.QLabel(self.widget)
        self.labelOriginalClass.setMinimumSize(QtCore.QSize(0, 25))
        self.labelOriginalClass.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelOriginalClass.setObjectName("labelOriginalClass")
        self.horizontalLayout_9.addWidget(self.labelOriginalClass)
        self.verticalLayout_2.addWidget(self.widget)
        self.labelNextInfo = QtWidgets.QLabel(self.widgetContainerDataset)
        self.labelNextInfo.setMinimumSize(QtCore.QSize(0, 25))
        self.labelNextInfo.setObjectName("labelNextInfo")
        self.verticalLayout_2.addWidget(self.labelNextInfo)
        self.pushButtonNext = QtWidgets.QPushButton(
            self.widgetContainerDataset)
        self.pushButtonNext.setMinimumSize(QtCore.QSize(0, 25))
        self.pushButtonNext.setObjectName("pushButtonNext")
        self.verticalLayout_2.addWidget(self.pushButtonNext)
        self.widgetContainerClass = QtWidgets.QWidget(
            self.widgetContainerDataset)
        self.widgetContainerClass.setObjectName("widgetContainerClass")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(
            self.widgetContainerClass)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2.addWidget(self.widgetContainerClass)
        self.horizontalLayout_3.addWidget(self.widgetContainerDataset)
        self.widgetFeatureImportance = QtWidgets.QWidget(self.tabScenario0)
        self.widgetFeatureImportance.setStyleSheet(
            "QWidget [objectName*=\"widgetFeatureImportance\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n"
            "}")
        self.widgetFeatureImportance.setObjectName("widgetFeatureImportance")
        self.verticalLayout = QtWidgets.QVBoxLayout(
            self.widgetFeatureImportance)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelFeatureImportance = QtWidgets.QLabel(
            self.widgetFeatureImportance)
        self.labelFeatureImportance.setMinimumSize(QtCore.QSize(0, 25))
        self.labelFeatureImportance.setMaximumSize(QtCore.QSize(16777215, 25))
        self.labelFeatureImportance.setObjectName("labelFeatureImportance")
        self.verticalLayout.addWidget(self.labelFeatureImportance)
        self.widgetContainerCanvas = DashView(self.widgetFeatureImportance)
        self.widgetContainerCanvas.setObjectName("widgetContainerCanvas")
        self.verticalLayout.addWidget(self.widgetContainerCanvas)
        self.horizontalLayout_3.addWidget(self.widgetFeatureImportance)
        self.tabWidget.addTab(self.tabScenario0, "")
        self.horizontalLayout.addWidget(self.tabWidget)

        self.retranslateUi(IterativeInterface)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(IterativeInterface)

    def retranslateUi(self, IterativeInterface):
        """
        Add text to interface Widgets.
        """
        _translate = QtCore.QCoreApplication.translate
        IterativeInterface.setWindowTitle(
            _translate("IterativeInterface", "Form"))
        self.labelSelectDatasetAndDatapoint.setText(
            _translate("IterativeInterface",
                       "Select Dataset and Datapoint: "
                       "select the dataset using the combobox below \n"
                       "and fill the following list features using the button "
                       "\"Random Point\" or manually"))
        self.pushButtonRandomPoint.setText(
            _translate("IterativeInterface", "Random Point"))
        self.labelClassInfo.setText(
            _translate("IterativeInterface",
                       "The button \"Calculate Class\" calculates "
                       "the class to the above-filled point"))
        self.pushButtonCalculateClass.setText(
            _translate("IterativeInterface", "Calculate Class"))
        self.labelOriginalClass.setText(
            _translate("IterativeInterface", "Original Class:"))
        self.labelNextInfo.setText(
            _translate("IterativeInterface",
                       "The button \"Next\" instantiates a new scenario"))
        self.pushButtonNext.setText(_translate("IterativeInterface", "Next"))
        self.labelFeatureImportance.setText(
            _translate("IterativeInterface",
                       "Feature Importance: a graph with the "
                       "prediction model features importance"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tabScenario0),
            _translate("IterativeInterface", "Scenario 0"))
