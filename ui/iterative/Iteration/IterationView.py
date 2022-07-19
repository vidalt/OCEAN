# Author: Moises Henrique Pereira

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QListWidgetItem
# Import UI functions
from ui.interface.InterfaceViewer import InterfaceViewer
from ui.canvas.MatplotLibCanvas import MatplotLibCanvas
from ui.canvas.MatplotLibCanvasDistribution import MatplotLibCanvasDistribution
from interface.CheckableComboBox.CheckableComboBox import CheckableComboBox


class IterationView(InterfaceViewer):
    """
    Import the UI file to be possible to access
    and interact with the interface components
    """

    selectedFeatures = pyqtSignal()
    nextIteration = pyqtSignal()
    finishIteration = pyqtSignal()
    outdatedGraph = pyqtSignal()

    def __init__(self):
        super(IterationView, self).__init__()
        self.setupIterationUi(self)

        self.__dictItems = {}

        self.pushButtonNext.clicked.connect(
            lambda: self.nextIteration.emit())
        self.pushButtonFinish.clicked.connect(
            lambda: self.finishIteration.emit())
        self.comboBoxCheckable.itemsChanged.connect(
            lambda: self.__onItemsChanged())

    def getCanvas(self):
        return self.widgetCanvas

    def getCanvasDistribution(self):
        return self.widgetCanvasDistribution

    def selectFeatures(self, options):
        self.comboBoxCheckable.selectItems(options)

    # this function is used to add the features components inside the main view
    def addFeatureWidget(self, featureName: str, feature: 'QWidget'):
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
        self.outdatedGraph.emit()

    def enabledNextIteration(self, enabled):
        self.pushButtonNext.setEnabled(enabled)
        self.pushButtonFinish.setEnabled(enabled)

    def setupIterationUi(self, Iteration):
        Iteration.setObjectName("Iteration")
        Iteration.resize(904, 548)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Iteration)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widgetContainerGraph = QtWidgets.QWidget(Iteration)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetContainerGraph.sizePolicy().hasHeightForWidth())
        self.widgetContainerGraph.setSizePolicy(sizePolicy)
        self.widgetContainerGraph.setStyleSheet(
            "QWidget [objectName*=\"widgetContainerGraph\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n}")
        self.widgetContainerGraph.setObjectName("widgetContainerGraph")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.widgetContainerGraph)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.labelSelectFeaturesToPlot = QtWidgets.QLabel(
            self.widgetContainerGraph)
        self.labelSelectFeaturesToPlot.setMinimumSize(QtCore.QSize(0, 25))
        self.labelSelectFeaturesToPlot.setMaximumSize(
            QtCore.QSize(16777215, 25))
        self.labelSelectFeaturesToPlot.setObjectName(
            "labelSelectFeaturesToPlot")
        self.verticalLayout_2.addWidget(self.labelSelectFeaturesToPlot)
        self.widgetContainerAxes = QtWidgets.QWidget(self.widgetContainerGraph)
        self.widgetContainerAxes.setMaximumSize(QtCore.QSize(16777215, 25))
        self.widgetContainerAxes.setObjectName("widgetContainerAxes")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(
            self.widgetContainerAxes)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.comboBoxCheckable = CheckableComboBox(self.widgetContainerAxes)
        self.comboBoxCheckable.setMinimumSize(QtCore.QSize(0, 25))
        self.comboBoxCheckable.setObjectName("comboBoxCheckable")
        self.horizontalLayout_9.addWidget(self.comboBoxCheckable)
        self.verticalLayout_2.addWidget(self.widgetContainerAxes)
        self.widgetContainer = QtWidgets.QWidget(self.widgetContainerGraph)
        self.widgetContainer.setObjectName("widgetContainer")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widgetContainer)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widgetContainerCanvas = QtWidgets.QWidget(self.widgetContainer)
        self.widgetContainerCanvas.setObjectName("widgetContainerCanvas")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widgetContainerCanvas)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelCanvasTitle = QtWidgets.QLabel(self.widgetContainerCanvas)
        self.labelCanvasTitle.setMinimumSize(QtCore.QSize(0, 26))
        self.labelCanvasTitle.setMaximumSize(QtCore.QSize(16777215, 26))
        self.labelCanvasTitle.setObjectName("labelCanvasTitle")
        self.verticalLayout.addWidget(self.labelCanvasTitle)
        self.widgetCanvas = MatplotLibCanvas(self.widgetContainerCanvas)
        self.widgetCanvas.setObjectName("widgetCanvas")
        self.verticalLayout.addWidget(self.widgetCanvas)
        self.horizontalLayout_3.addWidget(self.widgetContainerCanvas)
        self.widgetContainerFeatureInformations = QtWidgets.QWidget(
            self.widgetContainer)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetContainerFeatureInformations.sizePolicy().hasHeightForWidth())
        self.widgetContainerFeatureInformations.setSizePolicy(sizePolicy)
        self.widgetContainerFeatureInformations.setMinimumSize(
            QtCore.QSize(310, 0))
        self.widgetContainerFeatureInformations.setMaximumSize(
            QtCore.QSize(310, 16777215))
        self.widgetContainerFeatureInformations.setStyleSheet(
            "QWidget [objectName*=\"widgetContainerGraph\"]{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "\n"
            "}")
        self.widgetContainerFeatureInformations.setObjectName(
            "widgetContainerFeatureInformations")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(
            self.widgetContainerFeatureInformations)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.labelFeatureInformation = QtWidgets.QLabel(
            self.widgetContainerFeatureInformations)
        self.labelFeatureInformation.setMinimumSize(QtCore.QSize(0, 26))
        self.labelFeatureInformation.setMaximumSize(
            QtCore.QSize(16777215, 16777215))
        self.labelFeatureInformation.setObjectName("labelFeatureInformation")
        self.verticalLayout_3.addWidget(self.labelFeatureInformation)
        self.listWidgetFeatureInformations = QtWidgets.QListWidget(
            self.widgetContainerFeatureInformations)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.listWidgetFeatureInformations.sizePolicy().hasHeightForWidth())
        self.listWidgetFeatureInformations.setSizePolicy(sizePolicy)
        self.listWidgetFeatureInformations.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.listWidgetFeatureInformations.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.listWidgetFeatureInformations.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.listWidgetFeatureInformations.setObjectName(
            "listWidgetFeatureInformations")
        self.verticalLayout_3.addWidget(self.listWidgetFeatureInformations)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.labelFeatureDistribution = QtWidgets.QLabel(
            self.widgetContainerFeatureInformations)
        self.labelFeatureDistribution.setMinimumSize(QtCore.QSize(0, 25))
        self.labelFeatureDistribution.setMaximumSize(
            QtCore.QSize(16777215, 25))
        self.labelFeatureDistribution.setObjectName("labelFeatureDistribution")
        self.verticalLayout_3.addWidget(self.labelFeatureDistribution)
        self.widgetCanvasDistribution = MatplotLibCanvasDistribution(
            self.widgetContainerFeatureInformations)
        self.widgetCanvasDistribution.setMinimumSize(QtCore.QSize(0, 240))
        self.widgetCanvasDistribution.setMaximumSize(
            QtCore.QSize(16777215, 240))
        self.widgetCanvasDistribution.setObjectName("widgetCanvasDistribution")
        self.verticalLayout_3.addWidget(self.widgetCanvasDistribution)
        self.horizontalLayout_3.addWidget(
            self.widgetContainerFeatureInformations)
        self.verticalLayout_2.addWidget(self.widgetContainer)
        self.labelNextInfo = QtWidgets.QLabel(self.widgetContainerGraph)
        self.labelNextInfo.setMinimumSize(QtCore.QSize(0, 25))
        self.labelNextInfo.setObjectName("labelNextInfo")
        self.verticalLayout_2.addWidget(self.labelNextInfo)
        self.widgetNextFinish = QtWidgets.QWidget(self.widgetContainerGraph)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.widgetNextFinish.sizePolicy().hasHeightForWidth())
        self.widgetNextFinish.setSizePolicy(sizePolicy)
        self.widgetNextFinish.setObjectName("widgetNextFinish")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widgetNextFinish)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonNext = QtWidgets.QPushButton(self.widgetNextFinish)
        self.pushButtonNext.setMinimumSize(QtCore.QSize(0, 25))
        self.pushButtonNext.setObjectName("pushButtonNext")
        self.horizontalLayout_2.addWidget(self.pushButtonNext)
        self.pushButtonFinish = QtWidgets.QPushButton(self.widgetNextFinish)
        self.pushButtonFinish.setMinimumSize(QtCore.QSize(0, 25))
        self.pushButtonFinish.setObjectName("pushButtonFinish")
        self.horizontalLayout_2.addWidget(self.pushButtonFinish)
        self.verticalLayout_2.addWidget(self.widgetNextFinish)
        self.horizontalLayout.addWidget(self.widgetContainerGraph)

        self.retranslateUi(Iteration)
        QtCore.QMetaObject.connectSlotsByName(Iteration)

    def retranslateUi(self, Iteration):
        _translate = QtCore.QCoreApplication.translate
        Iteration.setWindowTitle(_translate("Iteration", "Form"))
        self.labelSelectFeaturesToPlot.setText(
            _translate("Iteration",
                       "Select Features To Plot: select the desired"
                       " features using the multiselection combobox below"))
        self.labelCanvasTitle.setText(_translate(
            "Iteration", "Drag the dots to change the point values: "
            "click on the dot over the axis that want to change and "
            "then pull it up or down"))
        self.labelFeatureInformation.setText(
            _translate("Iteration",
                       "Feature Information: shows the clicked feature "
                       "information \n"
                       "and allows to update its constraints"))
        self.labelFeatureDistribution.setText(_translate(
            "Iteration", "Feature Distribution: shows the clicked"
            " feature distribution"))
        self.labelNextInfo.setText(_translate(
            "Iteration", "The button \"Next\" instantiates a new scenario,"
            " and the button \"Finish\" instantiates the \"FinalScenario\""))
        self.pushButtonNext.setText(_translate("Iteration", "Next"))
        self.pushButtonFinish.setText(_translate("Iteration", "Finish"))
