import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import pyqtSignal, QObject
from sklearn.preprocessing import LabelEncoder
# Import OCEAN functions
from src.CounterFactualParameters import FeatureType


class MatplotLibCanvasDistribution(FigureCanvasQTAgg, QObject):

    # the updated current point values
    updatedPoint = pyqtSignal(list)
    # inform errors
    errorPlot = pyqtSignal(str)

    def __init__(self, parent=None):
        # [left, bottom, width, height]
        self.__distributionRectangle = [0.2, 0.5, 0.75, 0.48]

        self.dpi = 100

        self.figure = plt.figure()

        FigureCanvasQTAgg.__init__(self, self.figure)

        self.distributionAxes = self.figure.add_axes(
            self.__distributionRectangle)

        self.setParent(parent)

        self.controller = None
        self.__featureToPlot = None

    # this function update the canvas probability

    def __updateProbability(self):
        feature = self.__featureToPlot

        rotation = 0
        if self.controller.model.featuresType[feature] is FeatureType.Categorical:
            rotation = 45

        dataValues = self.controller.model.data[feature].to_numpy()
        for i in range(len(dataValues)):
            dataValues[i] = str(dataValues[i])

        encoder = LabelEncoder()
        encoder.fit(dataValues)

        dataToBoxPlot = dataValues
        dataToBoxPlotEncoded = encoder.transform(dataToBoxPlot)

        xTicksName = encoder.classes_
        xTicksValues = [i+0.5 for i in range(len(xTicksName))]

        self.distributionAxes.hist(dataToBoxPlotEncoded, bins=len(
            xTicksName), range=(0, len(xTicksName)))
        self.distributionAxes.set_xticks(xTicksValues)
        self.distributionAxes.set_xticklabels(xTicksName, rotation=rotation)
        self.distributionAxes.set_xlabel(feature)
        self.distributionAxes.set_ylabel('count')

    def updateGraphDistribution(self, parameters=None):
        self.clearAxesAndGraph()

        if parameters is not None:
            self.controller = parameters['controller']
            self.__featureToPlot = parameters['featureToPlot']

            self.distributionAxes.set_ylabel('count')

            # distribution graph
            self.__updateProbability()

            # draw the figure
            self.draw()

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi)

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.distributionAxes = self.figure.add_axes(
            self.__distributionRectangle)
        self.distributionAxes.clear()
