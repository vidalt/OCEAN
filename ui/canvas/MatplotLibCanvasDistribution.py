import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from sklearn.preprocessing import LabelEncoder
# Import OCEAN functions
from src.CounterFactualParameters import FeatureType


class MatplotLibCanvasDistribution(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        # Set dimensions: [left, bottom, width, height]
        self.__distributionRectangle = [0.2, 0.5, 0.75, 0.48]
        self.dpi = 100
        self.figure = plt.figure()
        FigureCanvasQTAgg.__init__(self, self.figure)
        self.distributionAxes = self.figure.add_axes(
            self.__distributionRectangle)
        self.setParent(parent)
        self.controller = None
        self.__featureToPlot = None

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

    def __clearAxesAndGraph(self):
        self.figure.clear()
        self.distributionAxes = self.figure.add_axes(
            self.__distributionRectangle)
        self.distributionAxes.clear()

    def updateGraphDistribution(self, parameters=None):
        self.__clearAxesAndGraph()
        if parameters is not None:
            self.controller = parameters['controller']
            self.__featureToPlot = parameters['featureToPlot']
            self.distributionAxes.set_ylabel('count')
            # distribution graph
            self.__updateProbability()
            # draw the figure
            self.draw()
