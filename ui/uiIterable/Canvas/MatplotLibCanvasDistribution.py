from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplcursors
from numpy.core.fromnumeric import size
import seaborn as sns
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  as FigureCanvas

from CounterfactualEngine.CounterfactualEngine import CounterfactualEngine

from CounterFactualParameters import FeatureType
from .PolygonInteractor import PolygonInteractor

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from sklearn.preprocessing import LabelEncoder 

import numpy as np
import pandas as pd

from PyQt5.QtCore import pyqtSignal, QObject

class MatplotLibCanvasDistribution(FigureCanvas, QObject):

    # the updated current point values
    updatedPoint = pyqtSignal(list)
    # inform errors
    errorPlot = pyqtSignal(str)

    def __init__(self, parent=None):
        self.__distributionRectangle = [0.2, 0.5, 0.75, 0.48] # [left, bottom, width, height]

        self.dpi=100

        self.figure = plt.figure()

        FigureCanvas.__init__(self, self.figure)

        self.distributionAxes = self.figure.add_axes(self.__distributionRectangle)

        self.setParent(parent)

        self.controller = None
        self.__featureToPlot = None

            
    # this function update the canvas probability
    def __updateProbability(self):
        feature = self.__featureToPlot

        rotation = 0
        if self.controller.model.featuresType[feature] is FeatureType.Categorical:
            rotation = 45

        encoder = LabelEncoder()
        encoder.fit(self.controller.model.data[feature].to_numpy())

        dataToBoxPlot = self.controller.model.data[feature].to_numpy()
        dataToBoxPlotEncoded = encoder.transform(dataToBoxPlot)

        xTicksName = encoder.classes_
        xTicksValues = [i+0.5 for i in range(len(xTicksName))]

        self.distributionAxes.hist(dataToBoxPlotEncoded, bins=len(xTicksName), range=(0,len(xTicksName)))
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
        self.distributionAxes = self.figure.add_axes(self.__distributionRectangle)
        self.distributionAxes.clear()
