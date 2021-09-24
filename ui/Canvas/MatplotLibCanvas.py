from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplcursors
from numpy.core.fromnumeric import size
import seaborn as sns
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

class MatplotLibCanvas(FigureCanvas, QObject):

    # the updated current point values
    updatedPoint = pyqtSignal(list)

    def __init__(self, parent=None):
        super(MatplotLibCanvas, self).__init__()

        self.dpi=100

        self.figure = plt.figure()

        FigureCanvas.__init__(self, self.figure)

        self.axes = self.figure.add_axes([0.1, 0.15, 0.65, 0.75])
        self.txt = None

        self.setParent(parent)

        self.__featuresToPlot = None
        self.__featuresUniqueValues = {}


    def createAxis(self, fig, ax, xRange, yRange, labels):
        # create axis
        line1 = Line2D(xRange, yRange, color='black', alpha=0.5)
        fig.axes[0].add_line(line1)

        # set annotation ("ytick labels") to each feature
        for i, label in enumerate(labels):
            ax.text(xRange[0], i, ' '+str(label), transform=ax.transData)

    def infToPlot(self, model, allFeaturesToPlot, datapoint):
        xs = []
        ys = []
        ranges = []
        decimals = []
        actionables = []
        xMaxRange = 0

        for i, f in enumerate(allFeaturesToPlot):
            uniqueValuesFeature = None

            if f == 'dist':
                pass

            elif f == 'prob1':
                pass

            elif f == 'Class':
                value = datapoint.iloc[0][f]

                # append the x, y, range, decimal plate, and actionability
                xs.append(i)
                ys.append(float(value))
                ranges.append(2)
                decimals.append(0)
                actionables.append(False)
                xMaxRange = 2 if 2 > xMaxRange else xMaxRange

                # uniqueValuesFeature
                uniqueValuesFeature = [0, 1]

                # use the unique values feature to plot the vertical axis
                self.createAxis(self.figure, self.axes, [i, i], [0, len(uniqueValuesFeature)-1], uniqueValuesFeature)

            elif f == 'color':
                pass

            else:
                # append ranges to move
                uniqueValuesFeature = model.data[f].unique()
                ranges.append(len(uniqueValuesFeature)-1)
                if model.featuresType[f] == FeatureType.Discrete or model.featuresType[f] == FeatureType.Numeric:
                    uniqueValuesFeature = pd.to_numeric(uniqueValuesFeature)
                    uniqueValuesFeature.sort()

                # append x, y
                value = datapoint.iloc[0][f]
                xs.append(i)
                if model.featuresType[f] == FeatureType.Discrete or model.featuresType[f] == FeatureType.Numeric:
                    ys.append(np.where(uniqueValuesFeature == float(value))[0][0])
                else:
                    ys.append(np.where(uniqueValuesFeature == value)[0][0])

                # get x max range
                if len(uniqueValuesFeature) > xMaxRange:
                    xMaxRange = len(uniqueValuesFeature)

                # use the unique values feature to plot the vertical axis
                self.createAxis(self.figure, self.axes, [i, i], [0, len(uniqueValuesFeature)-1], uniqueValuesFeature)

                # append decimal plates to move
                if model.featuresType[f] == FeatureType.Binary or model.featuresType[f] == FeatureType.Categorical or model.featuresType[f] == FeatureType.Discrete:
                    decimals.append(0)
                else:
                    decimals.append(1)

                # append actionability of the features
                actionables.append(True)

            if uniqueValuesFeature is not None:
                self.__featuresUniqueValues[f] = uniqueValuesFeature

        return xs, ys, ranges, decimals, xMaxRange, actionables

    def updateGraph(self, parameters=None):
        self.clearAxesAndGraph()

        if parameters is not None:
            model = parameters['model']
            currentPoint = parameters['currentPoint']
            originalPoint = parameters['originalPoint']
            selectedFeatures = parameters['selectedFeatures']

            allFeaturesToPlot = selectedFeatures.copy()
            allFeaturesToPlot.insert(0, 'dist')
            allFeaturesToPlot.insert(1, 'prob1')
            allFeaturesToPlot.append('Class')
            allFeaturesToPlot.append('color')

            # save the features to plot
            self.__featuresToPlot = allFeaturesToPlot

            # clear the canvas
            self.axes.cla()  

            # create the draggable line to current point
            xDraggable, yDraggable, ranges, decimals, xMaxRange, actionables = self.infToPlot(model, allFeaturesToPlot, currentPoint)

            # creating a polygon object
            poly = Polygon(np.column_stack([xDraggable, yDraggable]), closed=False, fill=False, animated=True)

            # set the draggable line to PolygonInteractor
            self.axes.add_patch(poly)
            p = PolygonInteractor(self.axes, poly, ranges, decimals, actionables)
            p.updatedPoint.connect(self.__onUpdatedCurrentPoint)

            # creating the line to original point
            xOriginal, yOriginal, _, _, _, _ = self.infToPlot(model, allFeaturesToPlot, originalPoint)
            lineOriginal = Line2D(xOriginal, yOriginal, color='green', animated=False)
            self.axes.add_line(lineOriginal)
            
            # legends
            self.axes.legend([p.line, lineOriginal], ['Current', 'Original'], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

            # boundary
            self.axes.set_xlim((-1, len(allFeaturesToPlot)))
            self.axes.set_ylim((-1, xMaxRange))

            # xtick value
            xTicksValues = [i for i in range(len(allFeaturesToPlot))]
            self.axes.set_xticks(xTicksValues)
            # xtick name
            self.axes.set_xticklabels(allFeaturesToPlot)
            # ytick name
            self.axes.set_yticklabels([])
            
            # draw the figure
            self.draw()

    # listen the updated point and emit the same to controller
    def __onUpdatedCurrentPoint(self, point):
        currentPoint = []
        indexAux = 0
        for f in self.__featuresToPlot:
            if f == 'dist' or f == 'prob1' or f == 'color':
                # quando colocar essas colunas: necessário atualizar o indexAux
                pass

            elif f == 'Class':
                indexAux += 1
            
            else:
                # only need to update the selected features
                # the other values are obtained by the controller
                currentPoint.append(self.__featuresUniqueValues[f][int(point[indexAux])])
                indexAux += 1

        self.updatedPoint.emit(currentPoint)

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi) 

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.axes = self.figure.add_axes([0.1, 0.15, 0.65, 0.75])
        self.axes.clear()
        self.__featuresToPlot = None
        self.__featuresUniqueValues = {}
