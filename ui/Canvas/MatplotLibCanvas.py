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

class MatplotLibCanvas(FigureCanvas, QObject):

    # the updated current point values
    updatedPoint = pyqtSignal(list)
    # inform errors
    errorPlot = pyqtSignal(str)

    def __init__(self, parent=None):
        # super(MatplotLibCanvas, self).__init__()

        self.dpi=100

        self.figure = plt.figure()

        FigureCanvas.__init__(self, self.figure)

        self.axes = self.figure.add_axes([0.1, 0.15, 0.65, 0.75])
        self.txt = None

        self.setParent(parent)

        self.__featuresToPlot = None
        self.__featuresUniqueValues = {}

        self.controller = None

        self.polygonInteractable = None


    def createAxis(self, fig, ax, xRange, yRange, labels):
        # create axis
        line1 = Line2D(xRange, yRange, color='#d3d3d3', alpha=0.5)
        fig.axes[0].add_line(line1)

        # set annotation ("ytick labels") to each feature
        for i, label in enumerate(labels):
            ax.text(xRange[0], i, ' '+str(label), transform=ax.transData, fontsize=8)

    def infToPlot(self, allFeaturesToPlot, datapoint):
        xs = []
        ys = []
        ranges = []
        decimals = []
        actionables = []
        xMaxRange = 0

        lastFeature = None
        try:
            for i, f in enumerate(allFeaturesToPlot):
                lastFeature = f
                
                uniqueValuesFeature = None

                if f == 'dist':
                    pass

                elif f == 'prob1':
                    value = datapoint.iloc[0][f]

                    # append the x, y, range, decimal plate, and actionability
                    xs.append(i)
                    ys.append(float(value)*4)
                    ranges.append(5)
                    decimals.append(0)
                    actionables.append(False)
                    xMaxRange = 5 if 5 > xMaxRange else xMaxRange

                    # uniqueValuesFeature
                    uniqueValuesFeature = [0, 0.25, 0.5, 0.75, 1]

                    # use the unique values feature to plot the vertical axis
                    self.createAxis(self.figure, self.axes, [i, i], [0, len(uniqueValuesFeature)-1], uniqueValuesFeature)

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

                else:
                    uniqueValuesFeature = None
                    if self.controller.model.featuresType[f] is FeatureType.Binary:
                        content = self.controller.dictControllersSelectedPoint[f].getContent()
                        uniqueValuesFeature = [content['value0'], content['value1']]

                    elif self.controller.model.featuresType[f] is FeatureType.Discrete or self.controller.model.featuresType[f] is FeatureType.Numeric:
                        content = self.controller.dictControllersSelectedPoint[f].getContent()
                        minimumValue = math.floor(content['minimumValue'])
                        maximumValue = math.ceil(content['maximumValue'])
                        uniqueValuesFeature = [i for i in range(minimumValue, maximumValue+1)]

                    elif self.controller.model.featuresType[f] is FeatureType.Categorical:
                        content = self.controller.dictControllersSelectedPoint[f].getContent()
                        uniqueValuesFeature = content['allowedValues']

                    # append ranges to move
                    ranges.append(len(uniqueValuesFeature)-1)

                    # append x, y
                    value = datapoint.iloc[0][f]
                    xs.append(i)
                    if self.controller.model.featuresType[f] == FeatureType.Discrete or self.controller.model.featuresType[f] == FeatureType.Numeric:
                        content = self.controller.dictControllersSelectedPoint[f].getContent()
                        maximumValue = math.ceil(content['maximumValue'])
                        ys.append(float(value)-minimumValue)
                    else:
                        ys.append(uniqueValuesFeature.index(value))

                    # get x max range
                    if len(uniqueValuesFeature) > xMaxRange:
                        xMaxRange = len(uniqueValuesFeature)

                    # use the unique values feature to plot the vertical axis
                    self.createAxis(self.figure, self.axes, [i, i], [0, len(uniqueValuesFeature)-1], uniqueValuesFeature)

                    # append decimal plates to move
                    if self.controller.model.featuresType[f] == FeatureType.Binary or self.controller.model.featuresType[f] == FeatureType.Categorical or self.controller.model.featuresType[f] == FeatureType.Discrete:
                        decimals.append(0)
                    else:
                        decimals.append(1)

                    # append actionability of the features
                    actionables.append(True)

                if uniqueValuesFeature is not None:
                    self.__featuresUniqueValues[f] = uniqueValuesFeature

            return xs, ys, ranges, decimals, xMaxRange, actionables
        
        except:
            self.errorPlot.emit(lastFeature)

    def updateGraph(self, parameters=None):
        self.clearAxesAndGraph()

        if parameters is not None:
            self.controller = parameters['controller']
            currentPoint = parameters['currentPoint']
            originalPoint = parameters['originalPoint']
            lastScenarioPoint = parameters['lastScenarioPoint']
            selectedFeatures = parameters['selectedFeatures']

            polygonColor = 'blue' if float(currentPoint.iloc[0].Class) == 0 else 'green'

            allFeaturesToPlot = selectedFeatures.copy()
            allFeaturesToPlot.insert(0, 'dist')
            allFeaturesToPlot.insert(1, 'prob1')
            allFeaturesToPlot.append('Class')

            # save the features to plot
            self.__featuresToPlot = allFeaturesToPlot

            # clear the canvas
            self.axes.cla()  

            # create the draggable line to current point
            returnedInfo = self.infToPlot(allFeaturesToPlot, currentPoint)
            if returnedInfo is None:
                return
            xDraggable, yDraggable, ranges, decimals, xMaxRange, actionables = returnedInfo

            # creating a polygon object
            poly = Polygon(np.column_stack([xDraggable, yDraggable]), closed=False, fill=False, animated=True)

            # set the draggable line to PolygonInteractor
            self.axes.add_patch(poly)
            self.polygonInteractable = PolygonInteractor(self.axes, poly, ranges, decimals, actionables, polygonColor)
            self.polygonInteractable.updatedPoint.connect(self.__onUpdatedCurrentPoint)

            # creating the line to the original point
            returnedOriginalInfo = self.infToPlot(allFeaturesToPlot, originalPoint)
            if returnedOriginalInfo is None:
                return
            xOriginal, yOriginal, _, _, _, _ = returnedOriginalInfo
            lineOriginal = Line2D(xOriginal, yOriginal, color='#a9a9a9', animated=False)
            self.axes.add_line(lineOriginal)

            # creating the line to the last scenario point
            if lastScenarioPoint is not None:
                returnedLastScenarioInfo = self.infToPlot(allFeaturesToPlot, lastScenarioPoint)
                if returnedLastScenarioInfo is None:
                    return
                xLastScenario, yLastScenario, _, _, _, _ = returnedLastScenarioInfo
                lineLastScenario = Line2D(xLastScenario, yLastScenario, color='#d3a27f', animated=False)
                self.axes.add_line(lineLastScenario)
            
            # legends
            if lastScenarioPoint is not None:
                self.axes.legend([lineOriginal, lineLastScenario, self.polygonInteractable.line], ['Original', 'Last Scenario', 'Current editable'], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            else:
                self.axes.legend([lineOriginal, self.polygonInteractable.line], ['Original', 'Current editable'], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

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
    def __onUpdatedCurrentPoint(self, currentPolygon, point):
        # to avoid the cached polygon
        if currentPolygon == self.polygonInteractable:
            currentPoint = []
            indexAux = 0
            for f in self.__featuresToPlot:            
                if f == 'prob1':
                    # quando colocar essas colunas: necessário atualizar o indexAux
                    pass

                elif f == 'dist' or f == 'Class':
                    indexAux += 1
                
                else:
                    # only need to update the selected features
                    # the other values are obtained by the controller
                    featureType = self.controller.model.featuresInformations[f]['featureType']

                    if featureType is FeatureType.Binary:
                        try:
                            currentPoint.append(self.__featuresUniqueValues[f][int(point[indexAux])])
                            indexAux += 1
                        except:
                            print('ERROR:', f)

                    elif featureType is FeatureType.Discrete:
                        minValue = min(self.__featuresUniqueValues[f])
                        currentPoint.append(int(point[indexAux] + minValue))
                        indexAux += 1

                    elif featureType is FeatureType.Numeric:
                        minValue = min(self.__featuresUniqueValues[f])
                        currentPoint.append(point[indexAux] + minValue)
                        indexAux += 1

                    elif featureType is FeatureType.Categorical:
                        try:
                            currentPoint.append(self.__featuresUniqueValues[f][int(point[indexAux])])
                            indexAux += 1
                        except:
                            print('!'*75)
                            print('ERROR:', f, '---', len(self.__featuresUniqueValues[f]))
                            print(int(point[indexAux]), self.__featuresUniqueValues[f])
                            print('!'*75)

            self.updatedPoint.emit(currentPoint)

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi) 

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.axes = self.figure.add_axes([0.1, 0.15, 0.65, 0.75])
        self.axes.clear()
        self.__featuresToPlot = None
        self.__featuresUniqueValues = {}
