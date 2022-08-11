import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import math
from PyQt5.QtCore import pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# Import UI functions
from .PolygonInteractor import PolygonInteractor
# Import OCEAN fucnctions
from src.CounterFactualParameters import FeatureType


class MatplotLibCanvas(FigureCanvasQTAgg, QObject):
    # the updated current point values
    updatedPoint = pyqtSignal(list)
    # the last feature clicked to plot the distribution and informations
    lastFeatureClicked = pyqtSignal(object)
    # inform errors
    errorPlot = pyqtSignal(str)

    def __init__(self, parent=None):
        # [left, bottom, width, height]
        self.__rectangle = [0, 0.25, 0.98, 0.65]

        self.dpi = 100

        self.figure = plt.figure()

        FigureCanvasQTAgg.__init__(self, self.figure)

        self.axes = self.figure.add_axes(self.__rectangle)
        self.txt = None

        self.setParent(parent)

        self.__featuresToPlot = None
        self.__featuresUniqueValues = {}

        self.controller = None

        self.polygonInteractable = None

        self.__currentDataframeAllowance = None
        self.__lastFeatureClicked = None

        self.__dictMinimumValues = {}
        self.__dictMaximumValues = {}

    def createAxis(self, fig, ax, xRange, yRange, labels,
                   editable, categorical, allowedAxis):
        axisColor = '#d3d3d3'
        if editable:
            axisColor = '#c0c0c0'
        if not allowedAxis:
            axisColor = 'red'
        # create axis
        line1 = Line2D(xRange, yRange, color=axisColor, alpha=0.5)
        fig.axes[0].add_line(line1)

        rotation = 0
        if categorical:
            rotation = 45

        # set annotation ("ytick labels") to each feature
        for i, label in enumerate(labels):
            ax.text(xRange[0], i, ' '+str(label), transform=ax.transData,
                    fontsize=8, color='#808080', rotation=rotation)

    def infToPlot(self, allFeaturesToPlot, datapoint):
        xs = []
        ys = []
        ranges = []
        decimals = []
        actionables = []
        xMaxRange = 0
        for i, f in enumerate(allFeaturesToPlot):
            uniqueValuesFeature = None
            if f == 'prob1':
                uniqueValuesFeature = [0, '-', 1]
                # use the unique values feature to plot the vertical axis
                self.createAxis(self.figure, self.axes, [i, i],
                                [0, len(uniqueValuesFeature)-1],
                                uniqueValuesFeature, False, False, True)
            else:
                uniqueValuesFeature = None
                rotation = False
                if self.controller.model.featuresType[f] is FeatureType.Binary:
                    content = self.controller.initPointFeatures[f].getContent(
                    )
                    uniqueValuesFeature = [
                        content['value0'], content['value1']]
                    rotation = False

                elif self.controller.model.featuresType[f] is FeatureType.Discrete or self.controller.model.featuresType[f] is FeatureType.Numeric:
                    content = self.controller.initPointFeatures[f].getContent(
                    )
                    minimumValue = math.floor(content['minimumValue'])
                    minimumValue = min(
                        minimumValue, self.__dictMinimumValues[f])

                    maximumValue = math.ceil(content['maximumValue'])
                    maximumValue = max(
                        maximumValue, self.__dictMaximumValues[f])

                    uniqueValuesFeature = [i for i in range(
                        minimumValue, maximumValue+1)]
                    rotation = False

                elif self.controller.model.featuresType[f] is FeatureType.Categorical:
                    content = self.controller.initPointFeatures[f].getContent(
                    )
                    uniqueValuesFeature = content['allPossibleValues']
                    rotation = True

                # append ranges to move
                ranges.append(len(uniqueValuesFeature)-1)

                # append x, y
                value = datapoint.iloc[0][f]
                xs.append(i)
                if self.controller.model.featuresType[f] == FeatureType.Discrete or self.controller.model.featuresType[f] == FeatureType.Numeric:
                    content = self.controller.initPointFeatures[f].getContent(
                    )
                    maximumValue = math.ceil(content['maximumValue'])
                    ys.append(float(value)-minimumValue)
                else:
                    if value == '-':  # this happens when the value doesn't matter
                        ys.append(0)
                    else:
                        ys.append(uniqueValuesFeature.index(value))

                # get x max range
                if len(uniqueValuesFeature) > xMaxRange:
                    xMaxRange = len(uniqueValuesFeature)

                # use the unique values feature to plot the vertical axis
                allowedAxis = self.__currentDataframeAllowance[f]
                self.createAxis(self.figure, self.axes, [i, i], [0, len(
                    uniqueValuesFeature)-1], uniqueValuesFeature, True, rotation, allowedAxis)

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

    def updateGraph(self, parameters=None):
        self.clearAxesAndGraph()

        # clear the polygon and the signal connections
        if self.polygonInteractable is not None:
            self.polygonInteractable.disconnectAll()

        if parameters is not None:
            self.controller = parameters['controller']
            currentPoint = parameters['currentPoint']
            originalPoint = parameters['originalPoint']
            lastScenarioPoint = parameters['lastScenarioPoint']
            lastScenarioName = parameters['lastScenarioName']
            counterfactualPoint = parameters['counterfactualPoint']
            selectedFeatures = parameters['selectedFeatures']

            self.__currentDataframeAllowance = parameters['currentDataframeAllowance']

            polygonColor = 'blue' if float(
                currentPoint.iloc[0].Class) == 0 else 'green'
            originalColor = '#696969'
            lastScenarioColor = '#d3a27f'
            counterfactualColor = '#98FB98'

            allFeaturesToPlot = selectedFeatures.copy()
            allFeaturesToPlot.insert(0, 'prob1')

            for i, f in enumerate(allFeaturesToPlot):
                if f == 'prob1':
                    pass

                else:
                    if self.controller.model.featuresType[f] is FeatureType.Discrete or self.controller.model.featuresType[f] is FeatureType.Numeric:
                        if lastScenarioPoint is not None:
                            self.__dictMinimumValues[f] = min(int(float(currentPoint.iloc[0][f])), int(float(originalPoint.iloc[0][f])), int(
                                float(lastScenarioPoint.iloc[0][f])), int(float(counterfactualPoint.iloc[0][f])))
                            self.__dictMaximumValues[f] = max(int(float(currentPoint.iloc[0][f])), int(float(originalPoint.iloc[0][f])), int(
                                float(lastScenarioPoint.iloc[0][f])), int(float(counterfactualPoint.iloc[0][f])))

                        else:
                            self.__dictMinimumValues[f] = min(int(float(currentPoint.iloc[0][f])), int(
                                float(originalPoint.iloc[0][f])), int(float(counterfactualPoint.iloc[0][f])))
                            self.__dictMaximumValues[f] = max(int(float(currentPoint.iloc[0][f])), int(
                                float(originalPoint.iloc[0][f])), int(float(counterfactualPoint.iloc[0][f])))

            # save the features to plot
            self.__featuresToPlot = allFeaturesToPlot

            # create the draggable line to current point
            returnedInfo = self.infToPlot(allFeaturesToPlot, currentPoint)
            if returnedInfo is None:
                return
            xDraggable, yDraggable, ranges, decimals, xMaxRange, actionables = returnedInfo

            # creating a polygon object
            poly = Polygon(np.column_stack(
                [xDraggable, yDraggable]), closed=False, fill=False, animated=True)

            # set the draggable line to PolygonInteractor
            self.axes.add_patch(poly)
            self.polygonInteractable = PolygonInteractor(
                self.axes, poly, ranges, decimals, actionables, polygonColor)
            self.polygonInteractable.updatedPoint.connect(
                self.__onUpdatedCurrentPoint)
            self.polygonInteractable.currentIndex.connect(
                self.__onLastFeatureClicked)

            # creating the line to the original point
            returnedOriginalInfo = self.infToPlot(
                allFeaturesToPlot, originalPoint)
            if returnedOriginalInfo is None:
                return
            xOriginal, yOriginal, _, _, _, _ = returnedOriginalInfo
            lineOriginal = Line2D(xOriginal, yOriginal,
                                  color=originalColor, animated=False)
            self.axes.add_line(lineOriginal)

            # creating the line to the last scenario point
            if lastScenarioPoint is not None:
                returnedLastScenarioInfo = self.infToPlot(
                    allFeaturesToPlot, lastScenarioPoint)
                if returnedLastScenarioInfo is None:
                    return
                xLastScenario, yLastScenario, _, _, _, _ = returnedLastScenarioInfo
                lineLastScenario = Line2D(
                    xLastScenario, yLastScenario, color=lastScenarioColor, animated=False)
                self.axes.add_line(lineLastScenario)

            # creating the line to the counterfactual point
            returnedCounterfactualInfo = self.infToPlot(
                allFeaturesToPlot, counterfactualPoint)
            if returnedCounterfactualInfo is None:
                return
            xCounterfactual, yCounterfactual, _, _, _, _ = returnedCounterfactualInfo
            lineCounterfactual = Line2D(
                xCounterfactual, yCounterfactual, color=counterfactualColor, animated=False)
            self.axes.add_line(lineCounterfactual)

            # probability axis
            xs = 0
            ys = float(currentPoint.iloc[0]['prob1'])*2
            self.axes.plot(xs-0.02, ys, color=polygonColor,
                           marker='o', markerfacecolor=polygonColor)

            ys = float(originalPoint.iloc[0]['prob1'])*2
            self.axes.plot(xs-0.01, ys, color=originalColor,
                           marker='o', markerfacecolor=originalColor)

            if lastScenarioPoint is not None:
                ys = float(lastScenarioPoint.iloc[0]['prob1'])*2
                self.axes.plot(xs+0.01, ys, color=lastScenarioColor,
                               marker='o', markerfacecolor=lastScenarioColor)

            ys = float(counterfactualPoint.iloc[0]['prob1'])*2
            self.axes.plot(xs+0.02, ys, color=counterfactualColor,
                           marker='o', markerfacecolor=counterfactualColor)

            # legends
            if lastScenarioPoint is not None:
                self.axes.legend([lineOriginal, lineLastScenario, self.polygonInteractable.line, lineCounterfactual],
                                 ['Scenario0', lastScenarioName, 'Current editable',
                                     'Counterfactual', 'Not editable axis'],
                                 loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=4)

            else:
                self.axes.legend([lineOriginal, self.polygonInteractable.line, lineCounterfactual],
                                 ['Scenario0', 'Current editable',
                                     'Counterfactual', 'Not editable axis'],
                                 loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=3)

            # boundary
            self.axes.set_xlim((-1, len(allFeaturesToPlot)))
            self.axes.set_ylim((-1, xMaxRange))

            # xtick value
            xTicksValues = [i for i in range(len(allFeaturesToPlot))]
            self.axes.set_xticks(xTicksValues)
            # xtick name
            xTicksLabels = ['class']+allFeaturesToPlot[1:]
            self.axes.set_xticklabels(xTicksLabels, rotation=45)
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
                    pass

                else:
                    # only need to update the selected features
                    # the other values are obtained by the controller
                    featureType = self.controller.model.featuresInformations[f]['featureType']

                    if featureType is FeatureType.Binary:
                        try:
                            currentPoint.append(
                                self.__featuresUniqueValues[f][int(point[indexAux])])
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
                            currentPoint.append(
                                self.__featuresUniqueValues[f][int(point[indexAux])])
                            indexAux += 1
                        except:
                            print('!'*75)
                            print('ERROR:', f, '---',
                                  len(self.__featuresUniqueValues[f]))
                            print(int(point[indexAux]),
                                  self.__featuresUniqueValues[f])
                            print('!'*75)

            self.updatedPoint.emit(currentPoint)

    # listen the current index to generate the distribution
    def __onLastFeatureClicked(self, currentPolygon, currentIndex):
        # to avoid the cached polygon
        if currentPolygon == self.polygonInteractable:
            if currentIndex >= 0:
                # self.__lastFeatureClicked = currentIndex+1
                self.__lastFeatureClicked = currentIndex

            else:
                self.__lastFeatureClicked = None

            self.lastFeatureClicked.emit(self.__lastFeatureClicked)

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi)

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.axes = self.figure.add_axes(self.__rectangle)
        self.axes.clear()
        self.__featuresToPlot = None
        self.__featuresUniqueValues = {}
        self.__lastFeatureClicked = None
