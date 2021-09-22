from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplcursors
from numpy.core.fromnumeric import size
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  as FigureCanvas

from CounterFactualParameters import FeatureType
from .PolygonInteractor import PolygonInteractor

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from sklearn.preprocessing import LabelEncoder 

import numpy as np
import pandas as pd

class MatplotLibCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.dpi=100

        self.figure = plt.figure()

        FigureCanvas.__init__(self, self.figure)

        self.axes = self.figure.add_axes([0.1, 0.15, 0.65, 0.75])
        self.txt = None

        self.setParent(parent)

        self.__dataframe = None


    def createAxis(self, fig, ax, xRange, yRange, labels):
        # create axis
        line1 = Line2D(xRange, yRange, color='r', alpha=0.5)
        fig.axes[0].add_line(line1)

        # set annotation ("ytick labels") to each feature
        for i, label in enumerate(labels):
            ax.text(xRange[0], i, ' '+str(label), transform=ax.transData)

    def updateGraph(self, parameters=None):
        self.clearAxesAndGraph()
        if parameters is not None:
            dataframe = parameters['dataframe']
            model = parameters['model']
            selectedFeatures = parameters['selectedFeatures']

            # dictEncoders = {}
            # for f in selectedFeatures:
            #     featureEncoder = LabelEncoder()
            #     featureEncoder.fit(dataframe[f])
            #     dictEncoders[f] = featureEncoder

            allFeaturesToPlot = selectedFeatures.copy()
            allFeaturesToPlot.insert(0, 'dist')
            allFeaturesToPlot.insert(1, 'prob1')
            allFeaturesToPlot.append('Class')
            allFeaturesToPlot.append('color')

            # clear the canvas
            self.axes.cla()  

            # create the draggable line to current point
            xs = []
            ys = []
            ranges = []
            decimals = []
            xMaxRange = 0

            for i, f in enumerate(allFeaturesToPlot):
                if f == 'dist':
                    pass
                elif f == 'prob1':
                    pass
                elif f == 'Class':
                    pass
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
                    value = dataframe.iloc[0][f]
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

            poly = Polygon(np.column_stack([xs, ys]), closed=False, fill=False, animated=True)

            # set the draggable line to PolygonInteractor
            self.axes.add_patch(poly)
            p = PolygonInteractor(self.axes, poly, ranges, decimals)

            # legends
            self.axes.legend([p.line], ['Current'], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

            # boundary
            self.axes.set_xlim((-1, len(allFeaturesToPlot)))
            self.axes.set_ylim((-1, xMaxRange))

            # xtick value
            xTicksValues = [i for i in range(len(allFeaturesToPlot))]
            self.axes.set_xticks(xTicksValues)
            # xtick name
            self.axes.set_xticklabels(allFeaturesToPlot)
            
            # draw the figure
            self.draw()

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi) 

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.axes = self.figure.add_axes([0.1, 0.15, 0.65, 0.75])
        self.axes.clear()
