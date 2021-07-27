from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplcursors
from numpy.core.fromnumeric import size
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  as FigureCanvas

class MatplotLibCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.dpi=100

        self.figure = Figure()

        FigureCanvas.__init__(self, self.figure)

        self.axes = self.figure.add_subplot(111)
        self.txt = None

        self.setParent(parent)

        self.__dataframe = None


    def updateGraph(self, parameters=None):
        if parameters is not None:
            self.__dataframe = parameters['dataframe']
            xVariable = parameters['xVariable']
            yVariable = parameters['yVariable']

            self.axes.cla()  # Clear the canvas.
            sns.scatterplot(x=self.__dataframe[xVariable],
                            y=self.__dataframe[yVariable],
                            hue=self.__dataframe['Class'],
                            data=self.__dataframe,
                            ax=self.axes,
                            size=self.__dataframe['distance'],
                            picker=True,
                            zorder=10)
            self.axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

            annot = self.axes.annotate("", xy=(0, 0), xytext=(-10, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))

            annot.set_visible(False)

            cursor = mplcursors.cursor(self.axes, hover=True)

            cursor.connect('add', lambda sel: sel.annotation.set_text('Class: {} \nDistance: {}'.
                format(self.__dataframe['Class'][sel.target.index], self.__dataframe['distance'][sel.target.index])))
            
            self.draw()

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi) 

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.clear()
