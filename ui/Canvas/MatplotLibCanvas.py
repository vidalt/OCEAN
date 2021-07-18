from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  as FigureCanvas


class MatplotLibCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.dpi=100

        self.figure = Figure()

        FigureCanvas.__init__(self, self.figure)

        self.axes = self.figure.add_subplot(111)
        self.axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        self.setParent(parent)


    def updateGraph(self, parameters=None):
        if parameters is not None:
            dataframe = parameters['dataframe']
            xVariable = parameters['xVariable']
            yVariable = parameters['yVariable']
            dataframe['distance'] = dataframe['distance']

            self.axes.cla()  # Clear the canvas.
            sns.scatterplot(x=dataframe[xVariable],
                            y=dataframe[yVariable],
                            hue=dataframe['Class'],
                            data=dataframe,
                            ax=self.axes,
                            size=dataframe['distance'],
                            picker=True,
                            zorder=10)
            self.axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            self.draw()

    def resizeCanvas(self, width, height):
        self.figure.set_size_inches(width/self.dpi, height/self.dpi)

    def clearAxesAndGraph(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.clear()
