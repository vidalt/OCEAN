import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class FeatureImportanceCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = plt.figure()
        self.dpi = 100
        FigureCanvasQTAgg.__init__(self, self.figure)

    def updateFeatureImportanceGraph(self, dataframe, xVariable, yVariable):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axes.bar(dataframe[xVariable], dataframe[yVariable])
        plt.xticks(rotation=70)
        self.figure.tight_layout()
        self.draw()
