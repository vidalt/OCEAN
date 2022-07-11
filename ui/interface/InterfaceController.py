# Author: Moises Henrique Pereira

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
# Import ui functions
from ui.interface.InterfaceViewer import InterfaceViewer
from ui.interface.InterfaceModel import InterfaceModel
from ui.interface.InterfaceEnums import InterfaceEnums


class InterfaceController():
    """ Handle the logic over the interface.

    Interact with model, viewer and worker.
    Take the selected dataset informations from model to send to counterfactual
    generator in worker class.
    """

    def __init__(self):
        self.interfaceViewer = InterfaceViewer()
        self.model = InterfaceModel()

        self.__chosenDataset = InterfaceEnums.SelectDataset.DEFAULT.value

        self.randomForestClassifier = None
        self.isolationForest = None

        self.initPointFeatures = {}
        self.chosenDataPoint = None
        self.transformedChosenDataPoint = None
        self.predictedOriginalClass = None

    def initializeView(self):
        """ Send the dataframe names to interface. """
        datasets = self.model.getDatasetsName()

        datasetsName = [InterfaceEnums.SelectDataset.DEFAULT.value]
        for datasetName in datasets:
            auxDatasetName = datasetName.split('.')[0]
            datasetsName.append(auxDatasetName)

        self.view.initializeView(datasetsName)

    def waitCursor(self):
        """
        Change the cursor to a wait cursor.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def restorCursor(self):
        """
        Restore cursor to default.
        """
        QApplication.restoreOverrideCursor()
