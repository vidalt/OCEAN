import numpy as np
from PyQt5.QtCore import QThread
# Import ui functions
from ui.interface.InterfaceController import InterfaceController
from ui.static.StaticViewer import StaticViewer
from ui.static.StaticWorker import StaticWorker
from ui.engine.CounterfactualEngine import CounterfactualEngine as engine
from ui.interface.InterfaceEnums import InterfaceEnums
# Import OCEAN functions
from src.CounterFactualParameters import FeatureType


class StaticController(InterfaceController):

    def __init__(self):
        super().__init__()
        self.view = StaticViewer()
        self.initializeView()

        self.view.chosenDataset.connect(self.__handlerChosenDataset)
        self.view.randomPoint.connect(self.__handlerRandomPoint)
        self.view.calculateClass.connect(self.__handlerCalculateClass)

        self.view.generateCounterfactual.connect(
            self.__handlerGenerateCounterfactual)

        self.featuresConstraints = {}

        # Set each view on a tab
        self.interfaceViewer.tabWidget.addTab(self.view,
                                              'Static Counterfactual')

    def __handlerChosenDataset(self):
        """ Opens the selected dataset, trains the random forest
        and the isolation forest, and present the features components
        and its respectives informations.
        """
        # getting the name of the desired dataset
        self.__chosenDataset = self.view.getChosenDataset()
        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            # cleaning the view
            self.view.clearView()
            self.initPointFeatures.clear()

            self.train_random_and_isolation_forests(self.__chosenDataset)
            self.show_feature_info_component()
        else:
            # cleaning the view
            self.view.clearView()
            self.initPointFeatures.clear()

    def __handlerRandomPoint(self):
        """
        Get a random datapoint from dataset.
        """
        self.view.clearCounterfactual()
        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            self.sample_random_point()

    def __handlerCalculateClass(self):
        """
        Calculate the class of the selected data point.
        """
        self.view.clearCounterfactual()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            _ = self.transform_and_predict_data_point()
            self.view.showOriginalClass(self.predictedOriginalClass[0])

    def __handlerGenerateCounterfactual(self):
        """
        Calculate the class of the selected data point
        generate the counterfactual explanation,
        and calculate the class of the explanation.
        """
        # Update cursor
        self.waitCursor()
        # Clean the view
        self.view.clearCounterfactual()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            # Show the steps
            self.view.showCounterfactualStatus(
                InterfaceEnums.Status.STEP1.value)

            # Get the datapoint
            datapoint = []
            featuresInformations = self.model.featuresInformations
            for feature in self.model.features:
                if feature != 'Class':
                    featureType = featuresInformations[feature]['featureType']
                    content = self.initPointFeatures[feature].getContent()

                    datapoint.append(content['value'])

                    if featureType is FeatureType.Binary:
                        notAllowedValue = content['notAllowedValue']
                        self.featuresConstraints[feature] = {
                            'featureType': featureType,
                            'notAllowedValue': notAllowedValue}

                    elif featureType in [FeatureType.Discrete,
                                         FeatureType.Numeric]:
                        minimum = self.model.transformSingleNumericalValue(
                            feature, content['minimumValue'])
                        maximum = self.model.transformSingleNumericalValue(
                            feature, content['maximumValue'])
                        self.featuresConstraints[feature] = {
                            'featureType': featureType,
                            'selectedMinimum': minimum,
                            'selectedMaximum': maximum}

                    elif featureType is FeatureType.Categorical:
                        self.featuresConstraints[feature] = {
                            'featureType': featureType,
                            'notAllowedValues': content['notAllowedValues']}

            self.chosenDataPoint = np.array(datapoint)

            # Show the steps
            self.view.showCounterfactualStatus(
                InterfaceEnums.Status.STEP2.value)
            # Transform the datapoint to predic its class
            # and generate its counterfactual explanation
            self.transformedChosenDataPoint = self.model.transformDataPoint(
                self.chosenDataPoint)
            # Predict the datapoint class
            self.predictedOriginalClass = engine.randomForestClassifierPredict(
                self.rfClassifier, [self.transformedChosenDataPoint])
            self.view.showOriginalClass(self.predictedOriginalClass[0])

            # Run the counterfactual generation in another thread
            self.thread = QThread()
            self.worker = StaticWorker(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.finished.connect(self.restorCursor)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)
            self.worker.couterfactualClass.connect(
                self.updateCounterfactualClass)
            self.worker.tableCounterfactualValues.connect(
                self.updateCounterfactualTable)
            self.thread.start()

    # this function is used to show the counterfactual step
    def reportProgress(self, status):
        assert isinstance(status, str)
        # showing the steps
        self.view.showCounterfactualStatus(status)

    # this function is used to update the counterfactual class text
    def updateCounterfactualClass(self, counterfactualClass):
        assert counterfactualClass is not None
        self.view.showCounterfactualClass(counterfactualClass)

    def updateCounterfactualTable(self, counterfactualComparison):
        """
        Update the comparison between the original datapoint
        and the counterfactual explanation
        """
        assert isinstance(counterfactualComparison, list)
        for item in counterfactualComparison:
            assert isinstance(item, list)
            assert len(item) == 3

        self.view.showCounterfactualComparison(counterfactualComparison)
