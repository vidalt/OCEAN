import numpy as np
from PyQt5.QtCore import QThread
# Import ui functions
from ui.interface.InterfaceController import InterfaceController
from ui.static.StaticViewer import StaticViewer
from ui.static.InterfaceWorker import InterfaceWorker
from ui.engine.CounterfactualEngine import CounterfactualEngine
from ui.interface.InterfaceEnums import InterfaceEnums
from ui.interface.ComboboxList.ComboboxListController import ComboboxListController
from ui.interface.DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ui.interface.Slider3Ranges.Slider3RangesController import Slider3RangesController
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

            # opening the desired dataset
            self.model.openChosenDataset(self.__chosenDataset)
            xTrain, yTrain = self.model.getTrainData()

            # training the random forest and isolation forest models
            if xTrain is not None and yTrain is not None:
                self.randomForestClassifier = CounterfactualEngine.trainRandomForestClassifier(
                    xTrain, yTrain)
                self.isolationForest = CounterfactualEngine.trainIsolationForest(
                    xTrain)

            # showing the features components and informations
            for feature in self.model.features:
                if feature != 'Class':
                    featureType = self.model.featuresInformations[feature]['featureType']
                    componentController = None
                    if featureType is FeatureType.Binary:
                        value0 = self.model.featuresInformations[feature]['value0']
                        value1 = self.model.featuresInformations[feature]['value1']

                        componentController = DoubleRadioButtonController(
                            self.view)
                        componentController.initializeView(
                            feature, str(value0), str(value1))

                    elif featureType is FeatureType.Discrete:
                        minValue = self.model.featuresInformations[feature]['min']
                        maxValue = self.model.featuresInformations[feature]['max']

                        # componentController = LineEditMinimumMaximumController(self.view)
                        componentController = Slider3RangesController(
                            self.view)
                        componentController.initializeView(
                            feature, minValue, maxValue, decimalPlaces=0)

                    elif featureType is FeatureType.Numeric:
                        minValue = self.model.featuresInformations[feature]['min']
                        maxValue = self.model.featuresInformations[feature]['max']

                        # componentController = LineEditMinimumMaximumController(self.view)
                        componentController = Slider3RangesController(
                            self.view)
                        componentController.initializeView(
                            feature, minValue, maxValue)

                    elif featureType is FeatureType.Categorical:
                        componentController = ComboboxListController(self.view)
                        componentController.initializeView(
                            feature, self.model.featuresInformations[feature]['possibleValues'])

                    # adding the view to selectedPoint component
                    self.view.addFeatureWidget(componentController.view)
                    # saving the controller to facilitate the access to components
                    self.initPointFeatures[feature] = componentController

        else:
            # cleaning the view
            self.view.clearView()
            self.initPointFeatures.clear()

    # this function get a random datapoint from dataset
    def __handlerRandomPoint(self):
        self.view.clearCounterfactual()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            randomDataPoint = self.model.getRandomPoint(
                self.randomForestClassifier)

            # showing the values in their respective component
            for index, feature in enumerate(self.model.features):
                if feature != 'Class':
                    self.initPointFeatures[feature].setSelectedValue(
                        randomDataPoint[index])

    # this function takes the selected data point and calculate the respective class
    def __handlerCalculateClass(self):
        self.view.clearCounterfactual()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            # getting the datapoint
            auxiliarDataPoint = []
            for feature in self.model.features:
                if feature != 'Class':
                    content = self.initPointFeatures[feature].getContent()
                    auxiliarDataPoint.append(content['value'])

            self.chosenDataPoint = np.array(auxiliarDataPoint)

            # transforming the datapoint to predict its class
            self.transformedChosenDataPoint = self.model.transformDataPoint(
                self.chosenDataPoint)

            # predicting the datapoint class and showing its value
            self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(
                self.randomForestClassifier, [self.transformedChosenDataPoint])
            self.view.showOriginalClass(self.predictedOriginalClass[0])

    # this function takes the selected data point,
    # calculate the class,
    # generate the counterfactual explanation,
    # and calculate the class for the counterfactual explanation
    def __handlerGenerateCounterfactual(self):
        # updating cursor
        self.waitCursor()
        # cleaning the view
        self.view.clearCounterfactual()

        if self.__chosenDataset != InterfaceEnums.SelectDataset.DEFAULT.value:
            # showing the steps
            self.view.showCounterfactualStatus(
                InterfaceEnums.Status.STEP1.value)

            # getting the datapoint
            auxiliarDataPoint = []
            for feature in self.model.features:
                if feature != 'Class':
                    featureType = self.model.featuresInformations[feature]['featureType']

                    content = self.initPointFeatures[feature].getContent(
                    )

                    auxiliarDataPoint.append(content['value'])

                    if featureType is FeatureType.Binary:
                        notAllowedValue = content['notAllowedValue']
                        self.featuresConstraints[feature] = {
                            'featureType': featureType, 'notAllowedValue': notAllowedValue}

                    elif featureType is FeatureType.Discrete or featureType is FeatureType.Numeric:
                        selectedMinimum = self.model.transformSingleNumericalValue(
                            feature, content['minimumValue'])
                        selectedMaximum = self.model.transformSingleNumericalValue(
                            feature, content['maximumValue'])
                        self.featuresConstraints[feature] = {'featureType': featureType,
                                                             'selectedMinimum': selectedMinimum,
                                                             'selectedMaximum': selectedMaximum}

                    elif featureType is FeatureType.Categorical:
                        self.featuresConstraints[feature] = {'featureType': featureType,
                                                             'notAllowedValues': content['notAllowedValues']}

            self.chosenDataPoint = np.array(auxiliarDataPoint)

            # showing the steps
            self.view.showCounterfactualStatus(
                InterfaceEnums.Status.STEP2.value)
            # transforming the datapoint to predic its class and generate its counterfactual explanation
            self.transformedChosenDataPoint = self.model.transformDataPoint(
                self.chosenDataPoint)
            # predicting the datapoint class
            self.predictedOriginalClass = CounterfactualEngine.randomForestClassifierPredict(
                self.randomForestClassifier, [self.transformedChosenDataPoint])
            self.view.showOriginalClass(self.predictedOriginalClass[0])

            # running the counterfactual generation in another thread
            self.thread = QThread()
            self.worker = InterfaceWorker(self)
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

    # this function is used to update the comparison between the original datapoint and the counterfactual explanation
    def updateCounterfactualTable(self, counterfactualComparison):
        assert isinstance(counterfactualComparison, list)
        for item in counterfactualComparison:
            assert isinstance(item, list)
            assert len(item) == 3

        self.view.showCounterfactualComparison(counterfactualComparison)
