from .FinalIterationView import FinalIterationView

class FinalIterationController:

    def __init__(self, parent):
        self.view = FinalIterationView()

        self.__initializeView()


    # this function sets the initial configurations to the view
    def __initializeView(self):
        pass

    # this function sets the scenarios names to the view
    def setViewScenariosName(self, originalScenarioName, counterfactualScenarioName):
        self.view.setViewScenariosName(originalScenarioName, counterfactualScenarioName)

    # this function is used to update the label to comparison
    def updateComparisonLabel(self, text):
        assert text is not None

        self.view.updateComparisonLabel(text)

    # this function is used to update the original and the counterfactual class text
    def updateClasses(self, originalClass, counterfactualClass):
        assert originalClass is not None
        assert counterfactualClass is not None

        self.view.showOriginalClass(originalClass)
        self.view.showCounterfactualClass(counterfactualClass)

    # this function is used to update the comparison between the original datapoint and the counterfactual explanation
    def updateCounterfactualTable(self, counterfactualComparison):
        assert isinstance(counterfactualComparison, list)
        for item in counterfactualComparison:
            assert isinstance(item, list)
            assert len(item) == 3

        self.view.showCounterfactualComparison(counterfactualComparison)
