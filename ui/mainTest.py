# Author: Moises Henrique Pereira
# this class handle to instantiate the classes
from ui.interface.CounterfactualInterfaceModel import CounterfactualInterfaceModel
from ui.interface.CounterfactualInterfaceView import CounterfactualInterfaceView
from ui.interface.CounterfactualInterfaceController import CounterfactualInterfaceController
from ui.interface.ComboboxList.ComboboxListView import ComboboxListView
from ui.interface.ComboboxList.ComboboxListController import ComboboxListController
from ui.interface.DoubleRadioButton.DoubleRadioButtonView import DoubleRadioButtonView
from ui.interface.DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ui.interface.LineEditMinimumMaximum.LineEditMinimumMaximumView import LineEditMinimumMaximumView
from ui.interface.LineEditMinimumMaximum.LineEditMinimumMaximumController import LineEditMinimumMaximumController
from ui.interface.Slider3Ranges.Slider3RangesController import Slider3RangesController
from ui.interface.Slider3Ranges.Slider3RangesView import Slider3RangesView
from ui.interface.Slider3Ranges.Slider import Slider

class StaticObjects:

    @staticmethod
    def staticCounterfactualInterfaceModel():
        return CounterfactualInterfaceModel()

    @staticmethod
    def staticCounterfactualInterfaceView():
        return CounterfactualInterfaceView()

    @staticmethod
    def staticCounterfactualInterfaceController():
        return CounterfactualInterfaceController()

    @staticmethod
    def staticCounterfactualInterfaceComboboxListView():
        return ComboboxListView()

    @staticmethod
    def staticCounterfactualInterfaceComboboxListController():
        return ComboboxListController()

    @staticmethod
    def staticCounterfactualInterfaceDoubleRadioButtonView():
        return DoubleRadioButtonView()

    @staticmethod
    def staticCounterfactualInterfaceDoubleRadioButtonController():
        return DoubleRadioButtonController()

    @staticmethod
    def staticCounterfactualInterfaceLineEditMinimumMaximumView():
        return LineEditMinimumMaximumView()

    @staticmethod
    def staticCounterfactualInterfaceLineEditMinimumMaximumController():
        return LineEditMinimumMaximumController()

    @staticmethod
    def staticCounterfactualInterfaceSlider3RangesController():
        return Slider3RangesController()

    @staticmethod
    def staticCounterfactualInterfaceSlider3RangesView():
        return Slider3RangesView()

    @staticmethod
    def staticCounterfactualInterfaceSlider():
        return Slider()
