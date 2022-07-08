# Author: Moises Henrique Pereira
# this class handle to instantiate the classes ensuring that the classes will behave properly considering
# the relative import used in some files

import sys
import os

# these three lines is needed to do the the correct reference
path0 = sys.path[0]
sys.path.insert(0,os.path.join(path0, 'src'))
sys.path.insert(1,os.path.join(path0, 'ui'))

from ui.CounterfactualInterface.CounterfactualInterfaceModel import CounterfactualInterfaceModel
from ui.CounterfactualInterface.CounterfactualInterfaceView import CounterfactualInterfaceView
from ui.CounterfactualInterface.CounterfactualInterfaceController import CounterfactualInterfaceController
from ui.CounterfactualInterface.ComboboxList.ComboboxListView import ComboboxListView
from ui.CounterfactualInterface.ComboboxList.ComboboxListController import ComboboxListController
from ui.CounterfactualInterface.DoubleRadioButton.DoubleRadioButtonView import DoubleRadioButtonView
from ui.CounterfactualInterface.DoubleRadioButton.DoubleRadioButtonController import DoubleRadioButtonController
from ui.CounterfactualInterface.LineEditMinimumMaximum.LineEditMinimumMaximumView import LineEditMinimumMaximumView
from ui.CounterfactualInterface.LineEditMinimumMaximum.LineEditMinimumMaximumController import LineEditMinimumMaximumController
from ui.CounterfactualInterface.Slider3Ranges.Slider3RangesController import Slider3RangesController
from ui.CounterfactualInterface.Slider3Ranges.Slider3RangesView import Slider3RangesView
from ui.CounterfactualInterface.Slider3Ranges.Slider import Slider

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