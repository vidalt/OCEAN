# Author: Moises Henrique Pereira
# this class handle the logic over the interface, interacting with model, view and worker
# also taking the selected dataset informations from model to send to counterfactual generator in worker class

import numpy as np

from .CounterfactualInterfaceView import CounterfactualInterfaceView
from .CounterfactualIterable.CounterfactualInterfaceControllerIterable import CounterfactualInterfaceControllerIterable

class CounterfactualInterfaceController():

    def __init__(self):
        self.view = CounterfactualInterfaceView()

        self.counterfactualInterfaceControllerIterable = CounterfactualInterfaceControllerIterable()

        # setar cada view em uma aba
        self.view.tabWidget.addTab(self.counterfactualInterfaceControllerIterable.view, 'Iterable Counterfactual')
