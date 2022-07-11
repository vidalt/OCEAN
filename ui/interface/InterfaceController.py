# Author: Moises Henrique Pereira
# Handle the logic over the interface, interacting with model, view and worker
# Take the selected dataset informations from model to send to counterfactual
# generator in worker class

# Import ui functions
from ui.interface.InterfaceView import CounterfactualInterfaceView
from ui.static.StaticController import StaticController
from ui.iterative.IterativeController import IterativeController


class InterfaceController():

    def __init__(self, interfaceType='static'):
        self.view = CounterfactualInterfaceView()

        if interfaceType == 'static':
            self.counterfactualInterfaceControllerStatic = StaticController()
            # Set each view on a tab
            self.view.tabWidget.addTab(
                self.counterfactualInterfaceControllerStatic.view,
                'Static Counterfactual')
        elif interfaceType == 'iterative':
            self.counterfactualInterfaceControllerIterable = IterativeController()
            # Set each view on a tab
            self.view.tabWidget.addTab(
                                       self.counterfactualInterfaceControllerIterable.view,
                                       'Iterative Counterfactual')
