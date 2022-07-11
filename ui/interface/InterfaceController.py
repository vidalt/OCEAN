# Author: Moises Henrique Pereira

# Import ui functions
from ui.interface.InterfaceView import CounterfactualInterfaceView
from ui.static.StaticController import StaticController
from ui.iterative.IterativeController import IterativeController


class InterfaceController():
    """ Handle the logic over the interface.

    Interact with model, view and worker.
    Take the selected dataset informations from model to send to counterfactual
    generator in worker class.
    """

    def __init__(self, interfaceType='static'):
        self.view = CounterfactualInterfaceView()

        if interfaceType == 'static':
            self.staticController = StaticController()
            # Set each view on a tab
            self.view.tabWidget.addTab(
                self.staticController.view, 'Static Counterfactual')
        elif interfaceType == 'iterative':
            self.iterativeController = IterativeController()
            # Set each view on a tab
            self.view.tabWidget.addTab(
                self.iterativeController.view, 'Iterative Counterfactual')
