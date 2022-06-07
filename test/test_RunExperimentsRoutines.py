import unittest
# Import local functions to test
from src.RunExperimentsRoutines import trainModelAndSolveCounterFactuals


class test_RunExperimentsRoutines(unittest.TestCase):
    def test_emptyCallOftrainModelAndSolveCounterFactualsHasTypeError(self):
        """ trainModelAndSolveCounterFactuals requires two arguments """
        self.assertRaises(TypeError, trainModelAndSolveCounterFactuals)
