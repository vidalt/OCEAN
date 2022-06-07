import unittest
# Import local functions to test
from src.RunExperimentsRoutines import trainModelAndSolveCounterFactuals


class test_trainModelAndSolveCounterFactuals(unittest.TestCase):
    def test_emptyCallHasTypeError(self):
        """ trainModelAndSolveCounterFactuals requires two arguments """
        self.assertRaises(TypeError, trainModelAndSolveCounterFactuals)
