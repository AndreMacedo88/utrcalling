import unittest
import os


def run_all_tests(verbosity=2):
    """
    Runs all tests in the test module in one go.
    Parameters
    ----------
    verbosity : int
        The amount of verbosity output by the tests.
    """
    verbosity = int(verbosity)
    # Gets this file directory, the tests dir
    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = unittest.TestLoader()  # Initiates the test loader
    # Adds all files starting in 'test' to the suite
    suite = loader.discover(start_dir=dir_path, pattern='test*.py')
    unittest.TextTestRunner(verbosity=verbosity).run(
        suite)  # Runs the test suite
