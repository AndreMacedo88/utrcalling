import os
import argparse
import unittest
import logging as log_module
from utrcalling.core import module_log

# os.environ["MODIN_CPUS"] = "1"


def run_all_tests(verbosity, logging):
    """
    Runs all tests in the test module in one go.
    Parameters
    ----------
    verbosity : int
        The amount of verbosity output by the tests.
    """

    verbosity = int(verbosity)

    if not logging:
        # disable logging for tests
        log_module.disable(log_module.CRITICAL)

    # Gets this file directory, the tests dir
    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = unittest.TestLoader()  # Initiates the test loader
    # Adds all files starting in 'test' to the suite
    suite = loader.discover(start_dir=dir_path, pattern="test_*.py")
    # Runs the test suite
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    module_log.log_end_run("Finished testing")


def create_parser(subparsers):

    help = "Runs all tests developed for the utrcalling module."
    description = (
        f"{help} Verbosity:"
        "| 0 (quiet): Get the total numbers of tests executed and the global result. "
        "| 1 (moderate): Get the same as quiet plus a dot for every successful test or a "
        "F for every failure. "
        "| 2 (verbose): Get the help string of every test and the respective result."
    )

    parser = subparsers.add_parser(
        name="run-tests",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling run-tests --verbosity 1 --logging
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        required=False,
        help="Verbosity level for the test run.",
        default=2,
    )
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        required=False,
        help="Should the tool's logs be also shown while testing?",
        default=False,
    )
    parser.set_defaults(func=run_all_tests)
