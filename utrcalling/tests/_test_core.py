import unittest
from utrcalling.main_module import core, module_log


class TestCore(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
