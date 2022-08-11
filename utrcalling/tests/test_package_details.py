import unittest
from contextlib import contextmanager
from utrcalling.core import module_log

class TestCase(unittest.TestCase):
    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))


class TestImport(TestCase):

    def test_utrcalling_import(self):
        with self.assertNotRaises(ModuleNotFoundError):
            import utrcalling


class TestVersion(unittest.TestCase):

    def test_version(self):
        import utrcalling
        self.assertTrue(utrcalling.__version__)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
