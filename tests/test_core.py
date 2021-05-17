import unittest
from utrcalling import core

class TestCore(unittest.TestCase):

    def test_core(self):
        r = core.addtest(2, 3)
        self.assertEqual(r, 5)

