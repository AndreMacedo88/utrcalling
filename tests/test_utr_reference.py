import os
import sys
from pathlib import Path

# make sure main module gets imported
file_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(file_dir)
# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

import unittest

import pandas as pd

from utrcalling import utr_reference

class TestOpenGtf(unittest.TestCase):
    def setUp(self):
        self.path = "./test_data/test_mus_musculus_GRCm38_102.gtf"
        self.path_gz = "./test_data/test_mus_musculus_GRCm38_102.gtf.gz"

    
    def test_file_opened(self):
        file_ = utr_reference.open_gtf(self.path)
        self.assertIsInstance(file_, pd.DataFrame)
    
    def test_gz_file_opened(self):
        file_ = utr_reference.open_gtf(self.path_gz)
        self.assertIsInstance(file_, pd.DataFrame)

class TestExtractFeature(unittest.TestCase):
    pass