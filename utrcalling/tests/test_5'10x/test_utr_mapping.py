import os
from pathlib import Path
import unittest
import pandas as pd
from utrcalling.tests.test_data import synthetic_data_3tenx
from utrcalling.core import protocol, utr_mapping, module_log

# we change working dir to the base test dir so that relative paths are sane
os.chdir(str(Path(__file__).resolve().parents[1]))
