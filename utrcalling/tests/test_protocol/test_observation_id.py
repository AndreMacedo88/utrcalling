import os
from pathlib import Path
import unittest

import pandas as pd

from utrcalling.core import protocol

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[1]))


class TestGetObservationIdTenx(unittest.TestCase):
    """Test get_observation_id() for 10x protocols with and without batch labeling."""

    @classmethod
    def setUpClass(cls):
        anno = pd.DataFrame({
            "CellBarcode": ["AAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBB"],
            "batch": [1, 2],
        })
        cls.prot_with_batch = protocol.create(
            platform="10x", version=3,
            annotations=anno,
            label_batches=True,
        )
        cls.prot_no_batch = protocol.create(
            platform="10x", version=3,
            annotations=anno,
            label_batches=False,
        )

    def test_with_batch_suffix(self):
        result = self.prot_with_batch.get_observation_id("AAAA", batch_idx="1")
        self.assertEqual(result, "AAAA-1")

    def test_with_batch_different_idx(self):
        result = self.prot_with_batch.get_observation_id("AAAA", batch_idx="2")
        self.assertEqual(result, "AAAA-2")

    def test_without_batch_ignores_idx(self):
        result = self.prot_no_batch.get_observation_id("AAAA", batch_idx="1")
        self.assertEqual(result, "AAAA")

    def test_with_batch_none_idx(self):
        result = self.prot_with_batch.get_observation_id("AAAA", batch_idx=None)
        self.assertEqual(result, "AAAA")

    def test_default_label_batches_is_true(self):
        """10x protocols default to label_batches=True when not specified."""
        anno = pd.DataFrame({
            "CellBarcode": ["AAAAAAAAAAAAAAAA"],
            "batch": [1],
        })
        prot = protocol.create(
            platform="10x", version=3,
            annotations=anno,
        )
        self.assertTrue(prot.label_batches)


class TestGetObservationIdSmartseq(unittest.TestCase):
    """Test get_observation_id() for SmartSeq3 protocol."""

    @classmethod
    def setUpClass(cls):
        anno = pd.DataFrame({
            "CellBarcode": ["AAAAAA", "BBBBBB"],
            "sample_id": ["sample1", "sample2"],
            "batch": [1, 2],
        })
        cls.prot = protocol.create(
            platform="smartseq", version=3,
            annotations=anno,
            alignments=["path/sample1.bam", "path/sample2.bam"],
        )

    def test_default_no_batch_suffix(self):
        result = self.prot.get_observation_id("AAAAAA", batch_idx="1")
        self.assertEqual(result, "AAAAAA")

    def test_label_batches_defaults_false(self):
        self.assertFalse(self.prot.label_batches)


class TestGetObservationIdCapseq(unittest.TestCase):
    """Test get_observation_id() for CapSeq protocol — always returns barcode directly."""

    @classmethod
    def setUpClass(cls):
        anno = pd.DataFrame({
            "CellBarcode": ["sampleA", "sampleB"],
            "sample_id": ["sampleA", "sampleB"],
            "batch": [1, 2],
        })
        cls.prot = protocol.create(
            platform="smart-seq_cap", version=3,
            annotations=anno,
            alignments=["path/sampleA.bam", "path/sampleB.bam"],
        )

    def test_returns_barcode_directly(self):
        result = self.prot.get_observation_id("my_sample.bam", batch_idx="1")
        self.assertEqual(result, "my_sample.bam")

    def test_ignores_batch_idx(self):
        result = self.prot.get_observation_id("sample_name", batch_idx=None)
        self.assertEqual(result, "sample_name")

    def test_label_batches_defaults_false(self):
        self.assertFalse(self.prot.label_batches)


class TestGetObservationIdCapseqReadName(unittest.TestCase):
    """Test get_observation_id() for CapSeqSmartseq3ReadName protocol."""

    @classmethod
    def setUpClass(cls):
        anno = pd.DataFrame({
            "CellBarcode": ["sampleX", "sampleY"],
            "sample_id": ["sampleX", "sampleY"],
            "batch": [1, 2],
        })
        cls.prot = protocol.create(
            platform="smart-seq_cap-readname", version=3,
            annotations=anno,
            alignments=["path/sampleX.bam", "path/sampleY.bam"],
        )

    def test_returns_barcode_directly(self):
        result = self.prot.get_observation_id("any_value", batch_idx="99")
        self.assertEqual(result, "any_value")


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
