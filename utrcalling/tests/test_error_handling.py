"""Error handling and negative tests — verify graceful failures for invalid inputs."""

import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from utrcalling.core import protocol, utr_mapping, utr_reference

os.chdir(str(Path(__file__).resolve().parents[0]))


class TestInvalidProtocol(unittest.TestCase):
    """Test that invalid protocol/version combinations raise errors."""

    def test_unsupported_platform(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(platform="unsupported", version=1)

    def test_unsupported_10x_version(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(platform="10x", version=99)

    def test_unsupported_smartseq_version(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(platform="smartseq", version=1)

    def test_unsupported_capseq_version(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(platform="smart-seq_cap", version=1)

    def test_unsupported_5prime10x_version(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(platform="5prime10x", version=99)

    def test_invalid_merge_method(self):
        annotations = pd.DataFrame({"sample": ["test"]})
        with self.assertRaises(ValueError):
            protocol.create(
                platform="10x",
                version=3,
                merge_method="invalid_method",
                annotations=annotations,
            )

    def test_invalid_label_batches_string(self):
        annotations = pd.DataFrame({"sample": ["test"]})
        with self.assertRaises(ValueError):
            protocol.create(
                platform="10x",
                version=3,
                label_batches="maybe",
                annotations=annotations,
            )


class TestEmptyDataFrames(unittest.TestCase):
    """Test behavior with empty DataFrames."""

    def test_generate_utr_name_wrong_type(self):
        with self.assertRaises(TypeError):
            utr_reference.generate_utr_name("not_a_dataframe")

    def test_generate_utr_intervals_empty(self):
        empty = pd.DataFrame(
            columns=["chrom", "start", "end", "strand"],
        )
        empty.index.name = "utr_name"
        result = utr_mapping.generate_utr_intervals(empty)
        self.assertEqual(len(result), 0)

    def test_subset_reference_empty_reads(self):
        ref = pd.DataFrame(
            {"chrom": ["1"], "start": [100], "end": [200]},
            index=pd.Index(["UTR1"], name="utr_name"),
        )
        empty_reads = pd.DataFrame({"utr_name": pd.Categorical([])})
        result = utr_reference.subset_reference_data(empty_reads, ref)
        self.assertEqual(len(result), 0)

    def test_deduplicate_reads_empty(self):
        result = utr_mapping.deduplicate_reads(
            read_umis=[], read_info=[[], []]
        )
        self.assertListEqual(result, [[], []])


class TestAnnotationValidation(unittest.TestCase):
    """Test annotation column detection edge cases."""

    def test_multiple_barcode_columns_raises(self):
        annotations = pd.DataFrame({
            "CellBarcode": ["AAAA"],
            "Sample_Barcode": ["BBBB"],
        })
        with self.assertRaises(ValueError):
            protocol.create(
                platform="10x",
                version=3,
                annotations=annotations,
            )

    def test_multiple_batch_columns_raises(self):
        annotations = pd.DataFrame({
            "batch_1": [1],
            "batch_2": [2],
        })
        with self.assertRaises(ValueError):
            protocol.create(
                platform="10x",
                version=3,
                annotations=annotations,
            )


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
