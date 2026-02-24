from utrcalling.core import core, protocol, utr_mapping
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

os.chdir(str(Path(__file__).resolve().parents[0]))


class TestFilterMappedReadsBase(unittest.TestCase):
    """Base class that sets up mapped reads for filter_mapped_reads tests."""

    @classmethod
    def setUpClass(cls):
        cls.utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        cls.annotations = pd.read_csv("./test_data/vivo_anno.csv")
        alignment = "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam"
        cls.protocol_ = protocol.create(
            platform="10x",
            version=2,
            annotations=cls.annotations,
            merge_method="intersection",
        )
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        cls.mapped = utr_mapping.map_reads_to_utr(
            bam=alignment, intervals=intervals, protocol=cls.protocol_, read="first",
            batch_idx="1",
        )


class TestFilterMappedReadsReturnType(TestFilterMappedReadsBase):

    def test_returns_tuple_of_two_dataframes(self):
        """filter_mapped_reads should return (filtered_reads, utr_reference_subset)."""
        result = core.filter_mapped_reads(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)

    def test_utr_reference_subset_matches_read_data(self):
        """Reference subset should only contain UTRs present in reads."""
        filtered, ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
        )
        read_utrs = set(filtered["utr_name"].unique())
        ref_utrs = set(ref_subset.index)
        # All UTRs in filtered reads should be in the reference subset
        self.assertTrue(read_utrs.issubset(ref_utrs))

    def test_utr_sizes_below_threshold_removed(self):
        """UTR sizes <=1 should be filtered out."""
        filtered, _ = core.filter_mapped_reads(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
        )
        self.assertTrue((filtered["utr_size"] > 1).all())

    def test_filtered_has_expected_columns(self):
        """Filtered DataFrame should have utr_name, observation_id, utr_size."""
        filtered, _ = core.filter_mapped_reads(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
        )
        self.assertListEqual(
            sorted(filtered.columns.tolist()),
            ["observation_id", "utr_name", "utr_size"],
        )

    def test_with_barcode_whitelist(self):
        """When whitelist is set, only matching barcodes should remain."""
        filtered, _ = core.filter_mapped_reads(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
        )
        if self.protocol_.barcode_whitelist is not None:
            obs_ids = set(filtered["observation_id"].unique())
            whitelist = set(self.protocol_.barcode_whitelist)
            # All observations should be in the whitelist (exact or fuzzy-matched)
            self.assertTrue(obs_ids.issubset(whitelist))


class TestFilterMappedReadsNoWhitelist(unittest.TestCase):

    def test_without_barcode_whitelist(self):
        """Without a whitelist, all observations should pass through."""
        utr_regions = pd.DataFrame({
            "chrom": ["1"], "start": [100], "end": [200],
            "name": ["G1"], "score": ["."], "strand": ["+"],
            "transcript_id": ["T1"], "gene_name": ["Gene1"],
        }, index=pd.Index(["UTR1"], name="utr_name"))

        mapped = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1", "UTR1"]),
            "observation_id": pd.Categorical(["CELL1", "CELL2"]),
            "utr_size": np.array([50, 100], dtype=np.int32),
        })

        protocol_ = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=False,
            annotations=pd.DataFrame({"Sample_ID": ["S1"]}),
            alignments=["dummy.bam"],
        )
        # Ensure no whitelist
        self.assertIsNone(protocol_.barcode_whitelist)

        filtered, ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=mapped,
            utr_reference_regions=utr_regions,
            protocol=protocol_,
        )
        # filter_utr_sizes with threshold=1 keeps sizes > 1
        # Both 50 and 100 are > 1 so both should pass
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
