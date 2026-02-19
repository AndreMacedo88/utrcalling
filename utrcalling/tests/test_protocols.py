import os
from pathlib import Path
import unittest
import copy

import pandas as pd

from utrcalling.core import protocol, utr_mapping, process_mapped_reads, module_log

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class Test10x(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        alignment_1 = "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam"
        alignment_2 = "./test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam"
        cls.protocol = protocol.create(
            platform="10x",
            version=2,
            alignments=[alignment_1, alignment_2],
            annotations="./test_data/smartseq3_anno.csv",
            merge_method="intersection",
        )
        cls.mappings = [
            utr_mapping.map_reads_to_utr(
                alignment, intervals, cls.protocol, read="first",
                batch_idx=str(i + 1)
            )
            for i, alignment in enumerate([alignment_1, alignment_2])
        ]


class TestFilterUtrSizes(Test10x):
    def setUp(self) -> None:
        merged = process_mapped_reads.merge_samples_utr_sizes(
            copy.deepcopy(self.mappings), method=self.protocol.merge_method
        )

        # modify the data to check for successful filtering.
        # Replace utr_size values for ENSMUSG00000001036 with alternating 1 and 798
        mask = merged["utr_name"] == "ENSMUSG00000001036"
        utr_rows = merged[mask].index
        new_sizes = [1, 798, 1, 798, 1, 798]
        for i, idx in enumerate(utr_rows):
            merged.at[idx, "utr_size"] = new_sizes[i]

        self.filtered = self.protocol.filter_utr_sizes(merged, threshold=1)

    def test_number_of_features(self) -> None:
        n_unique_utrs = len(self.filtered["utr_name"].unique())
        self.assertEqual(n_unique_utrs, 63)

    def test_is_filtered(self):
        utr_data = self.filtered[
            self.filtered["utr_name"] == "ENSMUSG00000001036"
        ]
        self.assertListEqual(
            utr_data["observation_id"].tolist(),
            [
                "CTAATGGCATTCGACA-1",
                "CCCAGTTTCGCTTGTC-1",
                "TATTACCAGGATTCGG-2",
            ],
        )
        self.assertListEqual(
            utr_data["utr_size"].tolist(),
            [798, 798, 798],
        )


class TestSmartSeqCapPhosphoReadname(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_parquet("./test_data/test_UTR_reference_5prime.parquet")
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        cls.alignment = "./test_data/test_CapPhospho_genome_102_coordsort.bam"
        cls.protocol = protocol.create(
            platform="smart-seq_cap-readname",
            version=3,
            alignments=cls.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        cls.mappings = utr_mapping.map_reads_to_utr(
            cls.alignment, intervals, cls.protocol, read="first"
        )


class TestFilterUtrSizesSmartSeqCap(TestSmartSeqCapPhosphoReadname):
    def setUp(self) -> None:
        mappings = self.mappings.copy()
        # modify the data to check for successful filtering. Original was [44, 23, 19].
        # Replace the second utr_size with 1 for ENSMUSG00000023905
        mask = mappings["utr_name"] == "ENSMUSG00000023905"
        utr_rows = mappings[mask].index
        new_sizes = [44, 1, 19]
        for i, idx in enumerate(utr_rows):
            mappings.at[idx, "utr_size"] = new_sizes[i]

        self.filtered = self.protocol.filter_utr_sizes(mappings)

    def test_number_of_features(self) -> None:
        # For 5'UTRs we don't want filtering by default — all rows stay
        total_rows = len(self.filtered)
        self.assertEqual(total_rows, len(self.mappings))

    def test_is_filtered(self):
        # For 5'UTRs we don't want filtering by default — all rows stay including size=1
        utr_data = self.filtered[
            self.filtered["utr_name"] == "ENSMUSG00000023905"
        ]
        self.assertListEqual(
            utr_data["observation_id"].tolist(),
            [self.alignment] * 3,
        )
        self.assertListEqual(
            utr_data["utr_size"].tolist(),
            [44, 1, 19],
        )


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
