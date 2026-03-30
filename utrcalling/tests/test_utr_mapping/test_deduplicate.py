import unittest

from utrcalling.core.utr_mapping import deduplicate_reads, deduplicate_reads_keep_max


class TestDeduplicateReads(unittest.TestCase):

    def test_basic_dedup(self):
        """Same barcode+UMI pair should keep only the first occurrence."""
        read_umis = ["UMI1", "UMI1", "UMI2"]
        read_info = [
            ["BC1", "BC1", "BC1"],  # barcodes
            [100, 200, 300],  # sizes
        ]
        result = deduplicate_reads(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC1"])
        self.assertEqual(result[1], [100, 300])

    def test_different_barcodes_same_umi(self):
        """Same UMI in different barcodes should be kept in both."""
        read_umis = ["UMI1", "UMI1"]
        read_info = [["BC1", "BC2"], [100, 200]]
        result = deduplicate_reads(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC2"])
        self.assertEqual(result[1], [100, 200])

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = deduplicate_reads([], [[], []])
        self.assertEqual(result, [[], []])

    def test_single_read(self):
        """Single read should pass through unchanged."""
        read_umis = ["UMI1"]
        read_info = [["BC1"], [100]]
        result = deduplicate_reads(read_umis, read_info)
        self.assertEqual(result[0], ["BC1"])
        self.assertEqual(result[1], [100])

    def test_all_duplicates(self):
        """All identical barcode+UMI should keep only the first."""
        read_umis = ["UMI1", "UMI1", "UMI1"]
        read_info = [["BC1", "BC1", "BC1"], [100, 200, 300]]
        result = deduplicate_reads(read_umis, read_info)
        self.assertEqual(result[0], ["BC1"])
        self.assertEqual(result[1], [100])

    def test_preserves_first_occurrence_order(self):
        """First occurrence of each unique (barcode, UMI) pair should be kept."""
        read_umis = ["UMI2", "UMI1", "UMI2", "UMI1"]
        read_info = [["BC1", "BC1", "BC1", "BC1"], [10, 20, 30, 40]]
        result = deduplicate_reads(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC1"])
        self.assertEqual(result[1], [10, 20])

    def test_multiple_barcodes_multiple_umis(self):
        """Complex case with multiple barcodes and UMIs."""
        read_umis = ["U1", "U2", "U1", "U1", "U2", "U3"]
        read_info = [
            ["A", "A", "A", "B", "B", "B"],  # barcodes
            [1, 2, 3, 4, 5, 6],  # sizes
        ]
        result = deduplicate_reads(read_umis, read_info)
        # A-U1 kept (idx 0), A-U2 kept (idx 1), A-U1 dup, B-U1 kept (idx 3),
        # B-U2 kept (idx 4), B-U3 kept (idx 5)
        self.assertEqual(result[0], ["A", "A", "B", "B", "B"])
        self.assertEqual(result[1], [1, 2, 4, 5, 6])

    def test_no_duplicates(self):
        """When all reads are unique, all should pass through."""
        read_umis = ["U1", "U2", "U3"]
        read_info = [["BC1", "BC2", "BC3"], [100, 200, 300]]
        result = deduplicate_reads(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC2", "BC3"])
        self.assertEqual(result[1], [100, 200, 300])


class TestDeduplicateReadsKeepMax(unittest.TestCase):
    """Tests for deduplicate_reads_keep_max — keeps largest UTR size per (BC, UMI)."""

    def test_basic_keep_max(self):
        """Same barcode+UMI pair should keep the one with largest size."""
        read_umis = ["UMI1", "UMI1", "UMI2"]
        read_info = [
            ["BC1", "BC1", "BC1"],  # barcodes
            [100, 200, 300],  # sizes — UMI1 max is 200
        ]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC1"])
        self.assertEqual(result[1], [200, 300])

    def test_different_barcodes_same_umi(self):
        """Same UMI in different barcodes: independent maximization."""
        read_umis = ["UMI1", "UMI1"]
        read_info = [["BC1", "BC2"], [100, 200]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC2"])
        self.assertEqual(result[1], [100, 200])

    def test_empty_input(self):
        result = deduplicate_reads_keep_max([], [[], []])
        self.assertEqual(result, [[], []])

    def test_single_read(self):
        read_umis = ["UMI1"]
        read_info = [["BC1"], [100]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[0], ["BC1"])
        self.assertEqual(result[1], [100])

    def test_all_same_umi_keeps_max(self):
        """All identical barcode+UMI: keep the entry with largest size."""
        read_umis = ["UMI1", "UMI1", "UMI1"]
        read_info = [["BC1", "BC1", "BC1"], [100, 300, 200]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[0], ["BC1"])
        self.assertEqual(result[1], [300])

    def test_all_same_size(self):
        """When all sizes are equal, still collapses to one per (BC, UMI)."""
        read_umis = ["UMI1", "UMI1", "UMI1"]
        read_info = [["BC1", "BC1", "BC1"], [100, 100, 100]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[1], [100])

    def test_complex_case(self):
        """Multiple barcodes and UMIs with varying sizes."""
        read_umis = ["U1", "U2", "U1", "U1", "U2", "U3"]
        read_info = [
            ["A", "A", "A", "B", "B", "B"],
            [10, 20, 30, 40, 50, 60],
        ]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        # A-U1: max(10,30)=30 → idx 2
        # A-U2: 20 → idx 1
        # B-U1: 40 → idx 3
        # B-U2: 50 → idx 4
        # B-U3: 60 → idx 5
        self.assertEqual(result[0], ["A", "A", "B", "B", "B"])
        self.assertEqual(result[1], [20, 30, 40, 50, 60])

    def test_no_duplicates(self):
        """All unique (BC, UMI) pairs: all pass through."""
        read_umis = ["U1", "U2", "U3"]
        read_info = [["BC1", "BC2", "BC3"], [100, 200, 300]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[0], ["BC1", "BC2", "BC3"])
        self.assertEqual(result[1], [100, 200, 300])

    def test_max_at_different_positions(self):
        """Max size can be at any position (first, middle, last)."""
        # Max at last position
        read_umis = ["U1", "U1", "U1"]
        read_info = [["A", "A", "A"], [10, 20, 30]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[1], [30])

        # Max at first position
        read_umis = ["U1", "U1", "U1"]
        read_info = [["A", "A", "A"], [30, 20, 10]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[1], [30])

        # Max at middle position
        read_umis = ["U1", "U1", "U1"]
        read_info = [["A", "A", "A"], [10, 30, 20]]
        result = deduplicate_reads_keep_max(read_umis, read_info)
        self.assertEqual(result[1], [30])


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
