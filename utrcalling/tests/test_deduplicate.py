import unittest

from utrcalling.core.utr_mapping import deduplicate_reads


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


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
