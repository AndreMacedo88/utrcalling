import gzip
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utrcalling.core import module_log, utr_reference

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class TestOpenGtf(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "./test_data/test_mus_musculus_GRCm38_102.gtf"
        self.path_gz = "./test_data/test_mus_musculus_GRCm38_102.gtf.gz"

    def test_file_opened(self) -> None:
        file_ = utr_reference.open_gtf(self.path)
        self.assertIsInstance(file_, pd.DataFrame)

    def test_gz_file_opened(self) -> None:
        file_ = utr_reference.open_gtf(self.path_gz)
        self.assertIsInstance(file_, pd.DataFrame)


# Lines that test the reading of GTF files: a comment, missing scores and
# frames, an attribute given twice, a value with spaces, an unquoted value,
# and attributes that only some rows have.
TRICKY_GTF = (
    "#!genome-build GRCm39\n"
    '1\tensembl\tgene\t100\t900\t.\t+\t.\tgene_id "G1"; gene_name "Abc";\n'
    '1\tensembl\tthree_prime_utr\t800\t900\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; '
    'gene_name "Abc"; tag "basic"; tag "CCDS"; '
    'transcript_support_level "1 (assigned to previous version 5)";\n'
    'X\thavana\tthree_prime_utr\t50\t60\t3.5\t-\t0\tgene_id "G2"; transcript_id "T2"; '
    "exon_number 2;\n"
    '2\tensembl\tCDS\t10\t20\t.\t+\t2\tgene_id "G3"; transcript_id "T3";\n'
)


class TestReadGtf(unittest.TestCase):
    """read_gtf reads GTF files as gtfparse 1.x did (checked against gtfparse 1.3.0)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.path = self.tmp / "tricky.gtf"
        self.path.write_text(TRICKY_GTF)

    def test_columns_and_attributes(self) -> None:
        table = utr_reference.read_gtf(str(self.path))
        expected = pd.DataFrame(
            {
                "seqname": ["1", "1", "X", "2"],
                "source": ["ensembl", "ensembl", "havana", "ensembl"],
                "feature": ["gene", "three_prime_utr", "three_prime_utr", "CDS"],
                "start": np.array([100, 800, 50, 10], dtype=np.int64),
                "end": np.array([900, 900, 60, 20], dtype=np.int64),
                "score": np.array([np.nan, np.nan, 3.5, np.nan], dtype=np.float32),
                "strand": ["+", "+", "-", "+"],
                "frame": ["0", "0", "0", "2"],  # "." is 0
                "gene_id": ["G1", "G1", "G2", "G3"],
                "gene_name": ["Abc", "Abc", "", ""],
                "transcript_id": ["", "T1", "T2", "T3"],
                "tag": ["", "basic,CCDS", "", ""],  # given twice: both, joined
                "transcript_support_level": ["", "1", "", ""],  # the first word
                "exon_number": ["", "", "2", ""],  # unquoted
            }
        )
        pd.testing.assert_frame_equal(table, expected)

    def test_one_feature(self) -> None:
        # Gzipped too. Only the attributes of the feature's rows are columns.
        gz = self.tmp / "tricky.gtf.gz"
        with gzip.open(gz, "wt") as gtf:
            gtf.write(TRICKY_GTF)
        table = utr_reference.read_gtf(str(gz), "three_prime_utr")
        self.assertListEqual(table.index.tolist(), [0, 1])
        self.assertListEqual(table.transcript_id.tolist(), ["T1", "T2"])
        self.assertNotIn("CDS", table.feature.tolist())
        self.assertListEqual(
            table.columns.tolist()[8:],
            [
                "gene_id",
                "transcript_id",
                "gene_name",
                "tag",
                "transcript_support_level",
                "exon_number",
            ],
        )

    def test_broken_quotes_of_some_ensembl_releases(self) -> None:
        # As gtfparse repairs them: ';"' and ';-' inside values.
        self.path.write_text(
            '1\tensembl\tthree_prime_utr\t1\t9\t.\t+\t.\tgene_id "G;"; gene_name "A;-1";\n'
        )
        table = utr_reference.read_gtf(str(self.path))
        self.assertListEqual([table.gene_id[0], table.gene_name[0]], ["G", "A-1"])


class TestExtractFeature(unittest.TestCase):
    def setUp(self) -> None:
        path = "./test_data/test_mus_musculus_GRCm38_102.gtf"
        self.gtf_df = utr_reference.open_gtf(path)

    def test_three_prime_unique(self) -> None:
        annotations = utr_reference.extract_feature(self.gtf_df, "three_prime_utr")
        feature_list = annotations["feature"].unique().tolist()
        self.assertListEqual(feature_list, ["three_prime_utr"])

    def test_three_prime_number(self) -> None:
        annotations = utr_reference.extract_feature(self.gtf_df, "three_prime_utr")
        expected_number_3utrs = 1032
        self.assertEqual(expected_number_3utrs, len(annotations["feature"]))

    def test_five_prime(self) -> None:
        annotations = utr_reference.extract_feature(self.gtf_df, "five_prime_utr")
        feature_list = annotations["feature"].unique().tolist()
        self.assertListEqual(feature_list, ["five_prime_utr"])


class TestPrepareAnnotationsIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.annotations = ["gene_id", "score", "strand", "transcript_id"]

    def test_columns(self) -> None:
        ann_columns, _ = utr_reference.prepare_annotations_index(self.annotations)
        self.assertListEqual(
            ann_columns,
            ["seqname", "start", "end", "gene_id", "score", "strand", "transcript_id"],
        )

    def test_index(self) -> None:
        _, ann_index = utr_reference.prepare_annotations_index(self.annotations)
        self.assertListEqual(ann_index, [4, 5, 6, 7])

    def test_different_annotations(self) -> None:
        annotations = [*self.annotations, "gene_name"]
        ann_columns, _ = utr_reference.prepare_annotations_index(annotations)
        self.assertListEqual(
            ann_columns,
            [
                "seqname",
                "start",
                "end",
                "gene_id",
                "score",
                "strand",
                "transcript_id",
                "gene_name",
            ],
        )


class TestReduceRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gtf_path = "../tests/test_data/test_mus_musculus_GRCm38_102.gtf"
        cls.utr = utr_reference.read_gtf(gtf_path, "three_prime_utr")
        cls.utr = cls.utr.sort_values(by=["seqname", "start"])  # sorted, for merging


class TestReduceRegionsStandard(TestReduceRegions):
    def setUp(self) -> None:
        annotations = ["gene_id", "score", "strand", "transcript_id"]
        columns, _ = utr_reference.prepare_annotations_index(annotations)
        self.utr_reduced = utr_reference.merge_regions(self.utr[columns])

    def test_number_of_features(self) -> None:
        """
        The test GTF file was generated with 1000 3'UTR regions, plus 1000 random regions.
        In the end it turned out to have 1016 3'UTR regions.
        """
        self.assertEqual(len(self.utr_reduced), 1016)

    def test_number_of_fields(self) -> None:
        """The merged regions have chrom, start and end, then the annotations."""
        self.assertEqual(self.utr_reduced.shape[1], 7)

    def test_fields_are_proper(self) -> None:
        proper_fields = [
            "1",
            7160892,
            7161063,
            "ENSMUSG00000051285",
            ".",
            "+",
            "ENSMUST00000182675",
        ]
        self.assertListEqual(self.utr_reduced.iloc[0].tolist(), proper_fields)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.utr_reduced.iloc[0].strand, "+")


class TestReduceRegionsExtended(TestReduceRegions):
    def setUp(self) -> None:
        annotations = ["gene_id", "score", "strand", "transcript_id", "gene_name"]
        columns, _ = utr_reference.prepare_annotations_index(annotations)
        self.utr_reduced = utr_reference.merge_regions(self.utr[columns])

    def test_number_of_features(self) -> None:
        """
        The test GTF file was generated with 1000 3'UTR regions, plus 1000 random regions.
        In the end it turned out to have 1016 3'UTR regions.
        """
        self.assertEqual(len(self.utr_reduced), 1016)

    def test_number_of_fields(self) -> None:
        """The merged regions have chrom, start and end, then the annotations."""
        self.assertEqual(self.utr_reduced.shape[1], 8)

    def test_fields_are_proper(self) -> None:
        proper_fields = [
            "1",
            7160892,
            7161063,
            "ENSMUSG00000051285",
            ".",
            "+",
            "ENSMUST00000182675",
            "Pcmtd1",
        ]
        self.assertListEqual(self.utr_reduced.iloc[0].tolist(), proper_fields)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.utr_reduced.iloc[0].strand, "+")


def _regions(*rows):
    """Regions as merge_regions takes them: chrom, start, end, gene, score, strand, transcript."""
    return pd.DataFrame(
        rows, columns=["seqname", "start", "end", "gene_id", "score", "strand", "transcript_id"]
    )


class TestMergeRegions(unittest.TestCase):
    """merge_regions merges as `bedtools merge -s -c 4,5,6,7 -o distinct` (v2.31) did."""

    def _merged(self, *rows):
        merged = utr_reference.merge_regions(_regions(*rows))
        return [tuple(row) for row in merged.itertuples(index=False)]

    def test_overlapping_and_touching_regions_join(self):
        # 150-250 overlaps 100-200; 250-300 touches it (BED ends are exclusive);
        # 301-400 begins 1 nt after: a region of its own.
        merged = self._merged(
            ("1", 100, 200, "B", ".", "+", "T2"),
            ("1", 150, 250, "A", ".", "+", "T1"),
            ("1", 250, 300, "C", ".", "+", "T3"),
            ("1", 301, 400, "D", ".", "+", "T4"),
        )
        self.assertListEqual(
            merged,
            [(1, 100, 300, "A,B,C", ".", "+", "T1,T2,T3"), (1, 301, 400, "D", ".", "+", "T4")],
        )

    def test_strands_merge_apart_and_regions_come_by_start(self):
        # Sorted as utrcalling sorts them: by chromosome name as text, then start.
        # The minus-strand regions lie inside the plus-strand one, which still
        # joins across them. Regions come by start, and as given at equal starts.
        merged = self._merged(
            ("10", 5, 9, "Q", ".", "-", "T5"),
            ("2", 100, 200, "B", ".", "+", "T2"),
            ("2", 100, 150, "Y", ".", "-", "T8"),
            ("2", 120, 180, "Z", ".", "-", "T9"),
            ("2", 150, 250, "A", ".", "+", "T1"),
            ("2", 400, 450, "A", ".", "+", "T1"),
        )
        self.assertListEqual(
            merged,
            [
                (10, 5, 9, "Q", ".", "-", "T5"),
                (2, 100, 250, "A,B", ".", "+", "T1,T2"),
                (2, 100, 180, "Y,Z", ".", "-", "T8,T9"),
                (2, 400, 450, "A", ".", "+", "T1"),
            ],
        )

    def test_distinct_values_are_sorted_and_missing_ones_are_dots(self):
        merged = self._merged(
            ("1", 10, 20, "G2", None, "+", "T1"),
            ("1", 15, 30, "G10", None, "+", "T1"),
        )
        self.assertListEqual(merged, [(1, 10, 30, "G10,G2", ".", "+", "T1")])

    def test_zero_length_regions(self):
        # bedtools widens a region whose start equals its end by 1 nt on each
        # side: alone, it keeps its coordinates; merged, the widened ones count,
        # from the first region's start.
        merged = self._merged(
            ("1", 50, 50, "A", ".", "+", "T1"),
            ("1", 80, 80, "B", ".", "+", "T2"),
            ("1", 80, 85, "B", ".", "+", "T3"),
            ("1", 120, 130, "C", ".", "+", "T4"),
            ("1", 120, 120, "C", ".", "+", "T5"),
            ("1", 130, 130, "C", ".", "+", "T6"),
            ("1", 200, 210, "D", ".", "+", "T7"),
            ("1", 211, 211, "D", ".", "+", "T8"),
        )
        self.assertListEqual(
            [(start, end) for _, start, end, *_ in merged],
            # 211 widened to 210-212 touches 200-210.
            [(50, 50), (79, 85), (120, 131), (200, 212)],
        )

    def test_records_set_aside_start_the_next_region_by_position(self):
        # The minus-strand region sets aside the plus-strand records that follow
        # it; the next region begins with the lowest of them, the zero-length
        # one widened to 99, as bedtools 2.31 merges them. Without the minus-
        # strand region, the first record read, at 100, begins it.
        plus = [("1", 100, 200, "A", ".", "+", "T1"), ("1", 100, 100, "B", ".", "+", "T2")]
        merged = self._merged(("1", 50, 300, "M", ".", "-", "T3"), *plus)
        self.assertListEqual(
            merged, [(1, 50, 300, "M", ".", "-", "T3"), (1, 99, 200, "A,B", ".", "+", "T1,T2")]
        )
        self.assertListEqual(self._merged(*plus), [(1, 100, 200, "A,B", ".", "+", "T1,T2")])

    def test_records_without_a_strand_are_left_out(self):
        # Before a region, within it and after it.
        merged = self._merged(
            ("1", 10, 20, "U", ".", ".", "T0"),
            ("1", 15, 30, "A", ".", "+", "T1"),
            ("1", 20, 25, "W", ".", ".", "T3"),
            ("1", 28, 35, "B", ".", "+", "T4"),
            ("1", 40, 50, "V", ".", ".", "T2"),
        )
        self.assertListEqual(merged, [(1, 15, 35, "A,B", ".", "+", "T1,T4")])

    def test_columns_are_read_back_as_from_a_bed_file(self):
        merged = utr_reference.merge_regions(
            _regions(("X", 1, 9, "G", ".", "+", "T"), ("1", 1, 9, "G", ".", "+", "T"))
        )
        self.assertListEqual(
            merged.columns.tolist(),
            ["chrom", "start", "end", "name", "score", "strand", "thickStart"],
        )
        self.assertListEqual(merged.chrom.tolist(), ["X", "1"])
        self.assertEqual(merged.start.dtype, np.int64)


class TestGenerateUtrName(TestReduceRegionsExtended):
    def setUp(self) -> None:
        super().setUp()
        self.utr_reduced = utr_reference.generate_utr_name(self.utr_reduced)

    def test_names_len(self):
        self.assertEqual(self.utr_reduced.index[2], "ENSMUSG00000025911_2")

    def test_number_of_features(self) -> None:
        """
        The test GTF file was generated with 1000 3'UTR regions, plus 1000 random regions.
        In the end it turned out to have 1016 3'UTR regions.
        """
        self.assertEqual(self.utr_reduced.shape[0], 1016)

    def test_number_of_fields(self) -> None:
        """
        Number of fields in the annotation BED file should be equal to the optional
        fields (in variable 'annotations') plus the mandatory fields 'seqname', 'start',
        'end' and finally the count field, that tell us how many features were merged
        at each new feature (row).
        """
        self.assertEqual(self.utr_reduced.shape[1], 8)

    def test_fields_are_proper(self) -> None:
        proper_fields = [
            "1",
            7160892,
            7161063,
            "ENSMUSG00000051285",
            ".",
            "+",
            "ENSMUST00000182675",
            "Pcmtd1",
        ]
        self.assertListEqual(self.utr_reduced.iloc[0].values.tolist(), proper_fields)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.utr_reduced.iloc[0].strand, "+")


class TestCorrectColumnNames(TestGenerateUtrName):
    def setUp(self) -> None:
        super().setUp()
        self.utr_reduced = utr_reference.correct_column_names(self.utr_reduced)

    def test_column_names(self):
        proper_names = [
            "chrom",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "transcript_id",
            "gene_name",
        ]
        self.assertListEqual(self.utr_reduced.columns.tolist(), proper_names)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
