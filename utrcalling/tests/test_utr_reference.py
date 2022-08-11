import os
from pathlib import Path
import unittest

import pandas as pd
import pybedtools
from utrcalling.core import utr_reference, module_log

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


class TestExtractFeature(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "./test_data/test_mus_musculus_GRCm38_102.gtf"

    def test_three_prime(self) -> None:
        annotations = utr_reference.extract_feature(
            self.path, 'three_prime_utr')
        feature_list = annotations["feature"].unique().tolist()
        self.assertListEqual(feature_list, ['three_prime_utr'])

    def test_five_prime(self) -> None:
        annotations = utr_reference.extract_feature(
            self.path, 'five_prime_utr')
        feature_list = annotations["feature"].unique().tolist()
        self.assertListEqual(feature_list, ['five_prime_utr'])


class TestPrepareAnnotationsIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.annotations = ['gene_id', 'score', 'strand', 'transcript_id']

    def test_columns(self) -> None:
        ann_columns, _ = utr_reference.prepare_annotations_index(
            self.annotations)
        self.assertListEqual(ann_columns, ['seqname', 'start', 'end', 'gene_id',
                                           'score', 'strand', 'transcript_id'])

    def test_index(self) -> None:
        _, ann_index = utr_reference.prepare_annotations_index(
            self.annotations)
        self.assertListEqual(ann_index, [4, 5, 6, 7])

    def test_different_annotations(self) -> None:
        annotations = self.annotations + ['gene_name']
        ann_columns, _ = utr_reference.prepare_annotations_index(
            annotations)
        self.assertListEqual(ann_columns, ['seqname', 'start', 'end', 'gene_id',
                                           'score', 'strand', 'transcript_id', 'gene_name'])


class TestReduceRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gtf_path = "../tests/test_data/test_mus_musculus_GRCm38_102.gtf"
        utr = utr_reference.extract_feature(gtf_path, 'three_prime_utr')
        cls.utr = utr.sort_values(
            by=['seqname', 'start'])  # sort for bed merge


class TestReduceRegionsStandard(TestReduceRegions):
    def setUp(self) -> None:
        annotations = ['gene_id', 'score', 'strand', 'transcript_id']
        columns, ann_idx = utr_reference.prepare_annotations_index(annotations)
        utr_bed = pybedtools.BedTool.from_dataframe(self.utr, columns=columns)
        self.utr_reduced = utr_reference.reduce_regions(
            utr_bed, annotations_index=ann_idx)

    def test_number_of_features(self) -> None:
        """
        The test GTF file was generated with 1000 3'UTR regions, plus 1000 random regions.
        In the end it turned out to have 1016 3'UTR regions.
        """
        self.assertEqual(self.utr_reduced.count(), 1016)

    def test_number_of_fields(self) -> None:
        """
        Number of fields in the annotation BED file should be equal to the optional 
        fields (in variable 'annotations') plus the mandatory fields 'seqname', 'start',
        'end' and finally the count field, that tell us how many features were merged 
        at each new feature (row).
        """
        self.assertEqual(self.utr_reduced.field_count(1), 7)

    def test_fields_are_proper(self) -> None:
        proper_fields = ['1', '7160892', '7161063', 'ENSMUSG00000051285', '.', '+',
                         'ENSMUST00000182675']
        self.assertListEqual(self.utr_reduced[0].fields, proper_fields)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.utr_reduced[0].strand, '+')


class TestReduceRegionsExtended(TestReduceRegions):
    def setUp(self) -> None:
        annotations = ['gene_id', 'score',
                       'strand', 'transcript_id', 'gene_name']
        columns, ann_idx = utr_reference.prepare_annotations_index(annotations)
        utr_bed = pybedtools.BedTool.from_dataframe(self.utr, columns=columns)
        self.utr_reduced = utr_reference.reduce_regions(
            utr_bed, annotations_index=ann_idx)

    def test_number_of_features(self) -> None:
        """
        The test GTF file was generated with 1000 3'UTR regions, plus 1000 random regions.
        In the end it turned out to have 1016 3'UTR regions.
        """
        self.assertEqual(self.utr_reduced.count(), 1016)

    def test_number_of_fields(self) -> None:
        """
        Number of fields in the annotation BED file should be equal to the optional 
        fields (in variable 'annotations') plus the mandatory fields 'seqname', 'start',
        'end' and finally the count field, that tell us how many features were merged 
        at each new feature (row).
        """
        self.assertEqual(self.utr_reduced.field_count(1), 8)

    def test_fields_are_proper(self) -> None:
        proper_fields = ['1', '7160892', '7161063', 'ENSMUSG00000051285', '.', '+',
                         'ENSMUST00000182675', 'Pcmtd1']
        self.assertListEqual(self.utr_reduced[0].fields, proper_fields)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.utr_reduced[0].strand, '+')


class TestGenerateUtrName(TestReduceRegionsExtended):
    def setUp(self) -> None:
        super().setUp()
        self.utr_reduced = utr_reference.generate_utr_name(self.utr_reduced)

    def test_names_len(self):
        self.assertEqual(self.utr_reduced.index[2], 'ENSMUSG00000025911_2')

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
        proper_fields = ['1', 7160892, 7161063, 'ENSMUSG00000051285', '.', '+',
                         'ENSMUST00000182675', 'Pcmtd1']
        self.assertListEqual(
            self.utr_reduced.iloc[0].values.tolist(), proper_fields)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.utr_reduced.iloc[0].strand, '+')


class TestCorrectColumnNames(TestGenerateUtrName):
    def setUp(self) -> None:
        super().setUp()
        self.utr_reduced = utr_reference.correct_column_names(self.utr_reduced)

    def test_column_names(self):
        proper_names = ['chrom', 'start', 'end', 'name', 'score', 'strand',
                        'transcript_id', 'gene_name']
        self.assertListEqual(self.utr_reduced.columns.tolist(), proper_names)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
