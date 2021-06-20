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
import pybedtools

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
    def setUp(self):
        self.path = "./test_data/test_mus_musculus_GRCm38_102.gtf"
    
    def test_three_prime(self):
        annotations = utr_reference.extract_feature(self.path, 'three_prime_utr')
        feature_list = annotations["feature"].unique().tolist()
        self.assertListEqual(feature_list, ['three_prime_utr'])
    
    def test_five_prime(self):
        annotations = utr_reference.extract_feature(self.path, 'five_prime_utr')
        feature_list = annotations["feature"].unique().tolist()
        self.assertListEqual(feature_list, ['five_prime_utr'])

class TestPrepareAnnotationsIndex(unittest.TestCase):
    def setUp(self):
        self.annotations = ['gene_id', 'score', 'strand', 'transcript_id']

    def test_columns(self):
        ann_columns, _ = utr_reference.prepare_annotations_index(self.annotations)
        self.assertListEqual(ann_columns, ['seqname', 'start', 'end', 'gene_id', 
            'score', 'strand', 'transcript_id'])
    
    def test_index(self):
        _, ann_index = utr_reference.prepare_annotations_index(self.annotations)
        self.assertListEqual(ann_index, [4, 5, 6, 7])

    def test_different_annotations(self):
        annotations = self.annotations + ['gene_name']
        ann_columns, ann_index = utr_reference.prepare_annotations_index(annotations)
        self.assertListEqual(ann_columns, ['seqname', 'start', 'end', 'gene_id', 
            'score', 'strand', 'transcript_id', 'gene_name'])

class TestReduceRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gtf_path = "../tests/test_data/test_mus_musculus_GRCm38_102.gtf"
        utr = utr_reference.extract_feature(gtf_path, 'three_prime_utr')
        cls.utr = utr.sort_values(by=['seqname', 'start'])  # sort for bed merge

    def test_number_of_features(self):
        annotations = ['gene_id', 'score', 'strand', 'transcript_id']
        columns, ann_idx = utr_reference.prepare_annotations_index(annotations)
        utr_bed = pybedtools.BedTool.from_dataframe(self.utr, columns=columns)
        utr_reduced = utr_reference.reduce_regions(utr_bed, annotations_index=ann_idx)
        self.assertEqual(utr_reduced.count(), 1016)
    
    def test_number_of_fields(self):
        annotations = ['gene_id', 'score', 'strand', 'transcript_id']
        columns, ann_idx = utr_reference.prepare_annotations_index(annotations)
        utr_bed = pybedtools.BedTool.from_dataframe(self.utr, columns=columns)
        utr_reduced = utr_reference.reduce_regions(utr_bed, annotations_index=ann_idx)
        self.assertEqual(utr_reduced.field_count(1), 8)
