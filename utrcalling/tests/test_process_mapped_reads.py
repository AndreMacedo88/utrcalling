from test_data import synthetic_data
from utrcalling.core import process_mapped_reads, protocol, utr_mapping, module_log, helpers

import os
from pathlib import Path
import unittest

import pandas as pd
import copy

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class TestPeakCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet(
            './test_data/test_UTR_reference.parquet')
        intervals = utr_mapping.generate_utr_intervals(
            utr_regions=cls.utr_regions)
        alignment_1 = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        alignment_2 = './test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam'
        cls.annotations = pd.read_csv("./test_data/vivo_anno.csv")
        cls.protocol_ = protocol.create(
            platform='10x',
            version=2,
            annotations=cls.annotations,
            merge_method="intersection",
        )
        cls.protocol_union = protocol.create(
            platform='10x',
            version=2,
            annotations=cls.annotations,
            merge_method="union",
        )
        cls.mappings = [
            utr_mapping.map_reads_to_utr(
                bam=alignment,
                intervals=intervals,
                protocol=cls.protocol_,
                read='first'
            ) for alignment in [alignment_1, alignment_2]
        ]


class TestMergeSamplesUtrSizesIntersection(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=self.mappings,
            method=self.protocol_.merge_method
        )

    def test_details(self):
        details_true = {
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1',
                              'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2',
                              'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2',
                              'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'],
            'utr_sizes_from_reads': [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000025917"].to_dict(),
            details_true
        )

    def test_length(self):
        self.assertEqual(self.merged.shape[0], 63)


class TestMergeSamplesUtrSizesUnion(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(self.mappings),
            method=self.protocol_union.merge_method
        )

    def test_identical_index(self):
        details_true = {
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1',
                              'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2',
                              'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2',
                              'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'],
            'utr_sizes_from_reads': [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000025917"].to_dict(),
            details_true
        )

    def test_index_on_first_df(self):
        details_true = {
            'read_barcodes': ['CGAGAAGGTGCAACGA-1', 'CAGTCCTAGAAGGACA-1'],
            'utr_sizes_from_reads': [53, 28]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000115423"].to_dict(),
            details_true
        )

    def test_index_on_second_df(self):
        details_true = {
            'read_barcodes': ['ACAGCTATCCTTAATC-2'],
            'utr_sizes_from_reads': [192]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000079334"].to_dict(),
            details_true
        )

    def test_length(self):
        self.assertEqual(
            self.merged.shape[0], self.mappings[0].index.union(
                self.mappings[1].index).shape[0]
        )


class TestSubsetReferenceData(TestPeakCalling):
    def setUp(self) -> None:
        merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(self.mappings)
        )
        self.subsetted = process_mapped_reads.subset_reference_data(
            utr_sizes=merged,
            utr_reference_regions=self.utr_regions
        )

    def test_number_of_features(self) -> None:
        self.assertEqual(self.subsetted.shape[0], 63)

    def test_number_of_fields(self) -> None:
        """
        Number of fields in the annotation BED file should be equal to the optional
        fields (in variable 'annotations') plus the mandatory fields 'seqname', 'start',
        'end' and finally the count field, that tell us how many features were merged
        at each new feature (row).
        """
        self.assertEqual(self.subsetted.shape[1], 8)

    def test_values_are_proper(self) -> None:
        proper_values = ['1', 10024601, 10024978, 'ENSMUSG00000025917', 0.0, '-',
                         'ENSMUST00000027050', 'Cops5']
        self.assertListEqual(
            self.subsetted.loc["ENSMUSG00000025917"].values.tolist(),
            proper_values
        )

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.subsetted.iloc[0].strand, '-')


class TestMergeUtrsPerGene(unittest.TestCase):
    def setUp(self) -> None:
        mapped_data = synthetic_data.MappedData()
        utr_regions = mapped_data.utr_regions
        mapped_reads = mapped_data.mapped_reads
        self.utr_regions, self.merged_mapped_reads = process_mapped_reads.merge_utrs_per_gene(
            utr_regions=utr_regions, read_mappings=mapped_reads
        )

    def test_index(self):
        self.assertListEqual(
            self.merged_mapped_reads.index.to_list(),
            ["ENSMUSG1", "ENSMUSG1,ENSMUSG4", "ENSMUSG2",
                "ENSMUSG3", "ENSMUSG4", "ENSMUSG5"]
        )

    def test_positive_strand_sizes(self):
        self.assertListEqual(
            self.merged_mapped_reads.loc[
                self.utr_regions.strand == "+", :
            ].utr_sizes_from_reads.to_list(),
            [[60], [60, 65], [90], [50, 45]]
        )

    def test_negative_strand_sizes(self):
        self.assertListEqual(
            self.merged_mapped_reads.loc[
                self.utr_regions.strand == "-", :
            ].utr_sizes_from_reads.to_list(),
            [[50, 230, 235], [20, 25, 80, 240, 350]]
        )

    def test_utr_regions_dataframe(self):
        expected = {
            'chrom': {
                'ENSMUSG1': '1',
                'ENSMUSG1,ENSMUSG4': '1',
                'ENSMUSG2': '1',
                'ENSMUSG3': '1',
                'ENSMUSG4': '1', 'ENSMUSG5': '1'
            },
            'start': {
                'ENSMUSG1': '1000',
                'ENSMUSG1,ENSMUSG4': '1150',
                'ENSMUSG2': '2000,2400',
                'ENSMUSG3': '3000',
                'ENSMUSG4': '500',
                'ENSMUSG5': '500,800,1100,1200'
            },
            'end': {
                'ENSMUSG1': '1100',
                'ENSMUSG1,ENSMUSG4': '1350',
                'ENSMUSG2': '2200,2600',
                'ENSMUSG3': '3200',
                'ENSMUSG4': '600', 'ENSMUSG5': '650,900,1250,1300'
            },
            'strand': {
                'ENSMUSG1': '+',
                'ENSMUSG1,ENSMUSG4': '+',
                'ENSMUSG2': '-',
                'ENSMUSG3': '+',
                'ENSMUSG4': '+',
                'ENSMUSG5': '-'
            },
            'transcript_id': {
                'ENSMUSG1': 'ENSMUST1,ENSMUST2',
                'ENSMUSG1,ENSMUSG4': 'ENSMUST1,ENSMUST2,ENSMUST6',
                'ENSMUSG2': 'ENSMUST3,ENSMUST4',
                'ENSMUSG3': 'ENSMUST5',
                'ENSMUSG4': 'ENSMUST6',
                'ENSMUSG5': 'ENSMUST7,ENSMUST8,ENSMUST9'
            },
            'gene_name': {
                'ENSMUSG1': 'Test1',
                'ENSMUSG1,ENSMUSG4': 'Test1,Test4',
                'ENSMUSG2': 'Test2',
                'ENSMUSG3': 'Test3',
                'ENSMUSG4': 'Test4',
                'ENSMUSG5': 'Test5'
            },
            'name': {
                'ENSMUSG1': 'ENSMUSG1',
                'ENSMUSG1,ENSMUSG4': 'ENSMUSG1,ENSMUSG4',
                'ENSMUSG2': 'ENSMUSG2',
                'ENSMUSG3': 'ENSMUSG3',
                'ENSMUSG4': 'ENSMUSG4',
                'ENSMUSG5': 'ENSMUSG5'
            },
        }
        self.assertDictEqual(expected, self.utr_regions.to_dict())

    def test_read_mappings_dataframe(self):
        expected = {
            'read_barcodes': {
                'ENSMUSG1': ['GAAGACGT'],
                'ENSMUSG1,ENSMUSG4': ['GAAGACGA', 'ATGATGA'],
                'ENSMUSG2': ['AAAGACGA', 'GTGATGA', 'AAAGACGG'],
                'ENSMUSG3': ['CAAGACGG'],
                'ENSMUSG4': ['CTAGACGT', 'CTGATGA'],
                'ENSMUSG5': ['GTAGACGT', 'GTAGCCGT', 'ATAGCCGT', 'TAAGACGA', 'TTGATGA']
            },
            'utr_sizes_from_reads': {
                'ENSMUSG1': [60],
                'ENSMUSG1,ENSMUSG4': [60, 65],
                'ENSMUSG2': [50, 230, 235],
                'ENSMUSG3': [90],
                'ENSMUSG4': [50, 45],
                'ENSMUSG5': [20, 25, 80, 240, 350]
            }
        }
        self.assertDictEqual(expected, self.merged_mapped_reads.to_dict())


class TestFilterBarcodes(TestPeakCalling):
    def setUp(self) -> None:
        self.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(self.mappings)
        )
        self.annotations = './test_data/vivo_anno.csv'

    @staticmethod
    def are_all_barcodes_in_anno(data, annotations):
        counter = 0
        control = 0
        barcodes_real = annotations.CellBarcode
        for barcodes in data.read_barcodes:
            control += 1
            if set(barcodes).issubset(barcodes_real):
                counter += 1
        return counter, control

    def test_before_filter(self):
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            self.merged,
            annotations
        )
        self.assertNotEqual(barcodes_in_data, barcodes_in_annotations)

    def test_filter(self):
        filtered = process_mapped_reads.filter_barcodes(
            utr_sizes=self.merged,
            whitelist=self.protocol_.barcode_whitelist
        )
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)

    def test_specific_utr(self):
        filtered = process_mapped_reads.filter_barcodes(
            utr_sizes=self.merged,
            whitelist=self.protocol_.barcode_whitelist
        )
        self.assertNotEqual(
            len(self.merged.loc["ENSMUSG00000025917"]['read_barcodes']),
            len(filtered.loc["ENSMUSG00000025917"]['read_barcodes'])
        )

    def test_specific_utr_2(self):
        barcodes_real = [
            'CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1', 'TGCGCAGAGGCATGTG-1',
            'CGACCTTTCTGCTGCT-2', 'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2',
            'GTGCGGTAGGACAGCT-2', 'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'
        ]
        filtered = process_mapped_reads.filter_barcodes(
            utr_sizes=self.merged,
            whitelist=self.protocol_.barcode_whitelist
        )
        self.assertListEqual(
            filtered.loc["ENSMUSG00000025917"]['read_barcodes'], barcodes_real)

    def test_using_annotations_as_dataframe(self):
        annotations = pd.read_csv(self.annotations)
        filtered = process_mapped_reads.filter_barcodes(
            utr_sizes=self.merged,
            whitelist=self.protocol_.barcode_whitelist
        )
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered,
            annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)


class TestFilterBarcodesSynthetic(unittest.TestCase):
    def setUp(self) -> None:
        data_synthetic = synthetic_data.MappedDataThreePrimeTenX()
        self.data = data_synthetic.data_synthetic
        self.anno = data_synthetic.anno
        self.filtered = process_mapped_reads.filter_barcodes(
            utr_sizes=self.data,
            whitelist=self.anno["barcodes"].unique()
        )
        self.barcodes_filtered = list(
            helpers.flatten(self.filtered.read_barcodes))

    def test_barcodes_in_anno(self):
        self.assertFalse(
            [barcode for barcode in self.barcodes_filtered
             if barcode not in self.anno.barcodes.tolist()]
        )

    def test_filtered_len(self):
        self.assertEqual(len(self.barcodes_filtered), 10)

    def test_filtered_id(self):
        correct = [
            "AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAA-1",
            "GAAAAAAAAAAAAAAT-2", "GAAAAAAAAAAAAAAT-2",
            "AAAAAAAAAAAAAAGG-2", "AAAAAAAAAAAAAAAG-1",
            "AAAAAAAAAAAAAAAG-1", "AAAAAAAAAAAAAAGG-2",
            "GAAAAAAAAAAAAAAT-2", "AAAAAAAAAAAAATTT-1"
        ]
        self.assertListEqual(correct, self.barcodes_filtered)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
