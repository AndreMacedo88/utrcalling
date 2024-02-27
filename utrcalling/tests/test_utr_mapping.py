from test_data import synthetic_data
from utrcalling.core import protocol, utr_mapping, module_log
import os
from pathlib import Path
import unittest

import pandas as pd

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class TestGenerateUtrIntervals(unittest.TestCase):
    def setUp(self) -> None:
        utr_regions = pd.read_parquet('./test_data/test_UTR_reference.parquet')
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)

    def test_correct_interval(self) -> None:
        """
        Here we test if the intervals are being generated correctly. Please note that
        since GTF files use 0 indexing and BAM files start indexing on 1, when converted
        to the HTSeq object the start location will be a smaller number by 1.
        """
        correct = ['ENSMUSG00000051285', 7160891, 7161063, '+']
        name, interval = list(self.intervals.items())[0]
        self.assertListEqual([name, interval.start_d, interval.end_d,
                              interval.strand], correct)


class TestMapReadsToUtr(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet(
            './test_data/test_UTR_reference.parquet')

        cls.utr_regions_smartseq = pd.read_parquet(
            './test_data/test_UTR_reference_smartseq.parquet')

        cls.utr_regions_5prime = pd.read_parquet(
            './test_data/test_UTR_reference_5prime.parquet'
        )

        # test data for 10X
        cls.alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'

        # test data for SmartSeq3
        cls.alignment_smartseq = './test_data/smartseq3_test_sorted_AS-740066-LR-59119.bam'

        # test data for 5'UTR
        cls.alignment_cap = './test_data/test_CapPhospho_genome_102_coordsort.bam'


class TestTenxTwo(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        protocol_ = protocol.create(
            platform='10x',
            version=2,
            annotations=pd.read_csv("./test_data/vivo_anno.csv")
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read='first'
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestTenxThree(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        protocol_ = protocol.create(
            platform='10x',
            version=3,
            annotations=pd.read_csv("./test_data/vivo_anno.csv")
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read='first'
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestTenxThreeNoBarcodes(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        annotations = pd.read_csv("./test_data/vivo_anno.csv")
        annotations = annotations.drop("CellBarcode", axis=1)

        protocol_ = protocol.create(
            platform='10x',
            version=3,
            annotations=annotations
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read='first'
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestSmartSeq3(TestMapReadsToUtr):
    def setUp(self) -> None:
        # test data for SmartSeq3
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions_smartseq
        )
        protocol_ = protocol.create(
            platform='smartseq',
            version=3,
            label_batches=True,
            alignments=self.alignment_smartseq,
            annotations="./test_data/smartseq3_anno.csv"
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment_smartseq,
            self.intervals,
            protocol_,
            read='first',
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': [['CACTGT-2'], ['CACTGT-2']],
            'utr_sizes_from_reads': [[300], [30]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_details_one(self):
        dict_true = {
            'read_barcodes': ['CACTGT-2'],
            'utr_sizes_from_reads': [300]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSG00000188976"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestSmartSeq3NoBarcodes(TestMapReadsToUtr):
    def setUp(self) -> None:
        # test data for SmartSeq3
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions_smartseq
        )
        annotations = pd.read_csv("./test_data/smartseq3_anno.csv")
        annotations = annotations.drop("Sample_Barcode", axis=1)
        protocol_ = protocol.create(
            platform='smartseq',
            version=3,
            label_batches=True,
            alignments=self.alignment_smartseq,
            annotations=annotations
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment_smartseq,
            self.intervals,
            protocol_,
            read='first',
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': [['CACTGT-2'], ['CACTGT-2']],
            'utr_sizes_from_reads': [[300], [30]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_details_one(self):
        dict_true = {
            'read_barcodes': ['CACTGT-2'],
            'utr_sizes_from_reads': [300]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSG00000188976"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestSmartSeq3Cap(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.intervals_5prime = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions_5prime
        )
        protocol_ = protocol.create(
            platform='smart-seq_cap-readname',
            version=3,
            alignments=self.alignment_cap,
            annotations="./test_data/smartseq3_anno.csv"
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment_cap,
            self.intervals_5prime,
            protocol_,
            read='first',
            batch="test_sample"
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': [self.alignment_cap, self.alignment_cap, self.alignment_cap],
            'utr_sizes_from_reads': [4, 25, 29]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000023905"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 31)


class TestProtocolNotSupported(TestMapReadsToUtr):
    def test_platform(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                platform='Zehn_x',
                version=3,
                annotations="./test_data/smartseq3_anno.csv",
            )

    def test_version_tenx(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                platform='10x',
                version=1,
                annotations="./test_data/smartseq3_anno.csv",
            )

    def test_version_smartseq(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                platform='Smart-seq',
                version=1,
                annotations="./test_data/smartseq3_anno.csv",
            )


class TestMapReadsToUtrMinimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR1-1", "UTR2", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [14190, 16400, 184700, 628000, 1200000],
                       "end": [14500, 16600, 184900, 630000, 1200500],
                       "name": ["UTR1", "UTR1", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "-", "+", "-"],
                       "transcript_id": ["UTR1_t", "UTR1_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/smartseq3_test_sorted_AS-740066-LR-59119.bam'


class TestSmartSeq3Minimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        protocol_ = protocol.create(
            platform='smartseq',
            version=3,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv"
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment, self.intervals,
            protocol_,
            read='first'
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': [['CACTGT'], ['CACTGT']],
            'utr_sizes_from_reads': [[223], [158]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestSmartSeq3LabelSamplesMinimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        protocol_ = protocol.create(
            platform='smartseq',
            version=3,
            label_batches=True,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv"
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read='first'
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': [['CACTGT-2'], ['CACTGT-2']],
            'utr_sizes_from_reads': [[223], [158]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestMapReadsToUtrMinimalCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [3102600, 3152900, 3154500, 4580000, 4846300],
                       "end": [3102750, 3153050, 3154600, 4760000, 4846300],
                       "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "+", "+", "+"],
                       "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/capseq_test_sorted.bam'


class TestSmartSeq3CapReadnameMinimal(TestMapReadsToUtrMinimalCap):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol_ = protocol.create(
            platform='smart-seq_cap-readname',
            version=3,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv"
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read='first',
            batch="Test"
        )

    def test_details(self):
        dict_true = {
            'read_barcodes': [[self.alignment], [self.alignment]],
            'utr_sizes_from_reads': [[68], [143]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestMapReadsToUtrSyntheticCap(unittest.TestCase):
    @ classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [3102600, 3340260, 3154500, 3536770, 4846300],
                       "end": [3102750, 3340670, 3154600, 3536840, 4846300],
                       "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "+", "+", "+"],
                       "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.data = synthetic_data.SmartSeqCapSeqBam()


class TestSmartSeqCapPhosphoSynthetic(TestMapReadsToUtrSyntheticCap):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol_ = protocol.create(
            platform='smart-seq_cap',
            version=3,
            alignments=self.data.alignment,
            annotations="./test_data/smartseq3_anno.csv"
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                protocol_,
                read='first',
                batch="Test",
            )

    def test_details(self):
        dict_true = {
            'read_barcodes': [[self.data.alignment], [self.data.alignment]],
            'utr_sizes_from_reads': [[404], [69]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestMapReadsToUtrSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [3102600, 3340260, 3154500, 3536770, 4846300],
                       "end": [3102750, 3340670, 3154600, 3536840, 4846300],
                       "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "+", "-", "+"],
                       "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.data = synthetic_data.SmartSeqBam()


# TODO: this was copied over from Cap seq, modify it to take into account that reads are
# matched from the end of the sequence and the read strand is assumed to be opposite of
# the genome
class TestSmartSeqPhosphoSynthetic(TestMapReadsToUtrSynthetic):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol_ = protocol.create(
            platform='smart-seq',
            version=3,
            alignments=self.data.alignment,
            annotations="./test_data/smartseq3_anno.csv"
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                protocol_,
                read='first',
                batch="Test",
            )

    def test_details(self):
        dict_true = {
            'read_barcodes': [["ATTGCG"]],
            'utr_sizes_from_reads': [[50]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)

# TODO: do similar class as TestSmartSeqPhosphoSynthetic, but for 10x to test UMI deduplication


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=False)
    module_log.log_end_run("Finished testing")
