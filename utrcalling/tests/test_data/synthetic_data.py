import os
from tempfile import mkstemp
import copy
import random
import string
import numpy as np
import pandas as pd
import pysam


class GtfEnsembl:
    def __init__(self) -> None:
        self.fd, self.file_path = mkstemp(suffix=".gtf", dir="./test_data/")

    def __enter__(self):
        """
        Note: GTF files are made of fields separated by tabs, but then the last field,
        the attributes, which can be composed of multiple attributes, must separate the
        multiple attributes by one space.
        """

        header = (
            '#!genome-build test_genome.p6\n'
            '#!genome-version test_genome\n'
            '#!genome-date 2023-05\n'
            '#!genome-build-accession NCBI:test_accession.8\n'
            '#!genebuild-last-updated 2023-05\n'
        )

        feature_1 = (
            '1	ensembl	three_prime_utr	1000	1100	.	+	0	'
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST1"; transcript_version "4"; '
            'gene_name "Test1"; transcript_name "Test1-201"; '
            'protein_id "ENSMUSP1"; protein_version "6";\n'
        )

        feature_2 = (
            '1	ensembl	three_prime_utr	1200	1350	.	+	0	'
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST1"; transcript_version "4"; '
            'gene_name "Test1"; transcript_name "Test1-201"; '
            'protein_id "ENSMUSP1"; protein_version "6";\n'
        )

        feature_3 = (
            '1	ensembl	three_prime_utr	1000	1100	.	+	0	'
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST2"; transcript_version "1"; '
            'gene_name "Test1"; transcript_name "Test1-202"; '
            'protein_id "ENSMUSP2"; protein_version "2";\n'
        )

        feature_4 = (
            '1	ensembl	three_prime_utr	1150	1300	.	+	0	'
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST2"; transcript_version "1"; '
            'gene_name "Test1"; transcript_name "Test1-202"; '
            'protein_id "ENSMUSP2"; protein_version "2";\n'
        )

        feature_5 = (
            '1	ensembl	three_prime_utr	2400	2600	.	-	0	'
            'gene_id "ENSMUSG2"; gene_version "2"; '
            'transcript_id "ENSMUST3"; transcript_version "4"; '
            'gene_name "Test2"; transcript_name "Test2-201"; '
            'protein_id "ENSMUSP3"; protein_version "3";\n'
        )

        feature_6 = (
            '1	ensembl	three_prime_utr	2000	2200	.	-	0	'
            'gene_id "ENSMUSG2"; gene_version "2"; '
            'transcript_id "ENSMUST3"; transcript_version "4"; '
            'gene_name "Test2"; transcript_name "Test2-201"; '
            'protein_id "ENSMUSP3"; protein_version "3";\n'
        )

        feature_7 = (
            '1	ensembl	three_prime_utr	2000	2100	.	-	0	'
            'gene_id "ENSMUSG2"; gene_version "2"; '
            'transcript_id "ENSMUST4"; transcript_version "3"; '
            'gene_name "Test2"; transcript_name "Test2-202"; '
            'protein_id "ENSMUSP4"; protein_version "2";\n'
        )

        feature_8 = (
            '1	ensembl	three_prime_utr	3000	3200	.	+	0	'
            'gene_id "ENSMUSG3"; gene_version "3"; '
            'transcript_id "ENSMUST5"; transcript_version "5"; '
            'gene_name "Test3"; transcript_name "Test3-201"; '
            'protein_id "ENSMUSP5"; protein_version "1";\n'
        )

        feature_9 = (
            '1	ensembl	three_prime_utr	500	600	.	+	0	'
            'gene_id "ENSMUSG4"; gene_version "1"; '
            'transcript_id "ENSMUST6"; transcript_version "2"; '
            'gene_name "Test4"; transcript_name "Test4-201"; '
            'protein_id "ENSMUSP6"; protein_version "1";\n'
        )

        feature_10 = (
            '1	ensembl	three_prime_utr	1200	1300	.	+	0	'
            'gene_id "ENSMUSG4"; gene_version "1"; '
            'transcript_id "ENSMUST6"; transcript_version "2"; '
            'gene_name "Test4"; transcript_name "Test4-201"; '
            'protein_id "ENSMUSP6"; protein_version "1";\n'
        )

        feature_11 = (
            '1	ensembl	three_prime_utr	500	650	.	-	0	'
            'gene_id "ENSMUSG5"; gene_version "7"; '
            'transcript_id "ENSMUST7"; transcript_version "7"; '
            'gene_name "Test5"; transcript_name "Test5-201"; '
            'protein_id "ENSMUSP7"; protein_version "1";\n'
        )

        feature_12 = (
            '1	ensembl	three_prime_utr	1200	1300	.	-	0	'
            'gene_id "ENSMUSG5"; gene_version "7"; '
            'transcript_id "ENSMUST7"; transcript_version "7"; '
            'gene_name "Test5"; transcript_name "Test5-201"; '
            'protein_id "ENSMUSP7"; protein_version "1";\n'
        )

        features = feature_1 + feature_2 + feature_3 + feature_4 + feature_5 + \
            feature_6 + feature_7 + feature_8 + feature_9 + feature_10 + \
            feature_11 + feature_12

        with os.fdopen(self.fd, "w") as f:
            f.writelines(header + features)

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.file_path)


UTR_REGIONS_MERGED = pd.DataFrame(
    columns=[
        "chrom", "start", "end", "name", "score", "strand", "transcript_id",
        "gene_name"
    ],
    index=[
        "ENSMUSG4",
        "ENSMUSG5",
        "ENSMUSG5_2",
        "ENSMUSG1",
        "ENSMUSG1,ENSMUSG4",
        "ENSMUSG5_3",
        "ENSMUSG5_4",
        "ENSMUSG2",
        "ENSMUSG2_2",
        "ENSMUSG3"
    ],
    data=np.array(
        [
            [1, 500, 600, 'ENSMUSG4', '.', '+', 'ENSMUST6', 'Test4'],
            [1, 500, 650, 'ENSMUSG5', '.', '-', 'ENSMUST7', 'Test5'],
            [1, 800, 900, 'ENSMUSG5', '.', '-', 'ENSMUST8', 'Test5'],
            [1, 1000, 1100, 'ENSMUSG1', '.', '+', 'ENSMUST1,ENSMUST2', 'Test1'],
            [1, 1150, 1350, 'ENSMUSG1,ENSMUSG4', '.',
                '+', 'ENSMUST1,ENSMUST2,ENSMUST6', 'Test1,Test4'],
            [1, 1100, 1250, 'ENSMUSG5', '.', '-', 'ENSMUST9', 'Test5'],
            [1, 1200, 1300, 'ENSMUSG5', '.', '-', 'ENSMUST7', 'Test5'],
            [1, 2000, 2200, 'ENSMUSG2', '.', '-', 'ENSMUST3,ENSMUST4', 'Test2'],
            [1, 2400, 2600, 'ENSMUSG2', '.', '-', 'ENSMUST3', 'Test2'],
            [1, 3000, 3200, 'ENSMUSG3', '.', '+', 'ENSMUST5', 'Test3']
        ]
    )
)
UTR_REGIONS_MERGED.index.name = "utr_name"
UTR_REGIONS_MERGED["start"] = UTR_REGIONS_MERGED["start"].astype(int)
UTR_REGIONS_MERGED["end"] = UTR_REGIONS_MERGED["end"].astype(int)

MAPPED_READS = pd.DataFrame(
    columns=[
        "read_barcodes",
        "utr_sizes_from_reads",
    ],
    index=[
        "ENSMUSG4",
        "ENSMUSG5",
        "ENSMUSG5_2",
        "ENSMUSG1",
        "ENSMUSG1,ENSMUSG4",
        "ENSMUSG5_3",
        "ENSMUSG5_4",
        "ENSMUSG2",
        "ENSMUSG2_2",
        "ENSMUSG3"
    ],
    data=np.array(
        [
            [["CTAGACGT", "CTGATGA"], [50, 45]],
            [["GTAGACGT"], [50]],
            [["GTAGCCGT"], [40]],
            [["GAAGACGT"], [60]],
            [["GAAGACGA", "ATGATGA"], [60, 65]],
            [["ATAGCCGT"], [30]],
            [["TAAGACGA", "TTGATGA"], [20, 25]],
            [["AAAGACGA", "GTGATGA"], [30, 35]],
            [["AAAGACGG"], [50]],
            [["CAAGACGG"], [90]],
        ], dtype="object"
    )
)
MAPPED_READS.name = "utr_name"


class SmartSeqCapSeqBam:
    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")

    def enter(self):
        # set-up the temporary alignment file
        header = {'HD': {'VN': '1.0'},
                  'SQ': [{'LN': 1575, 'SN': '1'},
                         {'LN': 1584, 'SN': '2'}]}

        with pysam.AlignmentFile(self.alignment, "wb", header=header) as outf:
            # First alignment read 1
            a1 = pysam.AlignedSegment()
            a1.query_name = "VH00211:1:AAAFH3LHV:1:1403:73731:41910"
            a1.query_sequence = (
                "TGAGAGATAACCTTTGATCACTCCACAGGGCCTGGAAATGAGGGTTTTCAGGAAGAAAGGAATTCCAAACAATGTTTTCAAACACGCAGACATTGCGCAAT"
            )
            a1.flag = 83  # proper pair, first in pair, reverse strand
            a1.reference_id = 0
            a1.reference_start = 3340581  # Add 82 to get start. Inside the UTR by 9/10bp
            a1.mapping_quality = 255
            a1.cigar = ((0, 82), (4, 19))
            a1.next_reference_id = 0
            a1.next_reference_start = 3340293  # inside the UTR
            a1.template_length = -370
            a1.query_qualities = pysam.qualitystring_to_array(
                "CCC;CCCCC;CCCCCC-;CCC-CCCCCCCC;CCCCCCCCCC-;CCCCCCCCCCCCCCCCCC-CCCCCCCCC-CCCCCCCCCCCCCCCCCCCC-CCCCC;C-"
            )
            a1.tags = (("NM", 1),
                       ("NM", 0))
            # First alignment read 2
            a2 = pysam.AlignedSegment()
            a2.query_name = "VH00211:1:AAAFH3LHV:1:1403:73731:41910"
            a2.query_sequence = (
                "GTGTACTCTGTTTCACCATTTCTCTGTCAACTTCTGCCAACCCCTGATTTTTTTTTTATTTCCTCTACAGTTTTACCATTTTTCAGGATATTCTCTGTTAC"
            )
            a2.flag = 163  # proper pair, second in pair, mate in reverse strand
            a2.reference_id = 0
            a2.reference_start = 3340293  # inside the UTR by 9/10bp
            a2.mapping_quality = 255
            a2.cigar = ((0, 101),)
            a2.next_reference_id = 0
            a2.next_reference_start = 3340581  # inside the UTR
            a2.template_length = 370
            a2.query_qualities = pysam.qualitystring_to_array(
                "CCCC;-CCCCCCCCCCCCC-CC;CCCCC;CC;;C-CCCC-CCCCCCCCCCCC;C-CC;CCCC-C-CC---CCCC;CC-CC-C----C-C-CC;CCCCCC;C"
            )
            a2.tags = (("NM", 1),
                       ("NM", 1))

            # Second alignment read 1
            b1 = pysam.AlignedSegment()
            b1.query_name = "VH00211:1:AAAFH3LHV:2:1202:74811:33600"
            b1.query_sequence = (
                "ATTGCGCAATGTTGGCAGGCTGAATGCTGAGCATTCCCTTGGAGAGGACACAATGCCCTGAAAATGGATAGCAGTTCCCTTAATCTTCTTGTACTTATGGG"
            )
            b1.flag = 99  # proper pair, first in pair, mate in reverse strand
            b1.reference_id = 0
            b1.reference_start = 3536771  # inside the UTR by 1bp
            b1.mapping_quality = 255
            b1.cigar = ((4, 19), (0, 82))  # 11bp TSO + 8bp UMI + 5'UTR
            b1.next_reference_id = 0
            b1.next_reference_start = 3536850  # outside the UTR
            b1.template_length = 46897
            b1.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCC;CCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCCCC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC-CCC-;CC;CCC-CCCCCC;C"
            )
            b1.tags = (("NM", 0),
                       ("NM", 0))
            # Second alignment read 2
            b2 = pysam.AlignedSegment()
            b2.query_name = "VH00211:1:AAAFH3LHV:2:1202:74811:33600"
            b2.query_sequence = (
                "GGGAGGAAAGAGAAGAGGTGGAGATATGCCCGTGTGCAGTGAGATCTGTTGGATTCTAACGGCTTTGCCTGGATGGCATTTGAAATCTCCAACTCAGACAC"
            )
            b2.flag = 147  # proper pair, second in pair, reverse strand
            b2.reference_id = 0
            b2.reference_start = 3536850  # outside the 5'UTR
            b2.mapping_quality = 255
            b2.cigar = ((0, 61), (3, 19), (0, 40))  # from the gene body
            b2.next_reference_id = 0
            b2.next_reference_start = 3536771  # outside the UTR
            b2.template_length = -46897
            b2.query_qualities = pysam.qualitystring_to_array(
                "CCC;CCCCCCCCCCCCCCCCCCC;-CCCCCCCCCCC;CCCC;CCCCCCCCCCCC-CC;CCCCC-CCCCCCCCCCCCC;C-CCCCC;CCCCCCCCCCCCCCC"
            )
            b2.tags = (("NM", 0),
                       ("NM", 0))

            # Write all read alignments to a bam file
            for align in [a2, a1, b1, b2]:  # alignment order must be position sorted
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)


ANNOTATION = pd.DataFrame(columns=["celltype", "barcodes",
                                   "cool_annotation"],
                          data=list(
    zip(
        ["A", "B", "A", "C", "B"],
        ["AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAG-1", "GAAAAAAAAAAAAAAT-2",
         "AAAAAAAAAAAAATTT-1", "AAAAAAAAAAAAAAGG-2"],
        [2.3, 4.4, 5.0, 2.1, 3.0]
    )
)
)


class ThreePrimeBarcoded:
    def __init__(self) -> None:
        # Build the data
        read_barcodes = [["AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAA-1",
                          "GAAAAAAAAAAAAAAT-2", "GAAAAAAAAAAAAAAT-2",
                          "AAAAAAAAAAAAAAAG-2", "AAAAAAAAAAAAAAAG-1"],
                         ["AAAAAAAAAAAAAAAG-1", "AAAAAAAAAAAAAGGG-2",
                          "AAAAAAAAAAAAAAAT-2", "AAAAAAAAAAAAAATT-1",
                          "ATATATATATATATAT-1"]]
        utr_sizes_from_reads = [
            [500, 499, 500, 400, 398, 300],
            [100, 200, 202, 300, 300]
        ]

        self._data_synthetic = pd.DataFrame(columns=[
            'read_barcodes',
            'utr_sizes_from_reads'
        ],
            index=["ENSMUSG00000000001", "ENSMUSG00000000002"],
            data=list(
            zip(
                read_barcodes,
                utr_sizes_from_reads
            )
        )
        )

        # Build the annotations
        self._anno = ANNOTATION

    @property
    def data_synthetic(self):
        return copy.deepcopy(self._data_synthetic)

    @property
    def anno(self):
        return copy.deepcopy(self._anno)


class ThreePrimeBarcodedFiltered:
    def __init__(self) -> None:
        # Build the data
        read_barcodes = [
            ['AAAAAAAAAAAAAAAA-1', 'AAAAAAAAAAAAAAAA-1',
             'GAAAAAAAAAAAAAAT-2', 'GAAAAAAAAAAAAAAT-2',
             'AAAAAAAAAAAAAAGG-2', 'AAAAAAAAAAAAAAAG-1'],
            ['AAAAAAAAAAAAAAAG-1', 'AAAAAAAAAAAAAAGG-2',
             'GAAAAAAAAAAAAAAT-2', 'AAAAAAAAAAAAATTT-1']
        ]
        utr_sizes_from_reads = [
            [500, 499, 500, 400, 398, 300],
            [100, 200, 202, 300]
        ]

        self._data_filtered = pd.DataFrame(columns=[
            'read_barcodes',
            'utr_sizes_from_reads'
        ],
            index=["ENSMUSG00000000001", "ENSMUSG00000000002"],
            data=list(
            zip(
                read_barcodes,
                utr_sizes_from_reads,
            )
        )
        )

        # Build the annotations
        self._anno = ANNOTATION

    @property
    def data_filtered(self):
        return copy.deepcopy(self._data_filtered)

    @property
    def anno(self):
        return copy.deepcopy(self._anno)


class LargeThreePrimeBarcoded:
    def __init__(self, size=(10000, 15)) -> None:
        # Build the data
        rng = np.random.default_rng(123)

        read_barcodes = [
            list(i) for i in rng.choice(list(string.ascii_uppercase), size=size)
        ]

        utr_sizes = rng.multinomial(
            n=100, pvals=[0.25, 0.75], size=size).astype(int).tolist()
        utr_sizes_from_reads = []
        for i in utr_sizes:
            i_list = []
            for z in i:
                i_list.append(int(rng.choice(z, 1)))
            utr_sizes_from_reads.append(i_list)

        utr_len = 15
        utr_bases = list(string.ascii_uppercase)
        utrs = [
            ''.join(random.choice(utr_bases) for v in range(utr_len))
            for i in range(size[0])
        ]

        self._data = pd.DataFrame(
            data=list(
                zip(
                    read_barcodes,
                    utr_sizes_from_reads,
                )
            ),
            columns=[
                'read_barcodes',
                'utr_sizes_from_reads'
            ],
            index=utrs
        )

        # Build the annotations
        barcodes = self._data.read_barcodes.explode().unique()
        anno_length = len(barcodes)
        sample_id = [
            i for i in rng.choice(
                list(string.ascii_lowercase),
                size=anno_length,
                replace=False,
            )
        ]
        batch = [1] * anno_length

        self._anno = pd.DataFrame(
            data=list(
                zip(
                    sample_id,
                    barcodes,
                    batch
                )
            ),
            columns=["sample_id", "barcodes", "batch"],
        )

        # Build the alignment paths
        self._alignments = sample_id

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def anno(self):
        return copy.deepcopy(self._anno)

    @anno.setter
    def anno(self, df):
        self.anno = df

    @property
    def alignments(self):
        return copy.deepcopy(self._alignments)
