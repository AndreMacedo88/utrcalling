import copy
import random
import string
import numpy as np
import pandas as pd


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
