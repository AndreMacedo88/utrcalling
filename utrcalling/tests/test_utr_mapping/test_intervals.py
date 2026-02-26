import os
from pathlib import Path
import unittest
import pandas as pd
from utrcalling.core import protocol, utr_mapping, module_log

# we change working dir to the base test dir so that relative paths are sane
os.chdir(str(Path(__file__).resolve().parents[1]))


class TestGenerateUtrIntervals(unittest.TestCase):
    def setUp(self) -> None:
        utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        self.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)

    def test_correct_interval(self) -> None:
        """
        Here we test if the intervals are being generated correctly. Please note that
        since GTF files use 0 indexing and BAM files start indexing on 1, the
        _IntervalData start location will be a smaller number by 1.
        """
        correct = ["ENSMUSG00000051285", 7160891, 7161063, "+"]
        name, interval = list(self.intervals.items())[0]
        self.assertListEqual(
            [name, interval.start_d, interval.end_d, interval.strand], correct
        )


class TestProtocolNotSupported(unittest.TestCase):
    def test_platform(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                platform="Zehn_x",
                version=3,
                annotations="./test_data/smartseq3_anno.csv",
            )

    def test_version_tenx(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                platform="10x",
                version=1,
                annotations="./test_data/smartseq3_anno.csv",
            )

    def test_version_smartseq(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                platform="Smart-seq",
                version=1,
                annotations="./test_data/smartseq3_anno.csv",
            )


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
