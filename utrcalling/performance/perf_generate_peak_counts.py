from utrcalling.tests.test_data import synthetic_data
from utrcalling.core import peak_calling, module_log, protocol
import os
from pathlib import Path
import cProfile
import pstats
import logging

os.chdir(str(Path(__file__).resolve().parents[0]))


class GeneratePeakCounts:
    def __init__(self) -> None:
        data_synthetic = synthetic_data.RandomThreePrimeBarcoded(
            size=(20000, 2000))
        self.data = data_synthetic.data
        self.anno = data_synthetic.anno
        self.alignments = data_synthetic.alignments
        self.protocol = protocol.create(
            platform='smartseq',
            version=3,
            annotations=self.anno,
            alignments=self.alignments,
            merge_method="union",
        )

    def perf_test(self):
        peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.data,
            annotations=self.anno,
            func=self.protocol.peak_calling
        )


def generate_peak_counts_perf(lines):
    logger = module_log.module_logger()
    logger.setLevel(logging.INFO)

    logger.info("Loading the test data")
    generate_counts = GeneratePeakCounts()

    logger.info("Profiling")
    profiler = cProfile.Profile()
    profiler.enable()
    generate_counts.perf_test()
    profiler.disable()

    logger.success("Total internal time:")
    total_time = pstats.Stats(profiler).sort_stats("tottime")
    total_time.print_stats(lines)

    logger.success("Cumulative time:")
    cum_time = pstats.Stats(profiler).sort_stats("cumtime")
    cum_time.print_stats(lines)


if __name__ == '__main__':
    generate_peak_counts_perf(lines=10)
