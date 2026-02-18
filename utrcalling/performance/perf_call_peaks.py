from utrcalling.tests.test_data import synthetic_data
from utrcalling.core import core, module_log, protocol
import os
from pathlib import Path
import cProfile
import pstats
import logging

import pandas as pd

os.chdir(str(Path(__file__).resolve().parents[0]))


class GeneratePeakCounts:
    def __init__(self) -> None:
        # create 2 sets of data
        data_synthetic = synthetic_data.RandomThreePrimeBarcoded(size=(20000, 1000))
        data_synthetic_2 = synthetic_data.RandomThreePrimeBarcoded(size=(20000, 1000))
        self.data = [data_synthetic.data, data_synthetic_2.data]
        # change annotation sample_id to include batch
        anno_1 = data_synthetic.anno
        anno_1.sample_id = anno_1.sample_id + anno_1.batch.astype(str)

        # change the annotation of the second to mimic a second batch
        anno_2 = data_synthetic_2.anno
        anno_2.batch = [2] * len(anno_2.batch)
        anno_2.sample_id = anno_2.sample_id + anno_2.batch.astype(str)
        self.anno = pd.concat([anno_1, anno_2], ignore_index=True)
        self.alignments = self.anno.sample_id.to_list()

        self.protocol = protocol.create(
            platform="smartseq",
            version=3,
            annotations=self.anno,
            alignments=self.alignments,
            merge_method="union",
            label_batches=True,
        )

    def perf_test(self):
        core.call_peaks(
            utr_sizes_counts=self.data,
            utr_reference_regions=self.anno,
            protocol=self.protocol,
        )


def generate_peak_counts_perf(lines):
    logger = module_log.module_logger()
    logger.setLevel(logging.INFO)

    logger.info("Generating the test data")
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


if __name__ == "__main__":
    generate_peak_counts_perf(lines=10)
