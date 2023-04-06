import os
from pathlib import Path
import cProfile
import pstats
import logging

import pandas as pd

from utrcalling.core import core, module_log, protocols

os.chdir(str(Path(__file__).resolve().parents[0]))


class MapAlignedReads:
    def __init__(self) -> None:
        self.utr_regions = pd.read_parquet(
            '../tests/test_data/test_UTR_reference.parquet')
        self.protocol = protocols.protocol(platform='10x', version=2)

        alignments_n = 5
        self.alignments = [
            '../tests/test_data/test_UTRoldNSCgenome_bwt2_102_duplicates_sort.bam',
        ] * alignments_n

    def perf_test(self):
        core.map_aligned_reads(
            self.alignments, self.utr_regions, self.protocol, store=False
        )


def map_aligned_reads_perf(lines):
    logger = module_log.module_logger()
    logger.setLevel(logging.WARNING)

    reads_to_map = MapAlignedReads()

    profiler = cProfile.Profile()
    profiler.enable()
    reads_to_map.perf_test()
    profiler.disable()

    logger.success("Total internal time:")
    total_time = pstats.Stats(profiler).sort_stats("tottime")
    total_time.print_stats(lines)

    logger.success("Cumulative time:")
    cum_time = pstats.Stats(profiler).sort_stats("cumtime")
    cum_time.print_stats(lines)


if __name__ == '__main__':
    map_aligned_reads_perf(lines=20)
