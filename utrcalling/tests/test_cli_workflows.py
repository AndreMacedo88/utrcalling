"""The command chain of the paper workflows, run end to end from the command line.

Each test runs `utrcalling <command> <flags>` in this process, with the flags the
Snakemake workflows of Macedo et al. 2026 use (here on the 10x v2 mouse test data,
at UTR level). These command lines must keep working when the CLI changes.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling import runtime

TEST_DATA = Path(__file__).resolve().parent / "test_data"
GTF = TEST_DATA / "test_mus_musculus_GRCm38_102.gtf.gz"
# The old cells are in the first BAM and the young cells in the second.
BAMS = [
    TEST_DATA / "test_UTRoldNSCgenome_bwt2_102_sort.bam",
    TEST_DATA / "test_UTRyoungNSCgenome_bwt2_102_sort.bam",
]
ANNOTATIONS = TEST_DATA / "vivo_anno.csv"
# Flags that `run` and `call-peaks` share in the paper's 10x workflow.
PIPELINE_FLAGS = ["-s", "utr", "-p", "10x", "-v", "2", "-m", "union"]
PIPELINE_FLAGS += ["-A", ANNOTATIONS, "-L", "True"]

# Output of the paper's `run` command, shared by the tests below (setUpModule).
WORKDIR = Path()


def utrcalling(*args, cwd):
    """Run `utrcalling <args>` from directory *cwd*, in this process."""
    previous = os.getcwd()
    os.chdir(cwd)
    try:
        with mock.patch.object(sys, "argv", ["utrcalling", *map(str, args)]):
            runtime.main()
    finally:
        os.chdir(previous)


def _dense(matrix):
    return matrix.toarray() if scipy.sparse.issparse(matrix) else np.asarray(matrix)


def assert_same_anndata(actual, expected):
    """Same cells, sites, annotations and counts, in the same order."""
    pd.testing.assert_index_equal(actual.obs_names, expected.obs_names)
    pd.testing.assert_index_equal(actual.var_names, expected.var_names)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    np.testing.assert_array_equal(_dense(actual.X), _dense(expected.X))


def setUpModule():
    global WORKDIR
    WORKDIR = Path(tempfile.mkdtemp())
    utrcalling(
        "run",
        GTF,
        "-a",
        *BAMS,
        "-f",
        "three_prime_utr",
        *PIPELINE_FLAGS,
        "-r",
        "-i",
        "-o",
        "raw.h5ad",
        cwd=WORKDIR,
    )


def tearDownModule():
    shutil.rmtree(WORKDIR)


class _InTempDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.tmp)


class TestPrepareReferenceThenCallPeaks(_InTempDir):
    """Running the pipeline in two halves gives the same result as `run`.

    prepare-reference-and-map-reads used to crash on every call: the protocol
    needs the annotation file, which the command did not accept.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling(
            "prepare-reference-and-map-reads",
            GTF,
            "-a",
            *BAMS,
            "-A",
            ANNOTATIONS,
            "-L",
            "True",
            "-f",
            "three_prime_utr",
            "-p",
            "10x",
            "-v",
            "2",
            cwd=cls.tmp,
        )
        intermediate = cls.tmp / "intermediate"
        utrcalling(
            "call-peaks",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            "-r",
            "-o",
            "halves.h5ad",
            cwd=cls.tmp,
        )

    def test_raw_sizes_match_run(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "halves.h5ad"), ad.read_h5ad(WORKDIR / "raw.h5ad")
        )

    def test_peaks_match_run(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "halves_kde.h5ad"), ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        )


if __name__ == "__main__":
    unittest.main()
