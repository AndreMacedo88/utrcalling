"""Tests for utrcalling.core.peak_calling.callers: peak callers by name, built in or added."""

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
from utrcalling.core import peak_calling_anndata as pca
from utrcalling.core import provenance
from utrcalling.core.peak_calling import callers, kde
from utrcalling.core.peak_calling.settings import PEAK_CALLERS


def mode_sites(sizes):
    """A peak caller for the tests: every molecule at its UTR's most frequent size."""
    sizes = np.asarray(sizes).ravel()
    if len(sizes) == 0:
        return np.zeros(0, dtype=np.int32)
    values, counts = np.unique(sizes, return_counts=True)
    return np.full(len(sizes), values[counts.argmax()], dtype=np.int32)


def mode_caller(bandwidth, kernel):
    """The factory of mode_sites, which has neither bandwidth nor kernel."""
    return mode_sites


class RegisteredModeCaller(unittest.TestCase):
    """Tests that register mode_caller as the peak caller "mode" while they run."""

    def setUp(self):
        PEAK_CALLERS.register("mode", mode_caller)
        self.addCleanup(PEAK_CALLERS.unregister, "mode")


class TestGetCaller(unittest.TestCase):
    def test_kde_caller_uses_the_bandwidth(self):
        sizes = np.array([100] * 10 + [130] * 10)
        narrow = callers.get_caller("KDE", bandwidth=2)(sizes)
        wide = callers.get_caller("kde", bandwidth=50)(sizes)
        self.assertEqual(len(np.unique(narrow)), 2)
        self.assertEqual(len(np.unique(wide)), 1)

    def test_kde_is_the_default(self):
        caller = callers.get_caller()
        self.assertIs(caller.func, kde.call_peaks_using_kernel_density_estimator)
        self.assertEqual(caller.keywords, {"bandwidth": 12, "kernel": "gaussian"})

    def test_unknown_method_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown peak caller 'gmm': expected one of kde"):
            callers.get_caller("gmm", bandwidth=12)

    def test_get_caller_spells_the_rule_as_the_record_does(self):
        self.assertEqual(
            callers.get_caller("kde", bandwidth="Scott:0.50").keywords["bandwidth"], "scott:0.5"
        )

    def test_an_invalid_bandwidth_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'wide'"):
            callers.get_caller("kde", bandwidth="wide")


class TestAnAddedCaller(RegisteredModeCaller):
    """A peak caller added by name works wherever peaks are called."""

    def setUp(self):
        super().setUp()
        # One UTR with sites at 40 (6 molecules) and 90 (9); another at 10.
        counts = np.array([[2, 2, 2, 3, 3, 3, 4]] * 2, dtype=np.int32)
        self.adata = ad.AnnData(
            X=scipy.sparse.csr_matrix(counts),
            obs=pd.DataFrame(index=["cell_a", "cell_b"]),
            var=pd.DataFrame(
                {
                    "utr_name": ["U0"] * 6 + ["U1"],
                    "gene_name": ["G0"] * 6 + ["G1"],
                    "utr_size": [39, 40, 41, 89, 90, 91, 10],
                    "chrom": "1",
                    "start": 1,
                    "end": 200,
                    "strand": "+",
                },
                index=[f"site{i}" for i in range(7)],
            ),
        )

    def test_get_caller(self):
        caller = callers.get_caller("Mode", bandwidth=7, kernel="linear")
        self.assertIs(caller, mode_sites)
        self.assertListEqual(caller(np.array([3, 5, 5, 9])).tolist(), [5] * 4)

    def test_peaks_of_a_matrix(self):
        # The UTR's most frequent sizes are 89-91 (6 molecules each): 89, the first.
        for n_jobs in (1, 2):  # in worker processes too
            with self.subTest(n_jobs=n_jobs):
                peaks = pca.peak_call_anndata_parallel(
                    self.adata, method="mode", bandwidth=12, n_jobs=n_jobs
                )
                self.assertListEqual(peaks.var.utr_size.tolist(), [89, 10])
                self.assertListEqual(np.asarray(peaks.X.sum(axis=0)).ravel().tolist(), [30, 8])
        sequential = pca.peak_call_anndata(self.adata, method="mode", bandwidth=12)
        self.assertListEqual(sequential.var.utr_size.tolist(), [89, 10])

    def test_peak_table(self):
        table = pca.peak_table(self.adata, bandwidth=12, method="mode", n_jobs=1)
        self.assertListEqual(table.utr_size.tolist(), [89, 10])
        self.assertListEqual(table.molecules.tolist(), [30, 8])

    def test_call_peaks_command(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.adata.write_h5ad(tmp / "raw.h5ad")
        argv = ["utrcalling", "call-peaks", str(tmp / "raw.h5ad"), str(tmp / "peaks")]
        argv += ["-b", "12", "--peak-caller", "MODE"]
        with mock.patch.object(sys, "argv", argv):
            runtime.main()
        # Non-KDE methods are named in the file name; Gaussian KDE files are not.
        peaks = ad.read_h5ad(tmp / "peaks_mode_bw12.h5ad")
        self.assertListEqual(peaks.var.utr_size.tolist(), [89, 10])
        self.assertEqual(provenance.recorded(peaks)["peak_caller"], "mode")

    def test_unknown_callers_are_refused_by_the_command_line(self):
        argv = ["utrcalling", "call-peaks", "raw.h5ad", "peaks", "--peak-caller", "gmm"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as exit_:
            runtime.main()
        self.assertEqual(exit_.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
