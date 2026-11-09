"""Tests for utrcalling.core.provenance: how a count matrix was made."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd

from utrcalling._metadata import __version__
from utrcalling.core import provenance


def _matrix():
    return ad.AnnData(
        X=np.ones((2, 3), dtype=np.int32),
        obs=pd.DataFrame(index=["cell_1", "cell_2"]),
        var=pd.DataFrame({"utr_size": [10, 20, 30]}, index=["A_1-10", "A_2-20", "A_3-30"]),
    )


def _record(adata, argv, **kwargs):
    with mock.patch.object(sys, "argv", argv):
        provenance.record(adata, **kwargs)


class TestRecord(unittest.TestCase):
    def test_records_version_command_line_and_matrix(self):
        adata = _matrix()
        _record(adata, ["/bin/utrcalling", "run", "my file.gtf", "-r"], matrix="raw_sizes")
        self.assertDictEqual(
            provenance.recorded(adata),
            {
                "version": __version__,
                "commands": ["utrcalling run 'my file.gtf' -r"],
                "matrix": "raw_sizes",
            },
        )

    def test_continues_the_record_of_its_source(self):
        source = _matrix()
        _record(source, ["utrcalling", "run", "a.gtf"], matrix="peaks", bandwidth=12)
        filtered = _matrix()
        _record(filtered, ["utrcalling", "process", "filter-sites", "in", "out"], source=source)
        info = provenance.recorded(filtered)
        self.assertListEqual(
            info["commands"], ["utrcalling run a.gtf", "utrcalling process filter-sites in out"]
        )
        self.assertEqual(info["matrix"], "peaks")
        self.assertEqual(info["bandwidth"], 12)
        # The source keeps its own record.
        self.assertListEqual(provenance.recorded(source)["commands"], ["utrcalling run a.gtf"])

    def test_a_new_matrix_kind_and_details_replace_the_old(self):
        adata = _matrix()
        _record(adata, ["utrcalling", "run"], matrix="raw_sizes", bandwidth=12)
        _record(adata, ["utrcalling", "call-peaks"], matrix="peaks", bandwidth=30)
        info = provenance.recorded(adata)
        self.assertEqual(info["matrix"], "peaks")
        self.assertEqual(info["bandwidth"], 30)
        self.assertEqual(len(info["commands"]), 2)

    def test_none_details_are_left_out(self):
        adata = _matrix()
        _record(adata, ["utrcalling", "run"], strandedness=None, protocol_platform="10x")
        info = provenance.recorded(adata)
        self.assertNotIn("strandedness", info)
        self.assertEqual(info["protocol_platform"], "10x")

    def test_unknown_matrix_kinds_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown matrix 'counts'"):
            _record(_matrix(), ["utrcalling"], matrix="counts")

    def test_matrices_without_a_record(self):
        self.assertDictEqual(provenance.recorded(_matrix()), {})

    def test_the_record_survives_writing_and_reading(self):
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir)
        adata = _matrix()
        _record(adata, ["utrcalling", "run", "a.gtf"], matrix="raw_sizes", bandwidth=12)
        adata.write_h5ad(tmpdir / "m.h5ad")
        read = ad.read_h5ad(tmpdir / "m.h5ad")
        self.assertListEqual(provenance.recorded(read)["commands"], ["utrcalling run a.gtf"])
        _record(read, ["utrcalling", "call-peaks", "m.h5ad", "p"], matrix="peaks")
        info = provenance.recorded(read)
        self.assertListEqual(
            info["commands"], ["utrcalling run a.gtf", "utrcalling call-peaks m.h5ad p"]
        )
        self.assertEqual(info["bandwidth"], 12)


if __name__ == "__main__":
    unittest.main()
