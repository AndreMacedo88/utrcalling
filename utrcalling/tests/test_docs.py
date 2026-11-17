"""The examples of the documentation run, and do what it says."""

import re
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling.core import protocol, provenance
from utrcalling.core.peak_calling import settings

DOCS = Path(__file__).resolve().parents[2] / "docs"
EXTENDING = DOCS / "extending.md"


def code_blocks(path, language):
    """The code blocks of a Markdown file in *language*, in order."""
    return re.findall(rf"```{language}\n(.*?)```", path.read_text(), flags=re.DOTALL)


def three_prime_sites(genome_size=400):
    """Raw sizes of one 3' UTR on 1-300 (+ strand), and a genome with a site after U's."""
    # Sites end at 101 (40 molecules) and 201 (20); after 101 come 30 T's.
    genome = list(np.random.default_rng(0).choice(list("ACG"), size=genome_size))
    genome[101:131] = "T" * 30
    var = pd.DataFrame(
        {
            "utr_name": "U",
            "gene_name": "G",
            "utr_size": [99, 100, 101, 199, 200, 201],
            "chrom": "1",
            "start": 1,
            "end": 300,
            "strand": "+",
        },
        index=[f"G_{i}" for i in range(6)],
    )
    counts = np.array([[5, 30, 5, 5, 10, 5]], dtype=np.int32)
    raw_sizes = ad.AnnData(X=scipy.sparse.csr_matrix(counts), var=var)
    with mock.patch.object(sys, "argv", ["utrcalling", "run"]):
        provenance.record(raw_sizes, matrix="raw_sizes", utr_end="three_prime")
    return raw_sizes, "".join(genome)


@unittest.skipUnless(EXTENDING.exists(), "the documentation is in the source repository")
class TestExtending(unittest.TestCase):
    """docs/extending.md: each example adds a method that works by name."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        raw_sizes, sequence = three_prime_sites()
        (cls.tmp / "genome.fa").write_text(f">1\n{sequence}\n")
        cls.namespace = {
            "__name__": "extending_example",
            "raw_sizes": raw_sizes,
            "genome_fasta": str(cls.tmp / "genome.fa"),
        }
        cls.registered = [
            (settings.SCORES, "downstream-u"),
            (settings.PEAK_CALLERS, "cluster"),
            (settings.BANDWIDTH_RULES, "mad"),
            (protocol.PROTOCOLS, "sense-bulk"),
        ]
        for block in code_blocks(EXTENDING, "python"):
            exec(block, cls.namespace)

    @classmethod
    def tearDownClass(cls):
        for registry, name in cls.registered:
            registry.unregister(name)
        shutil.rmtree(cls.tmp)

    def test_every_example_registers_its_method(self):
        for registry, name in self.registered:
            with self.subTest(name=name):
                self.assertIn(name, registry.names())

    def test_the_score(self):
        scores = self.namespace["scores"]
        self.assertListEqual(
            scores.columns.tolist(), ["downstream-u", "pas", "sites", "sites_per_utr"]
        )
        # Both bandwidths keep the two sites apart; the T's follow the one at
        # 101, with 40 of the 60 molecules.
        np.testing.assert_allclose(scores["downstream-u"], [40 / 60, 40 / 60])
        self.assertEqual(self.namespace["peak_calling_anndata"].best_setting(scores)[1], 4)

    def test_the_peak_caller(self):
        sites = self.namespace["sites"]
        # 99-101 and 199-201 are two clusters, each at its most used size.
        self.assertListEqual(sites.var.utr_size.tolist(), [100, 200])
        self.assertListEqual(np.asarray(sites.X.sum(axis=0)).ravel().tolist(), [40, 20])
        with self.assertRaisesRegex(ValueError, "takes a number of nt"):
            self.namespace["cluster_caller"]("scott", "gaussian")
        distance_clusters = self.namespace["distance_clusters"]
        self.assertListEqual(distance_clusters([5, 1, 2, 9, 2], gap=2).tolist(), [5, 2, 2, 9, 2])

    def test_the_bandwidth_rule(self):
        caller = self.namespace["caller"]
        self.assertEqual(caller.keywords, {"bandwidth": "mad:0.5", "kernel": "linear"})
        sizes = np.array([10, 11, 12, 13, 100])  # the MAD ignores the far size
        self.assertAlmostEqual(settings.BANDWIDTH_RULES.get("mad")(sizes), 1.4826 * 5**-0.2)

    def test_the_protocol(self):
        made = protocol.create("Sense-Bulk", 1, annotations=pd.DataFrame({"sample_id": ["S1"]}))
        self.assertTrue(made.check_strand_matches("+", "+"))
        self.assertFalse(made.check_strand_matches("-", "+"))

    def test_the_plugin_groups_are_the_registries(self):
        (pyproject,) = code_blocks(EXTENDING, "toml")
        groups = tomllib.loads(pyproject)["project"]["entry-points"]
        self.assertTrue(set(groups) <= {registry.group for registry, _ in self.registered})
        table = EXTENDING.read_text()
        for registry, _ in self.registered:
            with self.subTest(group=registry.group):
                self.assertIn(f"| `{registry.group}` |", table)

    def test_the_files_it_names_exist(self):
        (tree,) = re.findall(r"```\nutrcalling/core/\n(.*?)```", EXTENDING.read_text(), re.DOTALL)
        core = Path(protocol.__file__).parent
        folders = []
        for line in tree.splitlines():
            depth = (len(line) - len(line.lstrip("│├└─ "))) // 4 - 1  # "├── " is depth 0
            name = line.lstrip("│├└─ ").split()[0]
            folders = [*folders[:depth], name.rstrip("/")]
            with self.subTest(path="/".join(folders)):
                self.assertTrue(core.joinpath(*folders).exists())


if __name__ == "__main__":
    unittest.main()
