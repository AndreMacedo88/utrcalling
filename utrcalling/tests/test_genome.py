"""Tests for utrcalling.core.genome: where sites lie, and the genome around them."""

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

from utrcalling.core import genome, protocol, provenance
from utrcalling.core.utr_mapping import _IntervalData
from utrcalling.tests.test_utr_mapping.test_read_filters import (
    ANNOTATIONS,
    _alignment,
    _MappingTest,
)

# Chromosome 1 of a small genome, written so each base tells its 1-based
# position: positions 1-10 are ACGTACGTAC, then 11-20 GGGGGTTTTT, and so on.
CHROM_1 = "ACGTACGTAC" + "GGGGGTTTTT" + "AAAACCCCGG" + "TTGCAATGCA"


class TestSitePositions(_MappingTest):
    """site_positions undoes how the read loop measures UTR sizes."""

    def _position(self, platform, version, strand, flag):
        # A UTR on 1001-1500 (1-based); a read on 1101-1200.
        protocol_ = protocol.create(platform, version, annotations=ANNOTATIONS)
        interval = _IntervalData("1", 1000, 1500, strand)
        _, sizes, _ = self._map(
            [_alignment("read", 1100, flag=flag)], protocol_, {"UTR1": interval}
        )
        var = pd.DataFrame({"utr_size": sizes, "start": [1001], "end": [1500], "strand": [strand]})
        utr_end = (
            "five_prime" if isinstance(protocol_, protocol.ProtocolFivePrime) else "three_prime"
        )
        return genome.site_positions(var, utr_end)[0]

    def test_three_prime_protocols_measure_from_the_read_end_toward_the_polya(self):
        # Cell Ranger reads are sense: their 3' end is the site.
        self.assertEqual(self._position("standard_10x", 3, "+", 0), 1200)
        self.assertEqual(self._position("standard_10x", 3, "-", 0x10), 1101)

    def test_five_prime_protocols_measure_from_the_read_start(self):
        first_of_proper_pair = 0x1 | 0x2 | 0x40
        self.assertEqual(self._position("5prime10x", 1, "+", first_of_proper_pair), 1101)
        self.assertEqual(self._position("5prime10x", 1, "-", first_of_proper_pair | 0x10), 1200)

    def test_gene_level_coordinates_are_refused(self):
        var = pd.DataFrame(
            {"utr_size": [10], "start": ["101,301"], "end": ["200,400"], "strand": ["+"]}
        )
        with self.assertRaisesRegex(ValueError, "no genome positions"):
            genome.site_positions(var, "three_prime")

    def test_an_unknown_utr_end_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expected three_prime or five_prime"):
            genome.site_positions(pd.DataFrame(), "middle")


class TestReadSequences(unittest.TestCase):
    """read_sequences reads around each site, in transcript orientation."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.fasta = self.tmp / "genome.fa"
        self.fasta.write_text(f">chr1\n{CHROM_1}\n>chr3\nacgtnRYacgt\n")

    def _read(self, sites, start, stop):
        table = pd.DataFrame(sites, columns=["chrom", "position", "strand"])
        return genome.read_sequences(self.fasta, table, start, stop)

    def test_plus_strand_sites_read_the_genome_forwards(self):
        # Position 15 is the last G of GGGGG (11-15); offsets -2 to +2 are 13 to 17.
        sequences, found = self._read([("1", 15, "+")], -2, 3)
        self.assertListEqual(sequences.tolist(), ["GGGTT"])
        self.assertListEqual(found.tolist(), [True])

    def test_minus_strand_sites_read_the_reverse_complement(self):
        # On the minus strand, offsets -2 to +2 are positions 17 down to 13:
        # T, T, G, G, G on the plus strand, complemented A, A, C, C, C.
        sequences, _ = self._read([("1", 15, "-")], -2, 3)
        self.assertListEqual(sequences.tolist(), ["AACCC"])

    def test_before_and_after_the_site(self):
        sites = [("1", 15, "+"), ("1", 15, "-")]
        before, _ = self._read(sites, -3, 0)  # what 5' sites have before them
        after, _ = self._read(sites, 1, 4)  # what 3' sites have after them
        self.assertListEqual(before.tolist(), ["GGG", "AAA"])
        self.assertListEqual(after.tolist(), ["TTT", "CCC"])

    def test_beyond_the_chromosome_ends_is_n(self):
        sequences, _ = self._read([("chr1", 2, "+"), ("chr1", 39, "+"), ("chr1", 2, "-")], -2, 3)
        # 0 and 41 lie outside the 40 nt; the minus strand reads 4, 3, 2, 1, 0.
        self.assertListEqual(sequences.tolist(), ["NACGT", "TGCAN", "ACGTN"])

    def test_lower_case_and_other_letters(self):
        # Soft-masked bases count; ambiguity codes become N.
        sequences, _ = self._read([("3", 6, "+")], -5, 6)
        self.assertListEqual(sequences.tolist(), ["ACGTNNNACGT"])

    def test_chromosomes_the_genome_lacks(self):
        sequences, found = self._read([("2", 5, "+"), ("chr1", 5, "+")], 0, 2)
        self.assertListEqual(sequences.tolist(), ["NN", "AC"])
        self.assertListEqual(found.tolist(), [False, True])

    def test_a_genome_of_another_assembly_is_refused(self):
        with self.assertRaisesRegex(ValueError, "same genome assembly"):
            self._read([("NC_000001.11", 5, "+")], 0, 1)

    def test_at_least_one_nt(self):
        with self.assertRaisesRegex(ValueError, "offsets 2 to 2"):
            self._read([("1", 5, "+")], 2, 2)


class TestPositionedUtrEnd(unittest.TestCase):
    def setUp(self):
        var = pd.DataFrame({"utr_name": ["U"], "utr_size": [5]}, index=["s"])
        self.adata = ad.AnnData(X=scipy.sparse.csr_matrix(np.ones((1, 1))), var=var)

    def _record(self, **record):
        with mock.patch.object(sys, "argv", ["utrcalling", "run"]):
            provenance.record(self.adata, **record)

    def test_given_or_recorded(self):
        with self.assertRaisesRegex(ValueError, "does not record which end"):
            genome.positioned_utr_end(self.adata, None)
        self.assertEqual(genome.positioned_utr_end(self.adata, "five_prime"), "five_prime")
        self._record(utr_end="three_prime", summarization_level="utr")
        self.assertEqual(genome.positioned_utr_end(self.adata, None), "three_prime")

    def test_gene_level_matrices_have_no_positions(self):
        self._record(utr_end="three_prime", summarization_level="gene")
        with self.assertRaisesRegex(ValueError, "Gene-level sites"):
            genome.positioned_utr_end(self.adata, None)


class TestChromosomeName(unittest.TestCase):
    def test_without_chr(self):
        for name, expected in (("chr1", "1"), ("CHRX", "X"), ("1", "1"), (2, "2")):
            self.assertEqual(genome.chromosome_name(name), expected)


if __name__ == "__main__":
    unittest.main()
