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


class TestReadStretches(TestReadSequences):
    """read_stretches reads each site's own offsets, as read_sequences reads all alike."""

    def _stretches(self, sites, starts, stops):
        table = pd.DataFrame(sites, columns=["chrom", "position", "strand"])
        return genome.read_stretches(self.fasta, table, np.array(starts), np.array(stops))

    def test_each_its_own(self):
        sites = [("1", 15, "+"), ("1", 15, "-"), ("chr1", 2, "+"), ("2", 5, "+")]
        sequences, found = self._stretches(sites, [-2, 1, -2, 0], [3, 4, 0, 2])
        # As read_sequences reads each alone; empty when the stop is the start.
        self.assertListEqual(sequences, ["GGGTT", "CCC", "NA", "NN"])
        self.assertListEqual(found.tolist(), [True, True, True, False])
        empty, _ = self._stretches([("1", 15, "+"), ("1", 15, "-")], [4, 4], [4, 2])
        self.assertListEqual(empty, ["", ""])

    def test_wholly_beyond_the_chromosome_ends(self):
        # 40 nt: offsets -10 to -6 of position 3 are -7 to -3, before its start;
        # offsets 5 to 7 of position 38 are 43 to 45, after its end.
        sequences, _ = self._stretches([("1", 3, "+"), ("1", 38, "+")], [-10, 5], [-5, 8])
        self.assertListEqual(sequences, ["NNNNN", "NNN"])


def _sites(utr_end, sizes, strands, chrom="1", start=1, end=30):
    """A count matrix of sites in one UTR on chrom:start-end, recording its UTR end."""
    var = pd.DataFrame(
        {
            "utr_name": "U",
            "utr_size": sizes,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strands,
        },
        index=[f"G_{i}-{size}" for i, size in enumerate(sizes)],
    )
    adata = ad.AnnData(X=scipy.sparse.csr_matrix(np.ones((1, len(sizes)))), var=var)
    with mock.patch.object(sys, "argv", ["utrcalling", "run"]):
        provenance.record(adata, matrix="peaks", utr_end=utr_end, summarization_level="utr")
    return adata


class TestSiteSequences(unittest.TestCase):
    """Sequences around sites, numbered from the transcript's end."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.fasta = self.tmp / "genome.fa"
        self.fasta.write_text(f">1\n{CHROM_1}\n")

    def _sequences(self, adata, **lengths):
        table = genome.site_sequences(adata, self.fasta, **lengths)
        return list(zip(table.upstream, table.downstream, strict=True))

    def test_five_prime_sites_start_the_downstream_sequence(self):
        # Both sites at position 15 (G), the first transcribed base (+1). The
        # minus-strand transcript runs leftwards: its promoter is 16-18 (TTT).
        adata = _sites("five_prime", sizes=[16, 15], strands=["+", "-"])
        table = genome.site_sequences(adata, self.fasta, upstream=3, downstream=3)
        self.assertListEqual(table.position.tolist(), [15, 15])
        self.assertListEqual(
            list(zip(table.upstream, table.downstream, strict=True)),
            [("GGG", "GTT"), ("AAA", "CCC")],
        )

    def test_three_prime_sites_end_the_upstream_sequence(self):
        # Both sites at 15, the last transcribed base (-1).
        adata = _sites("three_prime", sizes=[14, 15], strands=["+", "-"])
        self.assertListEqual(
            self._sequences(adata, upstream=3, downstream=3), [("GGG", "TTT"), ("AAC", "CCC")]
        )

    def test_the_initiator_of_a_start_site(self):
        # A transcript starting at the A of CA, the most common initiator: C is
        # -1, A +1. On the minus strand, the genome reads TG there.
        sequence = list("G" * 60)
        sequence[19:21] = "CA"  # positions 20-21
        sequence[39:41] = "TG"  # positions 40-41
        self.fasta.write_text(f">1\n{''.join(sequence)}\n")
        # + strand: the site at 21 (size 60 - 21 + 1); - strand: at 40 (size 40).
        adata = _sites("five_prime", sizes=[40, 40], strands=["+", "-"], end=60)
        table = genome.site_sequences(adata, self.fasta, upstream=1, downstream=1)
        self.assertListEqual((table.upstream + table.downstream).tolist(), ["CA", "CA"])

    def test_the_polyadenylation_signal_of_a_cleavage_site(self):
        # AATAAA ending 20 nt before the cleavage site, as the PAS lies.
        sequence = list("C" * 100)
        sequence[54:60] = "AATAAA"  # positions 55-60; the site at 80
        self.fasta.write_text(f">1\n{''.join(sequence)}\n")
        adata = _sites("three_prime", sizes=[79], strands=["+"], end=100)
        table = genome.site_sequences(adata, self.fasta, upstream=30, downstream=5)
        self.assertEqual(table.position.iloc[0], 80)
        self.assertEqual(table.upstream.iloc[0].find("AATAAA") + 6, 30 - 20)
        self.assertEqual(table.downstream.iloc[0], "CCCCC")

    def test_one_side_only(self):
        adata = _sites("five_prime", sizes=[16], strands=["+"])
        self.assertListEqual(self._sequences(adata, upstream=0, downstream=2), [("", "GT")])
        self.assertListEqual(self._sequences(adata, upstream=2, downstream=0), [("GG", "")])

    def test_sites_on_chromosomes_the_genome_lacks(self):
        adata = _sites("five_prime", sizes=[16, 16], strands=["+", "+"], chrom=["1", "7"])
        table = genome.site_sequences(adata, self.fasta, upstream=1, downstream=1)
        self.assertEqual(table.upstream.dtype, "string")
        self.assertEqual(table.upstream.iloc[0], "G")
        self.assertTrue(pd.isna(table.upstream.iloc[1]) and pd.isna(table.downstream.iloc[1]))

    def test_to_the_ends_of_the_utr(self):
        # One UTR on 11-25. 5' sites at 15 (+1): before them the UTR's 11-14,
        # from them on 15-25 (+), or 15 down to 11 and 16-25 reversed (-).
        adata = _sites("five_prime", sizes=[11, 5], strands=["+", "-"], start=11, end=25)
        self.assertListEqual(genome.site_sequences(adata, self.fasta).position.tolist(), [15, 15])
        complement = str.maketrans("ACGT", "TGCA")

        def minus(first, last):  # positions first to last, read on the minus strand
            return CHROM_1[first - 1 : last].translate(complement)[::-1]

        expected = [
            (CHROM_1[10:14], CHROM_1[14:25]),  # + strand: 11-14 | 15-25
            (minus(16, 25), minus(11, 15)),  # - strand: 25 down to 16 | 15 down to 11
        ]
        self.assertListEqual(self._sequences(adata, upstream="utr", downstream="utr"), expected)
        # The UTR each start site makes: from the site to the UTR's end.
        isoforms = self._sequences(adata, upstream=0, downstream="utr")
        self.assertListEqual(isoforms, [("", expected[0][1]), ("", expected[1][1])])
        # One side to the UTR's end, the other a number of nt.
        self.assertListEqual(
            self._sequences(adata, upstream=2, downstream="utr"),
            [(CHROM_1[12:14], expected[0][1]), (minus(16, 17), expected[1][1])],
        )

    def test_three_prime_sites_to_the_ends_of_the_utr(self):
        # 3' sites at 15 (-1): the UTR up to them (11-15), and what follows (16-25).
        adata = _sites("three_prime", sizes=[4], strands=["+"], start=11, end=25)
        self.assertListEqual(
            self._sequences(adata, upstream="utr", downstream="utr"),
            [(CHROM_1[10:15], CHROM_1[15:25])],
        )

    def test_sites_beyond_the_utr(self):
        # A 5' site before the UTR's start (8, +) reads nothing before it within
        # the UTR, and from the site to the UTR's end after it; a site after the
        # UTR's end (28, +) reads from the UTR's start up to the site, and nothing after.
        adata = _sites("five_prime", sizes=[18, -2], strands=["+", "+"], start=11, end=25)
        self.assertListEqual(genome.site_sequences(adata, self.fasta).position.tolist(), [8, 28])
        self.assertListEqual(
            self._sequences(adata, upstream="utr", downstream="utr"),
            [("", CHROM_1[7:25]), (CHROM_1[10:27], "")],
        )

    def test_invalid_requests(self):
        adata = _sites("five_prime", sizes=[16], strands=["+"])
        invalid = [{"upstream": -1}, {"downstream": -5}, {"upstream": 0, "downstream": 0}]
        invalid.append({"upstream": "utr", "downstream": -1})
        for lengths in invalid:
            with self.subTest(**lengths), self.assertRaisesRegex(ValueError, "1 or more in all"):
                genome.site_sequences(adata, self.fasta, **lengths)
        with mock.patch.object(sys, "argv", ["utrcalling", "tools"]):
            provenance.record(adata, matrix="fractions")
        with self.assertRaisesRegex(ValueError, "holds fractions: give the count matrix"):
            genome.site_sequences(adata, self.fasta)


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
