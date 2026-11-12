"""Tests for utrcalling.core.qc: the quality control tables."""

import gzip
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import scipy.sparse

from utrcalling.core import protocol, provenance, qc, utr_mapping
from utrcalling.core.utr_mapping import MAPPING_STEPS, _IntervalData
from utrcalling.tests.test_utr_mapping.test_read_filters import (
    ANNOTATIONS,
    _alignment,
    _MappingTest,
)


def _matrix(counts, utr_names, sizes, gene_names=None, obs=None, **var_columns):
    """A count matrix (cells x sites) with the .var columns utrcalling writes."""
    counts = np.asarray(counts, dtype=np.int32)
    var = pd.DataFrame(
        {"utr_name": utr_names, "utr_size": sizes, **var_columns},
        index=[
            f"{utr}_{i}-{size}" for i, (utr, size) in enumerate(zip(utr_names, sizes, strict=True))
        ],
    )
    if gene_names is not None:
        var["gene_name"] = gene_names
    if obs is None:
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(counts.shape[0])])
    return ad.AnnData(X=scipy.sparse.csr_matrix(counts), obs=obs, var=var)


def _record(adata, **record):
    with mock.patch.object(sys, "argv", ["utrcalling", "run"]):
        provenance.record(adata, **record)


def _steps(**counts):
    return dict.fromkeys(MAPPING_STEPS, 0) | counts


class TestMappingSummary(unittest.TestCase):
    def test_a_row_per_input_and_a_total(self):
        summary = qc.mapping_summary(
            ["a.bam", "b.bam"],
            [
                _steps(alignments=10, wrong_strand=4, reads=6, molecules=5),
                _steps(alignments=8, outside_utr=1, reads=7, molecules=7),
            ],
            {
                "molecules_after_merge": 12,
                "molecules_after_size_filter": 11,
                "molecules_in_cell_list": 9,
                "molecules_counted": 9,
            },
        )
        self.assertListEqual(list(summary.index), ["a.bam", "b.bam", "total"])
        self.assertEqual(summary.index.name, "input")
        self.assertListEqual(list(summary.columns), [*MAPPING_STEPS, *qc.COUNTING_STEPS])
        self.assertTrue((summary.dtypes == "Int64").all())
        total = summary.loc["total"]
        self.assertEqual(total["alignments"], 18)
        self.assertEqual(total["wrong_strand"], 4)
        self.assertEqual(total["molecules"], 12)
        self.assertEqual(total["molecules_counted"], 9)
        # The counting steps are done on all inputs together.
        self.assertTrue(summary.loc["a.bam", list(qc.COUNTING_STEPS)].isna().all())

    def test_stored_mappings_without_their_mapping_steps(self):
        summary = qc.mapping_summary(
            ["a.parquet", "b.parquet"],
            [{"molecules": 5}, {"molecules": 7}],
            {"molecules_after_merge": 12, "molecules_counted": 10},
        )
        self.assertListEqual(
            list(summary.columns), ["molecules", "molecules_after_merge", "molecules_counted"]
        )
        self.assertListEqual(summary.molecules.tolist(), [5, 7, 12])

    def test_inputs_with_different_steps_share_only_the_common_ones(self):
        summary = qc.mapping_summary(
            ["a.parquet", "b.parquet"], [_steps(alignments=10, molecules=5), {"molecules": 7}], {}
        )
        self.assertEqual(summary.loc["a.parquet", "alignments"], 10)
        self.assertTrue(pd.isna(summary.loc["b.parquet", "alignments"]))
        self.assertTrue(pd.isna(summary.loc["total", "alignments"]))
        self.assertEqual(summary.loc["total", "molecules"], 12)


class TestJoinMappingSummaries(unittest.TestCase):
    """Concatenated experiments: the inputs of each, and their totals summed."""

    def setUp(self):
        self.first = qc.mapping_summary(
            ["a.bam"], [_steps(alignments=10, reads=6, molecules=5)], {"molecules_counted": 4}
        )
        self.second = qc.mapping_summary(
            ["b.bam", "c.bam"],
            [
                _steps(alignments=8, reads=7, molecules=7),
                _steps(alignments=2, reads=1, molecules=1),
            ],
            {"molecules_counted": 7},
        )

    def test_inputs_of_every_experiment_and_summed_totals(self):
        joined = qc.join_mapping_summaries([self.first, self.second])
        self.assertListEqual(list(joined.index), ["a.bam", "b.bam", "c.bam", "total"])
        self.assertListEqual(list(joined.columns), list(self.first.columns))
        self.assertTrue((joined.dtypes == "Int64").all())
        total = joined.loc["total"]
        self.assertEqual(total.alignments, 20)
        self.assertEqual(total.molecules, 13)
        self.assertEqual(total.molecules_counted, 11)

    def test_a_step_that_one_experiment_lacks_has_no_total(self):
        stored = qc.mapping_summary(["m.parquet"], [{"molecules": 3}], {"molecules_counted": 3})
        total = qc.join_mapping_summaries([self.first, stored]).loc["total"]
        self.assertTrue(pd.isna(total.alignments))
        self.assertEqual(total.molecules, 8)

    def test_an_experiment_without_a_summary(self):
        self.assertIsNone(qc.join_mapping_summaries([self.first, None]))


class TestCellMetrics(unittest.TestCase):
    """Each cell's molecules, sites, UTRs and relative UTR size."""

    def setUp(self):
        # UTR A: sites at 100 and 200. UTR B: 50. UTR C: a 5' UTR at size 0.
        self.adata = _matrix(
            [
                [2, 0, 1, 0],
                [1, 3, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 0, 0],
            ],
            utr_names=["A", "A", "B", "C"],
            sizes=[100, 200, 50, 0],
        )

    def test_counts_per_cell(self):
        cells = qc.cell_metrics(self.adata, min_molecules=2)
        self.assertEqual(cells.index.name, "cell")
        self.assertListEqual(list(cells.index), list(self.adata.obs_names))
        self.assertListEqual(cells.molecules.tolist(), [3, 4, 2, 0])
        self.assertListEqual(cells.sites.tolist(), [2, 2, 1, 0])
        self.assertListEqual(cells.utrs.tolist(), [2, 1, 1, 0])
        self.assertListEqual(cells.utrs_measured.tolist(), [1, 1, 1, 0])
        self.assertListEqual(cells.multi_site_utrs.tolist(), [0, 1, 0, 0])

    def test_relative_utr_size(self):
        # UTR A's molecules average 150 nt (6 molecules, 900 nt), B's 50 nt.
        # Cell 1: two molecules at 100/150 and one at 50/50; cell 2: one at
        # 100/150 and three at 200/150. UTR C's sizes are all 0: it does not count.
        relative = qc.cell_metrics(self.adata).relative_utr_size
        np.testing.assert_allclose(relative[:2], [(2 * 100 / 150 + 1) / 3, (100 + 600) / 150 / 4])
        self.assertTrue(relative[2:].isna().all())

    def test_dense_counts_give_the_same_table(self):
        dense = self.adata.copy()
        dense.X = dense.X.toarray()
        pd.testing.assert_frame_equal(qc.cell_metrics(dense), qc.cell_metrics(self.adata))

    def test_fractions_are_refused(self):
        _record(self.adata, matrix="fractions")
        with self.assertRaisesRegex(ValueError, "holds fractions: give a count matrix"):
            qc.cell_metrics(self.adata)


class TestDepthCorrelation(unittest.TestCase):
    def test_spearman_over_the_cells_with_a_relative_size(self):
        cells = pd.DataFrame(
            {"molecules": [1, 2, 3, 4, 0], "relative_utr_size": [0.5, 0.7, 0.9, 1.2, np.nan]}
        )
        self.assertEqual(qc.depth_correlation(cells), (1.0, 4))

    def test_fewer_than_three_cells(self):
        cells = pd.DataFrame({"molecules": [1, 2], "relative_utr_size": [0.5, 0.7]})
        rho, n_cells = qc.depth_correlation(cells)
        self.assertTrue(np.isnan(rho))
        self.assertEqual(n_cells, 2)


class TestBatchConsistency(unittest.TestCase):
    """Pseudobulk site usage of groups of cells, compared pair by pair."""

    def setUp(self):
        obs = pd.DataFrame(
            {"batch": ["b1", "b1", "b2", None]}, index=[f"cell_{i}" for i in range(4)]
        )
        # b1 sums to [12, 8, 10, 10, 1], b2 to [2, 8, 4, 16, 0]; cell_3 has no batch.
        self.adata = _matrix(
            [
                [6, 4, 10, 0, 0],
                [6, 4, 0, 10, 1],
                [2, 8, 4, 16, 0],
                [100, 0, 0, 0, 0],
            ],
            utr_names=["A", "A", "B", "B", "C"],
            sizes=[100, 200, 50, 80, 30],
            obs=obs,
        )

    def test_groups(self):
        groups, _ = qc.batch_consistency(self.adata, "batch")
        expected = pd.DataFrame(
            {
                "cells": [2, 1],
                "molecules": [41, 30],
                "sites": [5, 4],
                "sites_only_here": [1, 0],  # UTR C's site
                "molecules_only_here": [1, 0],
            },
            index=pd.Index(["b1", "b2"], name="group"),
        )
        pd.testing.assert_frame_equal(groups, expected)

    def test_pairs(self):
        # Usage of A's and B's sites: b1 [.6, .4, .5, .5], b2 [.2, .8, .2, .8].
        # UTR C has a single site, so it is not compared.
        _, pairs = qc.batch_consistency(self.adata, "batch")
        self.assertListEqual(list(pairs.index), [("b1", "b2")])
        self.assertListEqual(list(pairs.index.names), ["group_1", "group_2"])
        pair = pairs.loc[("b1", "b2")]
        self.assertEqual(pair.utrs, 2)
        self.assertAlmostEqual(pair.correlation, -np.sqrt(0.5))
        # Half the summed differences: A (.4 + .4) / 2, B (.3 + .3) / 2.
        self.assertAlmostEqual(pair.usage_difference, (0.4 + 0.3) / 2)

    def test_utrs_with_too_few_molecules_are_not_compared(self):
        _, pairs = qc.batch_consistency(self.adata, "batch", min_molecules=25)
        pair = pairs.loc[("b1", "b2")]
        self.assertEqual(pair.utrs, 0)
        self.assertTrue(np.isnan(pair.correlation))
        self.assertTrue(np.isnan(pair.usage_difference))

    def test_every_pair_of_groups(self):
        self.adata.obs["batch"] = ["b1", "b2", "b3", "b3"]
        groups, pairs = qc.batch_consistency(self.adata, "batch")
        self.assertListEqual(list(groups.index), ["b1", "b2", "b3"])
        self.assertListEqual(list(pairs.index), [("b1", "b2"), ("b1", "b3"), ("b2", "b3")])

    def test_an_unknown_column_is_refused(self):
        with self.assertRaisesRegex(KeyError, "No column 'sample'.*choose one of batch"):
            qc.batch_consistency(self.adata, "sample")


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
        return qc.site_positions(var, utr_end)[0]

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
            qc.site_positions(var, "three_prime")

    def test_an_unknown_utr_end_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expected three_prime or five_prime"):
            qc.site_positions(pd.DataFrame(), "middle")


class TestReadKnownSites(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_bed_with_header_lines_and_strands(self):
        text = (
            "track name=sites\n# a comment\nchr1\t10\t20\tsite1\t5\t+\nchr2\t30\t31\tsite2\t1\t-\n"
        )
        path = self.tmp / "sites.bed.gz"
        with gzip.open(path, "wt") as bed:
            bed.write(text)
        known = qc.read_known_sites(path)
        expected = pd.DataFrame(
            {"chrom": ["chr1", "chr2"], "start": [10, 30], "end": [20, 31], "strand": ["+", "-"]}
        )
        pd.testing.assert_frame_equal(known, expected)

    def test_lines_without_a_strand_among_lines_with_one(self):
        path = self.tmp / "sites.bed"
        path.write_text("1\t10\t20\tsite1\t5\t-\n1\t30\t31\n")
        self.assertListEqual(qc.read_known_sites(path).strand.tolist(), ["-", "."])

    def test_bed_without_strands(self):
        path = self.tmp / "sites.bed"
        path.write_text("1\t10\t20\n")
        self.assertListEqual(qc.read_known_sites(path).strand.tolist(), ["."])


class TestDistanceToNearest(unittest.TestCase):
    def test_inside_left_and_right_of_intervals(self):
        # [10, 20) contains 15, although [12, 14), which starts later, ends before.
        distances = qc._distance_to_nearest(
            np.array([5, 15, 25, 100]), np.array([30, 12, 10]), np.array([31, 14, 20])
        )
        np.testing.assert_array_equal(distances, [5, 0, 5, 70])


class TestKnownSiteDistances(unittest.TestCase):
    def setUp(self):
        # Three + strand sites on chromosome 1, at 1100, 1200 and 1300; one -
        # strand site on 2 at 1400; and one on 3, which no known site has.
        self.adata = _matrix(
            [[1, 2, 3, 4, 5]],
            utr_names=["A", "A", "A", "B", "C"],
            sizes=[99, 199, 299, 100, 10],
            gene_names=["GA", "GA", "GA", "GB", "GC"],
            chrom=["1", "1", "1", "2", "3"],
            start=[1001] * 5,
            end=[1500] * 5,
            strand=["+", "+", "+", "-", "+"],
        )
        _record(self.adata, utr_end="three_prime", summarization_level="utr")
        # In BED coordinates: 1196-1200 (+), 1096 on the - strand, and 1406 on
        # either strand of chr2.
        self.known = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [1195, 1095, 1405],
                "end": [1200, 1096, 1406],
                "strand": ["+", "-", "."],
            }
        )

    def test_distances_to_known_sites_on_the_same_strand(self):
        table = qc.known_site_distances(self.adata, self.known, window=20)
        self.assertEqual(table.index.name, "site")
        self.assertListEqual(table.position.tolist(), [1100, 1200, 1300, 1400, 1011])
        pd.testing.assert_extension_array_equal(
            table.known_site_distance.array, pd.array([96, 0, 100, 6, None], dtype="Int64")
        )
        self.assertListEqual(table.near_known_site.tolist(), [False, True, False, True, False])
        self.assertListEqual(table.molecules.tolist(), [1, 2, 3, 4, 5])
        self.assertListEqual(table.gene_name.tolist(), ["GA", "GA", "GA", "GB", "GC"])

    def test_the_utr_end_comes_from_the_record_unless_given(self):
        five_prime = qc.known_site_distances(self.adata, self.known, utr_end="five_prime")
        self.assertListEqual(five_prime.position.tolist()[:3], [1402, 1302, 1202])
        _record(self.adata, utr_end="five_prime")
        pd.testing.assert_frame_equal(qc.known_site_distances(self.adata, self.known), five_prime)

    def test_matrices_without_a_record_need_the_utr_end(self):
        del self.adata.uns["utrcalling"]
        with self.assertRaisesRegex(ValueError, "does not record which end"):
            qc.known_site_distances(self.adata, self.known)
        table = qc.known_site_distances(self.adata, self.known, utr_end="three_prime")
        self.assertEqual(table.position.iloc[0], 1100)

    def test_gene_level_matrices_are_refused(self):
        _record(self.adata, summarization_level="gene")
        with self.assertRaisesRegex(ValueError, "Gene-level sites have no genome positions"):
            qc.known_site_distances(self.adata, self.known)

    def test_known_sites_of_another_assembly_are_refused(self):
        known = self.known.assign(chrom="NC_000001.11")
        with self.assertRaisesRegex(ValueError, "same genome assembly"):
            qc.known_site_distances(self.adata, known)


def _codes(*sequences):
    """Equally long sequences as rows of ASCII codes, as _edit_distances takes them."""
    return np.frombuffer("".join(sequences).encode(), dtype=np.uint8).reshape(len(sequences), -1)


def _write_genome(path, chromosomes):
    with open(path, "w") as fasta:
        for name, sequence in chromosomes.items():
            fasta.write(f">{name}\n{sequence}\n")


def _plant(genome, start, sequence):
    """Write *sequence* into the list *genome* from 0-based *start*."""
    genome[start : start + len(sequence)] = sequence


class TestEditDistances(unittest.TestCase):
    def test_substitutions_insertions_and_deletions_count_one(self):
        distances = qc._edit_distances(
            _codes("TTATATGGG", "TTATACGGG", "TATATGGGA", "CCCCCCGGG"), "TTATATGGG"
        )
        # Identical; one substitution; shifted by one: a deletion and an insertion.
        self.assertListEqual(distances.tolist(), [0, 1, 2, 6])

    def test_sequences_and_patterns_of_different_lengths(self):
        self.assertListEqual(qc._edit_distances(_codes("TTATAGGG"), "TTATATGGG").tolist(), [1])
        self.assertListEqual(qc._edit_distances(_codes("TTATATGGGG"), "TTATATGGG").tolist(), [1])

    def test_n_in_the_pattern_matches_any_base_but_n_in_the_genome_matches_none(self):
        distances = qc._edit_distances(_codes("ACGTACGGG", "ACGTACCCC"), "NNNNNNGGG")
        self.assertListEqual(distances.tolist(), [0, 3])
        self.assertListEqual(qc._edit_distances(_codes("NNNNNNGGG"), "TTATATGGG").tolist(), [6])


class TestInternalPriming(unittest.TestCase):
    """3' sites with A's in the genome after them, on chromosome 1 of 1000 nt."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        genome = list("C" * 1000)
        # + strand sites at 100, 200, 300 and 400 (1-based): the nt after them
        # start at the same 0-based positions.
        _plant(genome, 100, "aaaaaa")  # 6 A's in a row; soft-masked
        _plant(genome, 110, "R")  # not A, C, G or T
        _plant(genome, 200, "AAGAAGAAAC")  # 7 A's in 10 nt
        _plant(genome, 300, "AAAAACACCC")  # 5 in a row, 6 in 10 nt
        _plant(genome, 420, "AAAAAAA")  # 20 nt after the site
        # The - strand site at 600 reads the genome before it, reverse complemented.
        _plant(genome, 590, "TTTTTTTT")
        _write_genome(self.tmp / "genome.fa", {"chr1": "".join(genome)})
        self.adata = _matrix(
            [[1, 2, 3, 4, 5, 6, 7]],
            utr_names=["P", "P", "P", "P", "M", "E", "X"],
            sizes=[99, 199, 299, 399, 400, 989, 10],
            gene_names=["GP", "GP", "GP", "GP", "GM", "GE", "GX"],
            chrom=["1", "1", "1", "1", "1", "1", "2"],
            start=[1] * 7,
            end=[1000] * 7,
            strand=["+", "+", "+", "+", "-", "+", "+"],
        )
        _record(self.adata, utr_end="three_prime", summarization_level="utr")

    def test_a_rich_genome_after_the_site(self):
        table = qc.priming_artefacts(self.adata, self.tmp / "genome.fa")
        self.assertListEqual(table.position.tolist(), [100, 200, 300, 400, 600, 990, 11])
        self.assertListEqual(table.molecules.tolist(), [1, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(
            table.downstream.tolist()[:6],
            [
                "AAAAAACCCCNCCCCCCCCC",
                "AAGAAGAAACCCCCCCCCCC",
                "AAAAACACCCCCCCCCCCCC",
                "C" * 20,
                "G" + "A" * 8 + "G" * 11,
                "C" * 10 + "N" * 10,  # past the chromosome's end
            ],
        )
        pd.testing.assert_extension_array_equal(
            table.longest_a_run.array, pd.array([6, 3, 5, 0, 8, 0, None], dtype="Int64")
        )
        pd.testing.assert_extension_array_equal(
            table.most_a_in_10_nt.array, pd.array([6, 7, 6, 0, 8, 0, None], dtype="Int64")
        )
        pd.testing.assert_extension_array_equal(
            table.internal_priming.array,
            pd.array([True, True, False, False, True, False, None], dtype="boolean"),
        )
        # Chromosome 2 is not in the genome.
        self.assertTrue(pd.isna(table.downstream.iloc[6]))

    def test_window(self):
        flags = qc.priming_artefacts(self.adata, self.tmp / "genome.fa", window=30).internal_priming
        self.assertTrue(flags.iloc[3])  # the A's 20 nt after the site
        short = qc.priming_artefacts(self.adata, self.tmp / "genome.fa", window=5)
        self.assertListEqual(short.downstream.tolist()[:3], ["AAAAA", "AAGAA", "AAAAA"])
        self.assertListEqual(short.most_a_in_10_nt.tolist()[:3], [5, 4, 5])
        with self.assertRaisesRegex(ValueError, "window 0"):
            qc.priming_artefacts(self.adata, self.tmp / "genome.fa", window=0)

    def test_chromosome_names_with_or_without_chr(self):
        expected = qc.priming_artefacts(self.adata, self.tmp / "genome.fa")
        fasta = (self.tmp / "genome.fa").read_text().replace(">chr1", ">1")
        (self.tmp / "unprefixed.fa").write_text(fasta)
        self.adata.var["chrom"] = ["chr1"] * 6 + ["chr2"]
        table = qc.priming_artefacts(self.adata, self.tmp / "unprefixed.fa")
        pd.testing.assert_frame_equal(table.drop(columns="chrom"), expected.drop(columns="chrom"))

    def test_a_genome_without_the_matrix_chromosomes_is_refused(self):
        _write_genome(self.tmp / "other.fa", {"NC_000001.11": "ACGT" * 10})
        with self.assertRaisesRegex(ValueError, "same genome assembly"):
            qc.priming_artefacts(self.adata, self.tmp / "other.fa")

    def test_gene_level_matrices_are_refused(self):
        _record(self.adata, summarization_level="gene")
        with self.assertRaisesRegex(ValueError, "Gene-level sites have no genome positions"):
            qc.priming_artefacts(self.adata, self.tmp / "genome.fa")


class TestStrandInvasion(unittest.TestCase):
    """5' sites with the genome before them like the TSO's 3' end, on chromosome 1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        genome = list("A" * 600)
        # + strand sites at 101, 201, 301 and 451 (1-based): the 9 nt before them
        # end at the 0-based position before.
        _plant(genome, 91, "TTATATGGG")  # the 3' end of 5' 10x's TSO
        _plant(genome, 191, "TTATACGGG")  # 1 substitution
        _plant(genome, 291, "CCCCCCGGG")  # 6 substitutions
        _plant(genome, 441, "TTATATGCC")  # 2 substitutions, 1 G before the site
        # The - strand site at 400 reads the genome after it, reverse complemented.
        _plant(genome, 400, "CCCATATAA")
        _write_genome(self.tmp / "genome.fa", {"chr1": "".join(genome)})
        self.adata = _matrix(
            [[1, 2, 3, 4, 5]],
            utr_names=["P", "P", "P", "P", "M"],
            sizes=[400, 300, 200, 50, 400],
            chrom=["1"] * 5,
            start=[1] * 5,
            end=[500] * 5,
            strand=["+", "+", "+", "+", "-"],
        )
        _record(
            self.adata,
            utr_end="five_prime",
            summarization_level="utr",
            protocol_platform="5prime10x",
            protocol_version=1,
        )

    def _table(self, **settings):
        return qc.priming_artefacts(self.adata, self.tmp / "genome.fa", **settings)

    def test_genome_before_the_site_like_the_tso_of_the_recorded_protocol(self):
        table = self._table()
        self.assertListEqual(table.position.tolist(), [101, 201, 301, 451, 400])
        self.assertListEqual(
            table.upstream.tolist(),
            ["TTATATGGG", "TTATACGGG", "CCCCCCGGG", "TTATATGCC", "TTATATGGG"],
        )
        self.assertListEqual(table.tso_edit_distance.tolist(), [0, 1, 6, 2, 0])
        self.assertListEqual(table.g_before_site.tolist(), [3, 3, 3, 1, 3])
        self.assertListEqual(table.strand_invasion.tolist(), [True, True, False, False, True])
        self.assertNotIn("internal_priming", table)

    def test_max_edit_distance(self):
        flags = self._table(max_edit_distance=0).strand_invasion
        self.assertListEqual(flags.tolist(), [True, False, False, False, True])

    def test_smart_seq3_matches_any_umi(self):
        _record(self.adata, protocol_platform="smartseq_cap", protocol_version=3)
        table = self._table()
        self.assertListEqual(table.tso_edit_distance.tolist(), [0, 0, 0, 2, 0])
        self.assertListEqual(table.strand_invasion.tolist(), [True, True, True, False, True])

    def test_a_tso_given_replaces_the_recorded_one(self):
        table = self._table(tso="aaaaaaaccccccggg")
        self.assertListEqual(table.tso_edit_distance.tolist(), [6, 5, 0, 8, 6])

    def test_a_tso_shorter_than_9_nt(self):
        table = self._table(tso="GGG")
        self.assertListEqual(table.tso_edit_distance.tolist(), [0, 0, 0, 2, 0])
        # Only the last 3 nt before the site count.
        self.assertEqual(table.upstream.iloc[0], "TTATATGGG")

    def test_no_tso(self):
        # A protocol without template switching, or no protocol recorded.
        _record(self.adata, protocol_platform="10x", protocol_version=3)
        with self.assertRaisesRegex(ValueError, "needs the template-switching oligo"):
            self._table()
        del self.adata.uns["utrcalling"]
        with self.assertRaisesRegex(ValueError, "needs the template-switching oligo"):
            self._table(utr_end="five_prime")
        flags = self._table(utr_end="five_prime", tso="GGG").strand_invasion
        self.assertListEqual(flags.tolist(), [True, True, True, False, True])

    def test_a_tso_that_is_not_dna_is_refused(self):
        for tso in ("TTXGGG", ""):
            with self.subTest(tso=tso), self.assertRaisesRegex(ValueError, "A, C, G, T and N"):
                self._table(tso=tso)


class TestMatrixReport(unittest.TestCase):
    def setUp(self):
        self.adata = _matrix(
            [[2, 0, 1], [1, 3, 0], [0, 0, 4], [5, 5, 5]],
            utr_names=["A", "A", "B"],
            sizes=[100, 200, 50],
            gene_names=["GA", "GA", "GB"],
        )

    def test_counts(self):
        summary = qc.mapping_summary(
            ["a.bam"],
            [_steps(alignments=100, reads=40, molecules=30)],
            {"molecules_counted": 26},
        )
        _record(self.adata, matrix="peaks", bandwidth=12, mapping_summary=summary)
        report = qc.matrix_report(self.adata, min_molecules=2)
        cells = qc.cell_metrics(self.adata, min_molecules=2)
        expected = {
            "matrix": "peaks",
            "version": provenance.recorded(self.adata)["version"],
            "steps": 1,
            "cells": 4,
            "sites": 3,
            "utrs": 2,
            "genes": 2,
            "bandwidth": 12,
            "molecules": 26,
            "median_molecules_per_cell": 4.0,  # 3, 4, 4 and 15
            "median_sites_per_cell": 2.0,
            "median_utrs_measured_per_cell": 1.0,
            "multi_site_utrs": 1,
            "molecules_in_multi_site_utrs": 16 / 26,
            "depth_correlation": qc.depth_correlation(cells)[0],
            "mapping_alignments": 100,
            "mapping_reads": 40,
            "mapping_molecules": 30,
            "mapping_molecules_counted": 26,
        }
        self.assertDictEqual(report, expected)

    def test_matrices_without_a_record(self):
        report = qc.matrix_report(self.adata)
        self.assertEqual(
            (report["matrix"], report["version"], report["steps"]), ("unknown", "unknown", 0)
        )
        self.assertEqual(report["molecules"], 26)
        self.assertNotIn("mapping_reads", report)
        del self.adata.var["gene_name"]
        self.assertNotIn("genes", qc.matrix_report(self.adata))

    def test_fractions_and_mean_sizes(self):
        fractions = self.adata.copy()
        _record(fractions, matrix="fractions")
        self.assertEqual(qc.matrix_report(fractions)["median_values_per_cell"], 2.0)
        fractions.X = fractions.X.toarray()
        self.assertEqual(qc.matrix_report(fractions)["median_values_per_cell"], 2.0)
        # As tools mean-sizes writes them: a column per UTR, named after it.
        means = ad.AnnData(
            X=np.array([[150.0, np.nan], [np.nan, np.nan], [100.0, 50.0]]),
            var=pd.DataFrame({"gene_name": ["GA", "GB"]}, index=["A", "B"]),
        )
        _record(means, matrix="mean_sizes")
        report = qc.matrix_report(means)
        self.assertEqual(report["median_values_per_cell"], 1.0)
        self.assertEqual(report["utrs"], 2)
        self.assertNotIn("sites", report)
        self.assertNotIn("molecules", report)

    def test_mapping_summaries_of_stored_read_mappings(self):
        # count, given read mappings without their mapping steps, knows only molecules.
        summary = qc.mapping_summary(["a.parquet"], [{"molecules": 30}], {"molecules_counted": 26})
        _record(self.adata, matrix="peaks", mapping_summary=summary)
        report = qc.matrix_report(self.adata)
        self.assertEqual(report["mapping_molecules"], 30)
        self.assertNotIn("mapping_alignments", report)


class TestReportTable(unittest.TestCase):
    def test_columns_in_order_and_whole_numbers(self):
        table = qc.report_table(
            {
                "peaks.h5ad": {"matrix": "peaks", "cells": 4, "bandwidth": 12, "steps": 1},
                "fractions.h5ad": {"matrix": "fractions", "cells": 4, "steps": 2},
            }
        )
        self.assertEqual(table.index.name, "file")
        self.assertListEqual(list(table.columns), ["matrix", "steps", "bandwidth", "cells"])
        self.assertEqual(table.bandwidth.dtype, "Int64")
        self.assertTrue(pd.isna(table.loc["fractions.h5ad", "bandwidth"]))


if __name__ == "__main__":
    unittest.main()
