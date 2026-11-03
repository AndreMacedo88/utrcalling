"""Tests for the unstranded paired-end bulk RNA-seq protocol (``bulk_pe``).

Covers the three behaviours that distinguish it from ``truseq``:
fragment-level positioning, one observation per fragment, and clamping of
fragments that overrun the annotated 3' end. The head-to-head test against
``truseq --strandedness unstranded`` on the *same* BAM is the regression that
motivates the protocol, so it is asserted rather than assumed.
"""

import os
import unittest
from pathlib import Path

import pandas as pd

from utrcalling.core import helpers, protocol, utr_mapping
from utrcalling.tests.test_data import synthetic_data_bulk_pe


class _BulkPeBase(unittest.TestCase):
    """Shared UTR regions and a mapping run over the synthetic PE BAM."""

    PLATFORM = "bulk_pe"
    STRANDEDNESS = None  # None -> the protocol default

    @classmethod
    def setUpClass(cls):
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        utr_regions = pd.DataFrame(
            {
                "utr_name": ["UTR1", "UTR2", "UTR3"],
                "chrom": [1, 1, 16],
                "start": [3102600, 172196728, 91270659],
                "end": [3102750, 172196828, 91271933],
                "name": ["UTR1", "UTR2", "UTR3"],
                "strand": ["+", "-", "+"],
                "transcript_id": ["UTR1_t", "UTR2_t", "UTR3_t"],
                "gene_name": ["Gene1", "Gene2", "Gene3"],
                "score": [0.0] * 3,
            }
        ).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)

    def setUp(self) -> None:
        self.maxDiff = None
        self.data = synthetic_data_bulk_pe.Bam()
        self.sample_id = Path(self.data.alignment).stem
        self.protocol = protocol.create(
            platform=self.PLATFORM,
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
            strandedness=self.STRANDEDNESS,
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                self.protocol,
                read=self.protocol.read_in_pair,
            )

    def _sizes(self, utr_name):
        return sorted(self.mappings[self.mappings["utr_name"] == utr_name]["utr_size"].tolist())


class TestBulkPeProperties(_BulkPeBase):
    """The protocol's configuration is what the pipeline branches on."""

    def test_is_bulk_pe_instance(self):
        self.assertIsInstance(self.protocol, protocol.ProtocolBulkPe)

    def test_defaults_to_unstranded(self):
        self.assertEqual(self.protocol.strandedness, "unstranded")

    def test_has_no_umi(self):
        self.assertFalse(self.protocol.has_umi)

    def test_accepts_any_mate(self):
        # narrowing to 'first' would drop fragments straddling the UTR start
        self.assertEqual(self.protocol.read_in_pair, "any")

    def test_dedups_by_fragment(self):
        self.assertEqual(self.protocol.dedup_strategy, "fragment")

    def test_uses_fragment_position(self):
        self.assertTrue(self.protocol.uses_fragment_position)

    def test_clamps_out_of_bounds(self):
        self.assertTrue(self.protocol.clamp_out_of_bounds)

    def test_explicit_strandedness_is_respected(self):
        prot = protocol.create(
            platform="bulk_pe",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
            strandedness="reverse",
        )
        self.assertEqual(prot.strandedness, "reverse")

    def test_all_aliases_resolve(self):
        for alias in ("bulk_pe", "bulk-pe", "standard_bulk_pe", "unstranded_bulk_pe"):
            with self.subTest(alias=alias):
                prot = protocol.create(
                    platform=alias,
                    version=1,
                    alignments=self.data.alignment,
                    annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
                )
                self.assertIsInstance(prot, protocol.ProtocolBulkPe)


class TestBulkPeMapping(_BulkPeBase):
    """End-to-end sizes produced from the synthetic paired-end BAM."""

    def test_utr3_sizes(self):
        # frag_proximal 141, frag_inside 551, read_unpaired 841,
        # frag_overrun clamped to the UTR end -> 1275
        self.assertEqual(self._sizes("UTR3"), [141, 551, 841, 1275])

    def test_utr2_minus_strand_size(self):
        self.assertEqual(self._sizes("UTR2"), [77])

    def test_utr_without_reads_is_absent(self):
        self.assertEqual(self._sizes("UTR1"), [])

    def test_one_observation_per_fragment(self):
        # 4 fragments/reads on UTR3 + 1 on UTR2, never 2 rows per pair
        self.assertEqual(len(self.mappings), 5)

    def test_observation_id_resolves_to_sample_id(self):
        self.assertEqual(set(self.mappings["observation_id"].unique()), {self.sample_id})

    def test_fragment_straddling_utr_start_is_kept(self):
        # only R2 of frag_proximal overlaps the interval; restricting to read 1
        # would silently drop it and deplete the proximal end of the UTR
        self.assertIn(141, self._sizes("UTR3"))

    def test_overrunning_fragment_is_clamped_not_dropped(self):
        utr3_len = 91271933 - (91270659 - 1)
        self.assertIn(utr3_len, self._sizes("UTR3"))


class TestBulkPeVersusTruseq(_BulkPeBase):
    """The head-to-head that motivates the protocol, on one BAM."""

    def _truseq_mappings(self):
        prot = protocol.create(
            platform="truseq",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
            strandedness="unstranded",
        )
        with self.data:
            return utr_mapping.map_reads_to_utr(
                self.data.alignment, self.intervals, prot, read=prot.read_in_pair
            )

    def test_truseq_unstranded_double_counts_a_fragment(self):
        truseq = self._truseq_mappings()
        sizes = sorted(truseq[truseq["utr_name"] == "UTR2"]["utr_size"].tolist())
        # both mates of frag_minus survive, at different distances from the UTR start
        self.assertEqual(sizes, [37, 77])

    def test_bulk_pe_counts_that_fragment_once(self):
        self.assertEqual(self._sizes("UTR2"), [77])

    def test_bulk_pe_produces_fewer_observations(self):
        truseq = self._truseq_mappings()
        self.assertLess(len(self.mappings), len(truseq))

    def test_bulk_pe_reaches_further_into_the_utr(self):
        # the fragment's distal base is downstream of either mate's own distal base
        truseq = self._truseq_mappings()
        truseq_inside = [s for s in truseq[truseq["utr_name"] == "UTR3"]["utr_size"].tolist()]
        self.assertIn(441, truseq_inside)  # R1's own distal base
        self.assertNotIn(441, self._sizes("UTR3"))  # superseded by the fragment end
        self.assertIn(551, self._sizes("UTR3"))


class TestNumericSampleIds(_BulkPeBase):
    """Annotation files with bare-integer sample IDs must work.

    The ITAN cohort names its samples with plain integers (102, 1003, ...), which
    pandas reads back as int64. ``re.escape`` rejects non-string input, so
    ``sample_in_path`` used to raise ``TypeError`` before the ids were coerced.
    Every previously shipped cohort had string ids (e.g. SRR21411313), so this
    path was never exercised.
    """

    def test_integer_sample_id_resolves(self):
        # the BAM stem is a temp name, so match on an integer token we inject
        int_id = 102
        bam = self.data.alignment
        self.assertFalse(helpers.sample_in_path(int_id, bam))  # must not raise
        self.assertTrue(helpers.sample_in_path(int_id, f"/data/star/{int_id}_wt_pe.bam"))

    def test_integer_sample_id_is_not_a_prefix_match(self):
        self.assertFalse(helpers.sample_in_path(10, "/data/star/102_wt_pe.bam"))
        self.assertFalse(helpers.sample_in_path(102, "/data/star/1023_wt_pe.bam"))

    def test_observation_ids_are_strings(self):
        self.assertTrue(all(isinstance(o, str) for o in self.mappings["observation_id"].unique()))

    def test_protocol_builds_with_integer_annotation_ids(self):
        prot = protocol.create(
            platform="bulk_pe",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [102, 1003]}),
        )
        self.assertIsInstance(prot, protocol.ProtocolBulkPe)


class TestKdeBandwidth(_BulkPeBase):
    """`kde_bandwidth` must reach the standard peak-calling path, not only --raw_sizes.

    Bandwidth 12 is tuned for 3'-end assays. On random-fragmentation data a wider
    kernel is needed, and before this the flag only affected the `--raw_sizes`
    output - the standard path silently used the default.
    """

    def test_default_bandwidth_is_unchanged(self):
        self.assertEqual(self.protocol.kde_bandwidth, 12)
        self.assertEqual(self.protocol.peak_calling.keywords["bandwidth"], 12)

    def test_custom_bandwidth_reaches_the_peak_caller(self):
        prot = protocol.create(
            platform="bulk_pe",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
            kde_bandwidth=150,
        )
        self.assertEqual(prot.kde_bandwidth, 150)
        self.assertEqual(prot.peak_calling.keywords["bandwidth"], 150)

    def test_bandwidth_accepted_by_every_protocol(self):
        for plat, ver in (("truseq", 1), ("smartseq", 3), ("10x", 3)):
            with self.subTest(platform=plat):
                prot = protocol.create(
                    platform=plat,
                    version=ver,
                    annotations=pd.DataFrame({"sample_id": [self.sample_id], "barcode": ["AAAA"]}),
                    kde_bandwidth=42,
                )
                self.assertEqual(prot.peak_calling.keywords["bandwidth"], 42)

    def test_no_peak_calling_leaves_transform_none(self):
        prot = protocol.create(
            platform="bulk_pe",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
            call_peaks=False,
            kde_bandwidth=150,
        )
        self.assertIsNone(prot.peak_calling)


class TestFragmentPositionArithmetic(unittest.TestCase):
    """Unit tests for the position function, independent of any BAM."""

    def setUp(self):
        self.fn = protocol.ProtocolBulkPe.get_fragment_position

    def test_plus_strand_uses_fragment_right_end(self):
        # fragment [1000, 1210): distal base 1209, from either mate
        from_left = self.fn(1000, 1099, False, "+", 1110, 210, True)
        from_right = self.fn(1110, 1209, True, "+", 1000, -210, True)
        self.assertEqual(from_left, 1209)
        self.assertEqual(from_right, 1209)

    def test_minus_strand_uses_fragment_left_end(self):
        from_left = self.fn(1000, 1099, False, "-", 1110, 210, True)
        from_right = self.fn(1110, 1209, True, "-", 1000, -210, True)
        self.assertEqual(from_left, 1000)
        self.assertEqual(from_right, 1000)

    def test_both_mates_agree(self):
        for strand in ("+", "-"):
            with self.subTest(strand=strand):
                self.assertEqual(
                    self.fn(1000, 1099, False, strand, 1110, 210, True),
                    self.fn(1110, 1209, True, strand, 1000, -210, True),
                )

    def test_falls_back_to_single_read_position_when_mate_unusable(self):
        for strand in ("+", "-"):
            with self.subTest(strand=strand):
                expected = protocol.ProtocolTruseq.get_read_position(1000, 1099, False, strand)
                self.assertEqual(self.fn(1000, 1099, False, strand, -1, 0, False), expected)

    def test_base_protocol_hook_delegates_to_get_read_position(self):
        # every existing protocol must be unaffected by the new hook
        expected = protocol.Protocol.get_read_position(1000, 1099, True, "+")
        self.assertEqual(
            protocol.Protocol.get_fragment_position(1000, 1099, True, "+", 1110, 210, True),
            expected,
        )


class TestDeduplicateFragments(unittest.TestCase):
    """Unit tests for the fragment de-duplicator."""

    def test_collapses_repeated_read_names(self):
        out = utr_mapping.deduplicate_fragments(
            read_names=["a", "a", "b"],
            read_info=[["S", "S", "S"], [10, 10, 20]],
        )
        self.assertEqual(out, [["S", "S"], [10, 20]])

    def test_keeps_distinct_names(self):
        out = utr_mapping.deduplicate_fragments(
            read_names=["a", "b", "c"],
            read_info=[["S", "S", "S"], [1, 2, 3]],
        )
        self.assertEqual(out, [["S", "S", "S"], [1, 2, 3]])

    def test_preserves_first_occurrence_and_order(self):
        out = utr_mapping.deduplicate_fragments(
            read_names=["b", "a", "b", "a"],
            read_info=[["S"] * 4, [1, 2, 3, 4]],
        )
        self.assertEqual(out, [["S", "S"], [1, 2]])

    def test_empty_input(self):
        self.assertEqual(
            utr_mapping.deduplicate_fragments(read_names=[], read_info=[[], []]),
            [[], []],
        )


if __name__ == "__main__":
    unittest.main()
