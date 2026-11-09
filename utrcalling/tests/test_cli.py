"""CLI tests: subcommands parse, --help works, and the tool entry points behave."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd

os.chdir(str(Path(__file__).resolve().parents[0]))


def _run_cli(*args, timeout=30):
    """Run utrcalling CLI with the given args and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "utrcalling.runtime", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse(*tokens):
    """The function and the arguments, defaults included, that `utrcalling <tokens>` runs."""
    from utrcalling import runtime

    command, bound, _ = runtime.app.parse_args(
        runtime.hyphenate_long_options(list(tokens)), exit_on_error=False, print_error=False
    )
    bound.apply_defaults()
    return command, bound.arguments


def _help(*command):
    """The help page of `utrcalling <command>`."""
    from rich.console import Console

    from utrcalling import runtime

    console = Console(file=io.StringIO(), width=100)
    runtime.app([*command, "--help"], console=console)
    return console.file.getvalue()


RUN = ["run", "annotation.gtf", "-a", "sample.bam", "-A", "anno.csv"]

COUNT = ["count", "utr_reference.parquet", "-c", "mappings.parquet"]
MAP_READS = ["map-reads", "annotation.gtf", "-a", "sample.bam", "-A", "anno.csv"]

# The commands outside the pipeline, by group, and the arguments they need.
GROUPED_COMMANDS = {
    ("process", "filter-sites"): ["in.h5ad", "out.h5ad"],
    ("process", "keep-multi-site"): ["in.h5ad", "out.h5ad"],
    ("process", "rename-sites"): ["in.h5ad", "out.h5ad"],
    ("process", "concatenate"): ["-i", "a.h5ad", "b.h5ad", "-o", "out.h5ad"],
    ("tools", "to-fractions"): ["in.h5ad", "out.h5ad"],
    ("tools", "mean-sizes"): ["in.h5ad", "out.h5ad"],
    ("qc", "pas-distance"): ["in.h5ad", "out.tsv", "genome.fa"],
    ("qc", "score-bandwidths"): ["in.h5ad", "out.tsv", "genome.fa"],
}
# Every command, with the arguments it needs.
COMMANDS = {
    ("run",): RUN[1:],
    ("map-reads",): MAP_READS[1:],
    ("count",): COUNT[1:],
    ("call-peaks",): ["in.h5ad", "out"],
    **GROUPED_COMMANDS,
}
# The names before version 4.0.0, which the paper's workflows use.
OLD_NAMES = {
    ("map-reads",): "prepare-reference-and-map-reads",
    ("call-peaks",): "call-peaks-different-bandwidths",
    ("process", "filter-sites"): "remove-peaks-below-threshold",
    ("process", "keep-multi-site"): "select-alternative-transcription-events",
    ("process", "rename-sites"): "generate-names",
    ("process", "concatenate"): "concatenate-experiments",
    ("tools", "to-fractions"): "counts-to-percentages",
    ("tools", "mean-sizes"): "compute-mean-utr-sizes",
    ("qc", "pas-distance"): "check-distance-to-pas",
    ("qc", "score-bandwidths"): "estimate-best-bandwidth",
}


class TestCLIHelp(unittest.TestCase):
    """`--help` describes every command, grouped by purpose."""

    def test_main_help(self):
        result = _run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Count the molecules at each UTR site", result.stdout)

    def test_version(self):
        result = _run_cli("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Software version:", result.stdout)

    def test_main_help_lists_the_pipeline_and_the_command_groups(self):
        page = _help()
        for panel in ("Pipeline", "Command groups"):
            self.assertIn(panel, page)
        for command in ("run", "map-reads", "count", "call-peaks"):
            self.assertIn(command, page)
        for group in ("process", "tools", "qc"):
            self.assertIn(group, page)

    def test_group_help_lists_its_commands(self):
        for group, command in GROUPED_COMMANDS:
            with self.subTest(group=group, command=command):
                self.assertIn(command, _help(group))

    def test_old_names_are_not_listed(self):
        page = _help()
        for old_name in OLD_NAMES.values():
            with self.subTest(old_name=old_name):
                self.assertNotIn(old_name, page)

    def test_every_command_has_a_help_page(self):
        for command in COMMANDS:
            with self.subTest(command=command):
                self.assertIn(f"Usage: utrcalling {' '.join(command)}", _help(*command))

    def test_pipeline_options_are_grouped_into_panels(self):
        page = _help("run")
        for panel in ("Inputs", "Protocol and samples", "Peak calling", "Output and resources"):
            self.assertIn(panel, page)

    def test_bandwidth_defaults_are_shown_as_typed(self):
        self.assertIn("[default: 1 7 12 15]", _help("qc", "score-bandwidths"))


class TestOldCommandNames(unittest.TestCase):
    """The names before version 4.0.0, which the paper's workflows use, still work."""

    def test_old_and_new_names_run_the_same_command(self):
        for new_name, old_name in OLD_NAMES.items():
            arguments = COMMANDS[new_name]
            with self.subTest(old_name=old_name):
                self.assertEqual(_parse(old_name, *arguments), _parse(*new_name, *arguments))

    def test_call_peaks_with_read_mappings_is_count(self):
        # Before 5.0.0, call-peaks counted the sites of the read mappings given
        # with -c; call-peaks now calls the peaks of a raw-sizes count matrix.
        from utrcalling.runtime import update_old_command_lines

        for flag in (["-c", "mappings.parquet"], ["--utr_sizes_counts", "mappings.parquet"]):
            old = ["call-peaks", "utr_reference.parquet", *flag, "-r"]
            with self.subTest(flag=flag[0]):
                self.assertEqual(_parse(*update_old_command_lines(old)), _parse(*COUNT, "-r"))
        self.assertListEqual(
            update_old_command_lines(["call-peaks", "--utr-sizes-counts=m.parquet", "u.parquet"]),
            ["count", "--utr-sizes-counts=m.parquet", "u.parquet"],
        )
        new = ["call-peaks", "raw.h5ad", "peaks", "-b", "12"]
        self.assertListEqual(update_old_command_lines(new), new)


class TestOptionSpellings(unittest.TestCase):
    """Long options are spelled with hyphens, and with underscores as before 4.0.0."""

    def test_underscores_and_hyphens_give_the_same_options(self):
        options = ["--protocol_version", "2", "--kde_bandwidth", "30", "--no_log_file"]
        self.assertEqual(
            _parse(*RUN, *options), _parse(*RUN, *[o.replace("_", "-") for o in options])
        )
        _, arguments = _parse(*RUN, *options)
        self.assertEqual(arguments["protocol_version"], 2)
        self.assertEqual(arguments["kde_bandwidth"], 30)
        self.assertIs(arguments["no_log_file"], True)

    def test_only_the_option_name_changes(self):
        from utrcalling.runtime import hyphenate_long_options

        tokens = ["--out_path=my_out.h5ad", "-o", "my_out.h5ad", "--", "--not_an_option"]
        self.assertListEqual(
            hyphenate_long_options(tokens),
            ["--out-path=my_out.h5ad", "-o", "my_out.h5ad", "--", "--not_an_option"],
        )

    def test_several_alignments_follow_one_flag(self):
        _, arguments = _parse("run", "g.gtf", "-a", "x.bam", "y.bam", "-A", "a.csv", "-L", "True")
        self.assertListEqual(arguments["alignments"], ["x.bam", "y.bam"])
        self.assertIs(arguments["label_batches"], True)


class TestCLIInvalidArgs(unittest.TestCase):
    """Test that invalid arguments produce appropriate errors."""

    def assertRefused(self, *tokens, message):
        from cyclopts import CycloptsError

        with self.assertRaisesRegex(CycloptsError, message):
            _parse(*tokens)

    def test_run_no_args_fails(self):
        result = _run_cli("run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires an argument", result.stdout + result.stderr)

    def test_call_peaks_no_args_fails(self):
        result = _run_cli("call-peaks")
        self.assertEqual(result.returncode, 2)

    def test_summarization_level_is_utr_or_gene_in_any_case(self):
        self.assertEqual(_parse(*RUN, "-s", "UTR")[1]["summarization_level"], "utr")
        self.assertEqual(_parse(*RUN, "-s", "Gene")[1]["summarization_level"], "gene")
        self.assertEqual(_parse(*RUN)[1]["summarization_level"], "utr")
        self.assertRefused(*RUN, "-s", "transcript", message="expected one of utr, gene")

    def test_label_batches_takes_true_or_false(self):
        cases = {"True": True, "yes": True, "1": True, "False": False, "no": False, "0": False}
        for text, expected in cases.items():
            with self.subTest(label_batches=text):
                self.assertIs(_parse(*RUN, "-L", text)[1]["label_batches"], expected)
        self.assertIs(_parse(*RUN, "-L", "-i")[1]["label_batches"], True)
        self.assertIsNone(_parse(*RUN)[1]["label_batches"])
        self.assertRefused(*RUN, "-L", "maybe", message="expected true or false, got 'maybe'")

    def test_choices_are_checked(self):
        self.assertRefused(*RUN, "--strandedness", "sideways", message="sideways")
        self.assertRefused(*RUN, "-m", "both", message="both")

    def test_a_flag_without_its_values_is_refused(self):
        self.assertRefused("run", "g.gtf", "-a", "-A", "a.csv", message="at least 1")

    def test_nonexistent_subcommand(self):
        result = _run_cli("nonexistent-command")
        self.assertNotEqual(result.returncode, 0)
        self.assertRefused("process", "nonexistent-command", message="Unknown command")

    def test_no_args_prints_usage(self):
        result = _run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage: utrcalling COMMAND", result.stdout)


class TestProtocolVersionDefault(unittest.TestCase):
    """Without -v, the commands and their tool functions default to version 3.

    Version 2 used to be the command default, although only 10x and standard_10x
    implement it, so e.g. `-p smartseq` failed unless the user passed `-v 3`.
    """

    def test_commands_default_to_version_3(self):
        for argv in (RUN, MAP_READS, COUNT):
            name = argv[0]
            with self.subTest(command=name):
                self.assertEqual(_parse(*argv)[1]["protocol_version"], 3)

    def test_tool_functions_default_to_version_3(self):
        import inspect

        from utrcalling.commands.count import count
        from utrcalling.commands.map_reads import map_reads
        from utrcalling.commands.run import run

        for tool in (run, map_reads, count):
            with self.subTest(tool=tool.__name__):
                default = inspect.signature(tool).parameters["protocol_version"].default
                self.assertEqual(default, 3)


class _StopAfterProtocol(Exception):
    """Raised by the patched protocol factory to end the tool run early."""


class TestToolsForwardKdeBandwidth(unittest.TestCase):
    """`--kde_bandwidth` must reach the protocol, which builds the standard-path peak caller.

    Without it, the standard (non --raw_sizes) path silently uses the default
    bandwidth of 12 whatever the user passes.
    """

    def _protocol_kwargs(self, tool, *args, **kwargs):
        with (
            mock.patch("utrcalling.core.protocol.create", side_effect=_StopAfterProtocol) as create,
            self.assertRaises(_StopAfterProtocol),
        ):
            tool(*args, kde_bandwidth=150, log_level="error", no_log_file=True, **kwargs)
        return create.call_args.kwargs

    def test_run_forwards_kde_bandwidth(self):
        from utrcalling.commands.run import run

        kwargs = self._protocol_kwargs(
            run, "annotation.gtf", alignments=["sample.bam"], annotations_file="anno.csv"
        )
        self.assertEqual(kwargs["kde_bandwidth"], 150)

    def test_count_forwards_kde_bandwidth(self):
        from utrcalling.commands.count import count

        kwargs = self._protocol_kwargs(
            count, "utr_reference.parquet", utr_sizes_counts=["mappings.parquet"]
        )
        self.assertEqual(kwargs["kde_bandwidth"], 150)


class TestCountRecordsTheUtrEnd(unittest.TestCase):
    """5' protocols count UTR sizes from the UTR's 3' end, and their matrices say so."""

    def test_five_prime_capseq_mappings(self):
        from utrcalling import runtime
        from utrcalling.core import protocol, provenance, utr_mapping

        test_data = Path(__file__).resolve().parent / "test_data"
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir)
        reference = pd.read_parquet(test_data / "test_UTR_reference_5prime.parquet")
        reference.to_parquet(tmpdir / "reference.parquet")
        bam = str(test_data / "test_CapPhospho_genome_102_coordsort.bam")
        protocol_ = protocol.create(
            platform="smart-seq_cap-readname",
            version=3,
            alignments=bam,
            annotations=str(test_data / "smartseq3_anno.csv"),
        )
        intervals = utr_mapping.generate_utr_intervals(reference)
        utr_mapping.map_reads_to_utr(bam, intervals, protocol_).to_parquet(
            tmpdir / "mappings.parquet"
        )
        # CapSeq names each sample after its BAM file, not after a barcode.
        annotations = pd.read_csv(test_data / "smartseq3_anno.csv")
        annotations.drop(columns="Sample_Barcode").to_csv(tmpdir / "anno.csv", index=False)

        argv = ["utrcalling", "count", str(tmpdir / "reference.parquet")]
        argv += ["-c", str(tmpdir / "mappings.parquet"), "-p", "smart-seq_cap-readname"]
        argv += ["-A", str(tmpdir / "anno.csv"), "-o", str(tmpdir / "sites.h5ad")]
        argv += ["--no-log-file", "-l", "warning", "--cores", "1"]
        with mock.patch.object(sys, "argv", argv):
            runtime.main()

        info = provenance.recorded(ad.read_h5ad(tmpdir / "sites.h5ad"))
        self.assertEqual(info["utr_end"], "five_prime")
        self.assertEqual(info["matrix"], "peaks")
        self.assertEqual(info["protocol_platform"], "smart-seq_cap-readname")
        self.assertListEqual(info["commands"], [" ".join(argv)])


class TestScoreBandwidthsCommand(unittest.TestCase):
    """`utrcalling qc score-bandwidths` writes the scorer's table, given the genome."""

    def test_genome_reaches_the_scorer_and_the_table_is_written(self):
        from utrcalling.commands.qc.score_bandwidths import score_bandwidths

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir)
        genome_path = tmpdir / "genome.fa"
        genome_path.write_text(">chr1\nACGT\n")
        out_path = tmpdir / "scores.tsv"
        scores = pd.DataFrame(
            {"score": [0.25, 0.5], "sites": [9, 4], "sites_per_utr": [3.0, 4 / 3]},
            index=pd.Index([7, 12], name="bandwidth"),
        )

        with (
            mock.patch("utrcalling.core.helpers.check_if_path_and_open"),
            mock.patch(
                "utrcalling.core.peak_calling_anndata.score_bandwidths", return_value=scores
            ) as scorer,
        ):
            score_bandwidths(
                input_path="experiment.h5ad",
                out_path=str(out_path),
                genome_path=str(genome_path),
                bandwidths=(7, 12),
                hamming=0,
            )

        self.assertEqual(scorer.call_args.kwargs["genome"].fn, str(genome_path))
        self.assertEqual(scorer.call_args.kwargs["bandwidths"], (7, 12))
        self.assertEqual(scorer.call_args.kwargs["hamming"], 0)
        pd.testing.assert_frame_equal(pd.read_csv(out_path, sep="\t", index_col=0), scores)


class TestPasDistanceCommand(unittest.TestCase):
    """`utrcalling qc pas-distance`: each site's distance to the nearest PAS hexamer."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)
        self._write_sites("+")

    def _write_sites(self, strand):
        # One UTR on chr1:100-200 (1-based) with molecules ending at UTR sizes
        # 40, 41 and 42: 5, 20 and 8 molecules over the four cells.
        var = pd.DataFrame(
            {
                "utr_name": ["UTR0"] * 3,
                "utr_size": [40, 41, 42],
                "gene_name": ["Gene0"] * 3,
                "chrom": ["chr1"] * 3,
                "start": [100] * 3,
                "end": [200] * 3,
                "strand": [strand] * 3,
            },
            index=["Gene0_1-40", "Gene0_2-41", "Gene0_3-42"],
        )
        counts = np.array([[2, 5, 2], [1, 6, 1], [2, 4, 3], [0, 5, 2]], dtype=np.int32)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(4)])
        ad.AnnData(X=counts, obs=obs, var=var).write_h5ad(self.tmpdir / "sites.h5ad")

    def _table(self, chr1):
        (self.tmpdir / "genome.fa").write_text(f">chr1\n{chr1}\n")
        argv = ["utrcalling", "qc", "pas-distance", str(self.tmpdir / "sites.h5ad")]
        argv += [str(self.tmpdir / "distances.tsv"), str(self.tmpdir / "genome.fa")]
        argv += ["-H", "0", "--cores", "1"]
        from utrcalling import runtime

        with mock.patch.object(sys, "argv", argv):
            runtime.main()
        return pd.read_csv(self.tmpdir / "distances.tsv", sep="\t", index_col="site")

    def test_distance_is_counted_from_the_end_of_the_hexamer(self):
        # The UTR sequence starts at 0-based genome position 99, so an AATAAA at
        # 119 ends 26 nt into the UTR: sites 40, 41 and 42 are 14, 15 and 16 nt
        # downstream of it.
        table = self._table("C" * 119 + "AATAAA" + "C" * 175)
        self.assertListEqual(list(table.index), ["Gene0_1-40", "Gene0_2-41", "Gene0_3-42"])
        self.assertListEqual(table.pas_distance.tolist(), [14, 15, 16])
        self.assertListEqual(table.molecules.tolist(), [5, 20, 8])
        self.assertListEqual(table.pas_hexamer.tolist(), ["AATAAA"] * 3)
        # Distances are whole numbers in the file too.
        first_row = (self.tmpdir / "distances.tsv").read_text().splitlines()[1]
        self.assertEqual(first_row, "Gene0_1-40\tGene0\tUTR0\t40\t5\tAATAAA\t14")
        self.assertListEqual(table.utr_size.tolist(), [40, 41, 42])
        self.assertListEqual(table.gene_name.tolist(), ["Gene0"] * 3)

    def test_minus_strand_utrs_give_the_same_distances(self):
        # UTR sizes count from the UTR's 5' end in transcript orientation, which
        # on the - strand is genome position 200. TTTATT at 0-based 174-179 is
        # an AATAAA ending 26 nt into the UTR on that strand, as above.
        self._write_sites("-")
        table = self._table("C" * 174 + "TTTATT" + "C" * 120)
        self.assertListEqual(table.pas_distance.tolist(), [14, 15, 16])

    def test_utrs_without_a_hexamer_have_no_distance(self):
        # Used to crash: numpy cannot concatenate an empty list of arrays.
        table = self._table("C" * 300)
        self.assertTrue(table.pas_distance.isna().all())
        self.assertTrue(table.pas_hexamer.isna().all())
        self.assertListEqual(table.molecules.tolist(), [5, 20, 8])


class TestPytestFailsOnUtrcallingDeprecations(unittest.TestCase):
    """The pytest config turns deprecations raised from utrcalling code into failures.

    They flag behaviour that changes in upcoming dependency releases (e.g. pandas 3).
    pandas and anndata attribute these warnings to the calling module, so warnings
    from inside other packages must not fail the run.
    """

    PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

    def _pytest_exit_code(self, module):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "test_probe.py"
            probe.write_text(
                "import warnings\n\n\n"
                "def test_probe():\n"
                f"    warnings.warn_explicit('deprecated', FutureWarning, 'caller.py', 1, module={module!r})\n"
            )
            command = [sys.executable, "-m", "pytest", "-c", str(self.PYPROJECT)]
            command += ["--rootdir", tmp, "-p", "no:cacheprovider", "-q", str(probe)]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )
        return result.returncode

    def test_deprecation_in_utrcalling_code_fails(self):
        self.assertEqual(self._pytest_exit_code("utrcalling.core.caller"), 1)

    def test_deprecation_in_other_packages_does_not_fail(self):
        self.assertEqual(self._pytest_exit_code("anndata._core.caller"), 0)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
