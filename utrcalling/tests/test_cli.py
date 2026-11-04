"""CLI tests: subcommands parse, --help works, and the tool entry points behave."""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import anndata as ad
import joblib
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


class TestCLIHelp(unittest.TestCase):
    """Test that --help works for the main parser and all subcommands."""

    def test_main_help(self):
        result = _run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Suite of tools", result.stdout)

    def test_version(self):
        result = _run_cli("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Software version:", result.stdout)

    def test_run_help(self):
        result = _run_cli("run", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("utr_annotation", result.stdout)

    def test_call_peaks_help(self):
        result = _run_cli("call-peaks", "--help")
        self.assertEqual(result.returncode, 0)

    def test_prepare_reference_help(self):
        result = _run_cli("prepare-reference-and-map-reads", "--help")
        self.assertEqual(result.returncode, 0)

    def test_run_tests_help(self):
        result = _run_cli("run-tests", "--help")
        self.assertEqual(result.returncode, 0)

    def test_remove_peaks_help(self):
        result = _run_cli("remove-peaks-below-threshold", "--help")
        self.assertEqual(result.returncode, 0)

    def test_select_alt_transcription_help(self):
        result = _run_cli("select-alternative-transcription-events", "--help")
        self.assertEqual(result.returncode, 0)

    def test_generate_names_help(self):
        result = _run_cli("generate-names", "--help")
        self.assertEqual(result.returncode, 0)

    def test_compute_mean_utr_sizes_help(self):
        result = _run_cli("compute-mean-utr-sizes", "--help")
        self.assertEqual(result.returncode, 0)

    def test_counts_to_percentages_help(self):
        result = _run_cli("counts-to-percentages", "--help")
        self.assertEqual(result.returncode, 0)

    def test_concatenate_experiments_help(self):
        result = _run_cli("concatenate-experiments", "--help")
        self.assertEqual(result.returncode, 0)

    def test_subset_experiments_help(self):
        result = _run_cli("subset-experiments", "--help")
        self.assertEqual(result.returncode, 0)

    def test_split_train_test_help(self):
        result = _run_cli("split-train-test", "--help")
        self.assertEqual(result.returncode, 0)


class TestCLIInvalidArgs(unittest.TestCase):
    """Test that invalid arguments produce appropriate errors."""

    def test_run_no_args_fails(self):
        result = _run_cli("run")
        self.assertNotEqual(result.returncode, 0)

    def test_call_peaks_no_args_fails(self):
        result = _run_cli("call-peaks")
        self.assertNotEqual(result.returncode, 0)

    def test_split_train_test_requires_out_path(self):
        # Without it, the tool used to crash on a None path after loading the data.
        from utrcalling.runtime import create_parser

        parser = create_parser()
        with (
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as ctx,
        ):
            parser.parse_args(["split-train-test", "experiment.h5ad"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--out_path", stderr.getvalue())

    def test_nonexistent_subcommand(self):
        result = _run_cli("nonexistent-command")
        self.assertNotEqual(result.returncode, 0)

    def test_no_args_prints_usage(self):
        result = _run_cli()
        # Should print usage and exit (returncode may be 0 or non-zero)
        # The important thing is it doesn't crash
        output = result.stdout.lower() + result.stderr.lower()
        self.assertTrue("usage" in output or "utrcalling" in output)


class TestCLIParserCreation(unittest.TestCase):
    """Test parser creation works without errors."""

    def test_create_parser(self):
        from utrcalling.runtime import create_parser

        parser = create_parser()
        self.assertIsNotNone(parser)

    def test_parse_version(self):
        from utrcalling.runtime import create_parser

        parser = create_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--version"])
        self.assertEqual(ctx.exception.code, 0)


class TestProtocolVersionDefault(unittest.TestCase):
    """Without -v, the commands and their tool functions default to version 3.

    Version 2 used to be the command default, although only 10x and standard_10x
    implement it, so e.g. `-p smartseq` failed unless the user passed `-v 3`.
    """

    def test_commands_default_to_version_3(self):
        from utrcalling.runtime import create_parser

        parser = create_parser()
        commands = {
            "run": ["run", "annotation.gtf", "-a", "sample.bam", "-A", "anno.csv"],
            "call-peaks": ["call-peaks", "utr_reference.bed", "-c", "mappings.parquet"],
            "prepare-reference-and-map-reads": [
                "prepare-reference-and-map-reads",
                "annotation.gtf",
                "-a",
                "sample.bam",
            ],
        }
        for name, argv in commands.items():
            with self.subTest(command=name):
                self.assertEqual(parser.parse_args(argv).protocol_version, 3)

    def test_tool_functions_default_to_version_3(self):
        import inspect

        from utrcalling.tools.call_peaks import call_peaks
        from utrcalling.tools.prepare_reference_and_map_reads import (
            generate_utr_reference_and_map_reads,
        )
        from utrcalling.tools.run import run

        for tool in (run, call_peaks, generate_utr_reference_and_map_reads):
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
        from utrcalling.tools.run import run

        kwargs = self._protocol_kwargs(run, "annotation.gtf", ["sample.bam"])
        self.assertEqual(kwargs["kde_bandwidth"], 150)

    def test_call_peaks_forwards_kde_bandwidth(self):
        from utrcalling.tools.call_peaks import call_peaks

        kwargs = self._protocol_kwargs(call_peaks, "utr_reference.bed", "mappings.parquet")
        self.assertEqual(kwargs["kde_bandwidth"], 150)


class TestSubsetExperimentsTool(unittest.TestCase):
    """`utrcalling subset-experiments` writes one subset per annotation file."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)

        self.input_path = self.tmpdir / "experiment.h5ad"
        ad.AnnData(
            X=np.arange(6, dtype=np.float32).reshape(3, 2),
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            var=pd.DataFrame(index=["p1", "p2"]),
        ).write_h5ad(self.input_path)

        self.obs_keep = self.tmpdir / "obs_keep.csv"
        pd.DataFrame({"val": [0, 1]}, index=["c1", "c3"]).to_csv(self.obs_keep)
        self.var_keep = self.tmpdir / "var_keep.csv"
        pd.DataFrame({"val": [0]}, index=["p2"]).to_csv(self.var_keep)

    def _subset(self, input_path, anno_types, out_paths):
        from utrcalling.tools.subset_experiments import subset_experiments

        subset_experiments(
            input_path=str(input_path),
            annotations_keep=[str(self.obs_keep), str(self.var_keep)],
            anno_types=anno_types,
            out_paths=[str(path) for path in out_paths],
        )

    def test_writes_one_subset_per_annotation_file(self):
        out_obs, out_var = self.tmpdir / "obs.h5ad", self.tmpdir / "var.h5ad"
        self._subset(self.input_path, ["obs", "var"], [out_obs, out_var])
        self.assertListEqual(list(ad.read_h5ad(out_obs).obs_names), ["c1", "c3"])
        self.assertListEqual(list(ad.read_h5ad(out_var).var_names), ["p2"])

    # The input path does not exist: these must fail before trying to read it.
    def test_fewer_output_paths_than_files_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "one output path per annotation file"):
            self._subset(self.tmpdir / "missing.h5ad", ["obs", "var"], [self.tmpdir / "a.h5ad"])

    def test_fewer_types_than_files_raises_value_error(self):
        out_paths = [self.tmpdir / "a.h5ad", self.tmpdir / "b.h5ad"]
        with self.assertRaisesRegex(ValueError, "one type per file"):
            self._subset(self.tmpdir / "missing.h5ad", ["obs"], out_paths)


class TestEstimateBestBandwidthTool(unittest.TestCase):
    """`utrcalling estimate-best-bandwidth` passes the genome to the scorer."""

    def test_genome_reaches_the_scorer(self):
        from utrcalling.tools.estimate_best_bandwidth import estimate_best_bandwidth

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir)
        genome_path = tmpdir / "genome.fa"
        genome_path.write_text(">chr1\nACGT\n")
        out_path = tmpdir / "scores.joblib"

        with (
            mock.patch("utrcalling.core.helpers.check_if_path_and_open"),
            mock.patch(
                "utrcalling.core.peak_calling_anndata.estimate_best_bandwidth",
                return_value=[0.25, 0.5],
            ) as scorer,
        ):
            estimate_best_bandwidth(
                input_path="experiment.h5ad",
                out_path=str(out_path),
                genome_path=str(genome_path),
                bandwidths=[7, 12],
            )

        self.assertEqual(scorer.call_args.kwargs["genome"].fn, str(genome_path))
        self.assertDictEqual(joblib.load(out_path), {7: 0.25, 12: 0.5})


class TestRunTestsExitCode(unittest.TestCase):
    """`utrcalling run-tests` must exit non-zero when tests fail, so scripts and CI notice."""

    def _run_suite(self, body):
        # Defined here, not at module level, so test discovery never collects it.
        class Probe(unittest.TestCase):
            def runTest(self):
                body(self)

        from utrcalling.tests import test_case

        with (
            mock.patch.object(
                unittest.TestLoader, "discover", return_value=unittest.TestSuite([Probe()])
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            test_case.run_all_tests(verbosity=0, logging=True)

    @staticmethod
    def _warn_from(module):
        # As pandas does: the warning is attributed to the module that called it.
        return lambda case: warnings.warn_explicit(
            "deprecated", FutureWarning, "caller.py", 1, module=module
        )

    def test_passing_suite_returns_normally(self):
        self._run_suite(lambda case: None)

    def test_failing_suite_exits_with_code_1(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_suite(lambda case: case.fail("deliberate failure"))
        self.assertEqual(ctx.exception.code, 1)

    def test_deprecation_in_utrcalling_code_exits_with_code_1(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_suite(self._warn_from("utrcalling.core.caller"))
        self.assertEqual(ctx.exception.code, 1)

    def test_deprecation_in_other_packages_does_not_fail(self):
        self._run_suite(self._warn_from("anndata._core.caller"))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
