"""CLI tests — verify that all subcommands parse correctly and --help works."""

import contextlib
import io
import os
import subprocess
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

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
