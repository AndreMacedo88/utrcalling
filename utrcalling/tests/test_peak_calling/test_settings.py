"""Tests for utrcalling.core.peak_calling.settings: the settings and methods of peak calling."""

import subprocess
import sys
import unittest

from utrcalling.core.peak_calling import settings


class TestKernels(unittest.TestCase):
    def test_the_kernels_are_scikit_learns(self):
        import numpy as np
        from sklearn.neighbors import KernelDensity

        sizes = np.array([[10], [11], [14]])
        for kernel in settings.KERNELS:
            with self.subTest(kernel=kernel):
                density = KernelDensity(kernel=kernel, bandwidth=2).fit(sizes).score_samples(sizes)
                self.assertTrue(np.isfinite(density).all())

    def test_check_kernel(self):
        self.assertEqual(settings.check_kernel("linear"), "linear")
        with self.assertRaisesRegex(ValueError, "Unknown kernel 'Linear': expected one of"):
            settings.check_kernel("Linear")


class TestParseBandwidth(unittest.TestCase):
    """A bandwidth is a number of nt, or a rule with an optional factor."""

    def test_numbers(self):
        cases = [("12", 12), (12, 12), ("12.0", 12), (12.0, 12), ("7.5", 7.5), (" 3 ", 3)]
        cases.append(("1e1", 10))
        for value, expected in cases:
            with self.subTest(value=value):
                parsed = settings.parse_bandwidth(value)
                self.assertEqual(parsed, expected)
                # Whole numbers name files as before: peaks_bw12.h5ad.
                self.assertIs(type(parsed), int if float(expected).is_integer() else float)

    def test_rules(self):
        cases = {
            "scott": "scott",
            "Scott": "scott",
            "scott:1": "scott",
            "scott:1.0": "scott",
            "scott:0.5": "scott:0.5",
            "SILVERMAN:2": "silverman:2",
            "silverman:0.25": "silverman:0.25",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(settings.parse_bandwidth(value), expected)

    def test_zero_only_where_allowed(self):
        self.assertEqual(settings.parse_bandwidth("0", allow_zero=True), 0)
        with self.assertRaisesRegex(ValueError, "bandwidth '0': must be above 0"):
            settings.parse_bandwidth("0")

    def test_other_values_are_refused(self):
        cases = {
            "-3": "must be above 0",
            "wide": "expected a number of nt or a rule \\(scott, silverman\\), as in scott:0.5",
            "inf": "expected a number of nt",
            "nan": "expected a number of nt",
            "scott:x": "the factor of scott must be a number above 0",
            "scott:0": "the factor of scott must be a number above 0",
            "scott:-1": "the factor of scott must be a number above 0",
            "scott:inf": "the factor of scott must be a number above 0",
            "gaussian:2": "expected a number of nt or a rule",
        }
        for value, message in cases.items():
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                settings.parse_bandwidth(value)

    def test_bandwidth_rule(self):
        self.assertEqual(settings.bandwidth_rule("scott:0.5"), ("scott", 0.5))
        self.assertEqual(settings.bandwidth_rule("Silverman"), ("silverman", 1.0))

    def test_setting_label(self):
        cases = {
            (12, "gaussian", "kde"): "bw12",
            ("scott:0.5", "linear", "kde"): "linear_bwscott0.5",
            (7.5, "gaussian", "mode"): "mode_bw7.5",
            (2, "linear", "mode"): "mode_linear_bw2",
        }
        for (bandwidth, kernel, method), expected in cases.items():
            with self.subTest(bandwidth=bandwidth, kernel=kernel, method=method):
                self.assertEqual(settings.setting_label(bandwidth, kernel, method), expected)
        self.assertEqual(settings.setting_label(12), "bw12")

    def test_bandwidth_label(self):
        cases = {12: "12", "12.0": "12", 7.5: "7.5", 0: "0", "scott": "scott"}
        cases |= {"scott:0.5": "scott0.5", "silverman:2": "silverman2"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(settings.bandwidth_label(value), expected)


class TestMethods(unittest.TestCase):
    """utrcalling's own methods, by name."""

    def test_names(self):
        self.assertEqual(settings.PEAK_CALLERS.names()[:1], ("kde",))
        self.assertEqual(settings.BANDWIDTH_RULES.names()[:2], ("scott", "silverman"))
        self.assertEqual(settings.SCORES.names()[:3], ("pas", "initiator", "known-sites"))

    def test_every_name_imports(self):
        from utrcalling.core.peak_calling.scores import SiteScore

        for name in ("pas", "initiator", "known-sites"):
            with self.subTest(score=name):
                self.assertTrue(issubclass(settings.SCORES.get(name), SiteScore))
        for name in ("scott", "silverman"):
            with self.subTest(rule=name):
                self.assertTrue(callable(settings.BANDWIDTH_RULES.get(name)))
        self.assertTrue(callable(settings.PEAK_CALLERS.get("kde")))

    def test_default_scores_are_scores(self):
        for utr_end, score in settings.DEFAULT_SCORES.items():
            with self.subTest(utr_end=utr_end):
                self.assertIn(score, settings.SCORES)


class TestImportsLightly(unittest.TestCase):
    def test_the_command_line_checks_the_settings_without_numpy(self):
        # The CLI imports these settings as it starts; numpy and scikit-learn
        # would slow every command, --help included.
        code = (
            "import sys, utrcalling.core.peak_calling.settings;"
            "print('numpy' in sys.modules or 'sklearn' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
