"""Tests for utrcalling.core.protocols.kde_options: the settings of KDE peak calling."""

import subprocess
import sys
import unittest

from utrcalling.core.protocols import kde_options


class TestKernels(unittest.TestCase):
    def test_the_kernels_are_scikit_learns(self):
        import numpy as np
        from sklearn.neighbors import KernelDensity

        sizes = np.array([[10], [11], [14]])
        for kernel in kde_options.KERNELS:
            with self.subTest(kernel=kernel):
                density = KernelDensity(kernel=kernel, bandwidth=2).fit(sizes).score_samples(sizes)
                self.assertTrue(np.isfinite(density).all())

    def test_check_kernel(self):
        self.assertEqual(kde_options.check_kernel("linear"), "linear")
        with self.assertRaisesRegex(ValueError, "Unknown kernel 'Linear': expected one of"):
            kde_options.check_kernel("Linear")


class TestParseBandwidth(unittest.TestCase):
    """A bandwidth is a number of nt, or a rule with an optional factor."""

    def test_numbers(self):
        cases = [("12", 12), (12, 12), ("12.0", 12), (12.0, 12), ("7.5", 7.5), (" 3 ", 3)]
        cases.append(("1e1", 10))
        for value, expected in cases:
            with self.subTest(value=value):
                parsed = kde_options.parse_bandwidth(value)
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
                self.assertEqual(kde_options.parse_bandwidth(value), expected)

    def test_zero_only_where_allowed(self):
        self.assertEqual(kde_options.parse_bandwidth("0", allow_zero=True), 0)
        with self.assertRaisesRegex(ValueError, "bandwidth '0': must be above 0"):
            kde_options.parse_bandwidth("0")

    def test_other_values_are_refused(self):
        cases = {
            "-3": "must be above 0",
            "wide": "expected a number of nt, scott or silverman",
            "inf": "expected a number of nt",
            "nan": "expected a number of nt",
            "scott:x": "the factor of scott must be a number",
            "scott:0": "the factor of scott must be above 0",
            "scott:-1": "the factor of scott must be above 0",
            "gaussian:2": "expected a number of nt",
        }
        for value, message in cases.items():
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                kde_options.parse_bandwidth(value)

    def test_bandwidth_rule(self):
        self.assertEqual(kde_options.bandwidth_rule("scott:0.5"), ("scott", 0.5))
        self.assertEqual(kde_options.bandwidth_rule("Silverman"), ("silverman", 1.0))

    def test_bandwidth_label(self):
        cases = {12: "12", "12.0": "12", 7.5: "7.5", 0: "0", "scott": "scott"}
        cases |= {"scott:0.5": "scott0.5", "silverman:2": "silverman2"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(kde_options.bandwidth_label(value), expected)


class TestImportsLightly(unittest.TestCase):
    def test_the_command_line_checks_the_settings_without_numpy(self):
        # The CLI imports these settings as it starts; numpy and scikit-learn
        # would slow every command, --help included.
        code = (
            "import sys, utrcalling.core.protocols.kde_options;"
            "print('numpy' in sys.modules or 'sklearn' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
