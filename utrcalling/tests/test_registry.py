"""Tests for utrcalling.core.registry: named methods, built in, registered or from plugins."""

import subprocess
import sys
import unittest
from importlib.metadata import EntryPoint
from unittest import mock

from utrcalling.core.registry import Registry


def double(value):
    """A method for the tests' registries, importable by reference."""
    return 2 * value


def triple(value):
    return 3 * value


def _registry():
    return Registry(
        "test method",
        "utrcalling.test_methods",
        {"Double": "utrcalling.tests.test_registry:double"},
    )


def _entry_points(*points):
    """Patch importlib.metadata.entry_points to give *points* for the registry's group."""

    def entry_points(group):
        return [point for point in points if point.group == group]

    return mock.patch("importlib.metadata.entry_points", side_effect=entry_points)


class TestBuiltins(unittest.TestCase):
    def test_names_are_found_in_any_case_and_imported_when_used(self):
        registry = _registry()
        self.assertIn("double", registry)
        self.assertIn("DOUBLE", registry)
        self.assertNotIn(2, registry)
        self.assertEqual(registry.check("DoUbLe"), "double")
        self.assertIs(registry.get("double"), double)
        self.assertIs(registry.get("Double"), double)  # the same, from the cache

    def test_unknown_names_are_refused_with_the_known_ones(self):
        registry = _registry()
        registry.register("triple", triple)
        with self.assertRaisesRegex(
            ValueError, "Unknown test method 'half': expected one of double, triple"
        ):
            registry.get("half")
        with self.assertRaisesRegex(ValueError, "Unknown test method 'half'"):
            registry.check("half")

    def test_builtins_stay(self):
        registry = _registry()
        with self.assertRaisesRegex(
            ValueError, "'DOUBLE' is utrcalling's own: choose another name"
        ):
            registry.register("DOUBLE", triple)
        with self.assertRaisesRegex(ValueError, "'double' is utrcalling's own: it stays"):
            registry.unregister("double")
        self.assertIs(registry.get("double"), double)


class TestRegister(unittest.TestCase):
    def test_register_and_unregister(self):
        registry = _registry()
        self.assertIs(registry.register("Triple", triple), triple)
        self.assertIs(registry.get("triple"), triple)
        self.assertEqual(registry.names(), ("double", "triple"))
        registry.unregister("TRIPLE")
        self.assertNotIn("triple", registry)
        with self.assertRaisesRegex(ValueError, "No test method 'triple' was registered"):
            registry.unregister("triple")

    def test_as_a_decorator(self):
        registry = _registry()

        @registry.register("square")
        def square(value):
            return value**2

        self.assertIs(registry.get("square"), square)
        self.assertEqual(registry.get("square")(3), 9)

    def test_registering_again_replaces(self):
        registry = _registry()
        registry.register("times", double)
        registry.register("times", triple)
        self.assertIs(registry.get("times"), triple)


class TestPlugins(unittest.TestCase):
    """Other packages add methods as entry points of the registry's group."""

    def test_entry_points_are_methods(self):
        registry = _registry()
        point = EntryPoint(
            "Triple", "utrcalling.tests.test_registry:triple", "utrcalling.test_methods"
        )
        other_group = EntryPoint("half", "math:floor", "utrcalling.other")
        with _entry_points(point, other_group) as entry_points:
            self.assertIn("triple", registry)
            self.assertEqual(registry.names(), ("double", "triple"))
            self.assertIs(registry.get("triple"), triple)
            self.assertNotIn("half", registry)
        # The plugins are looked for once.
        self.assertEqual(entry_points.call_count, 1)

    def test_plugins_cannot_take_names_already_known(self):
        registry = _registry()
        registry.register("square", triple)
        points = [
            EntryPoint("double", "math:floor", "utrcalling.test_methods"),
            EntryPoint("square", "math:floor", "utrcalling.test_methods"),
        ]
        with _entry_points(*points) as entry_points:
            self.assertEqual(registry.names(), ("double", "square"))
            self.assertIs(registry.get("double"), double)
            self.assertIs(registry.get("square"), triple)
        entry_points.assert_called_once()

    def test_known_names_need_no_plugins(self):
        registry = _registry()
        with _entry_points() as entry_points:
            registry.get("double")
        entry_points.assert_not_called()

    def test_the_command_line_does_not_load_importlib_metadata(self):
        # It takes 40 ms: only names that are not utrcalling's own need it.
        code = (
            "import sys; from utrcalling.core.peak_calling import settings;"
            "settings.parse_bandwidth('scott:0.5'); settings.SCORES.check('pas');"
            "print('importlib.metadata' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
