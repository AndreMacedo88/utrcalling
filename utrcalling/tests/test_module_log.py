import io
import logging
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

from rich.logging import RichHandler

import utrcalling
from utrcalling.core.module_log import log_end_run, module_logger


class TestModuleLogger(unittest.TestCase):
    def setUp(self):
        # Reset the global logging disable threshold — run-tests sets
        # logging.disable(CRITICAL) which silences all messages globally.
        self._prev_disable = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        logger = logging.getLogger("utrcalling")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    def tearDown(self):
        logger = logging.getLogger("utrcalling")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        # Restore the previous global disable level
        logging.disable(self._prev_disable)

    def test_returns_logger(self):
        logger = module_logger()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "utrcalling")

    def test_disable_returns_unconfigured_logger(self):
        logger = module_logger(disable=True)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(len(logger.handlers), 0)

    def test_store_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = module_logger(store_log=True, log_dir=tmpdir)
            logger.info("test message")
            log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
            self.assertEqual(len(log_files), 1)
            self.assertIn("utrcalling.log", log_files[0])

    def test_log_file_has_no_ansi_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = module_logger(store_log=True, log_dir=tmpdir)
            logger.info("test message for ANSI check")
            # Flush handlers
            for handler in logger.handlers:
                handler.flush()
            log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
            log_path = os.path.join(tmpdir, log_files[0])
            with open(log_path) as f:
                content = f.read()
            self.assertNotIn("\x1b[", content)
            self.assertNotRegex(content, re.compile(r"\x1b\["))
            self.assertIn("test message for ANSI check", content)

    def test_log_file_captures_debug_when_console_is_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = module_logger(store_log=True, log_dir=tmpdir, level="info")
            logger.debug("debug breadcrumb")
            for handler in logger.handlers:
                handler.flush()
            log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
            log_path = os.path.join(tmpdir, log_files[0])
            with open(log_path) as f:
                content = f.read()
            self.assertIn("debug breadcrumb", content)

    def test_no_file_when_store_log_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module_logger(store_log=False, log_dir=tmpdir)
            log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
            self.assertEqual(len(log_files), 0)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError) as ctx:
            module_logger(level="bogus")
        self.assertIn("Invalid log level", str(ctx.exception))

    def test_singleton_returns_same_logger(self):
        logger1 = module_logger(disable=True)
        logger2 = module_logger(disable=True)
        self.assertIs(logger1, logger2)

    def test_log_end_run(self):
        module_logger()  # initialize first
        with self.assertLogs("utrcalling", level="INFO") as logs:
            log_end_run("done")
        self.assertEqual(logs.output, ["INFO:utrcalling:Job ended: done"])

    def test_plain_lines_outside_a_terminal(self):
        # Such as a workflow's log: the lines of the log file, which do not wrap.
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            logger = module_logger(level="warning")
            logger.info("not shown")
            logger.warning("a long warning " * 10)
        (handler,) = logger.handlers
        self.assertIs(type(handler), logging.StreamHandler)
        self.assertEqual(handler.level, logging.WARNING)
        (line,) = stderr.getvalue().splitlines()
        self.assertRegex(line, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d utrcalling » WARNING - ")
        self.assertEqual(line.split(" - ", 1)[1], "a long warning " * 10)  # one line

    def test_rich_lines_in_a_terminal(self):
        with mock.patch.object(sys.stderr, "isatty", return_value=True):
            logger = module_logger(level="debug")
        (handler,) = logger.handlers
        self.assertIsInstance(handler, RichHandler)
        self.assertEqual(handler.level, logging.DEBUG)

    def test_later_calls_keep_the_handlers(self):
        logger = module_logger()
        handlers = list(logger.handlers)
        self.assertIs(module_logger(level="debug"), logger)
        self.assertListEqual(logger.handlers, handlers)


class TestShowLogo(unittest.TestCase):
    def test_name_and_version(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout):
            utrcalling.show_logo()
        self.assertIn(f"/// utrcalling \\\\\\ 🧬 v{utrcalling.__version__} 🧬", stdout.getvalue())
        self.assertNotIn("\x1b[", stdout.getvalue())  # no colours outside a terminal


if __name__ == "__main__":
    unittest.main()
