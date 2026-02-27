import logging
import os
import re
import tempfile
import unittest

from utrcalling.core.module_log import module_logger, log_end_run


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
            with open(log_path, "r") as f:
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
            with open(log_path, "r") as f:
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
        # Just ensure it runs without error
        module_logger()  # initialize first
        log_end_run("done")


if __name__ == "__main__":
    unittest.main()
