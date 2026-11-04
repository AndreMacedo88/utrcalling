import os
import tempfile
import unittest

import pandas as pd

from utrcalling.core.helpers import (
    check_if_path_and_open,
    shutdown_joblib_pool,
    to_string,
)


class TestToString(unittest.TestCase):
    def test_bytes_to_str(self):
        self.assertEqual(to_string(b"ACGT"), "ACGT")

    def test_empty_bytes(self):
        self.assertEqual(to_string(b""), "")

    def test_long_sequence(self):
        seq = b"ACGTACGTACGTACGT"
        self.assertEqual(to_string(seq), "ACGTACGTACGTACGT")


class TestCheckIfPathAndOpen(unittest.TestCase):
    def test_pass_through_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = check_if_path_and_open(df)
        pd.testing.assert_frame_equal(result, df)

    def test_pass_through_none(self):
        result = check_if_path_and_open(None)
        self.assertIsNone(result)

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            tmp_path = f.name
        try:
            result = check_if_path_and_open(tmp_path, method="csv")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertListEqual(list(result.columns), ["col1", "col2"])
        finally:
            os.unlink(tmp_path)

    def test_load_parquet(self):
        df = pd.DataFrame({"x": [10, 20]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        df.to_parquet(tmp_path)
        try:
            result = check_if_path_and_open(tmp_path, method="parquet")
            pd.testing.assert_frame_equal(result, df)
        finally:
            os.unlink(tmp_path)

    def test_unsupported_method_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name
        try:
            with self.assertRaises(RuntimeError):
                check_if_path_and_open(tmp_path, method="xml")
        finally:
            os.unlink(tmp_path)


class TestShutdownJoblibPool(unittest.TestCase):
    def test_idempotent_when_no_pool_exists(self):
        """Calling before any Parallel block ran must not raise."""
        # Force teardown of any pool from earlier tests, then call again.
        shutdown_joblib_pool()
        shutdown_joblib_pool()  # second call must also be a no-op

    def test_releases_pool_after_parallel_run(self):
        """After a Parallel block creates the loky pool, shutdown should
        terminate the worker processes so RAM/file descriptors return to
        the OS instead of being held by the reusable executor."""
        import joblib
        from joblib.externals.loky import get_reusable_executor

        # Run a tiny parallel block to ensure the executor exists and has
        # spawned at least one worker.
        joblib.Parallel(n_jobs=2, backend="loky")(
            joblib.delayed(lambda x: x * x)(i) for i in range(4)
        )
        executor_before = get_reusable_executor()
        self.assertIsNotNone(executor_before)

        shutdown_joblib_pool()
        # After shutdown, get_reusable_executor() returns either a fresh
        # executor or the old one in a shut state. The contract we care
        # about is that submitting work to the prior executor fails, with the
        # RuntimeError that concurrent.futures raises for a shut-down executor.
        with self.assertRaises(RuntimeError):
            executor_before.submit(lambda: 1).result(timeout=2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
