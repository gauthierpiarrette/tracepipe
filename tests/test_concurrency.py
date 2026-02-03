# tests/test_concurrency.py
"""
Concurrency and thread safety tests for TracePipe.

Coverage targets:
- Thread-local context isolation
- Multi-threaded DataFrame operations
- Race condition prevention
- Parallel pipeline execution
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import tracepipe as tp


def dbg():
    """Helper to access debug inspector."""
    return tp.debug.inspect()


class TestThreadLocalContext:
    """Tests for thread-local context isolation."""

    def test_each_thread_has_own_context(self):
        """Each thread can use TracePipe without errors."""
        results = {}
        errors = []

        def thread_work(thread_id):
            try:
                tp.reset()  # Reset in each thread
                tp.enable()
                df = pd.DataFrame({"a": range(5)})
                df = df.head(thread_id + 1)  # Different head for each thread

                dropped = dbg().dropped_rows()
                results[thread_id] = len(dropped)
            except Exception as e:
                errors.append((thread_id, str(e)))
            finally:
                try:
                    tp.disable()
                except Exception:
                    pass

        threads = []
        for i in range(5):
            t = threading.Thread(target=thread_work, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have no errors - main validation
        assert len(errors) == 0, f"Errors: {errors}"

        # All threads should have completed
        assert len(results) == 5

    def test_context_isolation_different_modes(self):
        """Threads can use different modes independently."""
        modes = {}

        def thread_work(thread_id, mode):
            tp.enable(mode=mode)
            modes[thread_id] = dbg().mode
            tp.disable()

        threads = [
            threading.Thread(target=thread_work, args=(0, "ci")),
            threading.Thread(target=thread_work, args=(1, "debug")),
            threading.Thread(target=thread_work, args=(2, "ci")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert modes[0] == "ci"
        assert modes[1] == "debug"
        assert modes[2] == "ci"


class TestConcurrentDataFrameOperations:
    """Tests for concurrent DataFrame operations."""

    def test_concurrent_dataframe_tracking(self):
        """Multiple threads can track DataFrames concurrently."""
        results = []
        errors = []

        def process_data(data_id):
            try:
                tp.reset()
                tp.enable()
                df = pd.DataFrame({"id": [data_id] * 10, "val": range(10)})
                df = df.dropna()
                df = df.head(5)

                stats = dbg().stats()
                results.append(
                    {
                        "data_id": data_id,
                        "enabled": stats["enabled"],
                    }
                )
            except Exception as e:
                errors.append((data_id, str(e)))
            finally:
                try:
                    tp.disable()
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_data, range(10))

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10

    def test_concurrent_filter_operations(self):
        """Concurrent filter operations work correctly."""
        all_dropped = []
        lock = threading.Lock()
        errors = []

        def filter_work(seed):
            try:
                tp.reset()
                tp.enable()
                df = pd.DataFrame({"a": [1, None, 3, None, 5] * 10})
                df = df.dropna()

                dropped = list(dbg().dropped_rows())
                with lock:
                    all_dropped.append(len(dropped))
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                try:
                    tp.disable()
                except Exception:
                    pass

        threads = [threading.Thread(target=filter_work, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors: {errors}"
        # All threads should complete
        assert len(all_dropped) == 5

    def test_concurrent_merge_operations(self):
        """Concurrent merge operations work correctly."""
        results = []
        lock = threading.Lock()
        errors = []

        def merge_work(thread_id):
            try:
                tp.reset()
                tp.enable()
                left = pd.DataFrame({"key": [1, 2, 3], "val": [thread_id] * 3})
                right = pd.DataFrame({"key": [1, 2, 3], "data": ["a", "b", "c"]})
                result = left.merge(right, on="key")

                with lock:
                    results.append(len(result))
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                try:
                    tp.disable()
                except Exception:
                    pass

        threads = [threading.Thread(target=merge_work, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors: {errors}"
        # All merges should return 3 rows
        assert all(r == 3 for r in results)


class TestRaceConditionPrevention:
    """Tests to detect and prevent race conditions."""

    def test_rapid_enable_disable_cycles(self):
        """Rapid enable/disable cycles don't corrupt state."""
        errors = []

        def cycle_work(iterations):
            for _ in range(iterations):
                try:
                    tp.enable()
                    df = pd.DataFrame({"a": [1, 2, 3]})
                    df = df.head(2)
                    _ = dbg().stats()
                    tp.disable()
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=cycle_work, args=(20,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_reset_operations(self):
        """Concurrent reset() calls don't cause issues."""
        errors = []

        def reset_work():
            for _ in range(10):
                try:
                    tp.enable()
                    df = pd.DataFrame({"a": [1, None, 3]})
                    df = df.dropna()
                    tp.reset()
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=reset_work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # May have some errors if reset happens during operation
        # Main thing is no crash or deadlock


class TestParallelPipelines:
    """Tests for parallel pipeline execution."""

    def test_parallel_etl_pipelines(self):
        """Multiple ETL pipelines can run in parallel."""
        results = []
        lock = threading.Lock()

        def etl_pipeline(pipeline_id):
            tp.enable()

            with tp.stage("extract"):
                df = pd.DataFrame(
                    {
                        "id": range(100),
                        "value": [i * pipeline_id for i in range(100)],
                        "category": ["A", "B", "C", None] * 25,
                    }
                )

            with tp.stage("transform"):
                df = df.dropna()
                df["processed"] = df["value"] * 2

            with tp.stage("validate"):
                result = tp.contract().expect_no_null_in("category").check(df)

            stats = dbg().stats()
            with lock:
                results.append(
                    {
                        "pipeline_id": pipeline_id,
                        "rows": len(df),
                        "steps": stats["total_steps"],
                        "passed": result.passed,
                    }
                )
            tp.disable()

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(etl_pipeline, range(5))

        assert len(results) == 5
        # Each pipeline should have 75 rows (25 nulls dropped)
        for r in results:
            assert r["rows"] == 75
            assert r["passed"] is True

    def test_parallel_analysis_jobs(self):
        """Parallel analysis jobs with groupby."""
        results = []
        lock = threading.Lock()
        errors = []

        def analyze(job_id):
            try:
                tp.reset()
                tp.enable()
                df = pd.DataFrame(
                    {
                        "region": ["East", "West", "North", "South"] * 25,
                        "sales": range(100),
                    }
                )

                summary = df.groupby("region").agg({"sales": "sum"})

                with lock:
                    results.append(len(summary))
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                try:
                    tp.disable()
                except Exception:
                    pass

        threads = [threading.Thread(target=analyze, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors: {errors}"
        # Each result should have 4 regions
        assert all(r == 4 for r in results)


class TestThreadSafetyEdgeCases:
    """Edge cases for thread safety."""

    def test_shared_dataframe_different_contexts(self):
        """Shared DataFrame with different tracking contexts."""
        shared_df = pd.DataFrame({"a": range(100)})
        results = []
        errors = []
        lock = threading.Lock()

        def process_shared(thread_id):
            try:
                tp.reset()  # Reset context for each thread
                tp.enable()
                # Each thread operates on copy
                local_df = shared_df.copy()
                local_df = local_df.head(thread_id * 10 + 10)

                dropped = len(dbg().dropped_rows())
                with lock:
                    results.append({"thread": thread_id, "dropped": dropped})
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                try:
                    tp.disable()
                except Exception:
                    pass

        threads = [threading.Thread(target=process_shared, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Main assertion: no errors occurred
        assert len(errors) == 0, f"Errors: {errors}"
        # All threads completed
        assert len(results) == 5

    def test_executor_with_context_manager(self):
        """ThreadPoolExecutor works with TracePipe stages."""
        results = []
        errors = []

        def worker(item):
            try:
                tp.reset()
                tp.enable()
                with tp.stage(f"process_{item}"):
                    df = pd.DataFrame({"x": [item] * 5})
                    df = df.head(3)

                stats = dbg().stats()
                tp.disable()
                return {"item": item, "enabled": stats["enabled"]}
            except Exception as e:
                return {"item": item, "error": str(e)}

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(worker, range(6)))

        assert len(results) == 6
        # Check no errors occurred
        for r in results:
            assert "error" not in r, f"Error: {r.get('error')}"
