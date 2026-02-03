# tests/conftest.py
"""
Shared pytest fixtures for TracePipe tests.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import tracepipe

# ============================================================================
# PANDAS VERSION INFO
# ============================================================================

PANDAS_VERSION_STR = pd.__version__
PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2])
PANDAS_MAJOR = PANDAS_VERSION[0]
PANDAS_MINOR = PANDAS_VERSION[1]

# Supported pandas version range
PANDAS_MIN_VERSION = (1, 5)  # pandas 1.5.0
PANDAS_MAX_VERSION = (2, 2)  # pandas 2.2.x


def pytest_configure(config):
    """Print pandas version info at test session start and register markers."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

    print("\n" + "=" * 70)
    print("TracePipe Test Session")
    print("=" * 70)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Pandas:     {pd.__version__}")
    print(f"NumPy:      {np.__version__}")
    print(f"TracePipe:  {tracepipe.__version__}")
    print("=" * 70)

    # Warn if using unsupported pandas version
    if PANDAS_VERSION < PANDAS_MIN_VERSION:
        print(
            f"⚠️  WARNING: pandas {PANDAS_VERSION_STR} is below minimum supported version {PANDAS_MIN_VERSION[0]}.{PANDAS_MIN_VERSION[1]}"
        )
    elif PANDAS_VERSION > PANDAS_MAX_VERSION:
        print(
            f"ℹ️  NOTE: pandas {PANDAS_VERSION_STR} is newer than tested version {PANDAS_MAX_VERSION[0]}.{PANDAS_MAX_VERSION[1]}.x"
        )


# ============================================================================
# CORE FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_tracepipe():
    """Reset TracePipe state before and after each test."""
    tracepipe.reset()
    yield
    try:
        tracepipe.disable()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def suppress_deprecation_warnings():
    """Suppress deprecation warnings during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        yield


@pytest.fixture
def enabled_tracepipe():
    """Provide an enabled TracePipe context."""
    tracepipe.enable()
    return tracepipe


@pytest.fixture
def debug_tracepipe():
    """Provide an enabled TracePipe context in debug mode."""
    tracepipe.enable(mode="debug")
    return tracepipe


@pytest.fixture
def config():
    """Provide a default TracePipeConfig."""
    from tracepipe import TracePipeConfig

    return TracePipeConfig()


# ============================================================================
# PANDAS VERSION FIXTURES
# ============================================================================


@pytest.fixture
def pandas_version():
    """Provide pandas version info as a tuple (major, minor)."""
    return PANDAS_VERSION


@pytest.fixture
def pandas_major():
    """Provide pandas major version."""
    return PANDAS_MAJOR


@pytest.fixture
def pandas_minor():
    """Provide pandas minor version."""
    return PANDAS_MINOR


@pytest.fixture
def requires_pandas_2():
    """Skip test if pandas < 2.0."""
    if PANDAS_MAJOR < 2:
        pytest.skip(f"Requires pandas 2.x, got {PANDAS_VERSION_STR}")


@pytest.fixture
def requires_pandas_21():
    """Skip test if pandas < 2.1."""
    if PANDAS_VERSION < (2, 1):
        pytest.skip(f"Requires pandas 2.1+, got {PANDAS_VERSION_STR}")


# ============================================================================
# DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_df():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, None, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def sample_df_with_nulls():
    """Provide a DataFrame with various null types."""
    return pd.DataFrame(
        {
            "a": [1, None, 3, None, 5],
            "b": [None, 2.0, None, 4.0, None],
            "c": ["x", None, "z", None, "w"],
        }
    )


@pytest.fixture
def large_df():
    """Provide a large DataFrame for performance testing."""
    n = 100_000
    return pd.DataFrame(
        {
            "id": range(n),
            "value": np.random.randn(n),
            "category": np.random.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture
def merge_dfs():
    """Provide two DataFrames suitable for merge testing."""
    left = pd.DataFrame(
        {
            "key": [1, 2, 3, 4],
            "left_val": ["a", "b", "c", "d"],
        }
    )
    right = pd.DataFrame(
        {
            "key": [2, 3, 4, 5],
            "right_val": ["w", "x", "y", "z"],
        }
    )
    return left, right


# ============================================================================
# SKIP MARKERS
# ============================================================================

# Custom markers for conditional tests
pandas_2_only = pytest.mark.skipif(PANDAS_MAJOR < 2, reason="Requires pandas 2.x")

pandas_21_only = pytest.mark.skipif(PANDAS_VERSION < (2, 1), reason="Requires pandas 2.1+")

slow_test = pytest.mark.slow


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def assert_dataframe_tracked(df, expected_drops=None):
    """Assert that a DataFrame is being tracked and optionally check drops."""
    inspector = tracepipe.debug.inspect()
    assert inspector.enabled, "TracePipe is not enabled"

    if expected_drops is not None:
        dropped = inspector.dropped_rows()
        assert (
            len(dropped) == expected_drops
        ), f"Expected {expected_drops} drops, got {len(dropped)}"


def create_test_pipeline():
    """Create a standard test pipeline for integration tests."""
    tracepipe.enable()

    with tracepipe.stage("extract"):
        df = pd.DataFrame(
            {
                "id": range(100),
                "value": [i * 1.5 if i % 3 != 0 else None for i in range(100)],
                "category": ["A", "B", "C"] * 33 + ["A"],
            }
        )

    with tracepipe.stage("transform"):
        df = df.dropna()
        df = df.query("value > 50")

    return df
