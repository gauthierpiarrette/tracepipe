# tests/conftest.py
"""
Shared pytest fixtures for TracePipe tests.
"""

import pytest

import tracepipe


@pytest.fixture(autouse=True)
def reset_tracepipe():
    """Reset TracePipe state before and after each test."""
    tracepipe.reset()
    yield
    try:
        tracepipe.disable()
    except Exception:
        pass


@pytest.fixture
def enabled_tracepipe():
    """Provide an enabled TracePipe context."""
    tracepipe.enable()
    return tracepipe


@pytest.fixture
def config():
    """Provide a default TracePipeConfig."""
    from tracepipe import TracePipeConfig

    return TracePipeConfig()
