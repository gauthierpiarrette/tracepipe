# tests/test_core.py
"""
Tests for tracepipe/core.py - Configuration, enums, and dataclasses.
"""

from tracepipe import TracePipeConfig
from tracepipe.core import ChangeType, CompletenessLevel, LineageGap, LineageGaps


class TestTracePipeConfig:
    """Tests for TracePipeConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = TracePipeConfig()

        assert config.max_diffs_in_memory == 500_000
        assert config.max_diffs_per_step == 100_000
        assert config.strict_mode is False
        assert config.auto_watch is False

    def test_custom_values(self):
        """Config accepts custom values."""
        config = TracePipeConfig(
            max_diffs_in_memory=1000,
            strict_mode=True,
            auto_watch=True,
        )

        assert config.max_diffs_in_memory == 1000
        assert config.strict_mode is True
        assert config.auto_watch is True

    def test_from_env(self, monkeypatch):
        """TracePipeConfig.from_env() reads environment variables."""
        monkeypatch.setenv("TRACEPIPE_MAX_DIFFS", "1000")
        monkeypatch.setenv("TRACEPIPE_STRICT", "1")

        config = TracePipeConfig.from_env()

        assert config.max_diffs_in_memory == 1000
        assert config.strict_mode is True


class TestEnums:
    """Tests for enum types."""

    def test_change_type_values(self):
        """ChangeType enum has expected values."""
        assert ChangeType.MODIFIED == 0
        assert ChangeType.DROPPED == 1
        assert ChangeType.ADDED == 2
        assert ChangeType.REORDERED == 3

    def test_completeness_level_values(self):
        """CompletenessLevel enum has expected values."""
        assert CompletenessLevel.FULL == 0
        assert CompletenessLevel.PARTIAL == 1
        assert CompletenessLevel.UNKNOWN == 2


class TestLineageGaps:
    """Tests for LineageGaps dataclass."""

    def test_empty_gaps(self):
        """Empty gaps indicate full tracking."""
        gaps = LineageGaps(gaps=[])

        assert gaps.is_fully_tracked is True
        assert gaps.has_gaps is False
        assert gaps.summary() == "Fully tracked"

    def test_single_gap(self):
        """Single gap is reported correctly."""
        gap = LineageGap(step_id=1, operation="apply", reason="Custom function")
        gaps = LineageGaps(gaps=[gap])

        assert gaps.is_fully_tracked is False
        assert gaps.has_gaps is True
        assert "1 step" in gaps.summary()

    def test_multiple_gaps(self):
        """Multiple gaps are reported correctly."""
        gap1 = LineageGap(step_id=1, operation="apply", reason="Custom function")
        gap2 = LineageGap(step_id=2, operation="merge", reason="Lineage reset")
        gaps = LineageGaps(gaps=[gap1, gap2])

        assert gaps.has_gaps is True
        assert "2 steps" in gaps.summary()
