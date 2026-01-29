# tests/test_html_export.py
"""
Tests for tracepipe/visualization/html_export.py - HTML report generation.
"""

import pandas as pd

import tracepipe


class TestHTMLExport:
    """Tests for HTML export functionality."""

    def test_save_creates_file(self, tmp_path):
        """save() creates an HTML file."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        filepath = tmp_path / "report.html"
        tracepipe.save(str(filepath))

        assert filepath.exists()

    def test_html_contains_tracepipe(self, tmp_path):
        """HTML contains TracePipe branding."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        filepath = tmp_path / "report.html"
        tracepipe.save(str(filepath))

        content = filepath.read_text()
        assert "TracePipe" in content

    def test_html_contains_pipeline_data(self, tmp_path):
        """HTML contains pipeline step data."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        filepath = tmp_path / "report.html"
        tracepipe.save(str(filepath))

        content = filepath.read_text()
        assert "dropna" in content.lower()

    def test_html_is_valid_html(self, tmp_path):
        """HTML has valid structure."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        filepath = tmp_path / "report.html"
        tracepipe.save(str(filepath))

        content = filepath.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content


class TestHTMLHelpers:
    """Tests for HTML helper functions."""

    def test_format_file_name(self):
        """_format_file_name extracts filename from path."""
        from tracepipe.visualization.html_export import _format_file_name

        assert _format_file_name("/path/to/file.py") == "file.py"
        assert _format_file_name("file.py") == "file.py"
        assert _format_file_name("a/b/c/d.py") == "d.py"

    def test_format_number(self):
        """_format_number formats with commas."""
        from tracepipe.visualization.html_export import _format_number

        assert _format_number(1000) == "1,000"
        assert _format_number(1234567) == "1,234,567"
        assert _format_number(42) == "42"

    def test_escape(self):
        """_escape handles HTML special characters."""
        from tracepipe.visualization.html_export import _escape

        assert "&lt;" in _escape("<script>")
        assert "&gt;" in _escape("<script>")
        assert "NULL" in _escape(None)
