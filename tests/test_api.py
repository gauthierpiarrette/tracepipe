"""
Tests for tracepipe high-level API.
"""
import pytest
import pandas as pd
import numpy as np

import tracepipe
from tracepipe.core import get_graph


class TestExplainAPI:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable()

    def teardown_method(self):
        tracepipe.disable()

    def test_explain_all(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        df = df.sort_values("a")
        
        result = tracepipe.explain()
        
        assert len(result) >= 2
        assert result.nodes is not None

    def test_explain_specific_data(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result_df = df.fillna(0)
        
        lineage = tracepipe.explain(output=result_df)
        
        assert len(lineage) >= 1

    def test_lineage_result_show(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        result = tracepipe.explain()
        filepath = result.show(open_browser=False)
        
        assert filepath.endswith(".html")

    def test_lineage_result_save(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        result = tracepipe.explain()
        
        save_path = tmp_path / "lineage.html"
        result.save(str(save_path))
        
        assert save_path.exists()
        content = save_path.read_text()
        assert "TracePipe" in content

    def test_lineage_result_to_json(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        result = tracepipe.explain()
        json_str = result.to_json()
        
        import json
        data = json.loads(json_str)
        
        assert "nodes" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_lineage_result_diff(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        
        with tracepipe.stage("cleaning"):
            df = df.fillna(0)
        
        with tracepipe.stage("transform"):
            df = df.astype({"a": int})
        
        result = tracepipe.explain()
        diff = result.diff("cleaning", "transform")
        
        assert "from_stage" in diff
        assert "to_stage" in diff
        assert diff["from_stage"] == "cleaning"
        assert diff["to_stage"] == "transform"


class TestSummaryAPI:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable()

    def teardown_method(self):
        tracepipe.disable()

    def test_summary_empty(self):
        result = tracepipe.summary()
        
        assert result["total_operations"] == 0
        assert result["stages"] == []

    def test_summary_with_operations(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        df = df.sort_values("a")
        
        result = tracepipe.summary()
        
        assert result["total_operations"] >= 2
        assert result["enabled"] is True

    def test_summary_with_stages(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with tracepipe.stage("stage1"):
            df = df.fillna(0)
        
        with tracepipe.stage("stage2"):
            df = df.sort_values("a")
        
        result = tracepipe.summary()
        
        assert "stage1" in result["stages"] or "stage2" in result["stages"]


class TestGetLineageAPI:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable()

    def teardown_method(self):
        tracepipe.disable()

    def test_get_lineage_alias(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        result1 = tracepipe.explain()
        result2 = tracepipe.get_lineage()
        
        assert len(result1) == len(result2)
