"""
Tests for tracepipe export functionality.
"""
import json
import pytest
import pandas as pd

import tracepipe
from tracepipe.export import export_to_json, export_to_openlineage, LineageExporter


class TestJSONExport:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable()

    def teardown_method(self):
        tracepipe.disable()

    def test_export_to_json_string(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        json_str = export_to_json()
        data = json.loads(json_str)
        
        assert "version" in data
        assert "nodes" in data
        assert "edges" in data
        assert "summary" in data
        assert data["generator"] == "tracepipe"

    def test_export_to_json_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        filepath = tmp_path / "lineage.json"
        export_to_json(str(filepath))
        
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "nodes" in data

    def test_export_without_samples(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        json_str = export_to_json(include_samples=False)
        data = json.loads(json_str)
        
        for node in data["nodes"]:
            if "output" in node:
                assert "sample_rows" not in node["output"] or node["output"]["sample_rows"] is None

    def test_lineage_exporter_to_dict(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        exporter = LineageExporter()
        data = exporter.to_dict()
        
        assert isinstance(data, dict)
        assert "nodes" in data
        assert "summary" in data


class TestOpenLineageExport:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable()

    def teardown_method(self):
        tracepipe.disable()

    def test_export_to_openlineage(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        events = export_to_openlineage()
        
        assert isinstance(events, list)
        assert len(events) >= 1
        
        for event in events:
            assert "eventType" in event
            assert "eventTime" in event
            assert "run" in event
            assert "job" in event

    def test_openlineage_namespace(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.fillna(0)
        
        events = export_to_openlineage(namespace="my_namespace", job_name="my_job")
        
        for event in events:
            assert event["job"]["namespace"] == "my_namespace"
            assert event["job"]["name"] == "my_job"

    def test_openlineage_schema_facets(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df = df.fillna(0)
        
        events = export_to_openlineage()
        
        has_schema = False
        for event in events:
            for output in event.get("outputs", []):
                if "schema" in output.get("facets", {}):
                    has_schema = True
                    schema = output["facets"]["schema"]
                    assert "fields" in schema
        
        assert has_schema
