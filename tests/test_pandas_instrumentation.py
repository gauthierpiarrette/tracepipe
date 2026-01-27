"""
Tests for pandas instrumentation.
"""
import pytest
import pandas as pd
import numpy as np

import tracepipe
from tracepipe.core import get_graph, OperationType


class TestPandasInstrumentation:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable(pandas=True, numpy=False, sklearn=False)

    def teardown_method(self):
        tracepipe.disable()

    def test_fillna_tracked(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = df.fillna(0)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        fillna_nodes = [n for n in nodes if "fillna" in n.operation_name]
        assert len(fillna_nodes) >= 1
        
        node = fillna_nodes[0]
        assert node.operation == OperationType.TRANSFORM
        assert node.output_snapshot is not None
        assert node.output_snapshot.shape == (3, 2)

    def test_dropna_tracked(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = df.dropna()
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        dropna_nodes = [n for n in nodes if "dropna" in n.operation_name]
        assert len(dropna_nodes) >= 1

    def test_merge_tracked(self):
        df1 = pd.DataFrame({"key": [1, 2], "value1": ["a", "b"]})
        df2 = pd.DataFrame({"key": [1, 2], "value2": ["x", "y"]})
        result = df1.merge(df2, on="key")
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        merge_nodes = [n for n in nodes if "merge" in n.operation_name]
        assert len(merge_nodes) >= 1
        
        node = merge_nodes[0]
        assert node.operation == OperationType.JOIN
        assert "on" in node.parameters

    def test_groupby_tracked(self):
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B"],
            "value": [1, 2, 3, 4]
        })
        result = df.groupby("category").sum()
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        assert len(nodes) >= 1
        assert any("DataFrame" in n.operation_name or "numpy" in n.operation_name for n in nodes)
    def test_sort_values_tracked(self):
        df = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        result = df.sort_values("a")
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        sort_nodes = [n for n in nodes if "sort_values" in n.operation_name]
        assert len(sort_nodes) >= 1
        
        node = sort_nodes[0]
        assert "by" in node.parameters

    def test_rename_tracked(self):
        df = pd.DataFrame({"old_name": [1, 2, 3]})
        result = df.rename(columns={"old_name": "new_name"})
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        rename_nodes = [n for n in nodes if "rename" in n.operation_name]
        assert len(rename_nodes) >= 1

    def test_getitem_tracked(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    def test_query_tracked(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        result = df.query("a > 2")
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        query_nodes = [n for n in nodes if "query" in n.operation_name]
        assert len(query_nodes) >= 1

    def test_filter_tracked(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = df.filter(items=["a", "b"])
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        filter_nodes = [n for n in nodes if "filter" in n.operation_name]
        assert len(filter_nodes) >= 1

    def test_sort_values_tracked(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        result = df.sort_values("a")
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        sort_nodes = [n for n in nodes if "sort" in n.operation_name]
        assert len(sort_nodes) >= 1

    def test_stage_annotation(self):
        df = pd.DataFrame({"a": [1, 2, None], "b": [4, None, 6]})
        
        with tracepipe.stage("cleaning"):
            df = df.fillna(0)
            df = df.dropna()
        
        with tracepipe.stage("transform"):
            df = df.sort_values("a")
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        cleaning_nodes = [n for n in nodes if n.stage == "cleaning"]
        transform_nodes = [n for n in nodes if n.stage == "transform"]
        
        assert len(cleaning_nodes) >= 1
        assert len(transform_nodes) >= 1

    def test_chained_operations(self):
        df = pd.DataFrame({
            "a": [1, None, 3, 4, None],
            "b": ["x", "y", "z", "w", "v"]
        })
        
        result = (
            df
            .fillna(0)
            .sort_values("a")
            .head(3)
        )
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        assert len(nodes) >= 3

    def test_disabled_no_tracking(self):
        tracepipe.disable()
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.fillna(0)
        
        graph = get_graph()
        initial_count = len(graph.get_all_nodes())
        
        result = df.dropna()
        
        assert len(graph.get_all_nodes()) == initial_count
