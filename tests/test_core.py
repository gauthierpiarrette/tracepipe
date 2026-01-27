"""
Tests for tracepipe core functionality.
"""
import pytest
import numpy as np
import pandas as pd

import tracepipe
from tracepipe.core import (
    DataSnapshot,
    LineageGraph,
    LineageNode,
    OperationType,
    get_context,
    get_graph,
    stage,
)


class TestDataSnapshot:
    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        snapshot = DataSnapshot.from_dataframe(df)
        
        assert snapshot.shape == (3, 2)
        assert "a" in snapshot.dtypes
        assert "b" in snapshot.dtypes
        assert snapshot.columns == ["a", "b"]
        assert snapshot.sample_rows is not None
        assert len(snapshot.sample_rows) == 3
        assert snapshot.checksum is not None

    def test_from_ndarray(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        snapshot = DataSnapshot.from_ndarray(arr)
        
        assert snapshot.shape == (3, 2)
        assert "array" in snapshot.dtypes
        assert snapshot.memory_bytes == arr.nbytes
        assert snapshot.checksum is not None

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        snapshot = DataSnapshot.from_dataframe(df)
        
        assert snapshot.shape == (0, 0)
        assert snapshot.columns == []


class TestLineageGraph:
    def setup_method(self):
        tracepipe.reset()

    def test_add_node(self):
        graph = LineageGraph()
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        node_id = graph.add_node(
            operation=OperationType.CREATE,
            operation_name="test_create",
            output_data=df,
        )
        
        assert node_id is not None
        assert len(node_id) == 8
        
        node = graph.get_node(node_id)
        assert node is not None
        assert node.operation == OperationType.CREATE
        assert node.operation_name == "test_create"

    def test_node_linking(self):
        graph = LineageGraph()
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        
        node1_id = graph.add_node(
            operation=OperationType.CREATE,
            operation_name="create",
            output_data=df1,
        )
        
        df2 = df1.copy()
        node2_id = graph.add_node(
            operation=OperationType.COPY,
            operation_name="copy",
            input_data=df1,
            output_data=df2,
        )
        
        node2 = graph.get_node(node2_id)
        assert node1_id in node2.parent_ids

    def test_get_lineage(self):
        graph = LineageGraph()
        
        node1_id = graph.add_node(
            operation=OperationType.CREATE,
            operation_name="step1",
        )
        
        node2_id = graph.add_node(
            operation=OperationType.TRANSFORM,
            operation_name="step2",
            parent_ids=[node1_id],
        )
        
        node3_id = graph.add_node(
            operation=OperationType.TRANSFORM,
            operation_name="step3",
            parent_ids=[node2_id],
        )
        
        lineage = graph.get_lineage(node3_id)
        assert len(lineage) == 3
        
        lineage = graph.get_lineage(node1_id)
        assert len(lineage) == 1


class TestStageContext:
    def setup_method(self):
        tracepipe.reset()

    def test_stage_context(self):
        ctx = get_context()
        
        assert ctx.current_stage is None
        
        with stage("preprocessing"):
            assert ctx.current_stage == "preprocessing"
            
            with stage("normalization"):
                assert ctx.current_stage == "normalization"
            
            assert ctx.current_stage == "preprocessing"
        
        assert ctx.current_stage is None

    def test_nested_stages(self):
        ctx = get_context()
        
        ctx.push_stage("a")
        ctx.push_stage("b")
        ctx.push_stage("c")
        
        assert ctx.current_stage == "c"
        
        ctx.pop_stage()
        assert ctx.current_stage == "b"
        
        ctx.pop_stage()
        assert ctx.current_stage == "a"


class TestTracePipeContext:
    def setup_method(self):
        tracepipe.reset()

    def test_enable_disable(self):
        ctx = get_context()
        
        assert not ctx.enabled
        
        ctx.enable()
        assert ctx.enabled
        
        ctx.disable()
        assert not ctx.enabled

    def test_reset(self):
        ctx = get_context()
        graph = get_graph()
        
        ctx.enable()
        ctx.push_stage("test")
        graph.add_node(
            operation=OperationType.CREATE,
            operation_name="test",
        )
        
        assert ctx.current_stage == "test"
        assert len(graph.get_all_nodes()) == 1
        
        ctx.reset()
        
        assert ctx.current_stage is None
        assert len(graph.get_all_nodes()) == 0
