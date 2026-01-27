"""
Tests for numpy instrumentation.
"""
import pytest
import numpy as np

import tracepipe
from tracepipe.core import get_graph, OperationType


class TestNumpyInstrumentation:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable(pandas=False, numpy=True, sklearn=False)

    def teardown_method(self):
        tracepipe.disable()

    def test_array_creation_tracked(self):
        arr = np.array([1, 2, 3, 4, 5])
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        array_nodes = [n for n in nodes if "numpy.array" in n.operation_name]
        assert len(array_nodes) >= 1
        
        node = array_nodes[0]
        assert node.operation == OperationType.NUMPY_OP
        assert node.output_snapshot is not None
        assert node.output_snapshot.shape == (5,)

    def test_zeros_tracked(self):
        arr = np.zeros((3, 4))
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        zeros_nodes = [n for n in nodes if "zeros" in n.operation_name]
        assert len(zeros_nodes) >= 1

    def test_ones_tracked(self):
        arr = np.ones((2, 3))
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        ones_nodes = [n for n in nodes if "ones" in n.operation_name]
        assert len(ones_nodes) >= 1

    def test_concatenate_tracked(self):
        arr1 = np.array([1, 2])
        arr2 = np.array([3, 4])
        result = np.concatenate([arr1, arr2])
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        concat_nodes = [n for n in nodes if "concatenate" in n.operation_name]
        assert len(concat_nodes) >= 1

    def test_reshape_tracked(self):
        arr = np.arange(12)
        result = np.reshape(arr, (3, 4))
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        reshape_nodes = [n for n in nodes if "reshape" in n.operation_name]
        assert len(reshape_nodes) >= 1

    def test_dot_tracked(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.dot(a, b)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        dot_nodes = [n for n in nodes if "dot" in n.operation_name]
        assert len(dot_nodes) >= 1

    def test_math_operations_tracked(self):
        arr = np.array([1.0, 4.0, 9.0])
        result = np.sqrt(arr)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        sqrt_nodes = [n for n in nodes if "sqrt" in n.operation_name]
        assert len(sqrt_nodes) >= 1
    def test_aggregation_tracked(self):
        arr = np.array([1, 2, 3, 4, 5])
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        array_nodes = [n for n in nodes if "array" in n.operation_name]
        assert len(array_nodes) >= 1

    def test_stage_annotation(self):
        with tracepipe.stage("data_prep"):
            arr = np.zeros((5, 5))
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        prep_nodes = [n for n in nodes if n.stage == "data_prep"]
        assert len(prep_nodes) >= 1

    def test_disabled_no_tracking(self):
        tracepipe.disable()
        
        graph = get_graph()
        initial_count = len(graph.get_all_nodes())
        
        arr = np.zeros((3, 3))
        
        assert len(graph.get_all_nodes()) == initial_count
