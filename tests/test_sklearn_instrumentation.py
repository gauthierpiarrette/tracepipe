"""
Tests for sklearn instrumentation.
"""
import pytest
import numpy as np

import tracepipe
from tracepipe.core import get_graph, OperationType

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestSklearnInstrumentation:
    def setup_method(self):
        tracepipe.reset()
        tracepipe.enable(pandas=False, numpy=False, sklearn=True)

    def teardown_method(self):
        tracepipe.disable()

    def test_standard_scaler_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        scaler_nodes = [n for n in nodes if "StandardScaler" in n.operation_name]
        assert len(scaler_nodes) >= 1
        
        node = scaler_nodes[0]
        assert node.operation in (OperationType.SKLEARN_TRANSFORM, OperationType.SKLEARN_FIT)

    def test_standard_scaler_fit_then_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scaler = StandardScaler()
        scaler.fit(X)
        result = scaler.transform(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        fit_nodes = [n for n in nodes if "fit" in n.operation_name and "StandardScaler" in n.operation_name]
        transform_nodes = [n for n in nodes if "transform" in n.operation_name and "StandardScaler" in n.operation_name and "fit" not in n.operation_name]
        
        assert len(fit_nodes) >= 1
        assert len(transform_nodes) >= 1

    def test_minmax_scaler_tracked(self):
        X = np.array([[1], [2], [3], [4], [5]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        scaler_nodes = [n for n in nodes if "MinMaxScaler" in n.operation_name]
        assert len(scaler_nodes) >= 1

    def test_pca_tracked(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        pca_nodes = [n for n in nodes if "PCA" in n.operation_name]
        assert len(pca_nodes) >= 1

    def test_logistic_regression_fit_predict(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = LogisticRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        fit_nodes = [n for n in nodes if "LogisticRegression.fit" in n.operation_name]
        predict_nodes = [n for n in nodes if "LogisticRegression.predict" in n.operation_name]
        
        assert len(fit_nodes) >= 1
        assert len(predict_nodes) >= 1

    def test_linear_regression_tracked(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        lr_nodes = [n for n in nodes if "LinearRegression" in n.operation_name]
        assert len(lr_nodes) >= 1

    def test_decision_tree_tracked(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        model = DecisionTreeClassifier()
        model.fit(X, y)
        predictions = model.predict(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        dt_nodes = [n for n in nodes if "DecisionTree" in n.operation_name]
        assert len(dt_nodes) >= 1

    def test_kmeans_tracked(self):
        X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [9, 8], [8, 9]])
        
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        model.fit(X)
        labels = model.predict(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        kmeans_nodes = [n for n in nodes if "KMeans" in n.operation_name]
        assert len(kmeans_nodes) >= 1

    def test_parameters_captured(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        pca = PCA(n_components=1)
        result = pca.fit_transform(X)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        pca_nodes = [n for n in nodes if "PCA" in n.operation_name]
        assert len(pca_nodes) >= 1
        
        node = pca_nodes[0]
        assert "n_components" in node.parameters

    def test_stage_annotation(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        with tracepipe.stage("preprocessing"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        with tracepipe.stage("modeling"):
            model = LogisticRegression()
            model.fit(X_scaled, y)
        
        graph = get_graph()
        nodes = graph.get_all_nodes()
        
        preproc_nodes = [n for n in nodes if n.stage == "preprocessing"]
        model_nodes = [n for n in nodes if n.stage == "modeling"]
        
        assert len(preproc_nodes) >= 1
        assert len(model_nodes) >= 1

    def test_disabled_no_tracking(self):
        tracepipe.disable()
        
        X = np.array([[1, 2], [3, 4]])
        scaler = StandardScaler()
        
        graph = get_graph()
        initial_count = len(graph.get_all_nodes())
        
        scaler.fit_transform(X)
        
        assert len(graph.get_all_nodes()) == initial_count
