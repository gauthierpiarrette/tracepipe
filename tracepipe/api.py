"""
High-level API for TracePipe lineage queries and explanations.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from tracepipe.core import (
    LineageGraph,
    LineageNode,
    get_context,
    get_graph,
)
from tracepipe.visualization.html_viz import LineageVisualizer


class LineageResult:
    def __init__(self, nodes: List[LineageNode], target_node_id: Optional[str] = None):
        self.nodes = nodes
        self.target_node_id = target_node_id
        self._visualizer: Optional[LineageVisualizer] = None

    @property
    def visualizer(self) -> LineageVisualizer:
        if self._visualizer is None:
            self._visualizer = LineageVisualizer(self.nodes)
        return self._visualizer

    def show(self, open_browser: bool = True) -> str:
        return self.visualizer.show(open_browser=open_browser)

    def save(self, filepath: str) -> None:
        self.visualizer.save(filepath)

    def to_html(self) -> str:
        return self.visualizer.to_html()

    def to_json(self, include_samples: bool = True, indent: int = 2) -> str:
        from tracepipe.export.json_export import LineageExporter
        
        graph = get_graph()
        exporter = LineageExporter(graph)
        return exporter.to_json(self.target_node_id, include_samples, indent)

    def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        from tracepipe.export.json_export import LineageExporter
        
        graph = get_graph()
        exporter = LineageExporter(graph)
        return exporter.to_dict(self.target_node_id, include_samples)

    def diff(self, from_stage: str, to_stage: str) -> Dict[str, Any]:
        from_nodes = [n for n in self.nodes if n.stage == from_stage]
        to_nodes = [n for n in self.nodes if n.stage == to_stage]
        
        if not from_nodes or not to_nodes:
            return {
                "error": f"Stage not found. Available stages: {set(n.stage for n in self.nodes if n.stage)}"
            }
        
        from_node = from_nodes[-1]
        to_node = to_nodes[-1]
        
        diff_result = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "from_operation": from_node.operation_name,
            "to_operation": to_node.operation_name,
            "operations_between": [],
        }
        
        from_idx = self.nodes.index(from_node)
        to_idx = self.nodes.index(to_node)
        
        if from_idx < to_idx:
            between_nodes = self.nodes[from_idx + 1:to_idx + 1]
            diff_result["operations_between"] = [
                {"operation": n.operation_name, "stage": n.stage}
                for n in between_nodes
            ]
        
        if from_node.output_snapshot and to_node.output_snapshot:
            from_snap = from_node.output_snapshot
            to_snap = to_node.output_snapshot
            
            diff_result["shape_change"] = {
                "from": from_snap.shape,
                "to": to_snap.shape,
            }
            
            if from_snap.columns and to_snap.columns:
                from_cols = set(from_snap.columns)
                to_cols = set(to_snap.columns)
                
                diff_result["columns"] = {
                    "added": list(to_cols - from_cols),
                    "removed": list(from_cols - to_cols),
                    "unchanged": list(from_cols & to_cols),
                }
            
            if from_snap.dtypes and to_snap.dtypes:
                type_changes = {}
                for col in set(from_snap.dtypes.keys()) & set(to_snap.dtypes.keys()):
                    if from_snap.dtypes[col] != to_snap.dtypes[col]:
                        type_changes[col] = {
                            "from": from_snap.dtypes[col],
                            "to": to_snap.dtypes[col],
                        }
                if type_changes:
                    diff_result["type_changes"] = type_changes
        
        return diff_result

    def __repr__(self) -> str:
        return f"LineageResult({len(self.nodes)} operations)"

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("TracePipe Lineage Summary")
        print(f"{'='*60}")
        print(f"Total operations: {len(self.nodes)}")
        
        stages = set(n.stage for n in self.nodes if n.stage)
        if stages:
            print(f"Stages: {', '.join(sorted(stages))}")
        
        print(f"\nOperation timeline:")
        print("-" * 60)
        
        for i, node in enumerate(self.nodes):
            stage_str = f"[{node.stage}] " if node.stage else ""
            shape_str = ""
            if node.output_snapshot:
                shape_str = f" → {node.output_snapshot.shape}"
            print(f"  {i+1}. {stage_str}{node.operation_name}{shape_str}")
        
        print(f"{'='*60}\n")


def explain(
    output: Optional[Any] = None,
    row_filter: Optional[Callable[[int], bool]] = None,
    node_id: Optional[str] = None,
) -> LineageResult:
    graph = get_graph()
    
    if node_id is not None:
        nodes = graph.get_lineage(node_id)
        return LineageResult(nodes, node_id)
    
    if output is not None:
        found_node_id = graph.find_node_for_data(output)
        if found_node_id:
            nodes = graph.get_lineage(found_node_id)
            return LineageResult(nodes, found_node_id)
    
    nodes = graph.get_all_nodes()
    return LineageResult(nodes, None)


def get_lineage(
    data: Optional[Any] = None,
    node_id: Optional[str] = None,
) -> LineageResult:
    return explain(output=data, node_id=node_id)


def summary() -> Dict[str, Any]:
    graph = get_graph()
    nodes = graph.get_all_nodes()
    
    if not nodes:
        return {
            "total_operations": 0,
            "stages": [],
            "operation_counts": {},
            "enabled": get_context().enabled,
        }
    
    stages = list(set(n.stage for n in nodes if n.stage))
    op_counts: Dict[str, int] = {}
    for node in nodes:
        op_type = node.operation.value
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    duration = nodes[-1].timestamp - nodes[0].timestamp if len(nodes) > 1 else 0
    
    return {
        "total_operations": len(nodes),
        "stages": stages,
        "operation_counts": op_counts,
        "duration_seconds": round(duration, 3),
        "enabled": get_context().enabled,
    }


def print_summary() -> None:
    result = explain()
    result.print_summary()
