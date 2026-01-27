"""
JSON and OpenLineage export for lineage data.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from tracepipe.core import (
    ColumnInfo,
    DataSnapshot,
    LineageGraph,
    LineageNode,
    get_graph,
)


class LineageExporter:
    def __init__(self, graph: Optional[LineageGraph] = None):
        self.graph = graph or get_graph()

    def to_dict(
        self,
        node_id: Optional[str] = None,
        include_samples: bool = True,
    ) -> Dict[str, Any]:
        if node_id:
            nodes = self.graph.get_lineage(node_id)
        else:
            nodes = self.graph.get_all_nodes()
        
        return {
            "version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "generator": "tracepipe",
            "nodes": [self._node_to_dict(n, include_samples) for n in nodes],
            "edges": self._extract_edges(nodes),
            "summary": self._generate_summary(nodes),
        }

    def _node_to_dict(self, node: LineageNode, include_samples: bool) -> Dict[str, Any]:
        result = {
            "node_id": node.node_id,
            "operation": node.operation.value,
            "operation_name": node.operation_name,
            "timestamp": datetime.fromtimestamp(node.timestamp).isoformat() + "Z",
            "stage": node.stage,
            "parameters": node.parameters,
            "code_location": node.code_location,
            "parent_ids": node.parent_ids,
            "metadata": node.metadata,
        }
        
        if node.input_snapshot:
            result["input"] = self._snapshot_to_dict(node.input_snapshot, include_samples)
        
        if node.output_snapshot:
            result["output"] = self._snapshot_to_dict(node.output_snapshot, include_samples)
        
        return result

    def _snapshot_to_dict(self, snapshot: DataSnapshot, include_samples: bool) -> Dict[str, Any]:
        result = {
            "shape": list(snapshot.shape),
            "dtypes": snapshot.dtypes,
            "memory_bytes": snapshot.memory_bytes,
            "checksum": snapshot.checksum,
        }
        
        if snapshot.columns:
            result["columns"] = snapshot.columns
        
        if include_samples and snapshot.sample_rows:
            result["sample_rows"] = snapshot.sample_rows
        
        if snapshot.column_info:
            result["column_info"] = {
                name: {
                    "name": info.name,
                    "dtype": info.dtype,
                    "null_count": info.null_count,
                    "unique_count": info.unique_count,
                    "sample_values": info.sample_values if include_samples else [],
                }
                for name, info in snapshot.column_info.items()
            }
        
        return result

    def _extract_edges(self, nodes: List[LineageNode]) -> List[Dict[str, str]]:
        edges = []
        node_ids = {n.node_id for n in nodes}
        
        for node in nodes:
            for parent_id in node.parent_ids:
                if parent_id in node_ids:
                    edges.append({
                        "source": parent_id,
                        "target": node.node_id,
                    })
        
        return edges

    def _generate_summary(self, nodes: List[LineageNode]) -> Dict[str, Any]:
        if not nodes:
            return {
                "total_operations": 0,
                "stages": [],
                "operation_counts": {},
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
            "start_time": datetime.fromtimestamp(nodes[0].timestamp).isoformat() + "Z",
            "end_time": datetime.fromtimestamp(nodes[-1].timestamp).isoformat() + "Z",
        }

    def to_json(
        self,
        node_id: Optional[str] = None,
        include_samples: bool = True,
        indent: int = 2,
    ) -> str:
        data = self.to_dict(node_id, include_samples)
        return json.dumps(data, indent=indent, default=str)

    def save(
        self,
        filepath: str,
        node_id: Optional[str] = None,
        include_samples: bool = True,
        indent: int = 2,
    ) -> None:
        json_str = self.to_json(node_id, include_samples, indent)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)

    def to_openlineage(
        self,
        namespace: str = "tracepipe",
        job_name: str = "pipeline",
    ) -> List[Dict[str, Any]]:
        nodes = self.graph.get_all_nodes()
        events = []
        
        for node in nodes:
            run_id = str(uuid4())
            
            event = {
                "eventType": "COMPLETE",
                "eventTime": datetime.fromtimestamp(node.timestamp).isoformat() + "Z",
                "run": {
                    "runId": run_id,
                    "facets": {
                        "tracepipe_operation": {
                            "_producer": "tracepipe",
                            "_schemaURL": "https://tracepipe.dev/schemas/operation.json",
                            "operation_type": node.operation.value,
                            "operation_name": node.operation_name,
                            "stage": node.stage,
                            "code_location": node.code_location,
                            "parameters": node.parameters,
                        }
                    },
                },
                "job": {
                    "namespace": namespace,
                    "name": job_name,
                    "facets": {},
                },
                "inputs": [],
                "outputs": [],
            }
            
            if node.input_snapshot:
                input_dataset = {
                    "namespace": namespace,
                    "name": f"input_{node.node_id}",
                    "facets": {
                        "schema": self._snapshot_to_openlineage_schema(node.input_snapshot),
                        "datasetStatistics": {
                            "_producer": "tracepipe",
                            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasetStatisticsFacet.json",
                            "rowCount": node.input_snapshot.shape[0] if node.input_snapshot.shape else 0,
                            "bytes": node.input_snapshot.memory_bytes,
                        },
                    },
                }
                event["inputs"].append(input_dataset)
            
            if node.output_snapshot:
                output_dataset = {
                    "namespace": namespace,
                    "name": f"output_{node.node_id}",
                    "facets": {
                        "schema": self._snapshot_to_openlineage_schema(node.output_snapshot),
                        "datasetStatistics": {
                            "_producer": "tracepipe",
                            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasetStatisticsFacet.json",
                            "rowCount": node.output_snapshot.shape[0] if node.output_snapshot.shape else 0,
                            "bytes": node.output_snapshot.memory_bytes,
                        },
                    },
                }
                event["outputs"].append(output_dataset)
            
            events.append(event)
        
        return events

    def _snapshot_to_openlineage_schema(self, snapshot: DataSnapshot) -> Dict[str, Any]:
        fields = []
        
        if snapshot.columns and snapshot.dtypes:
            for col in snapshot.columns:
                dtype = snapshot.dtypes.get(col, "unknown")
                fields.append({
                    "name": col,
                    "type": self._pandas_to_openlineage_type(dtype),
                })
        elif snapshot.dtypes:
            for name, dtype in snapshot.dtypes.items():
                fields.append({
                    "name": name,
                    "type": self._pandas_to_openlineage_type(dtype),
                })
        
        return {
            "_producer": "tracepipe",
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
            "fields": fields,
        }

    @staticmethod
    def _pandas_to_openlineage_type(dtype: str) -> str:
        dtype_lower = dtype.lower()
        if "int" in dtype_lower:
            return "INTEGER"
        elif "float" in dtype_lower:
            return "DOUBLE"
        elif "bool" in dtype_lower:
            return "BOOLEAN"
        elif "datetime" in dtype_lower or "timestamp" in dtype_lower:
            return "TIMESTAMP"
        elif "date" in dtype_lower:
            return "DATE"
        elif "object" in dtype_lower or "string" in dtype_lower:
            return "STRING"
        else:
            return "STRING"


def export_to_json(
    filepath: Optional[str] = None,
    node_id: Optional[str] = None,
    graph: Optional[LineageGraph] = None,
    include_samples: bool = True,
    indent: int = 2,
) -> Union[str, None]:
    exporter = LineageExporter(graph)
    
    if filepath:
        exporter.save(filepath, node_id, include_samples, indent)
        return None
    else:
        return exporter.to_json(node_id, include_samples, indent)


def export_to_openlineage(
    namespace: str = "tracepipe",
    job_name: str = "pipeline",
    graph: Optional[LineageGraph] = None,
) -> List[Dict[str, Any]]:
    exporter = LineageExporter(graph)
    return exporter.to_openlineage(namespace, job_name)
