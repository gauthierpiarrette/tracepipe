"""
Core lineage graph and context management for tracepipe.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd


class OperationType(Enum):
    CREATE = "create"
    TRANSFORM = "transform"
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    ASSIGN = "assign"
    COPY = "copy"
    SKLEARN_FIT = "sklearn_fit"
    SKLEARN_TRANSFORM = "sklearn_transform"
    SKLEARN_PREDICT = "sklearn_predict"
    NUMPY_OP = "numpy_op"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    sample_values: List[Any] = field(default_factory=list)
    null_count: int = 0
    unique_count: int = 0


@dataclass
class DataSnapshot:
    shape: Tuple[int, ...]
    dtypes: Dict[str, str]
    columns: Optional[List[str]] = None
    sample_rows: Optional[List[Dict[str, Any]]] = None
    column_info: Optional[Dict[str, ColumnInfo]] = None
    memory_bytes: int = 0
    checksum: Optional[str] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, sample_size: int = 3) -> DataSnapshot:
        ctx = _context
        was_enabled = ctx.enabled
        ctx._enabled = False
        
        try:
            columns = list(df.columns)
            dtypes = {col: str(df[col].dtype) for col in columns}
            
            sample_rows = None
            if len(df) > 0:
                sample_idx = df.head(min(sample_size, len(df))).index
                sample_rows = df.loc[sample_idx].to_dict("records")
            
            column_info = {}
            for col in columns:
                try:
                    null_count = int(df[col].isna().sum())
                    unique_count = min(df[col].nunique(), 100)
                    sample_vals = df[col].dropna().head(3).tolist()
                except Exception:
                    null_count = 0
                    unique_count = 0
                    sample_vals = []
                
                column_info[col] = ColumnInfo(
                    name=col,
                    dtype=str(df[col].dtype),
                    sample_values=sample_vals,
                    null_count=null_count,
                    unique_count=unique_count,
                )
            
            try:
                memory_bytes = df.memory_usage(deep=True).sum()
            except Exception:
                memory_bytes = 0
            
            checksum = cls._compute_checksum(df)
            
            return cls(
                shape=df.shape,
                dtypes=dtypes,
                columns=columns,
                sample_rows=sample_rows,
                column_info=column_info,
                memory_bytes=memory_bytes,
                checksum=checksum,
            )
        finally:
            ctx._enabled = was_enabled

    @classmethod
    def from_ndarray(cls, arr: np.ndarray, sample_size: int = 3) -> DataSnapshot:
        sample_rows = None
        if arr.size > 0:
            flat = arr.flatten()[:sample_size]
            sample_rows = [{"value": v} for v in flat.tolist()]
        
        checksum = cls._compute_checksum(arr)
        
        return cls(
            shape=arr.shape,
            dtypes={"array": str(arr.dtype)},
            sample_rows=sample_rows,
            memory_bytes=arr.nbytes,
            checksum=checksum,
        )

    @staticmethod
    def _compute_checksum(data: Any) -> Optional[str]:
        try:
            if isinstance(data, pd.DataFrame):
                content = pd.util.hash_pandas_object(data).values.tobytes()
            elif isinstance(data, np.ndarray):
                content = data.tobytes()
            else:
                return None
            return hashlib.md5(content).hexdigest()[:12]
        except Exception:
            return None


@dataclass
class LineageNode:
    node_id: str
    operation: OperationType
    operation_name: str
    timestamp: float
    stage: Optional[str]
    input_snapshot: Optional[DataSnapshot]
    output_snapshot: Optional[DataSnapshot]
    parameters: Dict[str, Any] = field(default_factory=dict)
    code_location: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation": self.operation.value,
            "operation_name": self.operation_name,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "input_shape": self.input_snapshot.shape if self.input_snapshot else None,
            "output_shape": self.output_snapshot.shape if self.output_snapshot else None,
            "parameters": self.parameters,
            "code_location": self.code_location,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
        }


class LineageGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, LineageNode] = {}
        self._data_to_node: Dict[int, str] = {}
        self._lock = threading.RLock()

    def add_node(
        self,
        operation: OperationType,
        operation_name: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None,
        code_location: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._lock:
            node_id = str(uuid.uuid4())[:8]
            
            input_snapshot = self._create_snapshot(input_data)
            output_snapshot = self._create_snapshot(output_data)
            
            inferred_parents = parent_ids or []
            if not inferred_parents and input_data is not None:
                parent_node_id = self._data_to_node.get(id(input_data))
                if parent_node_id:
                    inferred_parents = [parent_node_id]
            
            node = LineageNode(
                node_id=node_id,
                operation=operation,
                operation_name=operation_name,
                timestamp=time.time(),
                stage=stage or _context.current_stage,
                input_snapshot=input_snapshot,
                output_snapshot=output_snapshot,
                parameters=parameters or {},
                code_location=code_location,
                parent_ids=inferred_parents,
                metadata=metadata or {},
            )
            
            self._nodes[node_id] = node
            self._graph.add_node(node_id, data=node)
            
            for parent_id in inferred_parents:
                if parent_id in self._nodes:
                    self._graph.add_edge(parent_id, node_id)
            
            if output_data is not None:
                self._data_to_node[id(output_data)] = node_id
            
            return node_id

    def _create_snapshot(self, data: Any) -> Optional[DataSnapshot]:
        if data is None:
            return None
        try:
            if isinstance(data, pd.DataFrame):
                return DataSnapshot.from_dataframe(data)
            elif isinstance(data, np.ndarray):
                return DataSnapshot.from_ndarray(data)
            elif isinstance(data, pd.Series):
                return DataSnapshot.from_dataframe(data.to_frame())
            else:
                return DataSnapshot(
                    shape=(1,),
                    dtypes={"value": type(data).__name__},
                )
        except Exception:
            return None

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        return self._nodes.get(node_id)

    def get_lineage(self, node_id: str) -> List[LineageNode]:
        if node_id not in self._nodes:
            return []
        
        ancestors = nx.ancestors(self._graph, node_id)
        ancestors.add(node_id)
        
        nodes = [self._nodes[nid] for nid in ancestors if nid in self._nodes]
        nodes.sort(key=lambda n: n.timestamp)
        return nodes

    def get_all_nodes(self) -> List[LineageNode]:
        return sorted(self._nodes.values(), key=lambda n: n.timestamp)

    def find_node_for_data(self, data: Any) -> Optional[str]:
        return self._data_to_node.get(id(data))

    def get_subgraph(self, node_ids: Set[str]) -> nx.DiGraph:
        return self._graph.subgraph(node_ids).copy()

    def clear(self):
        with self._lock:
            self._graph.clear()
            self._nodes.clear()
            self._data_to_node.clear()


class TracePipeContext:
    def __init__(self):
        self._enabled = False
        self._graph = LineageGraph()
        self._stage_stack: List[str] = []
        self._lock = threading.RLock()
        self._instrumented_modules: Set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def graph(self) -> LineageGraph:
        return self._graph

    @property
    def current_stage(self) -> Optional[str]:
        return self._stage_stack[-1] if self._stage_stack else None

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def push_stage(self, name: str):
        with self._lock:
            self._stage_stack.append(name)

    def pop_stage(self) -> Optional[str]:
        with self._lock:
            return self._stage_stack.pop() if self._stage_stack else None

    def reset(self):
        with self._lock:
            self._graph.clear()
            self._stage_stack.clear()

    def mark_instrumented(self, module: str):
        self._instrumented_modules.add(module)

    def is_instrumented(self, module: str) -> bool:
        return module in self._instrumented_modules


_context = TracePipeContext()


def get_context() -> TracePipeContext:
    return _context


def get_graph() -> LineageGraph:
    return _context.graph


@contextmanager
def stage(name: str) -> Iterator[None]:
    _context.push_stage(name)
    try:
        yield
    finally:
        _context.pop_stage()


def get_code_location(depth: int = 2) -> Optional[str]:
    import inspect
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame:
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
    except Exception:
        pass
    return None
