"""
HTML visualization for lineage graphs.
"""
from __future__ import annotations

import html
import json
import os
import tempfile
import webbrowser
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from tracepipe.core import DataSnapshot, LineageGraph, LineageNode, get_graph


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TracePipe Lineage</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        /* Header */
        .header {{
            text-align: center;
            padding: 30px 0;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .subtitle {{ color: #666; font-size: 0.95em; }}
        
        /* Summary Cards */
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(145deg, #1e1e3f, #252550);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #2a2a5a;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: #00d9ff;
        }}
        .summary-card .label {{ color: #888; font-size: 0.85em; margin-top: 5px; }}
        .summary-card.rows-changed .value {{ color: #ff6b6b; }}
        .summary-card.cols-added .value {{ color: #00ff88; }}
        
        /* Stage Section */
        .stage-section {{
            background: #1a1a30;
            border-radius: 16px;
            margin-bottom: 20px;
            border: 1px solid #2a2a5a;
            overflow: hidden;
        }}
        .stage-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 24px;
            background: linear-gradient(90deg, rgba(0,217,255,0.1), transparent);
            cursor: pointer;
            user-select: none;
        }}
        .stage-header:hover {{ background: linear-gradient(90deg, rgba(0,217,255,0.15), transparent); }}
        .stage-title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .stage-title h2 {{
            font-size: 1.1em;
            font-weight: 600;
            color: #fff;
        }}
        .stage-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            background: rgba(0,217,255,0.2);
            color: #00d9ff;
        }}
        .stage-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            color: #888;
        }}
        .stage-stats span {{ display: flex; align-items: center; gap: 5px; }}
        .stat-positive {{ color: #00ff88 !important; }}
        .stat-negative {{ color: #ff6b6b !important; }}
        .expand-icon {{
            color: #666;
            transition: transform 0.3s ease;
            font-size: 1.2em;
        }}
        .stage-section.collapsed .expand-icon {{ transform: rotate(-90deg); }}
        .stage-section.collapsed .stage-content {{ display: none; }}
        
        /* Stage Content */
        .stage-content {{ padding: 0 24px 24px; }}
        .stage-summary {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
            padding: 16px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }}
        .stage-summary-item h4 {{ color: #888; font-size: 0.75em; text-transform: uppercase; margin-bottom: 8px; }}
        .shape-change {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: 'Monaco', monospace;
            font-size: 0.95em;
        }}
        .shape {{ padding: 6px 12px; background: #2a2a5a; border-radius: 6px; }}
        .shape.input {{ color: #888; }}
        .shape.output {{ color: #00d9ff; }}
        .arrow {{ color: #00ff88; }}
        
        /* Operation List */
        .operations-list {{ margin-top: 16px; }}
        .operation {{
            display: flex;
            align-items: flex-start;
            padding: 12px 16px;
            background: rgba(30,30,60,0.5);
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 3px solid #2a2a5a;
        }}
        .operation:hover {{ background: rgba(40,40,80,0.5); }}
        .operation.transform {{ border-left-color: #00d9ff; }}
        .operation.filter {{ border-left-color: #ff9f00; }}
        .operation.join {{ border-left-color: #ff00ff; }}
        .operation.aggregate {{ border-left-color: #ffff00; }}
        .operation.copy {{ border-left-color: #888; }}
        .op-number {{
            width: 24px;
            height: 24px;
            background: #2a2a5a;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            color: #888;
            margin-right: 12px;
            flex-shrink: 0;
        }}
        .op-content {{ flex: 1; }}
        .op-name {{ font-weight: 500; color: #fff; margin-bottom: 4px; }}
        .op-details {{ display: flex; gap: 16px; font-size: 0.8em; color: #666; }}
        .op-shape {{ font-family: monospace; color: #00d9ff; }}
        .op-params {{ color: #00ff88; }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 20px;
            color: #444;
            font-size: 0.85em;
        }}
        
        /* No stages message */
        .no-stages {{
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }}
        .no-stages h3 {{ color: #888; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 TracePipe</h1>
            <p class="subtitle">Data Lineage Report • {timestamp}</p>
        </div>
        
        <div class="summary-cards">
            <div class="summary-card">
                <div class="value">{num_stages}</div>
                <div class="label">Pipeline Stages</div>
            </div>
            <div class="summary-card">
                <div class="value">{num_operations}</div>
                <div class="label">Operations</div>
            </div>
            <div class="summary-card {rows_class}">
                <div class="value">{rows_delta}</div>
                <div class="label">Row Change</div>
            </div>
            <div class="summary-card {cols_class}">
                <div class="value">{cols_delta}</div>
                <div class="label">Column Change</div>
            </div>
        </div>
        
        {stages_html}
        
        <div class="footer">
            Generated by TracePipe v1.0.0
        </div>
    </div>
    
    <script>
        document.querySelectorAll('.stage-header').forEach(header => {{
            header.addEventListener('click', () => {{
                header.parentElement.classList.toggle('collapsed');
            }});
        }});
    </script>
</body>
</html>"""

STAGE_TEMPLATE = """
<div class="stage-section">
    <div class="stage-header">
        <div class="stage-title">
            <h2>{stage_icon} {stage_name}</h2>
            <span class="stage-badge">{num_ops} ops</span>
        </div>
        <div class="stage-stats">
            <span class="{shape_class}">{shape_change}</span>
            <span class="expand-icon">▼</span>
        </div>
    </div>
    <div class="stage-content">
        <div class="stage-summary">
            <div class="stage-summary-item">
                <h4>Input → Output Shape</h4>
                <div class="shape-change">
                    <span class="shape input">{input_shape}</span>
                    <span class="arrow">→</span>
                    <span class="shape output">{output_shape}</span>
                </div>
            </div>
            <div class="stage-summary-item">
                <h4>Operations</h4>
                <div>{op_types}</div>
            </div>
        </div>
        <div class="operations-list">
            {operations_html}
        </div>
    </div>
</div>
"""

OPERATION_TEMPLATE = """
<div class="operation {op_class}">
    <div class="op-number">{op_num}</div>
    <div class="op-content">
        <div class="op-name">{op_name}</div>
        <div class="op-details">
            <span class="op-shape">{shape}</span>
            {params_html}
        </div>
    </div>
</div>
"""


class LineageVisualizer:
    def __init__(self, nodes: List[LineageNode]):
        self.nodes = nodes

    @classmethod
    def from_graph(cls, graph: Optional[LineageGraph] = None) -> LineageVisualizer:
        if graph is None:
            graph = get_graph()
        return cls(graph.get_all_nodes())

    @classmethod
    def from_node_id(cls, node_id: str, graph: Optional[LineageGraph] = None) -> LineageVisualizer:
        if graph is None:
            graph = get_graph()
        return cls(graph.get_lineage(node_id))

    def to_html(self) -> str:
        if not self.nodes:
            return self._render_empty()
        
        stages = self._group_by_stage()
        stages_html = self._render_stages(stages)
        
        rows_delta, cols_delta = self._compute_deltas()
        rows_class = "rows-changed" if rows_delta.startswith("-") else ""
        cols_class = "cols-added" if cols_delta.startswith("+") else ""
        
        return HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            num_stages=len(stages),
            num_operations=len(self.nodes),
            rows_delta=rows_delta,
            cols_delta=cols_delta,
            rows_class=rows_class,
            cols_class=cols_class,
            stages_html=stages_html,
        )

    def _group_by_stage(self) -> Dict[str, List[LineageNode]]:
        stages = defaultdict(list)
        for node in self.nodes:
            stage_name = node.stage or "unnamed"
            stages[stage_name].append(node)
        return dict(stages)

    def _render_stages(self, stages: Dict[str, List[LineageNode]]) -> str:
        if not stages:
            return '<div class="no-stages"><h3>No operations captured</h3><p>Enable tracepipe and run your pipeline</p></div>'
        
        html_parts = []
        stage_icons = {
            "data_cleaning": "🧹",
            "cleaning": "🧹",
            "feature_engineering": "🔧",
            "features": "🔧",
            "preprocessing": "⚙️",
            "aggregation": "📊",
            "modeling": "🤖",
            "training": "🎯",
            "inference": "🔮",
            "unnamed": "📋",
        }
        
        for stage_name, nodes in stages.items():
            icon = stage_icons.get(stage_name.lower(), "📋")
            
            input_shape = "∅"
            output_shape = "∅"
            if nodes[0].input_snapshot:
                input_shape = f"{nodes[0].input_snapshot.shape}"
            elif nodes[0].output_snapshot:
                input_shape = f"{nodes[0].output_snapshot.shape}"
            if nodes[-1].output_snapshot:
                output_shape = f"{nodes[-1].output_snapshot.shape}"
            
            shape_change, shape_class = self._compute_shape_change(nodes)
            op_types = self._summarize_op_types(nodes)
            operations_html = self._render_operations(nodes)
            
            stage_html = STAGE_TEMPLATE.format(
                stage_icon=icon,
                stage_name=stage_name.replace("_", " ").title(),
                num_ops=len(nodes),
                shape_change=shape_change,
                shape_class=shape_class,
                input_shape=input_shape,
                output_shape=output_shape,
                op_types=op_types,
                operations_html=operations_html,
            )
            html_parts.append(stage_html)
        
        return "\n".join(html_parts)

    def _render_operations(self, nodes: List[LineageNode]) -> str:
        html_parts = []
        for i, node in enumerate(nodes, 1):
            op_class = node.operation.value
            if op_class.startswith("sklearn"):
                op_class = "transform"
            
            shape = ""
            if node.output_snapshot:
                shape = f"→ {node.output_snapshot.shape}"
            
            params_html = ""
            if node.parameters:
                key_params = {k: v for k, v in list(node.parameters.items())[:2] if k not in ("value",)}
                if key_params:
                    params_str = ", ".join(f"{k}={v}" for k, v in key_params.items())
                    params_html = f'<span class="op-params">{html.escape(params_str[:40])}</span>'
            
            op_html = OPERATION_TEMPLATE.format(
                op_num=i,
                op_class=op_class,
                op_name=html.escape(node.operation_name),
                shape=shape,
                params_html=params_html,
            )
            html_parts.append(op_html)
        
        return "\n".join(html_parts)

    def _compute_shape_change(self, nodes: List[LineageNode]) -> tuple:
        if not nodes:
            return "No change", ""
        
        first = nodes[0]
        last = nodes[-1]
        
        first_shape = first.input_snapshot.shape if first.input_snapshot else (first.output_snapshot.shape if first.output_snapshot else None)
        last_shape = last.output_snapshot.shape if last.output_snapshot else None
        
        if not first_shape or not last_shape:
            return "Shape tracked", ""
        
        row_diff = last_shape[0] - first_shape[0] if len(first_shape) > 0 and len(last_shape) > 0 else 0
        col_diff = last_shape[1] - first_shape[1] if len(first_shape) > 1 and len(last_shape) > 1 else 0
        
        parts = []
        css_class = ""
        
        if row_diff != 0:
            parts.append(f"{row_diff:+d} rows")
            css_class = "stat-negative" if row_diff < 0 else "stat-positive"
        if col_diff != 0:
            parts.append(f"{col_diff:+d} cols")
            css_class = "stat-positive" if col_diff > 0 else css_class
        
        return (", ".join(parts) if parts else "No change"), css_class

    def _summarize_op_types(self, nodes: List[LineageNode]) -> str:
        op_counts = defaultdict(int)
        for node in nodes:
            name = node.operation_name.split(".")[-1]
            op_counts[name] += 1
        
        parts = [f"{count}× {name}" for name, count in op_counts.items()]
        return ", ".join(parts[:4])

    def _compute_deltas(self) -> tuple:
        if not self.nodes:
            return "0", "0"
        
        first = self.nodes[0]
        last = self.nodes[-1]
        
        first_shape = first.input_snapshot.shape if first.input_snapshot else (first.output_snapshot.shape if first.output_snapshot else (0, 0))
        last_shape = last.output_snapshot.shape if last.output_snapshot else (0, 0)
        
        row_diff = last_shape[0] - first_shape[0] if len(first_shape) > 0 and len(last_shape) > 0 else 0
        col_diff = last_shape[1] - first_shape[1] if len(first_shape) > 1 and len(last_shape) > 1 else 0
        
        rows_str = f"{row_diff:+d}" if row_diff != 0 else "0"
        cols_str = f"{col_diff:+d}" if col_diff != 0 else "0"
        
        return rows_str, cols_str

    def _render_empty(self) -> str:
        return HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            num_stages=0,
            num_operations=0,
            rows_delta="0",
            cols_delta="0",
            rows_class="",
            cols_class="",
            stages_html='<div class="no-stages"><h3>No operations captured</h3><p>Enable tracepipe and run your pipeline</p></div>',
        )

    def show(self, open_browser: bool = True) -> str:
        html_content = self.to_html()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            filepath = f.name
        if open_browser:
            webbrowser.open(f"file://{filepath}")
        return filepath

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())


def render_lineage_html(
    nodes: Optional[List[LineageNode]] = None,
    node_id: Optional[str] = None,
    graph: Optional[LineageGraph] = None,
) -> str:
    if nodes is not None:
        viz = LineageVisualizer(nodes)
    elif node_id is not None:
        viz = LineageVisualizer.from_node_id(node_id, graph)
    else:
        viz = LineageVisualizer.from_graph(graph)
    return viz.to_html()
