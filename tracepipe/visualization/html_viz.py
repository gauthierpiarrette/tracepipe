"""
HTML visualization for lineage graphs.
"""
from __future__ import annotations

import html
import json
import os
import tempfile
import webbrowser
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from tracepipe.core import DataSnapshot, LineageGraph, LineageNode, get_graph


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TracePipe Lineage Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #2a2a4a;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #888;
            font-size: 1.1em;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00d9ff;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .timeline {{
            position: relative;
            padding-left: 40px;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(180deg, #00d9ff, #00ff88);
        }}
        .node {{
            position: relative;
            margin-bottom: 20px;
            background: #1e1e3f;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2a2a5a;
            transition: all 0.3s ease;
        }}
        .node:hover {{
            transform: translateX(5px);
            border-color: #00d9ff;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        }}
        .node::before {{
            content: '';
            position: absolute;
            left: -33px;
            top: 25px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #00d9ff;
            border: 3px solid #1a1a2e;
        }}
        .node.create::before {{ background: #00ff88; }}
        .node.transform::before {{ background: #00d9ff; }}
        .node.filter::before {{ background: #ff9f00; }}
        .node.join::before {{ background: #ff00ff; }}
        .node.aggregate::before {{ background: #ffff00; }}
        .node.sklearn::before {{ background: #ff6b6b; }}
        .node-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        .node-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #fff;
        }}
        .node-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-create {{ background: rgba(0, 255, 136, 0.2); color: #00ff88; }}
        .badge-transform {{ background: rgba(0, 217, 255, 0.2); color: #00d9ff; }}
        .badge-filter {{ background: rgba(255, 159, 0, 0.2); color: #ff9f00; }}
        .badge-join {{ background: rgba(255, 0, 255, 0.2); color: #ff00ff; }}
        .badge-aggregate {{ background: rgba(255, 255, 0, 0.2); color: #ffff00; }}
        .badge-sklearn {{ background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }}
        .badge-unknown {{ background: rgba(128, 128, 128, 0.2); color: #888; }}
        .node-meta {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 0.85em;
            color: #888;
        }}
        .node-meta span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .shape-change {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            background: #151530;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .shape {{
            font-family: 'Monaco', 'Menlo', monospace;
            padding: 5px 10px;
            background: #2a2a5a;
            border-radius: 4px;
            color: #00d9ff;
        }}
        .arrow {{
            color: #00ff88;
            font-size: 1.2em;
        }}
        .params {{
            background: #151530;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }}
        .params-title {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}
        .param {{
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: #1e1e3f;
            border-radius: 4px;
        }}
        .param-key {{
            color: #888;
        }}
        .param-value {{
            color: #00ff88;
            font-family: monospace;
        }}
        .stage-tag {{
            display: inline-block;
            padding: 3px 10px;
            background: rgba(0, 217, 255, 0.1);
            border: 1px solid #00d9ff;
            border-radius: 4px;
            font-size: 0.8em;
            color: #00d9ff;
            margin-right: 10px;
        }}
        .code-location {{
            font-family: monospace;
            font-size: 0.8em;
            color: #666;
            word-break: break-all;
        }}
        .sample-data {{
            margin-top: 15px;
            overflow-x: auto;
        }}
        .sample-data table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
        }}
        .sample-data th {{
            background: #2a2a5a;
            padding: 10px;
            text-align: left;
            color: #00d9ff;
        }}
        .sample-data td {{
            padding: 8px 10px;
            border-bottom: 1px solid #2a2a4a;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #555;
            font-size: 0.9em;
        }}
        .collapsible {{
            cursor: pointer;
        }}
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        .collapsible-content.show {{
            max-height: 1000px;
        }}
        .expand-icon {{
            transition: transform 0.3s ease;
        }}
        .expand-icon.rotated {{
            transform: rotate(90deg);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 TracePipe Lineage</h1>
            <p class="subtitle">Data transformation history</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{num_nodes}</div>
                    <div class="stat-label">Operations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{num_stages}</div>
                    <div class="stat-label">Stages</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{duration}</div>
                    <div class="stat-label">Duration</div>
                </div>
            </div>
        </div>
        <div class="timeline">
            {nodes_html}
        </div>
        <div class="footer">
            Generated by TracePipe • {timestamp}
        </div>
    </div>
    <script>
        document.querySelectorAll('.collapsible').forEach(elem => {{
            elem.addEventListener('click', () => {{
                const content = elem.nextElementSibling;
                const icon = elem.querySelector('.expand-icon');
                content.classList.toggle('show');
                if (icon) icon.classList.toggle('rotated');
            }});
        }});
    </script>
</body>
</html>"""

NODE_TEMPLATE = """
<div class="node {op_class}">
    <div class="node-header">
        <div>
            {stage_tag}
            <span class="node-title">{operation_name}</span>
        </div>
        <span class="node-badge badge-{badge_class}">{operation_type}</span>
    </div>
    <div class="node-meta">
        <span>🕐 {timestamp}</span>
        <span>🔑 {node_id}</span>
    </div>
    {shape_html}
    {params_html}
    {code_location_html}
    {sample_html}
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
        nodes = graph.get_lineage(node_id)
        return cls(nodes)

    def to_html(self) -> str:
        nodes_html = []
        for node in self.nodes:
            nodes_html.append(self._render_node(node))
        
        stages = set(n.stage for n in self.nodes if n.stage)
        
        if self.nodes:
            duration_sec = self.nodes[-1].timestamp - self.nodes[0].timestamp
            if duration_sec < 1:
                duration = f"{duration_sec * 1000:.1f}ms"
            elif duration_sec < 60:
                duration = f"{duration_sec:.2f}s"
            else:
                duration = f"{duration_sec / 60:.1f}m"
        else:
            duration = "0ms"
        
        return HTML_TEMPLATE.format(
            num_nodes=len(self.nodes),
            num_stages=len(stages),
            duration=duration,
            nodes_html="\n".join(nodes_html),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _render_node(self, node: LineageNode) -> str:
        op_type = node.operation.value
        op_class = op_type
        if op_type.startswith("sklearn"):
            op_class = "sklearn"
            badge_class = "sklearn"
        else:
            badge_class = op_type if op_type in ("create", "transform", "filter", "join", "aggregate") else "unknown"
        
        stage_tag = f'<span class="stage-tag">{html.escape(node.stage)}</span>' if node.stage else ""
        
        shape_html = ""
        if node.input_snapshot and node.output_snapshot:
            shape_html = f"""
            <div class="shape-change">
                <span class="shape">{node.input_snapshot.shape}</span>
                <span class="arrow">→</span>
                <span class="shape">{node.output_snapshot.shape}</span>
            </div>
            """
        elif node.output_snapshot:
            shape_html = f"""
            <div class="shape-change">
                <span class="shape">∅</span>
                <span class="arrow">→</span>
                <span class="shape">{node.output_snapshot.shape}</span>
            </div>
            """
        
        params_html = ""
        if node.parameters:
            params_items = []
            for k, v in node.parameters.items():
                params_items.append(f"""
                <div class="param">
                    <span class="param-key">{html.escape(str(k))}</span>
                    <span class="param-value">{html.escape(str(v)[:50])}</span>
                </div>
                """)
            params_html = f"""
            <div class="params">
                <div class="params-title">Parameters</div>
                <div class="params-grid">
                    {"".join(params_items)}
                </div>
            </div>
            """
        
        code_location_html = ""
        if node.code_location:
            code_location_html = f'<div class="code-location">📍 {html.escape(node.code_location)}</div>'
        
        sample_html = ""
        if node.output_snapshot and node.output_snapshot.sample_rows:
            rows = node.output_snapshot.sample_rows[:3]
            if rows:
                headers = list(rows[0].keys())[:6]
                header_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
                rows_html = ""
                for row in rows:
                    cells = "".join(f"<td>{html.escape(str(row.get(h, ''))[:30])}</td>" for h in headers)
                    rows_html += f"<tr>{cells}</tr>"
                sample_html = f"""
                <div class="sample-data collapsible-content">
                    <table>
                        <thead><tr>{header_html}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """
        
        timestamp_str = datetime.fromtimestamp(node.timestamp).strftime("%H:%M:%S.%f")[:-3]
        
        return NODE_TEMPLATE.format(
            op_class=op_class,
            badge_class=badge_class,
            operation_name=html.escape(node.operation_name),
            operation_type=op_type.upper(),
            stage_tag=stage_tag,
            timestamp=timestamp_str,
            node_id=node.node_id,
            shape_html=shape_html,
            params_html=params_html,
            code_location_html=code_location_html,
            sample_html=sample_html,
        )

    def show(self, open_browser: bool = True) -> str:
        html_content = self.to_html()
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            filepath = f.name
        
        if open_browser:
            webbrowser.open(f"file://{filepath}")
        
        return filepath

    def save(self, filepath: str) -> None:
        html_content = self.to_html()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)


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
