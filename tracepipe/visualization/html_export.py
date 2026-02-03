# tracepipe/visualization/html_export.py
"""
Interactive HTML dashboard for TracePipe lineage reports.

Design principles:
1. Dashboard-first: Key metrics visible immediately
2. Progressive disclosure: Summary → click to expand → full details
3. Searchable: Find any row by ID instantly
4. Scalable: Works with 1M+ rows (uses counts, not lists)
5. Visual pipeline: See data flow at a glance
"""

import html
import json
import linecache
import os
from typing import Optional

from ..context import get_context


def _escape(value) -> str:
    """Escape value for HTML."""
    if value is None:
        return '<span class="null">NULL</span>'
    s = str(value)
    if len(s) > 50:
        s = s[:47] + "..."
    return html.escape(s)


def _format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def _format_file_name(path: str) -> str:
    """Extract file name from path for display."""
    return path.split("/")[-1] if "/" in path else path


def _get_code_snippet(filepath: str, lineno: int, context: int = 2) -> Optional[str]:
    """Get source code snippet around a line number."""
    if not filepath or not lineno or not os.path.exists(filepath):
        return None

    try:
        start = max(1, lineno - context)
        end = lineno + context
        lines = []
        for i in range(start, end + 1):
            line = linecache.getline(filepath, i)
            if line:
                # Remove common indentation
                lines.append(line.rstrip())

        if not lines:
            return None

        # Dedent
        min_indent = float("inf")
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        if min_indent == float("inf"):
            min_indent = 0

        formatted_lines = []
        for i, line in enumerate(lines):
            curr_lineno = start + i
            content = line[int(min_indent) :] if len(line) >= min_indent else line
            is_target = curr_lineno == lineno
            marker = ">" if is_target else " "
            cls = "highlight" if is_target else ""
            formatted_lines.append(
                f'<div class="code-line {cls}"><span class="lineno">{curr_lineno}</span><span class="marker">{marker}</span><span class="content">{html.escape(content)}</span></div>'
            )

        return "\n".join(formatted_lines)
    except Exception:
        return None


def _get_pipeline_data(ctx) -> list[dict]:
    """Extract pipeline steps for visualization."""
    steps = []
    for step in ctx.store.steps:
        # Format code location
        code_loc = None
        snippet = None
        if step.code_file and step.code_line:
            code_loc = f"{_format_file_name(step.code_file)}:{step.code_line}"
            snippet = _get_code_snippet(step.code_file, step.code_line)

        step_data = {
            "id": step.step_id,
            "operation": step.operation,
            "stage": step.stage or "",
            "input_shape": list(step.input_shape) if step.input_shape else None,
            "output_shape": list(step.output_shape) if step.output_shape else None,
            "is_mass_update": step.is_mass_update,
            "rows_affected": step.rows_affected,
            "completeness": step.completeness.name,
            "code_file": step.code_file,
            "code_line": step.code_line,
            "code_loc": code_loc,
            "code_snippet": snippet,
            "timestamp": step.timestamp,
        }
        steps.append(step_data)
    return steps


def _get_dropped_summary(ctx) -> dict:
    """Get dropped rows summary."""
    dropped_by_step = ctx.store.get_dropped_by_step()
    total = sum(dropped_by_step.values())
    return {
        "total": total,
        "by_operation": dropped_by_step,
    }


def _get_changes_summary(ctx) -> dict:
    """Get cell changes summary."""
    changes_by_col = {}
    changes_by_step = {}

    for i in range(len(ctx.store.diff_cols)):
        col = ctx.store.diff_cols[i]
        step_id = ctx.store.diff_step_ids[i]
        change_type = ctx.store.diff_change_types[i]

        # Only count MODIFIED and ADDED
        if change_type in (0, 2):
            changes_by_col[col] = changes_by_col.get(col, 0) + 1

            # Find operation name for this step
            for step in ctx.store.steps:
                if step.step_id == step_id:
                    op = step.operation
                    changes_by_step[op] = changes_by_step.get(op, 0) + 1
                    break

    return {
        "total": ctx.store.total_diff_count,
        "by_column": changes_by_col,
        "by_operation": changes_by_step,
    }


def _get_groups_summary(ctx) -> list[dict]:
    """Get aggregation groups summary."""
    groups = []
    for mapping in ctx.store.aggregation_mappings:
        for group_key, row_ids in mapping.membership.items():
            # Count-only groups are stored as [-count] (list with one negative element)
            is_count_only = len(row_ids) == 1 and row_ids[0] < 0
            if is_count_only:
                row_count = abs(row_ids[0])
            else:
                row_count = len(row_ids)
            groups.append(
                {
                    "key": str(group_key),
                    "column": mapping.group_column,
                    "row_count": row_count,
                    "is_count_only": is_count_only,
                    "row_ids": [] if is_count_only else row_ids[:100],  # First 100 only
                    "agg_functions": mapping.agg_functions,
                }
            )
    return groups


def _build_row_index(ctx) -> dict[int, dict]:
    """Build searchable index of row events with full timeline for time-travel."""
    row_events = {}

    # Build step lookup for code locations
    step_lookup = {}
    for step in ctx.store.steps:
        code_loc = None
        if step.code_file and step.code_line:
            code_loc = f"{_format_file_name(step.code_file)}:{step.code_line}"
        step_lookup[step.step_id] = {
            "operation": step.operation,
            "stage": step.stage or "",
            "code_loc": code_loc,
            "code_file": step.code_file,
            "code_line": step.code_line,
        }

    # Index diffs
    for i in range(len(ctx.store.diff_row_ids)):
        row_id = ctx.store.diff_row_ids[i]
        if row_id not in row_events:
            row_events[row_id] = {"diffs": [], "dropped_at": None, "timeline": {}}

        step_id = ctx.store.diff_step_ids[i]
        change_type = ctx.store.diff_change_types[i]

        # Get step info
        step_info = step_lookup.get(
            step_id, {"operation": f"Step {step_id}", "stage": "", "code_loc": None}
        )

        change_names = {0: "MODIFIED", 1: "DROPPED", 2: "ADDED", 3: "REORDERED"}

        col = ctx.store.diff_cols[i]
        old_val = ctx.store.diff_old_vals[i]
        new_val = ctx.store.diff_new_vals[i]

        if change_type == 1:  # DROPPED
            row_events[row_id]["dropped_at"] = {
                "step_id": step_id,
                "operation": step_info["operation"],
                "stage": step_info["stage"],
                "code_loc": step_info["code_loc"],
            }
        else:
            row_events[row_id]["diffs"].append(
                {
                    "step_id": step_id,
                    "operation": step_info["operation"],
                    "stage": step_info["stage"],
                    "code_loc": step_info["code_loc"],
                    "column": col,
                    "old_val": old_val,
                    "new_val": new_val,
                    "change_type": change_names.get(change_type, "UNKNOWN"),
                }
            )

    return row_events


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    /* Colors - Cosmic Slate Theme */
    --bg-app: #0f1115;
    --bg-panel: #161b22;
    --bg-card: #1c2128;
    --bg-hover: #2d333b;
    --bg-input: #0d1117;

    --border-subtle: #30363d;
    --border-focus: #58a6ff;

    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;

    --code-bg: #0d1117;

    --accent-blue: #58a6ff;
    --accent-blue-dim: rgba(88, 166, 255, 0.15);
    --accent-green: #3fb950;
    --accent-green-dim: rgba(63, 185, 80, 0.15);
    --accent-red: #f85149;
    --accent-red-dim: rgba(248, 81, 73, 0.15);
    --accent-purple: #bc8cff;
    --accent-purple-dim: rgba(188, 140, 255, 0.15);
    --accent-orange: #d29922;
    --accent-cyan: #39c5cf;

    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);

    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

[data-theme="light"] {
    --bg-app: #ffffff;
    --bg-panel: #f6f8fa;
    --bg-card: #ffffff;
    --bg-hover: #f3f4f6;
    --bg-input: #ffffff;

    --border-subtle: #d0d7de;
    --border-focus: #0969da;

    --text-primary: #24292f;
    --text-secondary: #57606a;
    --text-muted: #6e7781;

    --code-bg: #f6f8fa;

    --accent-blue: #0969da;
    --accent-blue-dim: rgba(9, 105, 218, 0.1);
    --accent-green: #1a7f37;
    --accent-green-dim: rgba(26, 127, 55, 0.1);
    --accent-red: #cf222e;
    --accent-red-dim: rgba(207, 34, 46, 0.1);
    --accent-purple: #8250df;
    --accent-purple-dim: rgba(130, 80, 223, 0.1);
    --accent-orange: #bf8700;
    --accent-cyan: #0598a6;

    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.1);
}

* { box-sizing: border-box; }

body {
    font-family: var(--font-main);
    background: var(--bg-app);
    color: var(--text-primary);
    margin: 0;
    padding: 0;
    line-height: 1.5;
    height: 100vh;
    display: flex;
    overflow: hidden;
}

/* Layout */
.app-container {
    display: flex;
    width: 100%;
    height: 100%;
}

/* Sidebar */
.sidebar {
    width: 260px;
    background: var(--bg-panel);
    border-right: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}

.logo-area {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.theme-toggle {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}

.theme-toggle:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
}

.logo-text {
    font-size: 1.25rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}

.nav-menu {
    flex: 1;
    padding: 16px 12px;
    overflow-y: auto;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    margin-bottom: 4px;
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.nav-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
}

.nav-item.active {
    background: var(--accent-blue-dim);
    color: var(--accent-blue);
}

.nav-icon { width: 18px; text-align: center; }

/* Main Content */
.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
}

/* Top Bar */
.top-bar {
    height: 64px;
    background: var(--bg-app);
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    flex-shrink: 0;
}

.page-title {
    font-size: 1.1rem;
    font-weight: 600;
}

.search-wrapper {
    position: relative;
    width: 400px;
}

.search-input {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 8px 16px 8px 36px;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 0.9rem;
    transition: border-color 0.2s;
}

.search-input:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px var(--accent-blue-dim);
}

.search-icon-abs {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* Scrollable Canvas */
.canvas {
    flex: 1;
    overflow-y: auto;
    padding: 32px;
}

/* Dashboard Grid */
.grid-cols-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-bottom: 32px; }
.grid-cols-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-bottom: 32px; }

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
}

.card h3 {
    margin: 0 0 16px 0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    font-weight: 600;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 8px;
}

.metric-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
}

.metric-trend {
    font-size: 0.85rem;
    font-weight: 500;
}
.trend-up { color: var(--accent-green); }
.trend-down { color: var(--accent-red); }

/* Pipeline Timeline */
.pipeline-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 800px;
    margin: 0 auto;
    position: relative;
}

.pipeline-container::before {
    content: '';
    position: absolute;
    left: 24px;
    top: 20px;
    bottom: 20px;
    width: 2px;
    background: var(--border-subtle);
    z-index: 0;
}

.pipeline-step-card {
    display: flex;
    gap: 20px;
    position: relative;
    z-index: 1;
}

.step-marker {
    width: 50px;
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
}

.step-dot {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-app);
    border: 2px solid var(--accent-blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-blue);
    margin-bottom: 8px;
}

.pipeline-step-card.dropped .step-dot { border-color: var(--accent-red); color: var(--accent-red); }
.pipeline-step-card.transform .step-dot { border-color: var(--accent-purple); color: var(--accent-purple); }

.step-content {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}

.step-content:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--border-focus);
}

.step-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-hover);
}

.step-title {
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--accent-blue);
    font-size: 0.95rem;
}

.step-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.1);
    color: var(--text-secondary);
    font-weight: 500;
}

.step-body {
    padding: 16px;
}

.step-stats {
    display: flex;
    gap: 24px;
    margin-bottom: 12px;
    font-size: 0.85rem;
}

.stat-item { display: flex; flex-direction: column; }
.stat-lbl { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 2px; }
.stat-val { font-weight: 600; }

.code-snippet {
    background: var(--code-bg);
    border-radius: 6px;
    padding: 12px;
    margin-top: 12px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    overflow-x: auto;
    border: 1px solid var(--border-subtle);
}

.code-line { display: flex; gap: 12px; line-height: 1.5; }
.code-line.highlight { background: rgba(88, 166, 255, 0.15); }
.lineno { color: var(--text-muted); min-width: 24px; text-align: right; user-select: none; }
.marker { color: var(--accent-blue); font-weight: bold; width: 10px; user-select: none; }
.content { color: var(--text-primary); white-space: pre; }

/* Row Explorer View */
.row-explorer {
    display: none; /* Hidden by default */
    height: 100%;
}

.row-explorer.visible { display: flex; }

.row-sidebar {
    width: 350px;
    border-right: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    background: var(--bg-panel);
}

.row-main {
    flex: 1;
    padding: 32px;
    overflow-y: auto;
    background: var(--bg-app);
}

.timeline-list {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

.timeline-item {
    display: flex;
    gap: 12px;
    padding: 12px;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
    margin-bottom: 8px;
    transition: all 0.2s;
}

.timeline-item:hover { background: var(--bg-hover); }
.timeline-item.active { background: var(--bg-hover); border-color: var(--accent-blue); }

.tl-icon {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--bg-card);
    border: 2px solid var(--text-muted);
    flex-shrink: 0;
}

.timeline-item.dropped .tl-icon { border-color: var(--accent-red); background: var(--accent-red-dim); }
.timeline-item.modified .tl-icon { border-color: var(--accent-orange); background: rgba(210, 153, 34, 0.15); }
.timeline-item.added .tl-icon { border-color: var(--accent-green); background: var(--accent-green-dim); }

.tl-content { flex: 1; min-width: 0; }
.tl-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tl-meta { font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between; }

/* Data Grid */
.data-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
}

.data-cell {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
}

.cell-label { color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 6px; }
.cell-value { font-family: var(--font-mono); font-size: 0.95rem; word-break: break-all; }
.cell-value.changed { color: var(--accent-orange); font-weight: 600; }

.diff-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.diff-pill.mod { background: var(--accent-orange); color: #fff; }
.diff-pill.new { background: var(--accent-green); color: #fff; }

/* Empty States */
.empty-state {
    text-align: center;
    padding: 64px;
    color: var(--text-muted);
}
.empty-icon { font-size: 3rem; margin-bottom: 16px; opacity: 0.5; }

/* Helpers */
.text-green { color: var(--accent-green); }
.text-red { color: var(--accent-red); }
.text-mono { font-family: var(--font-mono); }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

@media (max-width: 1024px) {
    .grid-cols-4 { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""

JAVASCRIPT = """
<script>
// Data injected from Python
const pipelineData = __PIPELINE_DATA__;
const droppedSummary = __DROPPED_SUMMARY__;
const changesSummary = __CHANGES_SUMMARY__;
const groupsSummary = __GROUPS_SUMMARY__;
const rowIndex = __ROW_INDEX__;
const suggestedRows = __SUGGESTED_ROWS__;
const totalRegisteredRows = __TOTAL_REGISTERED_ROWS__;

// State
let activeView = 'dashboard';
let selectedRowId = null;
let currentStepIndex = -1; // -1 means end of time
let rowHistory = [];
let currentTheme = localStorage.getItem('tracepipe-theme') || 'dark';

function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme();
    localStorage.setItem('tracepipe-theme', currentTheme);
}

function applyTheme() {
    const root = document.documentElement;
    const icon = document.getElementById('theme-icon');

    if (currentTheme === 'light') {
        root.setAttribute('data-theme', 'light');
        icon.textContent = '🌙';
    } else {
        root.removeAttribute('data-theme');
        icon.textContent = '☀️';
    }
}

function switchView(viewName) {
    document.querySelectorAll('.view-section').forEach(el => el.style.display = 'none');
    document.getElementById(`view-${viewName}`).style.display = 'block';

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`nav-${viewName}`).classList.add('active');

    activeView = viewName;

    if (viewName === 'row-explorer' && !selectedRowId) {
        renderSuggestedRows();
    }
}

function renderSuggestedRows() {
    const container = document.getElementById('row-timeline-list');
    let html = '<div style="padding: 16px;">';

    // Helper for sections
    const renderSection = (title, items, icon, descFn) => {
        if (!items || items.length === 0) return '';
        let s = `<div style="margin-bottom: 20px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">
                ${title}
            </div>`;
        items.forEach(item => {
            s += `
            <div class="timeline-item" onclick="loadRow(${item.id})" style="margin-bottom: 4px; padding: 8px 12px;">
                <div class="tl-icon" style="width: 20px; height: 20px; font-size: 12px; display: flex; align-items: center; justify-content: center; border: none; background: transparent;">${icon}</div>
                <div class="tl-content">
                    <div class="tl-title" style="font-size: 0.85rem;">Row ${item.id}</div>
                    <div class="tl-meta" style="font-size: 0.75rem;">${descFn(item)}</div>
                </div>
            </div>`;
        });
        s += '</div>';
        return s;
    };

    if (suggestedRows) {
        html += renderSection('🚫 Dropped Samples', suggestedRows.dropped, '❌', i => ` at ${i.reason}`);
        html += renderSection('✏️ Most Modified', suggestedRows.modified, '📝', i => `${i.count} changes`);
        html += renderSection('✅ Survivors', suggestedRows.survivors, '🏁', i => 'Successfully processed');
    }

    if (html === '<div style="padding: 16px;">') {
        html = '<div class="empty-state" style="padding: 32px;">Search for a Row ID to inspect its journey.</div>';
    } else {
        html += '</div>';
    }

    container.innerHTML = html;
}

function searchRow(e) {
    if (e && e.key !== 'Enter') return;

    const input = document.getElementById('globalSearch');
    const val = input.value.trim();
    if (!val) return;

    const rowId = parseInt(val);
    if (!isNaN(rowId)) {
        loadRow(rowId);
    }
}

function loadRow(rowId) {
    selectedRowId = rowId;
    switchView('row-explorer');

    const rowData = rowIndex[rowId];
    const container = document.getElementById('row-timeline-list');
    const detailContainer = document.getElementById('row-detail-view');

    if (!rowData && (rowId < 0 || rowId >= totalRegisteredRows)) {
        container.innerHTML = '<div class="empty-state">Row ID not found</div>';
        detailContainer.innerHTML = '';
        return;
    }

    // Build History
    rowHistory = [];

    // Initial state (step 0) - empty or implied
    rowHistory.push({
        step_id: 0,
        operation: 'Initial State',
        state: {},
        changes: []
    });

    let currentState = {};
    let events = [];

    if (rowData) {
        // Collect all events
        if (rowData.diffs) events.push(...rowData.diffs);
        if (rowData.dropped_at) {
            events.push({
                ...rowData.dropped_at,
                change_type: 'DROPPED',
                is_drop: true
            });
        }

        events.sort((a, b) => a.step_id - b.step_id);

        // Replay history
        events.forEach(event => {
            const changes = [];

            if (event.change_type === 'DROPPED') {
                 // Mark as dropped in state
                 currentState['__status__'] = 'DROPPED';
            } else if (event.column) {
                // Determine previous value if not in state yet
                if (!(event.column in currentState) && event.old_val !== undefined) {
                    currentState[event.column] = event.old_val;
                    // Retrospectively update initial state if this is the first time we see it?
                    // Ideally we'd have full initial state, but we only have diffs.
                    // We can assume the "old_val" was the value at start if it hasn't changed before.
                    // Simplify: Just update current state.
                }

                currentState[event.column] = event.new_val;
                changes.push({
                    col: event.column,
                    old: event.old_val,
                    new: event.new_val
                });
            }

            rowHistory.push({
                step_id: event.step_id,
                operation: event.operation,
                code_loc: event.code_loc,
                is_drop: event.is_drop,
                state: {...currentState}, // Snapshot
                changes: changes
            });
        });
    } else {
        // Row exists but no tracked changes
        rowHistory.push({
            step_id: 999,
            operation: 'No Changes Tracked',
            state: {},
            changes: []
        });
    }

    renderTimeline();
    selectTimelineStep(rowHistory.length - 1);
}

function renderTimeline() {
    const container = document.getElementById('row-timeline-list');
    let html = '';

    rowHistory.forEach((step, index) => {
        let statusClass = '';
        if (step.is_drop) statusClass = 'dropped';
        else if (step.changes.length > 0) statusClass = 'modified';

        html += `
            <div class="timeline-item ${statusClass}" id="tl-step-${index}" onclick="selectTimelineStep(${index})">
                <div class="tl-icon"></div>
                <div class="tl-content">
                    <div class="tl-title">${escapeHtml(step.operation)}</div>
                    <div class="tl-meta">
                        <span>Step ${step.step_id}</span>
                        <span>${step.changes.length} changes</span>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function selectTimelineStep(index) {
    // UI Update
    document.querySelectorAll('.timeline-item').forEach(el => el.classList.remove('active'));
    const item = document.getElementById(`tl-step-${index}`);
    if (item) item.classList.add('active');

    const step = rowHistory[index];
    const prevState = index > 0 ? rowHistory[index-1].state : {};
    const currState = step.state;

    // Merge keys from current and all history to show full picture
    const allKeys = new Set([...Object.keys(currState), ...Object.keys(prevState)]);
    // Filter out internal
    allKeys.delete('__status__');

    const container = document.getElementById('row-detail-view');
    let gridHtml = '';

    if (step.is_drop) {
        gridHtml = `
            <div style="grid-column: 1/-1; background: rgba(248, 81, 73, 0.1); border: 1px solid var(--accent-red); padding: 24px; border-radius: 8px; text-align: center;">
                <h3 style="color: var(--accent-red); margin-top: 0;">🚫 Row Dropped</h3>
                <p>This row was removed from the pipeline at this step.</p>
                <div class="code-loc-badge">${escapeHtml(step.code_loc || '')}</div>
            </div>
        `;
    } else {
        if (allKeys.size === 0) {
             gridHtml = '<div style="grid-column: 1/-1; color: var(--text-muted); font-style: italic;">No column values tracked.</div>';
        } else {
            Array.from(allKeys).sort().forEach(key => {
                const val = currState[key];
                const prev = prevState[key];
                const changed = val !== prev && prev !== undefined; // Simple check

                // Check specific changes list for accuracy
                const changeRecord = step.changes.find(c => c.col === key);
                const isChanged = !!changeRecord;

                gridHtml += `
                    <div class="data-cell" style="${isChanged ? 'border-color: var(--accent-orange); background: rgba(210, 153, 34, 0.05);' : ''}">
                        <div class="cell-label">${escapeHtml(key)}</div>
                        <div class="cell-value ${isChanged ? 'changed' : ''}">
                            ${formatValue(val)}
                            ${isChanged ? '<span class="diff-pill mod">MOD</span>' : ''}
                        </div>
                        ${isChanged ? `<div style="margin-top:4px; font-size:0.75rem; color:var(--text-muted);">Was: ${formatValue(changeRecord.old)}</div>` : ''}
                    </div>
                `;
            });
        }
    }

    container.innerHTML = `
        <div style="margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin: 0;">State at Step ${step.step_id}</h2>
            <span style="color: var(--text-muted);">${escapeHtml(step.operation)}</span>
        </div>
        <div class="data-grid">
            ${gridHtml}
        </div>
    `;
}

// Utils
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function formatValue(val) {
    if (val === null || val === undefined) return '<span style="color: var(--text-muted);">NULL</span>';
    return escapeHtml(String(val));
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    applyTheme();

    // Parse URL for direct linking ?row=123
    const urlParams = new URLSearchParams(window.location.search);
    const rowParam = urlParams.get('row');
    if (rowParam) {
        loadRow(parseInt(rowParam));
    }
});
</script>
"""


def save(filepath: str, title: str = "TracePipe Dashboard") -> None:
    """
    Save interactive lineage report as HTML.

    Args:
        filepath: Path to save the HTML file
        title: Title for the report (shown in browser tab and header)
    """
    ctx = get_context()

    # Gather data
    pipeline_data = _get_pipeline_data(ctx)
    dropped_summary = _get_dropped_summary(ctx)
    changes_summary = _get_changes_summary(ctx)
    groups_summary = _get_groups_summary(ctx)
    row_index = _build_row_index(ctx)

    # Total registered rows (approximate)
    total_registered = (
        ctx.row_manager._next_row_id if hasattr(ctx.row_manager, "_next_row_id") else 0
    )

    # Identify Suggested Rows for UX
    suggested_rows = {"dropped": [], "modified": [], "survivors": []}

    # 1. Dropped Rows (Sample up to 5 unique operations)
    dropped_sample_map = {}
    for i in range(len(ctx.store.diff_row_ids)):
        if ctx.store.diff_change_types[i] == 1:  # DROPPED
            step_id = ctx.store.diff_step_ids[i]
            row_id = ctx.store.diff_row_ids[i]
            if step_id not in dropped_sample_map:
                dropped_sample_map[step_id] = row_id

    for step_id, row_id in list(dropped_sample_map.items())[:5]:
        step = next((s for s in ctx.store.steps if s.step_id == step_id), None)
        op_name = step.operation if step else f"Step {step_id}"
        suggested_rows["dropped"].append({"id": int(row_id), "reason": op_name})

    # 2. Heavily Modified Rows (Top 5 by change count)
    change_counts = {}
    for i in range(len(ctx.store.diff_row_ids)):
        if ctx.store.diff_change_types[i] == 0:  # MODIFIED
            rid = ctx.store.diff_row_ids[i]
            change_counts[rid] = change_counts.get(rid, 0) + 1

    top_changed = sorted(change_counts.items(), key=lambda x: -x[1])[:5]
    for rid, count in top_changed:
        suggested_rows["modified"].append({"id": int(rid), "count": count})

    # 3. Survivors (Sample 5 that are not dropped)
    dropped_ids = set(ctx.store.get_dropped_rows())
    survivors = []
    # Try a range of potential IDs
    import random

    potential_ids = list(range(max(0, total_registered - 100), total_registered))  # Last 100
    if not potential_ids and total_registered > 0:
        potential_ids = list(range(total_registered))

    random.shuffle(potential_ids)
    for rid in potential_ids:
        if rid not in dropped_ids:
            survivors.append({"id": rid})
            if len(survivors) >= 5:
                break
    suggested_rows["survivors"] = survivors

    # Initial/Final rows for health calc
    initial_rows = (
        pipeline_data[0]["input_shape"][0]
        if pipeline_data and pipeline_data[0]["input_shape"]
        else 0
    )
    final_rows = (
        pipeline_data[-1]["output_shape"][0]
        if pipeline_data and pipeline_data[-1]["output_shape"]
        else 0
    )

    # HTML Generation
    pipeline_html = ""
    for step in pipeline_data:
        snippet_html = (
            f'<div class="code-snippet">{step["code_snippet"]}</div>'
            if step["code_snippet"]
            else ""
        )

        status_cls = ""
        if "dropped" in step["operation"].lower() or step["rows_affected"] < 0:
            status_cls = "dropped"
        elif "setitem" in step["operation"] or "replace" in step["operation"]:
            status_cls = "transform"

        shape_info = ""
        if step["input_shape"] and step["output_shape"]:
            shape_info = f"""
            <div class="stat-item">
                <span class="stat-lbl">Flow</span>
                <span class="stat-val">{step["input_shape"][0]} → {step["output_shape"][0]} rows</span>
            </div>
            """

        pipeline_html += f"""
        <div class="pipeline-step-card {status_cls}">
            <div class="step-marker">
                <div class="step-dot">{step["id"]}</div>
            </div>
            <div class="step-content">
                <div class="step-header">
                    <span class="step-title">{_escape(step["operation"])}</span>
                    {f'<span class="step-badge">{_escape(step["stage"])}</span>' if step["stage"] else ""}
                </div>
                <div class="step-body">
                    <div class="step-stats">
                        {shape_info}
                        <div class="stat-item">
                            <span class="stat-lbl">Location</span>
                            <span class="stat-val" style="font-family: var(--font-mono);">{_escape(step["code_loc"] or "Unknown")}</span>
                        </div>
                    </div>
                    {snippet_html}
                </div>
            </div>
        </div>
        """

    # Escape title for HTML
    escaped_title = html.escape(title)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escaped_title}</title>
    {CSS}
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="logo-area">
                <div class="logo-text">TracePipe</div>
                <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
                    <span id="theme-icon">☀️</span>
                </button>
            </div>
            <div class="nav-menu">
                <div class="nav-item active" id="nav-dashboard" onclick="switchView('dashboard')">
                    <span class="nav-icon">📊</span> Dashboard
                </div>
                <div class="nav-item" id="nav-pipeline" onclick="switchView('pipeline')">
                    <span class="nav-icon">⚡</span> Pipeline Flow
                </div>
                <div class="nav-item" id="nav-row-explorer" onclick="switchView('row-explorer')">
                    <span class="nav-icon">🔍</span> Row Inspector
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Top Bar -->
            <div class="top-bar">
                <div class="page-title">{escaped_title}</div>
                <div class="search-wrapper">
                    <i class="search-icon-abs">🔍</i>
                    <input type="text" id="globalSearch" class="search-input"
                           placeholder="Search Row ID (e.g. 12)"
                           onkeydown="searchRow(event)">
                </div>
            </div>

            <!-- View: Dashboard -->
            <div id="view-dashboard" class="view-section canvas">
                <div class="grid-cols-4">
                    <div class="card">
                        <h3>Pipeline Steps</h3>
                        <div class="metric-value">{len(pipeline_data)}</div>
                        <div class="metric-sub">Total Operations</div>
                    </div>
                    <div class="card">
                        <h3>Retention</h3>
                        <div class="metric-value">{
        (final_rows / initial_rows * 100) if initial_rows else 0:.1f}%</div>
                        <div class="metric-sub">{_format_number(final_rows)} of {
        _format_number(initial_rows)
    } rows</div>
                    </div>
                    <div class="card">
                        <h3>Rows Dropped</h3>
                        <div class="metric-value" style="color: var(--accent-red);">{
        _format_number(dropped_summary["total"])
    }</div>
                        <div class="metric-sub">Across {
        len(dropped_summary["by_operation"])
    } filters</div>
                    </div>
                    <div class="card">
                        <h3>Cell Changes</h3>
                        <div class="metric-value" style="color: var(--accent-orange);">{
        _format_number(changes_summary["total"])
    }</div>
                        <div class="metric-sub">In watched columns</div>
                    </div>
                </div>

                <div class="grid-cols-2">
                    <div class="card">
                        <h3>Top Drop Reasons</h3>
                        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 16px;">
                            {
        "".join(
            f'<div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border-subtle); padding-bottom:8px;"><span>{_escape(k)}</span><span style="font-weight:600;">{_format_number(v)}</span></div>'
            for k, v in list(dropped_summary["by_operation"].items())[:5]
        )
    }
                        </div>
                    </div>
                    <div class="card">
                         <h3>Most Changed Columns</h3>
                         <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 16px;">
                            {
        "".join(
            f'<div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border-subtle); padding-bottom:8px;"><span>{_escape(k)}</span><span style="font-weight:600;">{_format_number(v)}</span></div>'
            for k, v in list(changes_summary["by_column"].items())[:5]
        )
    }
                        </div>
                    </div>
                </div>
            </div>

            <!-- View: Pipeline -->
            <div id="view-pipeline" class="view-section canvas" style="display: none;">
                <div class="pipeline-container">
                    {pipeline_html}
                </div>
            </div>

            <!-- View: Row Explorer -->
            <div id="view-row-explorer" class="view-section row-explorer">
                <div class="row-sidebar">
                    <div style="padding: 16px; border-bottom: 1px solid var(--border-subtle);">
                        <div style="font-size: 0.85rem; color: var(--text-muted);">Event Timeline</div>
                    </div>
                    <div id="row-timeline-list" class="timeline-list">
                        <div class="empty-state" style="padding: 32px;">
                            Search for a Row ID to inspect its journey.
                        </div>
                    </div>
                </div>
                <div id="row-detail-view" class="row-main">
                    <!-- Details will be injected here -->
                    <div class="empty-state">
                        <div class="empty-icon">👈</div>
                        <h3>Select a step</h3>
                        <p>Click a step in the timeline to see the row's state at that point.</p>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Inject Data -->
    {
        JAVASCRIPT.replace("__PIPELINE_DATA__", json.dumps(pipeline_data))
        .replace("__DROPPED_SUMMARY__", json.dumps(dropped_summary))
        .replace("__CHANGES_SUMMARY__", json.dumps(changes_summary))
        .replace("__GROUPS_SUMMARY__", json.dumps(groups_summary))
        .replace("__ROW_INDEX__", json.dumps(row_index))
        .replace("__SUGGESTED_ROWS__", json.dumps(suggested_rows))
        .replace("__TOTAL_REGISTERED_ROWS__", str(total_registered))
    }
</body>
</html>
"""

    with open(filepath, "w") as f:
        f.write(html_content)
