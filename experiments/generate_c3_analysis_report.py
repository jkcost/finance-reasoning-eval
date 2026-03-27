"""
C3 Deep Analysis Report Generator

변환에도 불구하고 정답을 맞춘 C3 케이스를 상세 분석하는 HTML 리포트 생성.
각 케이스의 원인, 증거, 모델 응답을 시각적으로 제공.

Usage:
    python experiments/generate_c3_analysis_report.py
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from apply_transformations_full import _extract_numbers_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CAUSE_META = {
    "ic_ignored_contradiction": {
        "label": "IC Contradiction Ignored",
        "color": "#e53935",
        "icon": "&#x26A0;",
        "fixable": False,
        "description": "Model ignores contradictory 1.5x value and uses the original value.",
    },
    "possible_memorization": {
        "label": "Possible Memorization",
        "color": "#7b1fa2",
        "icon": "&#x1F9E0;",
        "fixable": False,
        "description": "Model produces correct answer despite critical data removal, likely from training data.",
    },
    "removed_irrelevant_data": {
        "label": "Removed Irrelevant Data",
        "color": "#f57c00",
        "icon": "&#x1F4CA;",
        "fixable": True,
        "description": "Transformation removed data not needed for the solution (CVRR=0).",
    },
    "question_leaks_data": {
        "label": "Question Data Leakage",
        "color": "#0288d1",
        "icon": "&#x1F4A7;",
        "fixable": True,
        "description": "Removed values also appear in the question text.",
    },
    "ta_temporal_still_solvable": {
        "label": "TA Still Solvable",
        "color": "#455a64",
        "icon": "&#x23F0;",
        "fixable": False,
        "description": "Temporal ambiguity insufficient - model infers period from context clues.",
    },
}


def _esc(text: str) -> str:
    return escape(str(text)) if text else ""


def classify_cause(r: Dict, p: Dict) -> str:
    ttype = r["transformation_type"]
    t = p["transformations"][ttype]
    question = p["question"]
    orig_ctx = p["context_original"]
    sol = p.get("python_solution", "")

    if ttype == "IC":
        return "ic_ignored_contradiction"
    if ttype == "TA":
        return "ta_temporal_still_solvable"

    trans_ctx = t.get("context_transformed", orig_ctx)
    orig_nums = _extract_numbers_from_text(orig_ctx)
    trans_nums = _extract_numbers_from_text(trans_ctx)
    removed = orig_nums - trans_nums
    q_nums = _extract_numbers_from_text(question)
    leaked = removed & q_nums if removed else set()
    sol_vals = _extract_numbers_from_text(sol)
    critical_removed = sol_vals & removed if sol_vals else set()

    if removed and leaked == removed:
        return "question_leaks_data"
    if not removed or len(critical_removed) == 0:
        return "removed_irrelevant_data"
    return "possible_memorization"


def extract_evidence(r: Dict, p: Dict) -> Dict:
    """Extract detailed evidence for a C3 case."""
    ttype = r["transformation_type"]
    t = p["transformations"][ttype]
    question = p["question"]
    orig_ctx = p["context_original"]
    sol = p.get("python_solution", "")

    trans_ctx = t.get("context_transformed", orig_ctx) if ttype != "TA" else orig_ctx
    trans_q = t.get("question_transformed", question) if ttype == "TA" else question

    orig_nums = _extract_numbers_from_text(orig_ctx)
    trans_nums = _extract_numbers_from_text(trans_ctx)
    removed = orig_nums - trans_nums
    q_nums = _extract_numbers_from_text(question)
    sol_vals = _extract_numbers_from_text(sol)

    # Extract value assignments from model response
    raw = r.get("raw_response", "")
    val_assignments = []
    for line in raw.split("\n"):
        line_s = line.strip()
        if (
            "=" in line_s
            and any(c.isdigit() for c in line_s)
            and not line_s.startswith("#")
        ):
            val_assignments.append(line_s)

    # IC-specific: extract contradiction values
    ic_original = ""
    ic_contradictory = ""
    if ttype == "IC":
        desc = t.get("description", "")
        # Parse "Inserted contradictory sentence: X vs Y" or similar
        vs_match = re.search(r":\s*(.+?)\s+vs\s+(.+?)$", desc)
        if vs_match:
            ic_original = vs_match.group(1).strip()
            ic_contradictory = vs_match.group(2).strip()

    return {
        "removed": sorted(removed),
        "leaked": sorted(removed & q_nums) if removed else [],
        "critical": sorted(sol_vals & removed) if sol_vals else [],
        "sol_vals": sorted(sol_vals)[:12],
        "q_nums": sorted(q_nums),
        "trans_ctx": trans_ctx,
        "trans_q": trans_q,
        "val_assignments": val_assignments[:8],
        "raw_response": raw,
        "ic_original": ic_original,
        "ic_contradictory": ic_contradictory,
    }


def _build_unified_diff(original: str, transformed: str) -> str:
    """Build a unified diff view with line-by-line comparison.

    Shows all lines from original context. Lines removed are highlighted red,
    lines added are highlighted green, unchanged lines shown normally.
    For text (non-table) contexts, also does word-level diff within changed lines.
    """
    import difflib

    if not original and not transformed:
        return ""

    orig_lines = (original or "").split("\n")
    trans_lines = (transformed or "").split("\n")

    matcher = difflib.SequenceMatcher(None, orig_lines, trans_lines)
    html_lines: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in orig_lines[i1:i2]:
                html_lines.append(
                    f'<div class="diff-line diff-equal">{_esc(line) or "&nbsp;"}</div>'
                )
        elif tag == "delete":
            for line in orig_lines[i1:i2]:
                html_lines.append(
                    f'<div class="diff-line diff-del">'
                    f'<span class="diff-marker">-</span>{_esc(line)}</div>'
                )
        elif tag == "insert":
            for line in trans_lines[j1:j2]:
                html_lines.append(
                    f'<div class="diff-line diff-ins">'
                    f'<span class="diff-marker">+</span>{_esc(line)}</div>'
                )
        elif tag == "replace":
            # Show word-level diff for replaced lines
            for line in orig_lines[i1:i2]:
                html_lines.append(
                    f'<div class="diff-line diff-del">'
                    f'<span class="diff-marker">-</span>{_esc(line)}</div>'
                )
            for line in trans_lines[j1:j2]:
                html_lines.append(
                    f'<div class="diff-line diff-ins">'
                    f'<span class="diff-marker">+</span>{_esc(line)}</div>'
                )

    return "\n".join(html_lines)


def generate_report(eval_data: Dict, trans_data: Dict, output_path: Path) -> None:
    trans_idx = {p["question_id"]: p for p in trans_data["problems"]}
    all_results = eval_data["results"]
    c3_all = [r for r in all_results if r["case_type"] == 3]

    # Classify causes
    c3_with_cause = []
    for r in c3_all:
        p = trans_idx[r["question_id"]]
        cause = classify_cause(r, p)
        c3_with_cause.append((r, p, cause))

    cause_counts = Counter(c for _, _, c in c3_with_cause)
    total = len(all_results)
    total_c3 = len(c3_all)

    # Group by cause -> question_id+ttype -> models
    grouped = defaultdict(lambda: defaultdict(list))
    for r, p, cause in c3_with_cause:
        key = (r["question_id"], r["transformation_type"])
        grouped[cause][key].append(r)

    # Model stats
    model_c3 = Counter(r["model"] for r in c3_all)
    models = eval_data["metadata"]["models"]

    # IC per-model stats
    ic_results = [r for r in all_results if r["transformation_type"] == "IC"]
    ic_c3_by_model = Counter(
        r["model"]
        for r in all_results
        if r["transformation_type"] == "IC" and r["case_type"] == 3
    )
    ic_total_per_model = len(ic_results) // len(models) if models else 1

    # Start HTML
    html_parts = [_build_head()]
    html_parts.append("<body>")
    html_parts.append('<div class="container">')

    # Header
    html_parts.append(f"""
    <h1>C3 Deep Analysis Report</h1>
    <p class="subtitle">
        Total evaluations: {total} | C3 cases: {total_c3} ({total_c3 / total * 100:.1f}%)
        | Non-fixable: {total_c3 - cause_counts.get("removed_irrelevant_data", 0) - cause_counts.get("question_leaks_data", 0)}/{total_c3}
    </p>
    """)

    # Summary cards
    html_parts.append(_build_summary_cards(cause_counts, total_c3))

    # Model comparison
    html_parts.append(
        _build_model_comparison(
            models, model_c3, ic_c3_by_model, ic_total_per_model, all_results
        )
    )

    # Each cause section
    cause_order = [
        "ic_ignored_contradiction",
        "possible_memorization",
        "removed_irrelevant_data",
        "question_leaks_data",
        "ta_temporal_still_solvable",
    ]

    for cause_name in cause_order:
        cases = grouped.get(cause_name, {})
        if not cases:
            continue
        if cause_name == "ic_ignored_contradiction":
            html_parts.append(
                _build_ic_detail_section(cases, trans_idx, c3_all, all_results)
            )
        else:
            html_parts.append(
                _build_cause_section(cause_name, cases, trans_idx, c3_all)
            )

    # Cross-model patterns
    html_parts.append(_build_cross_model_section(c3_all, trans_idx))

    html_parts.append("</div></body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info(f"Report: {output_path}")


def _build_head() -> str:
    return """<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8">
<title>C3 Deep Analysis</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #333; line-height: 1.6; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { font-size: 28px; margin-bottom: 4px; }
.subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }
h2 { font-size: 22px; margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 2px solid #ddd; }
h3 { font-size: 16px; margin: 16px 0 8px; }

.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px; margin-bottom: 32px; }
.summary-card { background: white; border-radius: 12px; padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
.summary-card .count { font-size: 36px; font-weight: 700; }
.summary-card .label { font-size: 13px; color: #666; margin-top: 4px; }
.summary-card .pct { font-size: 14px; color: #999; }

.cause-section { margin-bottom: 40px; }
.cause-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.cause-badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
               color: white; font-size: 13px; font-weight: 600; }
.cause-desc { font-size: 13px; color: #666; margin-left: auto; }
.fixable-tag { font-size: 11px; padding: 2px 8px; border-radius: 10px; }
.fixable-yes { background: #e8f5e9; color: #2e7d32; }
.fixable-no { background: #fbe9e7; color: #bf360c; }

.case-card { background: white; border-radius: 12px; margin-bottom: 20px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; }
.case-header { padding: 16px 20px; cursor: pointer; display: flex;
               align-items: center; gap: 12px; border-bottom: 1px solid #eee; }
.case-header:hover { background: #fafafa; }
.case-qid { font-weight: 700; font-size: 14px; }
.case-type { font-size: 12px; padding: 2px 8px; border-radius: 4px;
             background: #e3f2fd; color: #1565c0; }
.case-models { font-size: 12px; color: #666; margin-left: auto; }
.case-body { padding: 20px; display: none; }
.case-card.open .case-body { display: block; }

.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
@media (max-width: 900px) { .info-grid { grid-template-columns: 1fr; } }

.info-box { background: #fafafa; border-radius: 8px; padding: 14px; }
.info-box h4 { font-size: 12px; color: #999; text-transform: uppercase;
               letter-spacing: 0.5px; margin-bottom: 8px; }
.info-box pre { font-size: 12px; white-space: pre-wrap; word-break: break-all;
                max-height: 300px; overflow-y: auto; line-height: 1.5; }

.evidence-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.evidence-tag { font-size: 11px; padding: 2px 8px; border-radius: 4px; }
.tag-removed { background: #ffcdd2; color: #b71c1c; }
.tag-critical { background: #f3e5f5; color: #6a1b9a; }
.tag-leaked { background: #e1f5fe; color: #01579b; }
.tag-sol { background: #e8f5e9; color: #1b5e20; }

.model-response { margin-top: 12px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }
.model-resp-header { padding: 10px 14px; background: #f5f5f5; font-size: 13px;
                     font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; }
.model-resp-header:hover { background: #eeeeee; }
.model-resp-body { padding: 14px; display: none; }
.model-response.open .model-resp-body { display: block; }
.model-name { font-size: 12px; padding: 2px 8px; border-radius: 4px; }
.model-gpt { background: #e8f5e9; color: #2e7d32; }
.model-claude { background: #fff3e0; color: #e65100; }
.model-gemini { background: #e3f2fd; color: #1565c0; }

.val-line { font-family: 'Fira Code', monospace; font-size: 12px;
            padding: 2px 8px; background: #fff8e1; border-left: 3px solid #ffc107;
            margin: 2px 0; }
.val-highlight { background: #ffeb3b; padding: 0 2px; font-weight: 600; }

.response-full { font-family: 'Fira Code', monospace; font-size: 11px;
                 white-space: pre-wrap; word-break: break-all;
                 max-height: 400px; overflow-y: auto; background: #fafafa;
                 padding: 12px; border-radius: 6px; line-height: 1.5; }

.diff-removed { background: #ffcdd2; text-decoration: line-through; }
.diff-added { background: #c8e6c9; }

.diff-container { font-family: 'Fira Code', 'Consolas', monospace; font-size: 12px;
                  line-height: 1.6; max-height: 500px; overflow-y: auto;
                  background: #fafafa; border-radius: 8px; padding: 8px 0; }
.diff-line { padding: 1px 12px; white-space: pre-wrap; word-break: break-all; }
.diff-equal { color: #555; }
.diff-del { background: #ffeef0; color: #b31d28; }
.diff-ins { background: #e6ffed; color: #22863a; }
.diff-marker { display: inline-block; width: 16px; font-weight: 700; user-select: none; }

table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
th { background: #f5f5f5; padding: 8px 12px; text-align: left; font-weight: 600; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; }

.bar-container { display: flex; align-items: center; gap: 8px; }
.bar { height: 20px; border-radius: 4px; min-width: 2px; }

.toggle-arrow { transition: transform 0.2s; font-size: 12px; }
.open .toggle-arrow { transform: rotate(90deg); }

.cross-table td { text-align: center; font-size: 12px; }
.cross-3 { background: #ffcdd2; font-weight: 700; }
.cross-2 { background: #fff9c4; }
.cross-1 { background: #f5f5f5; }
.cross-0 { color: #ccc; }
</style>
<script>
function toggle(el) {
  el.closest('.case-card, .model-response').classList.toggle('open');
}
function toggleAll(sectionId, open) {
  document.querySelectorAll('#' + sectionId + ' .case-card').forEach(c => {
    if (open) c.classList.add('open'); else c.classList.remove('open');
  });
}
</script>
</head>"""


def _build_summary_cards(cause_counts: Counter, total_c3: int) -> str:
    html = '<div class="summary-grid">'
    html += f"""
    <div class="summary-card">
      <div class="count" style="color:#f44336;">{total_c3}</div>
      <div class="label">Total C3 Cases</div>
      <div class="pct">out of 345 evaluations</div>
    </div>"""

    for cause_name, meta in CAUSE_META.items():
        count = cause_counts.get(cause_name, 0)
        if count == 0:
            continue
        html += f"""
    <div class="summary-card">
      <div class="count" style="color:{meta["color"]};">{count}</div>
      <div class="label">{meta["icon"]} {meta["label"]}</div>
      <div class="pct">{count / total_c3 * 100:.1f}% of C3</div>
    </div>"""

    nonfixable = sum(
        cause_counts.get(c, 0)
        for c in cause_counts
        if not CAUSE_META.get(c, {}).get("fixable", True)
    )
    html += f"""
    <div class="summary-card">
      <div class="count" style="color:#37474f;">{nonfixable}</div>
      <div class="label">Non-fixable (Model Behavior)</div>
      <div class="pct">{nonfixable / total_c3 * 100:.1f}% of C3</div>
    </div>"""

    html += "</div>"
    return html


def _build_model_comparison(
    models: List[str],
    model_c3: Counter,
    ic_c3_by_model: Counter,
    ic_total_per_model: int,
    all_results: List[Dict],
) -> str:
    html = "<h2>Model Comparison</h2>"
    html += "<table><tr><th>Model</th><th>Total C3</th><th>C3 Rate</th>"
    html += "<th>IC C3</th><th>IC Ignore Rate</th><th>C3 Bar</th></tr>"

    per_model = len(all_results) // len(models) if models else 1
    max_c3 = max(model_c3.values()) if model_c3 else 1

    for model in models:
        c3 = model_c3.get(model, 0)
        ic_c3 = ic_c3_by_model.get(model, 0)
        bar_w = int(c3 / max_c3 * 200) if max_c3 else 0

        model_cls = (
            "model-gpt"
            if "gpt" in model
            else "model-claude"
            if "claude" in model
            else "model-gemini"
        )
        html += f"""<tr>
        <td><span class="model-name {model_cls}">{_esc(model)}</span></td>
        <td>{c3}</td>
        <td>{c3 / per_model * 100:.1f}%</td>
        <td>{ic_c3}/{ic_total_per_model}</td>
        <td>{ic_c3 / ic_total_per_model * 100:.1f}%</td>
        <td><div class="bar-container">
            <div class="bar" style="width:{bar_w}px;background:#f44336;"></div>
            <span style="font-size:12px;">{c3}</span>
        </div></td>
        </tr>"""

    html += "</table>"
    return html


def _model_class(model: str) -> str:
    if "gpt" in model:
        return "model-gpt"
    if "claude" in model:
        return "model-claude"
    return "model-gemini"


def _build_cause_section(
    cause_name: str,
    cases: Dict[Tuple, List[Dict]],
    trans_idx: Dict,
    c3_all: List[Dict],
) -> str:
    meta = CAUSE_META[cause_name]
    total_cases = sum(len(v) for v in cases.values())
    fix_cls = "fixable-yes" if meta["fixable"] else "fixable-no"
    fix_text = "Fixable" if meta["fixable"] else "Non-fixable"
    section_id = cause_name.replace("_", "-")

    html = f"""
    <div class="cause-section" id="{section_id}">
    <div class="cause-header">
      <h2 style="border:none;margin:0;padding:0;">{meta["icon"]} {meta["label"]}</h2>
      <span class="cause-badge" style="background:{meta["color"]};">{total_cases} cases</span>
      <span class="fixable-tag {fix_cls}">{fix_text}</span>
      <span class="cause-desc">{meta["description"]}</span>
    </div>
    <div style="margin-bottom:12px;">
      <button onclick="toggleAll('{section_id}', true)" style="font-size:12px;cursor:pointer;">Expand All</button>
      <button onclick="toggleAll('{section_id}', false)" style="font-size:12px;cursor:pointer;margin-left:4px;">Collapse All</button>
    </div>
    """

    for (qid, ttype), result_list in sorted(cases.items()):
        p = trans_idx[qid]
        model_names = [r["model"] for r in result_list]
        t = p["transformations"][ttype]

        html += f"""
    <div class="case-card">
      <div class="case-header" onclick="toggle(this)">
        <span class="toggle-arrow">&#9654;</span>
        <span class="case-qid">{_esc(qid)}</span>
        <span class="case-type">{_esc(ttype)}</span>
        <span style="font-size:12px;color:#999;">GT: {_esc(str(p.get("ground_truth", "")))}</span>
        <span class="case-models">
          {" ".join(f'<span class="model-name {_model_class(m)}">{_esc(m.split("-")[0] if "-" in m else m)}</span>' for m in model_names)}
        </span>
      </div>
      <div class="case-body">
        <div class="info-grid">
          <div class="info-box">
            <h4>Question</h4>
            <pre>{_esc(p["question"])}</pre>
          </div>
          <div class="info-box">
            <h4>Transformation</h4>
            <pre>{_esc(t.get("description", ""))}</pre>
            <p style="font-size:12px;color:#999;margin-top:8px;">
              Format: {_esc(p["context_format"])} |
              Hardcoded: {p.get("is_hardcoded_solution", False)} |
              Ground Truth: {_esc(str(p.get("ground_truth", "")))}
            </p>
          </div>
        </div>"""

        # Evidence section
        evidence = extract_evidence(result_list[0], p)

        if ttype not in ("IC", "TA"):
            html += (
                '<div class="info-box" style="margin-bottom:16px;"><h4>Evidence</h4>'
            )
            html += '<div class="evidence-row">'
            if evidence["removed"]:
                html += f'<span class="evidence-tag tag-removed">Removed: {", ".join(evidence["removed"][:6])}</span>'
            if evidence["critical"]:
                html += f'<span class="evidence-tag tag-critical">Critical (in solution): {", ".join(evidence["critical"][:6])}</span>'
            if evidence["leaked"]:
                html += f'<span class="evidence-tag tag-leaked">Leaked (in question): {", ".join(evidence["leaked"][:6])}</span>'
            html += "</div>"
            if evidence["sol_vals"]:
                html += f'<div class="evidence-row"><span class="evidence-tag tag-sol">Solution values: {", ".join(evidence["sol_vals"][:10])}</span></div>'
            html += "</div>"

        # Context diff
        if ttype == "TA":
            ta_diff = _build_unified_diff(p["question"], evidence["trans_q"])
            html += f"""
            <div class="info-box" style="margin-bottom:16px;">
              <h4>Question Transformation (Diff)</h4>
              <div class="diff-container">{ta_diff}</div>
            </div>"""
        elif ttype == "IC":
            ic_diff = _build_unified_diff(p["context_original"], evidence["trans_ctx"])
            html += f"""
            <div class="info-box" style="margin-bottom:16px;">
              <h4>Context Diff (IC: added contradictory data)</h4>
              <div class="diff-container">{ic_diff}</div>
            </div>"""
        else:
            ctx_diff = _build_unified_diff(p["context_original"], evidence["trans_ctx"])
            html += f"""
            <div class="info-box" style="margin-bottom:16px;">
              <h4>Context Diff (red = removed, green = added)</h4>
              <div class="diff-container">{ctx_diff}</div>
            </div>"""

        # Model responses
        html += '<h3 style="margin-top:16px;">Model Responses</h3>'
        for r in result_list:
            ev = extract_evidence(r, p)
            model = r["model"]
            mcls = _model_class(model)

            html += f"""
        <div class="model-response">
          <div class="model-resp-header" onclick="toggle(this)">
            <span class="toggle-arrow">&#9654;</span>
            <span class="model-name {mcls}">{_esc(model)}</span>
            <span style="font-size:12px;color:#999;">Case {r["case_type"]}</span>
          </div>
          <div class="model-resp-body">"""

            # Value assignments (key evidence)
            if ev["val_assignments"]:
                html += '<div style="margin-bottom:12px;"><strong style="font-size:12px;">Key Value Assignments:</strong>'
                for vl in ev["val_assignments"]:
                    # Highlight numbers that match removed values
                    highlighted = _esc(vl)
                    for rv in evidence.get("removed", []) + evidence.get(
                        "critical", []
                    ):
                        if rv in vl:
                            highlighted = highlighted.replace(
                                _esc(rv),
                                f'<span class="val-highlight">{_esc(rv)}</span>',
                            )
                    html += f'<div class="val-line">{highlighted}</div>'
                html += "</div>"

            # Full response
            html += f"""
            <details>
              <summary style="font-size:12px;color:#666;cursor:pointer;">Full Response</summary>
              <div class="response-full">{_esc(ev["raw_response"])}</div>
            </details>"""

            html += "</div></div>"  # model-resp-body, model-response

        html += "</div></div>"  # case-body, case-card

    html += "</div>"  # cause-section
    return html


def _detect_value_used(raw_response: str, orig_val: str, contra_val: str) -> str:
    """Detect which value the model actually used in its response."""
    if not orig_val or not contra_val:
        return "unknown"

    # Normalize for matching
    def _normalize(v: str) -> List[str]:
        clean = v.replace("$", "").replace(",", "").replace("%", "").strip()
        variants = [clean, v.strip()]
        try:
            fv = float(clean)
            if fv == int(fv):
                variants.append(str(int(fv)))
                variants.append(f"{int(fv):,}")
            variants.append(str(fv))
        except ValueError:
            pass
        return variants

    orig_variants = _normalize(orig_val)
    contra_variants = _normalize(contra_val)

    has_orig = any(v in raw_response for v in orig_variants if len(v) > 1)
    has_contra = any(v in raw_response for v in contra_variants if len(v) > 1)

    if has_orig and not has_contra:
        return "original"
    if has_contra and not has_orig:
        return "contradictory"
    if has_orig and has_contra:
        return "both_mentioned"
    return "neither"


def _build_ic_detail_section(
    cases: Dict[Tuple, List[Dict]],
    trans_idx: Dict,
    c3_all: List[Dict],
    all_results: List[Dict],
) -> str:
    """Build a detailed IC analysis section with per-model breakdown."""
    meta = CAUSE_META["ic_ignored_contradiction"]
    total_evals = sum(len(v) for v in cases.values())
    unique_problems = len(cases)

    # IC overall stats
    ic_all = [r for r in all_results if r["transformation_type"] == "IC"]
    ic_c1 = sum(1 for r in ic_all if r["case_type"] == 1)
    ic_c2 = sum(1 for r in ic_all if r["case_type"] == 2)
    ic_c3 = sum(1 for r in ic_all if r["case_type"] == 3)

    html = f"""
    <div class="cause-section" id="ic-ignored-contradiction">
    <div class="cause-header">
      <h2 style="border:none;margin:0;padding:0;">{meta["icon"]} {meta["label"]} — Detailed Analysis</h2>
      <span class="cause-badge" style="background:{meta["color"]};">{total_evals} evals / {unique_problems} problems</span>
      <span class="fixable-tag fixable-no">Non-fixable</span>
    </div>

    <div style="background:white;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
      <h3 style="margin-bottom:12px;">IC Overall Distribution</h3>
      <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#4caf50;">{ic_c1}</div>
          <div style="font-size:12px;color:#666;">C1 (Detected)</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#ff9800;">{ic_c2}</div>
          <div style="font-size:12px;color:#666;">C2 (Wrong Answer)</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#f44336;">{ic_c3}</div>
          <div style="font-size:12px;color:#666;">C3 (Correct Despite IC)</div>
        </div>
        <div style="flex:1;min-width:200px;">
          <div style="height:24px;border-radius:12px;overflow:hidden;display:flex;background:#eee;">
            <div style="width:{ic_c1 / len(ic_all) * 100:.1f}%;background:#4caf50;" title="C1"></div>
            <div style="width:{ic_c2 / len(ic_all) * 100:.1f}%;background:#ff9800;" title="C2"></div>
            <div style="width:{ic_c3 / len(ic_all) * 100:.1f}%;background:#f44336;" title="C3"></div>
          </div>
          <div style="font-size:11px;color:#999;margin-top:4px;">
            Detection rate: {ic_c1 / len(ic_all) * 100:.1f}% |
            C3 rate: {ic_c3 / len(ic_all) * 100:.1f}% (total {len(ic_all)} evals)
          </div>
        </div>
      </div>
    </div>
    """

    # Per-problem detailed cards — all expanded by default
    for (qid, ttype), result_list in sorted(cases.items()):
        p = trans_idx[qid]
        t = p["transformations"][ttype]
        evidence = extract_evidence(result_list[0], p)
        model_names = [r["model"] for r in result_list]

        # Extract contradiction info
        ic_orig = evidence["ic_original"]
        ic_contra = evidence["ic_contradictory"]
        desc = t.get("description", "")

        html += f"""
    <div class="case-card open">
      <div class="case-header" onclick="toggle(this)">
        <span class="toggle-arrow">&#9654;</span>
        <span class="case-qid">{_esc(qid)}</span>
        <span class="case-type">IC</span>
        <span style="font-size:12px;color:#999;">GT: {_esc(str(p.get("ground_truth", "")))}</span>
        <span style="font-size:12px;font-weight:600;color:#e53935;margin-left:8px;">
          {len(result_list)}/{len([r for r in ic_all if r["question_id"] == qid])} models = C3
        </span>
        <span class="case-models">
          {" ".join(f'<span class="model-name {_model_class(m)}">{_esc(m)}</span>' for m in model_names)}
        </span>
      </div>
      <div class="case-body">
        <div class="info-grid">
          <div class="info-box">
            <h4>Question</h4>
            <pre>{_esc(p["question"])}</pre>
          </div>
          <div class="info-box">
            <h4>Contradiction Setup</h4>
            <pre>{_esc(desc)}</pre>
            <p style="font-size:12px;color:#999;margin-top:8px;">
              Format: {_esc(p["context_format"])} |
              Ground Truth: {_esc(str(p.get("ground_truth", "")))}
            </p>"""

        # Show contradiction values prominently
        if ic_orig and ic_contra:
            html += f"""
            <div style="margin-top:12px;padding:12px;background:#fff3e0;border-radius:8px;border-left:4px solid #ff9800;">
              <div style="font-size:13px;font-weight:600;margin-bottom:6px;">Contradiction Values</div>
              <div style="display:flex;gap:16px;align-items:center;">
                <div style="padding:6px 12px;background:#e8f5e9;border-radius:6px;font-family:monospace;font-size:14px;">
                  Original: <strong>{_esc(ic_orig)}</strong>
                </div>
                <span style="font-size:16px;color:#999;">vs</span>
                <div style="padding:6px 12px;background:#ffcdd2;border-radius:6px;font-family:monospace;font-size:14px;">
                  Contradictory: <strong>{_esc(ic_contra)}</strong> (1.5x)
                </div>
              </div>
            </div>"""

        html += """
          </div>
        </div>"""

        # Context diff
        ic_diff = _build_unified_diff(p["context_original"], evidence["trans_ctx"])
        html += f"""
        <div class="info-box" style="margin-bottom:16px;">
          <h4>Context Diff (green = contradictory data added)</h4>
          <div class="diff-container">{ic_diff}</div>
        </div>"""

        # Per-model analysis — each model gets a detailed card, open by default
        html += f'<h3 style="margin-top:16px;">Model-by-Model Analysis ({len(result_list)} C3 evaluations)</h3>'

        for r in result_list:
            ev = extract_evidence(r, p)
            model = r["model"]
            mcls = _model_class(model)
            raw = ev["raw_response"]

            # Detect which value the model used
            value_used = _detect_value_used(raw, ic_orig, ic_contra)
            if value_used == "original":
                val_badge = '<span style="padding:2px 8px;background:#e8f5e9;color:#2e7d32;border-radius:4px;font-size:11px;">Used ORIGINAL value</span>'
                val_explanation = "Model ignored the contradictory value and used the original — correct answer but no conflict detection."
            elif value_used == "contradictory":
                val_badge = '<span style="padding:2px 8px;background:#ffcdd2;color:#b71c1c;border-radius:4px;font-size:11px;">Used CONTRADICTORY value</span>'
                val_explanation = "Model adopted the contradictory value but still reached the correct answer (likely coincidence or calculation path)."
            elif value_used == "both_mentioned":
                val_badge = '<span style="padding:2px 8px;background:#fff9c4;color:#f57f17;border-radius:4px;font-size:11px;">Mentioned BOTH values</span>'
                val_explanation = "Model referenced both values but did not flag the contradiction as problematic."
            else:
                val_badge = '<span style="padding:2px 8px;background:#e0e0e0;color:#616161;border-radius:4px;font-size:11px;">Value source unclear</span>'
                val_explanation = "Could not determine which value the model used."

            # Extract the model's final answer
            answer = r.get("extracted_answer", r.get("model_answer", ""))

            html += f"""
        <div class="model-response open">
          <div class="model-resp-header" onclick="toggle(this)">
            <span class="toggle-arrow">&#9654;</span>
            <span class="model-name {mcls}">{_esc(model)}</span>
            {val_badge}
            <span style="font-size:12px;color:#999;margin-left:auto;">
              Answer: {_esc(str(answer))} (GT: {_esc(str(p.get("ground_truth", "")))})
            </span>
          </div>
          <div class="model-resp-body">
            <div style="padding:10px;background:#f5f5f5;border-radius:6px;margin-bottom:12px;font-size:13px;">
              <strong>Analysis:</strong> {val_explanation}
            </div>"""

            # Key value assignments
            if ev["val_assignments"]:
                html += '<div style="margin-bottom:12px;"><strong style="font-size:12px;">Key Value Assignments:</strong>'
                for vl in ev["val_assignments"]:
                    highlighted = _esc(vl)
                    # Highlight original value in green, contradictory in red
                    if ic_orig:
                        for v in [ic_orig, ic_orig.replace("$", "").replace(",", "")]:
                            if v in vl and len(v) > 1:
                                highlighted = highlighted.replace(
                                    _esc(v),
                                    f'<span style="background:#c8e6c9;padding:0 2px;">{_esc(v)}</span>',
                                )
                    if ic_contra:
                        for v in [
                            ic_contra,
                            ic_contra.replace("$", "").replace(",", ""),
                        ]:
                            if v in vl and len(v) > 1:
                                highlighted = highlighted.replace(
                                    _esc(v),
                                    f'<span style="background:#ffcdd2;padding:0 2px;">{_esc(v)}</span>',
                                )
                    html += f'<div class="val-line">{highlighted}</div>'
                html += "</div>"

            # Full response
            html += f"""
            <details open>
              <summary style="font-size:12px;color:#666;cursor:pointer;">Full Response</summary>
              <div class="response-full">{_esc(raw)}</div>
            </details>"""

            html += "</div></div>"  # model-resp-body, model-response

        html += "</div></div>"  # case-body, case-card

    html += "</div>"  # cause-section
    return html


def _build_cross_model_section(c3_all: List[Dict], trans_idx: Dict) -> str:
    html = "<h2>Cross-Model C3 Agreement Matrix</h2>"
    html += "<p style='font-size:13px;color:#666;margin-bottom:12px;'>How many models got C3 for each problem-type combination</p>"

    by_key = defaultdict(set)
    for r in c3_all:
        by_key[(r["question_id"], r["transformation_type"])].add(r["model"])

    # Count by agreement level
    agree_3 = [(k, v) for k, v in by_key.items() if len(v) == 3]
    agree_2 = [(k, v) for k, v in by_key.items() if len(v) == 2]
    agree_1 = [(k, v) for k, v in by_key.items() if len(v) == 1]

    html += f"""
    <div class="summary-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:20px;">
      <div class="summary-card">
        <div class="count" style="color:#d32f2f;">{len(agree_3)}</div>
        <div class="label">3/3 Models (Universal)</div>
      </div>
      <div class="summary-card">
        <div class="count" style="color:#f57c00;">{len(agree_2)}</div>
        <div class="label">2/3 Models (Partial)</div>
      </div>
      <div class="summary-card">
        <div class="count" style="color:#1976d2;">{len(agree_1)}</div>
        <div class="label">1/3 Models (Individual)</div>
      </div>
    </div>"""

    # Detailed table
    html += "<table><tr><th>Problem</th><th>Type</th><th>Models</th><th>Cause</th><th>Agreement</th></tr>"

    for level, items, cls in [
        (3, agree_3, "cross-3"),
        (2, agree_2, "cross-2"),
        (1, agree_1, "cross-1"),
    ]:
        for (qid, ttype), models in sorted(items):
            cause = classify_cause(
                next(
                    r
                    for r in c3_all
                    if r["question_id"] == qid and r["transformation_type"] == ttype
                ),
                trans_idx[qid],
            )
            cause_label = CAUSE_META.get(cause, {}).get("label", cause)
            cause_color = CAUSE_META.get(cause, {}).get("color", "#999")
            model_str = ", ".join(
                f'<span class="model-name {_model_class(m)}">{m.split("-")[0]}</span>'
                for m in sorted(models)
            )
            html += f"""<tr class="{cls}">
            <td>{_esc(qid)}</td><td>{_esc(ttype)}</td>
            <td>{model_str}</td>
            <td><span style="color:{cause_color};font-size:12px;">{cause_label}</span></td>
            <td style="text-align:center;font-weight:700;">{level}/3</td>
            </tr>"""

    html += "</table>"
    return html


def main():
    base = Path(__file__).parent / "results" / "metacognitive"

    eval_files = sorted(base.glob("batch_evaluation_*.json"))
    trans_files = sorted(base.glob("batch_transformations_*.json"))

    if not eval_files or not trans_files:
        logger.error("No data files found")
        return

    with open(eval_files[-1], "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(trans_files[-1], "r", encoding="utf-8") as f:
        trans_data = json.load(f)

    logger.info(f"Loaded: {eval_files[-1].name}, {trans_files[-1].name}")

    output_path = base / "c3_analysis_report.html"
    generate_report(eval_data, trans_data, output_path)


if __name__ == "__main__":
    main()
