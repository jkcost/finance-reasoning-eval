"""
Human Review HTML Generator v2

배치 변환 결과를 여러 작업자가 협업으로 리뷰할 수 있는 인터랙티브 HTML 생성.

v2 주요 기능:
  - 변환 결과 직접 편집 (editable textarea)
  - 작업자 분배 (URL 파라미터 ?assignee=이름&start=0&end=10)
  - JSON 내보내기/가져오기 (작업자별 파일, merge 지원)
  - 풀이 변수 의존관계 하이라이팅

Usage:
    python experiments/generate_human_review.py
    python experiments/generate_human_review.py --input batch_transformations_0_30.json
"""

import argparse
import html
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TYPE_KEYS = ["EA-partial", "EA-full", "SA", "IC", "TA"]
TYPE_DESCRIPTIONS = {
    "EA-partial": "명시적 부재 (부분) — N/A 마커로 특정 값 제거",
    "EA-full": "명시적 부재 (전체) — 키/컬럼 전체 삭제",
    "SA": "무표지 부재 — 마커 없이 조용히 삭제",
    "IC": "정보 충돌 — 1.5x 모순값 삽입",
    "TA": "시간적 모호성 — 연도를 모호한 표현으로 대체",
}
TYPE_COLORS = {
    "EA-partial": "#3b82f6",
    "EA-full": "#8b5cf6",
    "SA": "#f59e0b",
    "IC": "#ef4444",
    "TA": "#6b7280",
}


def _esc(text: str) -> str:
    """HTML escape."""
    return html.escape(str(text)) if text else ""


def _extract_critical_values_from_solution(python_solution: str) -> Dict[str, Any]:
    """Extract variable dependency info from python_solution for visualization.

    Returns dict with:
      - assignments: {var_name: {value, rhs_expr, is_critical}}
      - answer_var: str
      - critical_values: set of numeric strings
    """
    if not python_solution:
        return {"assignments": {}, "answer_var": "", "critical_values": set()}

    lines = python_solution.strip().split("\n")
    assignments: Dict[str, Dict[str, Any]] = {}
    var_values: Dict[str, Set[str]] = {}

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("#") or line_s.startswith("def ") or not line_s:
            continue
        m = re.match(r"(\w+)\s*=\s*(?!=)(.+)", line_s)
        if m:
            var_name = m.group(1)
            rhs = m.group(2)
            nums = set()
            for n in re.findall(r"\$?([\d,]+(?:\.\d+)?)", rhs):
                clean = n.replace(",", "")
                if len(clean.replace(".", "")) < 2:
                    continue
                try:
                    val = float(clean)
                    nums.add(str(int(val)) if val == int(val) else str(val))
                except ValueError:
                    pass
            assignments[var_name] = {"rhs": rhs, "nums": nums}
            var_values[var_name] = nums

    # Find answer variable
    answer_vars: Set[str] = set()
    answer_var = ""
    for line in reversed(lines):
        line_s = line.strip()
        if line_s.startswith("return "):
            answer_vars.update(re.findall(r"\b([a-zA-Z_]\w*)\b", line_s[7:]))
            answer_var = "return"
            break
        m = re.match(r"(answer|result)\s*=\s*(.+)", line_s)
        if m:
            answer_var = m.group(1)
            answer_vars.update(re.findall(r"\b([a-zA-Z_]\w*)\b", m.group(2)))
            break

    # BFS trace
    visited: Set[str] = set()
    queue = list(answer_vars)
    while queue:
        var = queue.pop(0)
        if var in visited:
            continue
        visited.add(var)
        if var in assignments:
            refs = re.findall(r"\b([a-zA-Z_]\w*)\b", assignments[var]["rhs"])
            for ref in refs:
                if ref in assignments and ref not in visited:
                    queue.append(ref)

    # Mark critical
    critical_values: Set[str] = set()
    result_assignments = {}
    for var_name, info in assignments.items():
        is_critical = var_name in visited
        result_assignments[var_name] = {
            "rhs": info["rhs"],
            "nums": list(info["nums"]),
            "is_critical": is_critical,
        }
        if is_critical:
            critical_values.update(info["nums"])

    return {
        "assignments": result_assignments,
        "answer_var": answer_var,
        "critical_values": list(critical_values),
    }


def _render_problem_card(problem: Dict, idx: int) -> str:
    """Render a single problem review card."""
    qid = problem["question_id"]
    question = problem["question"]
    context = problem["context_original"]
    ctx_format = problem["context_format"]
    ground_truth = problem["ground_truth"]
    is_hardcoded = problem["is_hardcoded_solution"]
    python_solution = problem.get("python_solution", "")
    transformations = problem["transformations"]

    # Extract critical values for highlighting
    sol_info = _extract_critical_values_from_solution(python_solution)
    critical_json = json.dumps(sol_info["critical_values"])

    # Build solution visualization
    sol_lines_html = ""
    if sol_info["assignments"]:
        for var_name, info in sol_info["assignments"].items():
            cls = "sol-critical" if info["is_critical"] else "sol-non-critical"
            marker = "●" if info["is_critical"] else "○"
            sol_lines_html += (
                f'<div class="{cls}">'
                f'<span class="sol-marker">{marker}</span>'
                f"<code>{_esc(var_name)} = {_esc(info['rhs'])}</code>"
                f"</div>"
            )

    # Count successful transformations
    success_types = [k for k in TYPE_KEYS if transformations.get(k, {}).get("success")]

    # Build transformation tabs and panels
    tabs_html = ""
    panels_html = ""
    for ttype in TYPE_KEYS:
        tdata = transformations.get(ttype, {})
        is_success = tdata.get("success", False)
        tab_class = "tab-success" if is_success else "tab-fail"
        type_desc = TYPE_DESCRIPTIONS.get(ttype, "")

        tabs_html += (
            f'<button class="tab-btn {tab_class}" '
            f"onclick=\"showTab('{qid}', '{ttype}')\" "
            f'id="tab-{qid}-{ttype}" title="{_esc(type_desc)}">'
            f"{ttype}</button>"
        )

        if is_success:
            desc = tdata.get("description", "")
            has_ctx = "context_transformed" in tdata
            has_q = "question_transformed" in tdata
            transformed_ctx = tdata.get("context_transformed", "")
            transformed_q = tdata.get("question_transformed", "")

            panel_content = (
                f'<div class="transform-desc">'
                f'<span class="type-info">{_esc(type_desc)}</span>'
                f"<br>{_esc(desc)}</div>"
            )

            if has_q:
                panel_content += (
                    f'<div class="diff-container">'
                    f'<div class="diff-panel">'
                    f'<div class="diff-label">원본 Question</div>'
                    f'<pre class="diff-content">{_esc(question)}</pre></div>'
                    f'<div class="diff-panel diff-transformed">'
                    f'<div class="diff-label">변환 후 Question</div>'
                    f'<pre class="diff-content">{_esc(transformed_q)}</pre>'
                    f"</div></div>"
                )

            if has_ctx:
                # Escaped versions for display
                orig_esc = _esc(context)
                trans_esc = _esc(transformed_ctx)
                # Raw for editing
                trans_raw = (
                    transformed_ctx.replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$")
                )

                panel_content += (
                    f'<div class="diff-container">'
                    f'<div class="diff-panel">'
                    f'<div class="diff-label">원본 Context</div>'
                    f'<pre class="diff-content" id="orig-ctx-{qid}-{ttype}">{orig_esc}</pre></div>'
                    f'<div class="diff-panel diff-transformed">'
                    f'<div class="diff-label">변환 후'
                    f"<button class=\"edit-btn\" onclick=\"toggleEdit('{qid}', '{ttype}')\">편집</button>"
                    f"</div>"
                    f'<pre class="diff-content" id="view-ctx-{qid}-{ttype}">{trans_esc}</pre>'
                    f'<textarea class="edit-area" id="edit-ctx-{qid}-{ttype}" '
                    f'style="display:none;" '
                    f"onchange=\"saveEditedCtx('{qid}', '{ttype}')\">"
                    f"{_esc(transformed_ctx)}</textarea>"
                    f"</div></div>"
                )

            # Judgment UI
            panel_content += f"""
            <div class="judgment-box" id="judgment-{qid}-{ttype}">
              <div class="judgment-row">
                <div class="judgment-buttons">
                  <button class="j-btn j-approve" onclick="setJudgment('{qid}', '{ttype}', 'approved')">승인</button>
                  <button class="j-btn j-modify" onclick="setJudgment('{qid}', '{ttype}', 'needs_modification')">수정 필요</button>
                  <button class="j-btn j-reject" onclick="setJudgment('{qid}', '{ttype}', 'rejected')">부적절</button>
                </div>
                <div class="judgment-selects">
                  <select id="unsolvable-{qid}-{ttype}" onchange="saveMeta('{qid}', '{ttype}')">
                    <option value="">unsolvable?</option>
                    <option value="yes">예 - 풀 수 없음</option>
                    <option value="no">아니오 - 여전히 풀 수 있음</option>
                    <option value="partial">부분적</option>
                  </select>
                  <select id="essential-{qid}-{ttype}" onchange="saveMeta('{qid}', '{ttype}')">
                    <option value="">제거 정보 필수?</option>
                    <option value="critical">필수</option>
                    <option value="helpful">보조</option>
                    <option value="irrelevant">무관</option>
                  </select>
                </div>
              </div>
              <textarea class="note-area" id="note-{qid}-{ttype}" rows="2"
                        placeholder="판정 이유 / 수정 방향 메모..."
                        onchange="saveMeta('{qid}', '{ttype}')"></textarea>
              <div class="judgment-status" id="status-{qid}-{ttype}"></div>
            </div>"""
        else:
            reason = tdata.get("reason", "unknown")
            panel_content = (
                f'<div class="transform-fail">'
                f"변환 실패: <code>{_esc(reason)}</code></div>"
            )

        panels_html += (
            f'<div class="tab-panel" id="panel-{qid}-{ttype}" style="display:none;">'
            f"{panel_content}</div>"
        )

    hardcoded_badge = (
        '<span class="badge badge-warn">하드코딩</span>' if is_hardcoded else ""
    )

    return f"""
    <div class="problem-card" id="card-{qid}" data-qid="{qid}" data-index="{idx}">
      <div class="card-header" onclick="toggleCard('{qid}')">
        <div class="card-title">
          <span class="card-index">#{idx}</span>
          <span class="card-qid">{qid}</span>
          <span class="badge badge-format">{ctx_format}</span>
          {hardcoded_badge}
          <span class="badge badge-count">{len(success_types)}/{len(TYPE_KEYS)}</span>
          <span class="review-status" id="review-{qid}"></span>
        </div>
        <div class="card-arrow" id="arrow-{qid}">&#9654;</div>
      </div>
      <div class="card-body" id="body-{qid}" style="display:none;">
        <div class="question-box">
          <div class="section-label">Question</div>
          <div class="question-text">{_esc(question)}</div>
        </div>
        <div class="meta-row">
          <span><strong>Ground Truth:</strong> {_esc(str(ground_truth))}</span>
          <span><strong>Format:</strong> {ctx_format}</span>
        </div>
        <details class="collapsible">
          <summary>Original Context</summary>
          <pre class="context-content">{_esc(context)}</pre>
        </details>
        <details class="collapsible">
          <summary>Python Solution <span class="sol-legend">● 풀이에 사용 &nbsp; ○ 미사용</span></summary>
          <div class="sol-viz" data-critical='{critical_json}'>{sol_lines_html}</div>
        </details>
        <div class="tabs-container">
          <div class="tabs-bar">{tabs_html}</div>
          {panels_html}
        </div>
      </div>
    </div>"""


def generate_html(data: Dict, preloaded_annotations: Dict = None) -> str:
    """Generate the full HTML review page.

    Args:
        data: Batch transformation result JSON.
        preloaded_annotations: Optional merged annotations to embed in HTML.
            Format: {qid: {ttype: {judgment, note, ...}}}
    """
    metadata = data["metadata"]
    coverage = data["coverage_summary"]
    problems = data["problems"]

    total = coverage["total_problems"]
    transformable = coverage["transformable"]

    # Problem cards
    cards_html = ""
    for problem in problems:
        cards_html += _render_problem_card(problem, problem["index"])

    # Coverage rows
    coverage_rows = ""
    for ttype in TYPE_KEYS:
        info = coverage["by_type"][ttype]
        color = TYPE_COLORS.get(ttype, "#666")
        coverage_rows += (
            f'<tr><td><span class="type-badge" style="background:{color}">{ttype}</span></td>'
            f"<td>{info['success']}/{total}</td>"
            f'<td><div class="bar-bg"><div class="bar-fill" style="width:{info["rate"]}%;background:{color}"></div></div></td>'
            f"<td>{info['rate']}%</td></tr>"
        )

    # Embed problem data as JSON for export
    problems_meta = []
    for p in problems:
        problems_meta.append(
            {
                "question_id": p["question_id"],
                "index": p["index"],
                "context_format": p["context_format"],
                "success_types": [
                    k
                    for k in TYPE_KEYS
                    if p["transformations"].get(k, {}).get("success")
                ],
            }
        )
    problems_meta_json = json.dumps(problems_meta)

    # Preloaded annotations JSON
    preloaded_json = json.dumps(preloaded_annotations or {}, ensure_ascii=False)

    range_start = metadata["range"][0]
    range_end = metadata["range"][1]

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Human Review v2 [{range_start}-{range_end})</title>
<style>
:root {{
  --bg: #0a0a0a; --surface: #141414; --surface2: #1e1e1e;
  --border: #2a2a2a; --text: #e5e5e5; --text2: #a0a0a0;
  --blue: #3b82f6; --purple: #8b5cf6; --amber: #f59e0b;
  --red: #ef4444; --green: #22c55e; --gray: #6b7280;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; font-size: 14px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 22px; font-weight: 600; }}
.header-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }}
.assignee-bar {{ display: flex; align-items: center; gap: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 14px; }}
.assignee-bar label {{ font-size: 13px; color: var(--text2); }}
.assignee-bar input {{ background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 4px 8px; font-size: 13px; width: 120px; }}
.assignee-bar .range-info {{ font-size: 12px; color: var(--text2); }}

/* Summary */
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 16px; }}
.summary-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }}
.summary-card .value {{ font-size: 24px; font-weight: 700; }}
.summary-card .label {{ font-size: 11px; color: var(--text2); }}

/* Coverage */
.coverage-table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; background: var(--surface); border-radius: 8px; overflow: hidden; font-size: 13px; }}
.coverage-table th {{ background: var(--surface2); padding: 8px 10px; text-align: left; color: var(--text2); }}
.coverage-table td {{ padding: 8px 10px; border-top: 1px solid var(--border); }}
.type-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: white; font-size: 12px; font-weight: 600; }}
.bar-bg {{ background: var(--surface2); border-radius: 4px; height: 6px; width: 100px; }}
.bar-fill {{ height: 6px; border-radius: 4px; }}

/* Toolbar */
.toolbar {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
.toolbar button {{ padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; font-size: 12px; }}
.toolbar button:hover {{ background: var(--surface2); }}
.toolbar button.active {{ background: var(--blue); border-color: var(--blue); color: white; }}
.toolbar .sep {{ width: 1px; height: 20px; background: var(--border); }}
.progress-text {{ font-size: 12px; color: var(--text2); margin-left: auto; }}

/* Problem card */
.problem-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 6px; }}
.problem-card.hidden {{ display: none; }}
.card-header {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; cursor: pointer; }}
.card-header:hover {{ background: var(--surface2); }}
.card-title {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
.card-index {{ font-weight: 700; color: var(--text2); font-size: 13px; min-width: 32px; }}
.card-qid {{ font-family: monospace; font-size: 13px; }}
.card-arrow {{ color: var(--text2); font-size: 11px; transition: transform 0.15s; }}
.card-arrow.open {{ transform: rotate(90deg); }}
.card-body {{ padding: 0 14px 14px; }}
.badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
.badge-format {{ background: #1e3a5f; color: var(--blue); }}
.badge-warn {{ background: #3d2608; color: var(--amber); }}
.badge-count {{ background: var(--surface2); color: var(--text2); }}

/* Question & context */
.question-box {{ margin-bottom: 10px; }}
.section-label {{ font-size: 12px; font-weight: 600; color: var(--text2); margin-bottom: 4px; }}
.question-text {{ background: var(--surface2); padding: 10px; border-radius: 6px; font-size: 13px; }}
.meta-row {{ display: flex; gap: 20px; margin-bottom: 10px; font-size: 12px; color: var(--text2); }}
.collapsible {{ margin-bottom: 10px; }}
.collapsible summary {{ font-size: 12px; font-weight: 600; color: var(--text2); cursor: pointer; padding: 4px 0; }}
.collapsible summary:hover {{ color: var(--text); }}
.context-content {{ background: var(--surface2); padding: 10px; border-radius: 6px; font-size: 11px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }}

/* Solution visualization */
.sol-viz {{ background: var(--surface2); padding: 10px; border-radius: 6px; font-size: 12px; }}
.sol-critical {{ padding: 2px 4px; margin: 1px 0; border-left: 3px solid var(--green); padding-left: 8px; }}
.sol-non-critical {{ padding: 2px 4px; margin: 1px 0; border-left: 3px solid var(--border); padding-left: 8px; opacity: 0.5; }}
.sol-marker {{ margin-right: 4px; font-size: 10px; }}
.sol-critical .sol-marker {{ color: var(--green); }}
.sol-non-critical .sol-marker {{ color: var(--text2); }}
.sol-legend {{ font-size: 11px; color: var(--text2); font-weight: 400; }}

/* Tabs */
.tabs-container {{ margin-top: 10px; }}
.tabs-bar {{ display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }}
.tab-btn {{ padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface2); color: var(--text); cursor: pointer; font-size: 12px; font-weight: 500; }}
.tab-btn:hover {{ border-color: var(--text2); }}
.tab-btn.active {{ border-color: var(--blue); background: #1e3a5f; color: white; }}
.tab-success {{ border-left: 3px solid var(--green); }}
.tab-fail {{ border-left: 3px solid var(--red); opacity: 0.4; }}

/* Diff */
.diff-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }}
.diff-panel {{ background: var(--surface2); border-radius: 6px; overflow: hidden; }}
.diff-transformed {{ border: 1px solid var(--amber); }}
.diff-label {{ font-size: 11px; font-weight: 600; color: var(--text2); padding: 5px 10px; background: rgba(255,255,255,0.03); display: flex; align-items: center; gap: 6px; }}
.diff-content {{ padding: 10px; font-size: 11px; white-space: pre-wrap; word-break: break-word; max-height: 350px; overflow-y: auto; margin: 0; background: transparent; }}
.transform-desc {{ font-size: 12px; padding: 6px 10px; background: #1a1a2e; border-radius: 6px; margin-bottom: 4px; }}
.transform-desc .type-info {{ color: var(--text2); font-size: 11px; }}
.transform-fail {{ font-size: 12px; color: var(--red); padding: 10px; background: #1a0a0a; border-radius: 6px; }}

/* Edit button & area */
.edit-btn {{ font-size: 10px; padding: 2px 8px; border-radius: 4px; border: 1px solid var(--amber); background: transparent; color: var(--amber); cursor: pointer; margin-left: auto; }}
.edit-btn:hover {{ background: var(--amber); color: black; }}
.edit-btn.editing {{ background: var(--amber); color: black; }}
.edit-area {{ width: 100%; min-height: 200px; max-height: 400px; background: var(--bg); border: 2px solid var(--amber); border-radius: 6px; color: var(--text); padding: 10px; font-size: 11px; font-family: monospace; white-space: pre-wrap; resize: vertical; }}

/* Judgment */
.judgment-box {{ margin-top: 10px; padding: 10px; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); }}
.judgment-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.judgment-buttons {{ display: flex; gap: 4px; }}
.judgment-selects {{ display: flex; gap: 4px; }}
.judgment-selects select {{ background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 4px 6px; font-size: 11px; }}
.j-btn {{ padding: 5px 14px; border-radius: 6px; border: 2px solid transparent; cursor: pointer; font-size: 12px; font-weight: 500; }}
.j-approve {{ background: #052e16; color: var(--green); border-color: #14532d; }}
.j-approve:hover, .j-approve.selected {{ background: var(--green); color: white; }}
.j-modify {{ background: #422006; color: var(--amber); border-color: #713f12; }}
.j-modify:hover, .j-modify.selected {{ background: var(--amber); color: black; }}
.j-reject {{ background: #1a0505; color: var(--red); border-color: #7f1d1d; }}
.j-reject:hover, .j-reject.selected {{ background: var(--red); color: white; }}
.note-area {{ width: 100%; margin-top: 6px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 6px; font-size: 12px; resize: vertical; }}
.judgment-status {{ margin-top: 4px; font-size: 11px; }}

/* Review status on header */
.review-status {{ font-size: 10px; padding: 1px 5px; border-radius: 4px; }}
.review-complete {{ background: #052e16; color: var(--green); }}
.review-partial {{ background: #422006; color: var(--amber); }}

/* Keyboard hint */
.kb-hint {{ position: fixed; bottom: 16px; right: 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 11px; color: var(--text2); z-index: 100; }}
.kb-hint kbd {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 3px; padding: 1px 4px; font-size: 10px; }}

/* Guide panel */
.guide-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }}
.guide-toggle {{ width: 100%; padding: 10px 14px; background: none; border: none; color: var(--text); cursor: pointer; text-align: left; font-size: 14px; font-weight: 600; display: flex; justify-content: space-between; }}
.guide-toggle:hover {{ background: var(--surface2); }}
.guide-body {{ padding: 0 14px 14px; display: none; }}
.guide-body.open {{ display: block; }}
.guide-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin-bottom: 14px; }}
.guide-card {{ background: var(--surface2); border-radius: 8px; padding: 12px; border-left: 4px solid; }}
.guide-card h3 {{ font-size: 13px; margin-bottom: 6px; }}
.guide-card p {{ font-size: 12px; color: var(--text2); margin-bottom: 4px; }}
.guide-card .guide-tag {{ display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 3px; margin-right: 4px; }}
.guide-card .example {{ font-size: 11px; background: var(--bg); padding: 6px 8px; border-radius: 4px; margin-top: 6px; font-family: monospace; white-space: pre-wrap; }}
.guide-section {{ margin-top: 14px; }}
.guide-section h3 {{ font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text2); }}
.guide-section table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.guide-section th {{ text-align: left; padding: 6px 8px; background: var(--bg); color: var(--text2); }}
.guide-section td {{ padding: 6px 8px; border-top: 1px solid var(--border); }}

@media (max-width: 768px) {{
  .diff-container {{ grid-template-columns: 1fr; }}
  .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
  .guide-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
  <div class="header-row">
    <h1>Human Review v2 [{range_start}, {range_end})</h1>
    <div class="assignee-bar">
      <label>작업자:</label>
      <input type="text" id="assignee-input" placeholder="이름 입력" onchange="setAssignee()">
      <span class="range-info" id="range-info"></span>
    </div>
  </div>

  <div class="summary-grid">
    <div class="summary-card"><div class="value">{total}</div><div class="label">전체</div></div>
    <div class="summary-card"><div class="value">{transformable}</div><div class="label">변환 가능</div></div>
    <div class="summary-card"><div class="value" id="reviewed-count">0</div><div class="label">리뷰 완료</div></div>
    <div class="summary-card"><div class="value" id="approved-count">0</div><div class="label">승인</div></div>
    <div class="summary-card"><div class="value" id="modify-count">0</div><div class="label">수정 필요</div></div>
    <div class="summary-card"><div class="value" id="rejected-count">0</div><div class="label">부적절</div></div>
    <div class="summary-card"><div class="value" id="edited-count">0</div><div class="label">직접 편집</div></div>
  </div>

  <table class="coverage-table">
    <thead><tr><th>유형</th><th>성공</th><th>비율</th><th>%</th></tr></thead>
    <tbody>{coverage_rows}</tbody>
  </table>

  <!-- Transformation Guide -->
  <div class="guide-panel">
    <button class="guide-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
      변환 유형 가이드 &amp; 리뷰 기준 <span>&#9660;</span>
    </button>
    <div class="guide-body">

      <h3 style="font-size:14px;margin-bottom:10px;">연구 배경</h3>
      <p style="font-size:12px;color:var(--text2);margin-bottom:14px;">
        이 리뷰의 목적은 <strong>LLM이 금융 문제에서 정보 부족/충돌을 인식하는 능력(메타인지)</strong>을 평가하기 위한 데이터셋을 만드는 것입니다.
        원본 문제에서 풀이에 필요한 데이터를 제거/변형하여 "unsolvable" 문제를 만들고, LLM이 이를 탐지하여 답변을 거부하는지 테스트합니다.
        변환이 올바르려면: (1) 제거된 정보가 풀이에 <strong>필수</strong>여야 하고, (2) 남은 정보만으로는 답을 <strong>도출할 수 없어야</strong> 합니다.
      </p>

      <div class="guide-grid">
        <div class="guide-card" style="border-color:#3b82f6;">
          <h3 style="color:#3b82f6;">EA-partial (명시적 부재 - 부분)</h3>
          <p><strong>방법:</strong> 특정 셀/값을 "N/A"로 교체</p>
          <p><strong>목적:</strong> 명시적 마커가 있을 때 LLM이 탐지하는지 (기본선)</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#052e16;color:#22c55e;">LOW</span> 마커가 보임</p>
          <div class="example">원본: Revenue 2024 = $100M
변환: Revenue 2024 = N/A</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> N/A로 바뀐 값이 풀이에 실제로 필요한가?</p>
        </div>

        <div class="guide-card" style="border-color:#8b5cf6;">
          <h3 style="color:#8b5cf6;">EA-full (명시적 부재 - 전체)</h3>
          <p><strong>방법:</strong> 테이블 컬럼 또는 JSON 키 전체 삭제</p>
          <p><strong>목적:</strong> 구조적 부재를 LLM이 인식하는지</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#422006;color:#f59e0b;">MODERATE</span> 마커는 없지만 구조 변화</p>
          <div class="example">원본: | Issued | In Treasury | Outstanding |
변환: | Issued | In Treasury |  (컬럼 삭제)</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 삭제된 컬럼의 값이 다른 컬럼으로부터 역산 가능한지 확인! (예: Outstanding = Issued + Treasury이면 삭제해도 풀 수 있음 = 부적절)</p>
        </div>

        <div class="guide-card" style="border-color:#f59e0b;">
          <h3 style="color:#f59e0b;">SA (무표지 부재 - Silent Absence)</h3>
          <p><strong>방법:</strong> 마커 없이 조용히 삭제. 문장/행/값을 제거하되 남은 문맥이 자연스러움</p>
          <p><strong>목적:</strong> 아무 단서 없이 LLM이 누락을 눈치채는지 (가장 어려운 부재 탐지)</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#7f1d1d;color:#ef4444;">HIGH</span> 마커 없음</p>
          <div class="example">원본: "순이익은 $2.3M이었다. 영업이익률은 15%였다."
변환: "영업이익률은 15%였다."  (순이익 문장 삭제)</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 삭제 후 문맥이 자연스러운지, 삭제된 정보 없이 정말 풀 수 없는지</p>
        </div>

        <div class="guide-card" style="border-color:#ef4444;">
          <h3 style="color:#ef4444;">IC (정보 충돌 - Information Conflict)</h3>
          <p><strong>방법:</strong> 원래 값의 1.5배인 모순 데이터를 삽입</p>
          <p><strong>목적:</strong> LLM이 수치 모순을 탐지하는지 (부재와 다른 메타인지 능력)</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#7f1d1d;color:#ef4444;">VERY HIGH</span> 현재 전 모델 실패</p>
          <div class="example">원본: Revenue 2024 = $100M
변환: Revenue 2024 = $150M
      Revenue 2024 (conflicting report) = $100M</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 모순이 자연스럽게 삽입되었는지, 모순 대상이 풀이에 사용되는 값인지</p>
        </div>

        <div class="guide-card" style="border-color:#6b7280;">
          <h3 style="color:#6b7280;">TA (시간적 모호성 - Temporal Ambiguity) [보조]</h3>
          <p><strong>방법:</strong> question의 연도를 "the end of the period"로 교체</p>
          <p><strong>목적:</strong> 다중 연도 데이터에서 시점 모호성을 인식하는지</p>
          <p><strong>적용 범위:</strong> ~11개 hard 문제에만 적용 가능</p>
          <div class="example">원본: "What was the revenue in 2024?"
변환: "What was the revenue at the end of the period?"</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> context에 여러 연도 데이터가 있어서 실제로 모호해지는지</p>
        </div>
      </div>

      <div class="guide-section">
        <h3>판정 기준</h3>
        <table>
          <tr><th>판정</th><th>기준</th><th>예시</th></tr>
          <tr>
            <td style="color:var(--green);">승인</td>
            <td>제거된 정보가 풀이에 필수적이며, 남은 정보로는 답을 도출할 수 없음</td>
            <td>ROE 계산에 필요한 순이익을 삭제 = 풀 수 없음</td>
          </tr>
          <tr>
            <td style="color:var(--amber);">수정 필요</td>
            <td>방향은 맞지만 제거 대상이나 방식에 개선이 필요</td>
            <td>풀이와 무관한 값을 삭제함 / 너무 많이 삭제하여 부자연스러움</td>
          </tr>
          <tr>
            <td style="color:var(--red);">부적절</td>
            <td>변환 후에도 문제를 풀 수 있거나, 변환이 의미 없음</td>
            <td>삭제한 컬럼의 값이 다른 컬럼에서 역산 가능</td>
          </tr>
        </table>
      </div>

      <div class="guide-section">
        <h3>Python Solution 읽는 법</h3>
        <p style="font-size:12px;color:var(--text2);">
          각 문제 하단의 Python Solution에서 <span style="color:var(--green);">● 초록색</span> 변수는 최종 답(answer)을 계산하는 데 사용되는 값이고,
          <span style="color:var(--text2);">○ 회색</span> 변수는 사용되지 않는 값입니다.
          변환이 올바르려면 <span style="color:var(--green);">● 초록색</span> 변수에 해당하는 context 데이터가 제거되어야 합니다.
        </p>
      </div>

    </div>
  </div>

  <div class="toolbar">
    <button class="active" onclick="filterCards('all', this)">전체</button>
    <button onclick="filterCards('pending', this)">미리뷰</button>
    <button onclick="filterCards('approved', this)">승인</button>
    <button onclick="filterCards('needs_modification', this)">수정 필요</button>
    <button onclick="filterCards('rejected', this)">부적절</button>
    <button onclick="filterCards('edited', this)">편집됨</button>
    <div class="sep"></div>
    <button onclick="expandAll()">모두 펼치기</button>
    <button onclick="collapseAll()">모두 접기</button>
    <div class="sep"></div>
    <button onclick="exportJSON()" style="background:#1e3a5f;color:var(--blue);">JSON 내보내기</button>
    <button onclick="document.getElementById('import-file').click()">JSON 가져오기</button>
    <input type="file" id="import-file" style="display:none;" accept=".json" onchange="handleImport(event)">
    <button onclick="exportMerged()" style="background:#14532d;color:var(--green);">통합 내보내기</button>
    <span class="progress-text" id="progress-text"></span>
  </div>

  <div id="cards-container">{cards_html}</div>
</div>

<div class="kb-hint">
  <kbd>J</kbd> 다음 문제 &nbsp;
  <kbd>K</kbd> 이전 문제 &nbsp;
  <kbd>1</kbd>~<kbd>5</kbd> 탭 전환 &nbsp;
  <kbd>A</kbd> 승인 &nbsp;
  <kbd>R</kbd> 부적절
</div>

<script>
const STORAGE_KEY = 'hr2_{range_start}_{range_end}';
const PROBLEMS_META = {problems_meta_json};
const PRELOADED = {preloaded_json};
let annotations = {{}};
let currentCardIdx = -1;
let currentAssignee = '';

// ===== Init =====
function init() {{
  // URL params
  const params = new URLSearchParams(window.location.search);
  const assignee = params.get('assignee') || '';
  const rangeStart = params.get('start');
  const rangeEnd = params.get('end');

  if (assignee) {{
    document.getElementById('assignee-input').value = assignee;
    currentAssignee = assignee;
  }}

  if (rangeStart !== null || rangeEnd !== null) {{
    const s = parseInt(rangeStart || '0');
    const e = parseInt(rangeEnd || '999');
    document.getElementById('range-info').textContent = `[${{s}}-${{e}})`;
    // Hide cards outside range
    document.querySelectorAll('.problem-card').forEach(card => {{
      const idx = parseInt(card.dataset.index);
      if (idx < s || idx >= e) card.classList.add('hidden');
    }});
  }}

  loadAnnotations();
}}

// ===== Storage =====
function getStorageKey() {{
  return currentAssignee ? `${{STORAGE_KEY}}_${{currentAssignee}}` : STORAGE_KEY;
}}

function loadAnnotations() {{
  // Start from preloaded (merged from previous review round)
  annotations = JSON.parse(JSON.stringify(PRELOADED));
  try {{
    const saved = localStorage.getItem(getStorageKey());
    if (saved) {{
      // Merge: localStorage overrides preloaded (local work takes priority)
      const local = JSON.parse(saved);
      for (const [qid, types] of Object.entries(local)) {{
        if (!annotations[qid]) annotations[qid] = {{}};
        for (const [ttype, tdata] of Object.entries(types)) {{
          annotations[qid][ttype] = {{ ...(annotations[qid][ttype] || {{}}), ...tdata }};
        }}
      }}
    }}
  }} catch(e) {{ console.error('Load failed:', e); }}
  applyAnnotations();
  updateCounts();
}}

function saveAnnotations() {{
  try {{
    localStorage.setItem(getStorageKey(), JSON.stringify(annotations));
  }} catch(e) {{ console.error('Save failed:', e); }}
  updateCounts();
}}

function setAssignee() {{
  const prev = currentAssignee;
  currentAssignee = document.getElementById('assignee-input').value.trim();
  if (prev && prev !== currentAssignee) {{
    // Save current under old key, load new
    localStorage.setItem(`${{STORAGE_KEY}}_${{prev}}`, JSON.stringify(annotations));
    annotations = {{}};
  }}
  loadAnnotations();
  // Update URL
  const url = new URL(window.location);
  if (currentAssignee) url.searchParams.set('assignee', currentAssignee);
  else url.searchParams.delete('assignee');
  history.replaceState(null, '', url);
}}

function applyAnnotations() {{
  for (const [qid, types] of Object.entries(annotations)) {{
    for (const [ttype, data] of Object.entries(types)) {{
      if (data.judgment) highlightJudgment(qid, ttype, data.judgment);
      if (data.note) {{
        const el = document.getElementById(`note-${{qid}}-${{ttype}}`);
        if (el) el.value = data.note;
      }}
      if (data.unsolvable) {{
        const el = document.getElementById(`unsolvable-${{qid}}-${{ttype}}`);
        if (el) el.value = data.unsolvable;
      }}
      if (data.essential) {{
        const el = document.getElementById(`essential-${{qid}}-${{ttype}}`);
        if (el) el.value = data.essential;
      }}
      if (data.edited_context) {{
        const el = document.getElementById(`edit-ctx-${{qid}}-${{ttype}}`);
        if (el) el.value = data.edited_context;
      }}
    }}
    updateCardStatus(qid);
  }}
}}

// ===== Judgment =====
function setJudgment(qid, ttype, judgment) {{
  if (!annotations[qid]) annotations[qid] = {{}};
  if (!annotations[qid][ttype]) annotations[qid][ttype] = {{}};
  annotations[qid][ttype].judgment = judgment;
  annotations[qid][ttype].assignee = currentAssignee;
  annotations[qid][ttype].timestamp = new Date().toISOString();
  highlightJudgment(qid, ttype, judgment);
  saveAnnotations();
  updateCardStatus(qid);
}}

function highlightJudgment(qid, ttype, judgment) {{
  const box = document.getElementById(`judgment-${{qid}}-${{ttype}}`);
  if (!box) return;
  box.querySelectorAll('.j-btn').forEach(b => b.classList.remove('selected'));
  const map = {{'approved':'j-approve','needs_modification':'j-modify','rejected':'j-reject'}};
  const cls = map[judgment];
  if (cls) box.querySelector(`.${{cls}}`)?.classList.add('selected');

  const status = document.getElementById(`status-${{qid}}-${{ttype}}`);
  const labels = {{'approved':'승인','needs_modification':'수정 필요','rejected':'부적절'}};
  const colors = {{'approved':'var(--green)','needs_modification':'var(--amber)','rejected':'var(--red)'}};
  if (status) status.innerHTML = `<span style="color:${{colors[judgment]}}">${{labels[judgment]}}</span>`;
}}

function saveMeta(qid, ttype) {{
  if (!annotations[qid]) annotations[qid] = {{}};
  if (!annotations[qid][ttype]) annotations[qid][ttype] = {{}};
  const note = document.getElementById(`note-${{qid}}-${{ttype}}`);
  const unsolvable = document.getElementById(`unsolvable-${{qid}}-${{ttype}}`);
  const essential = document.getElementById(`essential-${{qid}}-${{ttype}}`);
  if (note) annotations[qid][ttype].note = note.value;
  if (unsolvable) annotations[qid][ttype].unsolvable = unsolvable.value;
  if (essential) annotations[qid][ttype].essential = essential.value;
  saveAnnotations();
}}

// ===== Edit =====
function toggleEdit(qid, ttype) {{
  const view = document.getElementById(`view-ctx-${{qid}}-${{ttype}}`);
  const edit = document.getElementById(`edit-ctx-${{qid}}-${{ttype}}`);
  const btn = event.target;
  if (!view || !edit) return;

  const isEditing = edit.style.display !== 'none';
  view.style.display = isEditing ? 'block' : 'none';
  edit.style.display = isEditing ? 'none' : 'block';
  btn.classList.toggle('editing', !isEditing);
  btn.textContent = isEditing ? '편집' : '편집 완료';

  if (isEditing) {{
    // Apply edit to view
    view.textContent = edit.value;
    saveEditedCtx(qid, ttype);
  }}
}}

function saveEditedCtx(qid, ttype) {{
  const edit = document.getElementById(`edit-ctx-${{qid}}-${{ttype}}`);
  if (!edit) return;
  if (!annotations[qid]) annotations[qid] = {{}};
  if (!annotations[qid][ttype]) annotations[qid][ttype] = {{}};
  annotations[qid][ttype].edited_context = edit.value;
  annotations[qid][ttype].assignee = currentAssignee;
  annotations[qid][ttype].timestamp = new Date().toISOString();
  saveAnnotations();
}}

// ===== Card Status =====
function updateCardStatus(qid) {{
  const badge = document.getElementById(`review-${{qid}}`);
  if (!badge || !annotations[qid]) return;

  const card = document.getElementById(`card-${{qid}}`);
  const totalSuccess = card?.querySelectorAll('.tab-success').length || 0;
  const judgments = Object.values(annotations[qid]).filter(d => d.judgment).length;

  if (judgments >= totalSuccess && totalSuccess > 0) {{
    badge.className = 'review-status review-complete';
    badge.textContent = `${{judgments}}/${{totalSuccess}}`;
  }} else if (judgments > 0) {{
    badge.className = 'review-status review-partial';
    badge.textContent = `${{judgments}}/${{totalSuccess}}`;
  }} else {{
    badge.className = 'review-status';
    badge.textContent = '';
  }}
}}

function updateCounts() {{
  let reviewed = 0, approved = 0, modify = 0, rejected = 0, edited = 0;
  for (const types of Object.values(annotations)) {{
    for (const data of Object.values(types)) {{
      if (data.judgment) {{
        reviewed++;
        if (data.judgment === 'approved') approved++;
        else if (data.judgment === 'needs_modification') modify++;
        else if (data.judgment === 'rejected') rejected++;
      }}
      if (data.edited_context) edited++;
    }}
  }}
  document.getElementById('reviewed-count').textContent = reviewed;
  document.getElementById('approved-count').textContent = approved;
  document.getElementById('modify-count').textContent = modify;
  document.getElementById('rejected-count').textContent = rejected;
  document.getElementById('edited-count').textContent = edited;

  const totalTabs = document.querySelectorAll('.problem-card:not(.hidden) .tab-success').length;
  document.getElementById('progress-text').textContent =
    `${{reviewed}}/${{totalTabs}} (${{Math.round(reviewed/Math.max(totalTabs,1)*100)}}%)`;
}}

// ===== Navigation =====
function toggleCard(qid) {{
  const body = document.getElementById(`body-${{qid}}`);
  const arrow = document.getElementById(`arrow-${{qid}}`);
  const visible = body.style.display !== 'none';
  body.style.display = visible ? 'none' : 'block';
  arrow.classList.toggle('open', !visible);
  if (!visible) {{
    const card = document.getElementById(`card-${{qid}}`);
    const first = card.querySelector('.tab-success');
    if (first) first.click();
    // Track current
    const cards = [...document.querySelectorAll('.problem-card:not(.hidden)')];
    currentCardIdx = cards.indexOf(card);
  }}
}}

function showTab(qid, ttype) {{
  const card = document.getElementById(`card-${{qid}}`);
  card.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById(`panel-${{qid}}-${{ttype}}`);
  const tab = document.getElementById(`tab-${{qid}}-${{ttype}}`);
  if (panel) panel.style.display = 'block';
  if (tab) tab.classList.add('active');
}}

function expandAll() {{
  document.querySelectorAll('.problem-card:not(.hidden)').forEach(card => {{
    const qid = card.dataset.qid;
    document.getElementById(`body-${{qid}}`).style.display = 'block';
    document.getElementById(`arrow-${{qid}}`).classList.add('open');
  }});
}}

function collapseAll() {{
  document.querySelectorAll('.problem-card:not(.hidden)').forEach(card => {{
    const qid = card.dataset.qid;
    document.getElementById(`body-${{qid}}`).style.display = 'none';
    document.getElementById(`arrow-${{qid}}`).classList.remove('open');
  }});
}}

function navigateCard(delta) {{
  const cards = [...document.querySelectorAll('.problem-card:not(.hidden)')];
  if (!cards.length) return;
  // Close current
  if (currentCardIdx >= 0 && currentCardIdx < cards.length) {{
    const prevQid = cards[currentCardIdx].dataset.qid;
    document.getElementById(`body-${{prevQid}}`).style.display = 'none';
    document.getElementById(`arrow-${{prevQid}}`).classList.remove('open');
  }}
  currentCardIdx = Math.max(0, Math.min(cards.length - 1, currentCardIdx + delta));
  const card = cards[currentCardIdx];
  const qid = card.dataset.qid;
  document.getElementById(`body-${{qid}}`).style.display = 'block';
  document.getElementById(`arrow-${{qid}}`).classList.add('open');
  card.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  const first = card.querySelector('.tab-success');
  if (first) first.click();
}}

// ===== Filter =====
function filterCards(filter, btn) {{
  document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  document.querySelectorAll('.problem-card').forEach(card => {{
    if (card.classList.contains('hidden')) return; // respect range filter
    const qid = card.dataset.qid;
    if (filter === 'all') {{ card.style.display = 'block'; return; }}

    const ann = annotations[qid] || {{}};
    const judgments = Object.values(ann).map(d => d.judgment).filter(Boolean);
    const hasEdits = Object.values(ann).some(d => d.edited_context);

    if (filter === 'pending') {{
      const successTabs = card.querySelectorAll('.tab-success').length;
      card.style.display = judgments.length < successTabs ? 'block' : 'none';
    }} else if (filter === 'edited') {{
      card.style.display = hasEdits ? 'block' : 'none';
    }} else {{
      card.style.display = judgments.includes(filter) ? 'block' : 'none';
    }}
  }});
}}

// ===== Export/Import =====
function exportJSON() {{
  const output = {{
    assignee: currentAssignee,
    range: [{range_start}, {range_end}],
    exported_at: new Date().toISOString(),
    total_judgments: Object.values(annotations).reduce(
      (sum, t) => sum + Object.values(t).filter(d => d.judgment).length, 0),
    annotations: annotations,
  }};
  const name = currentAssignee || 'review';
  const blob = new Blob([JSON.stringify(output, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `review_${{name}}_{range_start}_{range_end}_${{new Date().toISOString().slice(0,10)}}.json`;
  a.click();
}}

function handleImport(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    try {{
      const imported = JSON.parse(e.target.result);
      const data = imported.annotations || imported;
      let count = 0;
      for (const [qid, types] of Object.entries(data)) {{
        if (!annotations[qid]) annotations[qid] = {{}};
        for (const [ttype, tdata] of Object.entries(types)) {{
          // Merge: imported overwrites only if has judgment or edit
          if (tdata.judgment || tdata.edited_context) {{
            annotations[qid][ttype] = {{ ...annotations[qid][ttype], ...tdata }};
            count++;
          }}
        }}
      }}
      saveAnnotations();
      applyAnnotations();
      alert(`${{count}}개 annotation 가져오기 완료 (from: ${{imported.assignee || 'unknown'}})`);
    }} catch(err) {{
      alert('파싱 실패: ' + err.message);
    }}
  }};
  reader.readAsText(file);
  event.target.value = '';  // reset for re-import
}}

function exportMerged() {{
  // Collect all assignee data from localStorage
  const allKeys = [];
  for (let i = 0; i < localStorage.length; i++) {{
    const key = localStorage.key(i);
    if (key.startsWith(STORAGE_KEY)) allKeys.push(key);
  }}
  const merged = {{}};
  const sources = [];
  for (const key of allKeys) {{
    try {{
      const data = JSON.parse(localStorage.getItem(key));
      const assignee = key.replace(STORAGE_KEY + '_', '') || 'default';
      sources.push(assignee);
      for (const [qid, types] of Object.entries(data)) {{
        if (!merged[qid]) merged[qid] = {{}};
        for (const [ttype, tdata] of Object.entries(types)) {{
          if (!merged[qid][ttype]) merged[qid][ttype] = [];
          merged[qid][ttype].push({{ ...tdata, assignee: assignee }});
        }}
      }}
    }} catch(e) {{ continue; }}
  }}
  const output = {{
    merged_at: new Date().toISOString(),
    sources: sources,
    range: [{range_start}, {range_end}],
    data: merged,
  }};
  const blob = new Blob([JSON.stringify(output, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `review_merged_{range_start}_{range_end}_${{new Date().toISOString().slice(0,10)}}.json`;
  a.click();
}}

// ===== Keyboard Shortcuts =====
document.addEventListener('keydown', function(e) {{
  // Skip if typing in input/textarea
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

  const cards = [...document.querySelectorAll('.problem-card:not(.hidden)')];
  if (!cards.length) return;

  switch(e.key.toLowerCase()) {{
    case 'j': navigateCard(1); break;
    case 'k': navigateCard(-1); break;
    case '1': case '2': case '3': case '4': case '5': {{
      const tabIdx = parseInt(e.key) - 1;
      const types = ['EA-partial', 'EA-full', 'SA', 'IC', 'TA'];
      if (currentCardIdx >= 0 && currentCardIdx < cards.length) {{
        const qid = cards[currentCardIdx].dataset.qid;
        showTab(qid, types[tabIdx]);
      }}
      break;
    }}
    case 'a': {{
      if (currentCardIdx >= 0 && currentCardIdx < cards.length) {{
        const qid = cards[currentCardIdx].dataset.qid;
        const activeTab = cards[currentCardIdx].querySelector('.tab-btn.active');
        if (activeTab) {{
          const ttype = activeTab.id.replace(`tab-${{qid}}-`, '');
          setJudgment(qid, ttype, 'approved');
        }}
      }}
      break;
    }}
    case 'r': {{
      if (currentCardIdx >= 0 && currentCardIdx < cards.length) {{
        const qid = cards[currentCardIdx].dataset.qid;
        const activeTab = cards[currentCardIdx].querySelector('.tab-btn.active');
        if (activeTab) {{
          const ttype = activeTab.id.replace(`tab-${{qid}}-`, '');
          setJudgment(qid, ttype, 'rejected');
        }}
      }}
      break;
    }}
  }}
}});

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate Human Review HTML v2")
    parser.add_argument(
        "--input",
        type=str,
        default="batch_transformations_0_30.json",
        help="Input batch transformations JSON filename",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/metacognitive",
        help="Results directory",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Pre-merged annotations JSON to embed (e.g., annotations/merged.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML filename",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    input_path = results_dir / args.input

    if not input_path.exists():
        logger.error(f"입력 파일 없음: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"로드: {input_path}")
    logger.info(f"문제 수: {len(data['problems'])}")

    # Load pre-merged annotations if provided
    preloaded = None
    if args.annotations:
        ann_path = project_root / args.annotations
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
            # Support both formats: {annotations: {...}} and raw dict
            preloaded = ann_data.get("annotations", ann_data)
            count = sum(
                1
                for types in preloaded.values()
                for t in types.values()
                if isinstance(t, dict) and t.get("judgment")
            )
            logger.info(f"Annotations 로드: {ann_path} ({count}개 판정)")
        else:
            logger.warning(f"Annotations 파일 없음: {ann_path} (빈 상태로 생성)")

    html_content = generate_html(data, preloaded_annotations=preloaded)

    if args.output:
        output_name = args.output
    else:
        r = data["metadata"]["range"]
        output_name = f"human_review_{r[0]}_{r[1]}.html"

    output_path = results_dir / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML 생성 완료: {output_path}")
    logger.info("사용법:")
    logger.info(f"  기본: 브라우저에서 {output_name} 열기")
    logger.info(f"  작업자 분배: {output_name}?assignee=홍길동&start=0&end=10")
    if preloaded:
        logger.info("  이전 리뷰 결과가 임베딩되어 브라우저에서 자동 로드됩니다.")


if __name__ == "__main__":
    main()
