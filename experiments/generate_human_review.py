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

TYPE_KEYS = [
    "EA-partial",
    "EA-full",
    "SA",
    "IC-L1",
    "IC-L2",
    "IC-L3",
    "IC-L4",
    "TA",
]
TYPE_DESCRIPTIONS = {
    "EA-partial": "명시적 부재 (부분) — N/A 마커로 특정 값 제거",
    "EA-full": "명시적 부재 (전체) — 키/컬럼 전체 삭제",
    "SA": "무표지 부재 — 마커 없이 조용히 삭제",
    "IC-L1": "IC 타이포 — 10x 자릿수 오류 (가장 쉬움)",
    "IC-L2": "IC 단위 불일치 — million/billion, %/bps 혼동",
    "IC-L3": "IC 권위 충돌 — 1.5x 모순값 + 권위적 출처",
    "IC-L4": "IC 기간 합산 — 분기합 ≠ 연간총계",
    "TA": "시간적 모호성 — 연도를 모호한 표현으로 대체",
}
TYPE_COLORS = {
    "EA-partial": "#3b82f6",
    "EA-full": "#8b5cf6",
    "SA": "#f59e0b",
    "IC-L1": "#22c55e",
    "IC-L2": "#3b82f6",
    "IC-L3": "#ef4444",
    "IC-L4": "#f59e0b",
    "TA": "#6b7280",
}
# Legacy IC key mapping for backward compatibility with old batch_transformations
LEGACY_IC_TO_L3 = {"IC": "IC-L3"}


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


def _render_enrich_block(
    qid: str,
    ttype: str,
    resp_id: str,
    enrich_reasons: Dict = None,
    enrich_mems: Dict = None,
) -> str:
    """Render the A1 (reason category) + F1 (Memorization) overlay.

    Only attaches to the "metacognitive" response card, which is the source
    of the refusal reason text and the basis of Memorization Score computation.
    Returns empty string when no enrich data exists for this (qid, ttype) pair.
    """
    if resp_id != "metacognitive":
        return ""
    reason_meta = (enrich_reasons or {}).get(qid, {}).get(ttype)
    mem_meta = (enrich_mems or {}).get(qid, {}).get(ttype)
    if not reason_meta and not mem_meta:
        return ""

    parts: list[str] = []
    if reason_meta:
        cat = reason_meta.get("category", "?")
        reason = _esc(reason_meta.get("reason", ""))
        conf = _esc(reason_meta.get("confidence", ""))
        parts.append(
            f'<div class="enrich-row">'
            f'<span class="enrich-label">📋 Reason</span>'
            f'<span class="enrich-reason">{reason}</span>'
            f"</div>"
            f'<div class="enrich-row">'
            f'<span class="enrich-label">🏷 Category</span>'
            f'<span class="enrich-cat enrich-cat-{cat}">{cat}</span>'
            f'<span class="enrich-conf">({conf})</span>'
            f"</div>"
        )
    if mem_meta:
        rate = float(mem_meta.get("memorization_rate", 0.0) or 0.0)
        counts = mem_meta.get("counts", {}) or {}
        total = int(mem_meta.get("total_values", 0) or 0)
        from_removed = int(counts.get("from_removed_data", 0) or 0)
        rate_color = "#ef4444" if rate > 0 else "#9ca3af"
        parts.append(
            f'<div class="enrich-row">'
            f'<span class="enrich-label">🧠 Memorization</span>'
            f'<span class="enrich-value" style="color:{rate_color}">{rate:.3f}</span>'
            f'<span class="enrich-conf">'
            f"from_removed={from_removed} / total={total}</span>"
            f"</div>"
        )
    return '<div class="enrich-block">' + "".join(parts) + "</div>"


def _render_problem_card(
    problem: Dict,
    idx: int,
    enrich_reasons: Dict = None,
    enrich_mems: Dict = None,
) -> str:
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

            # Reverse calculation warning for EA-full
            reverse_warning = ""
            if tdata.get("reverse_calculable"):
                reverse_warning = (
                    f'<div style="background:#422006;border:1px solid #f59e0b;border-radius:6px;'
                    f'padding:8px 12px;margin-bottom:8px;font-size:12px;">'
                    f'<strong style="color:#f59e0b;">&#9888; 역산 가능 경고</strong><br>'
                    f"{_esc(tdata.get('reverse_explanation', ''))}<br>"
                    f'<code style="color:#fbbf24;">{_esc(tdata.get("reverse_formula", ""))}</code>'
                    f"</div>"
                )

            panel_content = (
                f"{reverse_warning}"
                f'<div class="transform-desc">'
                f'<span class="type-info">{_esc(type_desc)}</span>'
                f"<br>{_esc(desc)}</div>"
            )

            if has_q:
                panel_content += (
                    f'<div class="q-compare">'
                    f'<div class="q-col">'
                    f'<div class="label">원본 Question</div>'
                    f'<div class="q-text">{_esc(question)}</div>'
                    f"</div>"
                    f'<div class="q-col">'
                    f'<div class="q-text q-changed">{_esc(transformed_q)}</div>'
                    f'<div class="label">변환 Question</div>'
                    f"</div>"
                    f"</div>"
                )

            if has_ctx:
                # Ensure transformed_ctx is a string
                if not isinstance(transformed_ctx, str):
                    transformed_ctx = str(transformed_ctx)
                # Escaped versions for display
                orig_esc = _esc(context)
                trans_esc = _esc(transformed_ctx)

                panel_content += (
                    f'<div class="diff-container">'
                    f'<div class="diff-panel">'
                    f'<div class="diff-label">원본 Context</div>'
                    f'<pre class="diff-content" id="orig-ctx-{qid}-{ttype}">{orig_esc}</pre></div>'
                    f'<div class="diff-panel diff-transformed">'
                    f'<div class="diff-label">변환 후'
                    f"<button class=\"edit-btn\" onclick=\"toggleEdit('{qid}', '{ttype}', this)\">편집</button>"
                    f"</div>"
                    f'<pre class="diff-content" id="view-ctx-{qid}-{ttype}">{trans_esc}</pre>'
                    f'<textarea class="edit-area" id="edit-ctx-{qid}-{ttype}" '
                    f'style="display:none;" '
                    f"onchange=\"saveEditedCtx('{qid}', '{ttype}')\">"
                    f"{_esc(transformed_ctx)}</textarea>"
                    f"</div></div>"
                )

            # Model responses (3 types: original, transformed+standard, transformed+metacognitive)
            gt_val = problem.get("ground_truth", "")
            case_labels = {1: "거부 (C1)", 2: "오답 (C2)", 3: "정답 (C3)"}
            case_colors = {1: "#22c55e", 2: "#f59e0b", 3: "#ef4444"}

            response_configs = [
                (
                    "original",
                    "원본 Context",
                    tdata.get("original_response", ""),
                    tdata.get("original_predicted", ""),
                    tdata.get("original_is_correct", False),
                    tdata.get("original_case_type", 0),
                    tdata.get("original_response_type", ""),
                    tdata.get("original_execution_error", ""),
                    "#3b82f6",
                ),
                (
                    "transformed",
                    "변환 Context (Standard)",
                    tdata.get("model_response", ""),
                    tdata.get("predicted_answer", ""),
                    tdata.get("is_correct", False),
                    tdata.get("case_type", 0),
                    tdata.get("response_type", ""),
                    tdata.get("execution_error", ""),
                    "#f59e0b",
                ),
                (
                    "metacognitive",
                    "변환 Context (Metacognitive)",
                    tdata.get("metacognitive_response", ""),
                    tdata.get("metacognitive_predicted", ""),
                    tdata.get("metacognitive_is_correct", False),
                    tdata.get("metacognitive_case_type", 0),
                    tdata.get("metacognitive_response_type", ""),
                    tdata.get("metacognitive_execution_error", ""),
                    "#a78bfa",
                ),
            ]

            has_any_response = any(cfg[2] for cfg in response_configs)
            if has_any_response:
                panel_content += '<div class="responses-grid">'

                for (
                    resp_id,
                    label,
                    raw_resp,
                    predicted,
                    is_correct,
                    case_type,
                    resp_type,
                    exec_err,
                    accent,
                ) in response_configs:
                    if not raw_resp:
                        panel_content += (
                            f'<div class="response-card response-empty" style="border-color:{accent}30">'
                            f'<div class="response-card-header" style="color:{accent}">{label}</div>'
                            f'<div class="response-card-body">실험 대기</div>'
                            f"</div>"
                        )
                        continue

                    case_label = case_labels.get(case_type, "?")
                    case_color = case_colors.get(case_type, "#666")
                    correct_icon = "O" if is_correct else "X"
                    correct_color = "#22c55e" if is_correct else "#ef4444"

                    error_html = ""
                    if exec_err:
                        error_html = f'<div class="result-error">에러: {_esc(str(exec_err)[:80])}</div>'

                    enrich_html = _render_enrich_block(
                        qid, ttype, resp_id, enrich_reasons, enrich_mems
                    )
                    panel_content += (
                        f'<div class="response-card" style="border-color:{accent}60">'
                        f'<div class="response-card-header" style="color:{accent}">{label}</div>'
                        f'<div class="response-card-result">'
                        f'<span class="case-badge" style="background:{case_color}20;color:{case_color};'
                        f'border:1px solid {case_color};padding:2px 6px;border-radius:4px;font-size:10px;">'
                        f"{case_label}</span>"
                        f'<span class="result-value" style="color:{correct_color};font-size:13px;">'
                        f"{_esc(str(predicted))} ({correct_icon})</span>"
                        f'<span class="result-gt" style="font-size:11px;">정답: {_esc(str(gt_val))}</span>'
                        f"</div>"
                        f"{error_html}"
                        f"{enrich_html}"
                        f'<details class="model-response-details">'
                        f"<summary>응답 코드</summary>"
                        f'<pre class="model-response-content">{_esc(raw_resp[:1200])}</pre>'
                        f"</details>"
                        f"</div>"
                    )

                panel_content += "</div>"

            # Judgment UI — Two-Stage Annotation
            panel_content += f"""
            <div class="judgment-box" id="judgment-{qid}-{ttype}">
              <div class="stage-box">
                <div class="stage-label">1단계: 변환 품질</div>
                <div class="j-buttons">
                  <button class="j-btn j-approve" onclick="setStage1('{qid}', '{ttype}', 'approved')">승인</button>
                  <button class="j-btn j-modify" onclick="setStage1('{qid}', '{ttype}', 'needs_modification')">수정필요</button>
                  <button class="j-btn j-reject" onclick="setStage1('{qid}', '{ttype}', 'rejected')">부적절</button>
                </div>
              </div>
              <div class="stage-box stage2" id="stage2-{qid}-{ttype}" style="display:none">
                <div class="stage-label">2단계: Solvability</div>
                <div class="j-buttons">
                  <button class="j-btn j-unsolvable" onclick="setStage2('{qid}', '{ttype}', 'unsolvable')">Unsolvable(통과)</button>
                  <button class="j-btn j-solvable" onclick="setStage2('{qid}', '{ttype}', 'solvable')">Solvable(답변가능)</button>
                  <button class="j-btn j-review" onclick="setStage2('{qid}', '{ttype}', 'review_needed')">확인필요</button>
                </div>
                <input type="text" class="solvable-reason" id="solvable-reason-{qid}-{ttype}"
                       placeholder="Solvable 사유 입력..."
                       style="display:none"
                       onchange="saveMeta('{qid}', '{ttype}')">
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
    # Detect question-only problem
    has_question_transform = any(
        transformations.get(k, {}).get("is_question_transform") for k in TYPE_KEYS
    )
    qonly_badge = (
        '<span class="badge badge-qonly">Q변환</span>' if has_question_transform else ""
    )

    return f"""
    <div class="problem-card{"  q-only" if has_question_transform else ""}" id="card-{qid}" data-qid="{qid}" data-index="{idx}">
      <div class="card-header" onclick="toggleCard('{qid}')">
        <div class="card-title">
          <span class="card-index">#{idx}</span>
          <span class="card-qid">{qid}</span>
          <span class="badge badge-format">{ctx_format}</span>
          {hardcoded_badge}
          {qonly_badge}
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
        <div class="gt-display">정답 (Ground Truth): <strong>{_esc(str(ground_truth))}</strong></div>
        <div class="meta-row">
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


def generate_html(
    data: Dict,
    preloaded_annotations: Dict = None,
    enrich_reasons: Dict = None,
    enrich_mems: Dict = None,
) -> str:
    """Generate the full HTML review page.

    Args:
        data: Batch transformation result JSON.
        preloaded_annotations: Optional merged annotations to embed in HTML.
            Format: {qid: {ttype: {judgment, note, ...}}}
        enrich_reasons: Optional {qid: {ttype: {reason, category, confidence}}}
            overlay for the metacognitive response card.
        enrich_mems: Optional {qid: {ttype: {memorization_rate, counts, total_values}}}.
    """
    metadata = data["metadata"]
    coverage = data["coverage_summary"]
    problems = data["problems"]

    total = coverage["total_problems"]
    transformable = coverage["transformable"]

    # Problem cards
    cards_html = ""
    for problem in problems:
        cards_html += _render_problem_card(
            problem, problem["index"], enrich_reasons, enrich_mems
        )

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
.badge-qonly {{ background: #1e1e3d; color: var(--purple); }}
.badge-count {{ background: var(--surface2); color: var(--text2); }}
.problem-card.q-only {{ border-left: 3px solid var(--purple); }}

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

/* Model Responses — 3-column grid */
.responses-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 10px; }}
.response-card {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px; border-top: 3px solid; }}
.response-card-header {{ font-size: 11px; font-weight: 700; margin-bottom: 6px; }}
.response-card-result {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }}
.response-card-body {{ font-size: 12px; color: var(--text2); padding: 8px; text-align: center; }}
.response-empty {{ opacity: 0.4; }}
.model-response-content {{ font-size: 11px; white-space: pre-wrap; word-break: break-word; max-height: 250px; overflow-y: auto; color: #c9d1d9; line-height: 1.4; }}
.result-value {{ font-weight: 700; font-family: monospace; }}
.result-gt {{ font-weight: 600; color: var(--blue); font-family: monospace; font-size: 11px; }}
.result-error {{ color: #f97583; font-size: 10px; margin-top: 4px; }}
.model-response-details {{ margin-top: 6px; }}
.model-response-details summary {{ font-size: 10px; color: var(--text2); cursor: pointer; }}
@media (max-width: 1200px) {{ .responses-grid {{ grid-template-columns: 1fr; }} }}

/* Judgment — Two-Stage */
.judgment-box {{ margin-top: 10px; padding: 10px; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); }}
.stage-box {{ margin-bottom: 8px; }}
.stage-label {{ font-size: 11px; font-weight: 600; color: var(--text2); margin-bottom: 4px; }}
.j-buttons {{ display: flex; gap: 4px; }}
.j-btn {{ padding: 5px 14px; border-radius: 6px; border: 2px solid transparent; cursor: pointer; font-size: 12px; font-weight: 500; }}
.j-approve {{ background: #052e16; color: var(--green); border-color: #14532d; }}
.j-approve:hover, .j-approve.selected {{ background: var(--green); color: white; }}
.j-modify {{ background: #422006; color: var(--amber); border-color: #713f12; }}
.j-modify:hover, .j-modify.selected {{ background: var(--amber); color: black; }}
.j-reject {{ background: #1a0505; color: var(--red); border-color: #7f1d1d; }}
.j-reject:hover, .j-reject.selected {{ background: var(--red); color: white; }}
.j-unsolvable {{ background: #052e16; color: var(--green); border-color: #14532d; }}
.j-unsolvable:hover, .j-unsolvable.selected {{ background: var(--green); color: white; }}
.j-solvable {{ background: #422006; color: var(--amber); border-color: #713f12; }}
.j-solvable:hover, .j-solvable.selected {{ background: var(--amber); color: black; }}
.j-review {{ background: #1e1e3d; color: var(--purple); border-color: #3b3070; }}
.j-review:hover, .j-review.selected {{ background: var(--purple); color: white; }}
.stage2 {{ border-top: 1px solid var(--border); padding-top: 8px; }}
.solvable-reason {{ width: 100%; margin-top: 6px; background: var(--bg); border: 1px solid var(--amber); border-radius: 6px; color: var(--text); padding: 6px; font-size: 12px; }}
.note-area {{ width: 100%; margin-top: 6px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 6px; font-size: 12px; resize: vertical; }}
.judgment-status {{ margin-top: 4px; font-size: 11px; }}

/* Ground Truth display */
.gt-display {{ background: #1a2332; border: 1px solid var(--blue); border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; font-size: 13px; color: var(--blue); }}
.gt-display strong {{ color: #60a5fa; font-size: 15px; }}

/* Question comparison */
.q-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; margin-bottom: 6px; }}
.q-col {{ background: var(--surface2); border-radius: 6px; padding: 10px; }}
.q-col .label {{ font-size: 11px; font-weight: 600; color: var(--text2); margin-bottom: 4px; }}
.q-text {{ font-size: 13px; white-space: pre-wrap; word-break: break-word; }}
.q-changed {{ color: var(--amber); border-left: 3px solid var(--amber); padding-left: 8px; }}

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
.prompt-comparison {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }}
.prompt-card {{ background: var(--surface2); border-radius: 8px; padding: 12px; border-top: 3px solid; }}
.prompt-card h4 {{ font-size: 12px; margin-bottom: 6px; }}
.prompt-desc {{ font-size: 11px; color: var(--text2); margin-bottom: 8px; line-height: 1.5; }}
.prompt-details summary {{ font-size: 11px; color: var(--text2); cursor: pointer; }}
.prompt-pre {{ font-size: 10px; background: var(--bg); padding: 8px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; color: #c9d1d9; margin-top: 4px; line-height: 1.4; }}
@media (max-width: 1200px) {{ .prompt-comparison {{ grid-template-columns: 1fr; }} }}
.guide-section {{ margin-top: 14px; }}
.guide-section h3 {{ font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text2); }}
.guide-section table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.guide-section th {{ text-align: left; padding: 6px 8px; background: var(--bg); color: var(--text2); }}
.guide-section td {{ padding: 6px 8px; border-top: 1px solid var(--border); }}

@media (max-width: 768px) {{
  .diff-container {{ grid-template-columns: 1fr; }}
  .q-compare {{ grid-template-columns: 1fr; }}
  .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
  .guide-grid {{ grid-template-columns: 1fr; }}
}}

/* Enrich overlay — A1 (reason category) + F1 (Memorization) */
.enrich-block {{ margin-top: 10px; padding: 10px 12px; background: var(--bg); border-radius: 6px; border: 1px dashed var(--border); font-size: 12px; }}
.enrich-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; flex-wrap: wrap; }}
.enrich-label {{ color: var(--text2); font-size: 11px; min-width: 110px; font-weight: 600; flex-shrink: 0; }}
.enrich-value {{ color: var(--text); font-family: monospace; font-weight: 600; }}
.enrich-reason {{ color: var(--text); font-size: 12px; line-height: 1.5; flex: 1; min-width: 0; }}
.enrich-conf {{ color: var(--text2); font-size: 10px; font-style: italic; margin-left: auto; }}
.enrich-cat {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; color: white; font-weight: 600; }}
.enrich-cat-MISSING_REQUIRED_VALUE {{ background: #3b82f6; }}
.enrich-cat-MISSING_SILENT {{ background: #06b6d4; }}
.enrich-cat-CONFLICTING_VALUES {{ background: #ef4444; }}
.enrich-cat-UNIT_AMBIGUITY {{ background: #8b5cf6; }}
.enrich-cat-TEMPORAL_MISMATCH {{ background: #a855f7; }}
.enrich-cat-UNDERSPECIFIED {{ background: #6b7280; }}
.enrich-cat-UNCATEGORIZABLE {{ background: #111827; border: 1px solid #ef4444; }}
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
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 아래 Python Solution의 <span style="color:#22c55e;">● 초록색</span> 변수에 해당하는 칼럼이 <strong>context에서 정말 삭제됐는지</strong> 확인하세요. 초록 변수와 일치하는 칼럼이 아직 남아있다면 모델이 값을 읽을 수 있으므로 변환이 불충분합니다.</p>
          <p style="font-size:11px;color:var(--text2);margin-top:4px;">&#9888; 역산 가능 경고가 표시된 경우: 삭제된 칼럼 값이 남은 칼럼들로 계산될 수 있다는 자동 탐지 결과입니다. 경고가 있으면 더 신중히 확인해주세요.</p>
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

        <div class="guide-card" style="border-color:#22c55e;">
          <h3 style="color:#22c55e;">IC-L1 (10x 타이포)</h3>
          <p><strong>방법:</strong> 숫자를 10배 틀리게 교체 (자릿수 오류)</p>
          <p><strong>목적:</strong> 가장 명백한 수치 오류를 LLM이 잡는지 (IC 기본선)</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#052e16;color:#22c55e;">LOW</span> 10배 차이는 명백함</p>
          <div class="example">원본: Revenue 2024 = $10M
변환: Revenue 2024 = $100M  (10배 오류)</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 10배 오류가 문맥상 눈에 띄는지, 풀이에 사용되는 값인지</p>
        </div>

        <div class="guide-card" style="border-color:#3b82f6;">
          <h3 style="color:#3b82f6;">IC-L2 (단위 불일치)</h3>
          <p><strong>방법:</strong> 같은 수치를 다른 단위(million/billion, %/bps)로 삽입하되, 1.5x 오류 포함</p>
          <p><strong>목적:</strong> 단위 변환을 교차 검증하는 능력 테스트</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#422006;color:#f59e0b;">MODERATE</span> 단위 이해 필요</p>
          <div class="example">원본: "Total revenue was $500 million"
변환: + "According to a separate filing, this figure was reported as 0.75 billion."
 (실제론 0.5B여야 하는데 0.75B = 1.5x 오류)</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 단위 변환이 실제로 틀린지 (올바른 변환이면 충돌이 아님!), 삽입이 자연스러운지</p>
        </div>

        <div class="guide-card" style="border-color:#ef4444;">
          <h3 style="color:#ef4444;">IC-L3 (권위 충돌)</h3>
          <p><strong>방법:</strong> "감사보고서" 등 권위 있는 출처를 언급하며 원래 값의 1.5배인 모순 데이터 삽입</p>
          <p><strong>목적:</strong> 권위적 출처에서 오는 모순을 LLM이 탐지하는지 (핵심 IC 테스트)</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#7f1d1d;color:#ef4444;">HIGH</span> 모델이 권위 출처를 무조건 신뢰하는 경향</p>
          <div class="example">원본: Revenue 2024 = $100M
변환: Revenue 2024 = $150M
      "Note: discrepancy — one source reports $150M, while another shows $100M."</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 모순이 자연스럽게 삽입되었는지, 모순 대상이 풀이에 사용되는 값인지</p>
        </div>

        <div class="guide-card" style="border-color:#f59e0b;">
          <h3 style="color:#f59e0b;">IC-L4 (기간 합산 불일치)</h3>
          <p><strong>방법:</strong> 분기별 값의 합계와 다른 연간 총계를 삽입 (또는 한 분기 값을 수정)</p>
          <p><strong>목적:</strong> 시계열 데이터의 내부 일관성 검증 능력 테스트</p>
          <p><strong>탐지 난이도:</strong> <span class="guide-tag" style="background:#7f1d1d;color:#ef4444;">HIGH</span> 산술 검증 필요</p>
          <div class="example">원본: Q1=$10M, Q2=$12M, Q3=$11M, Q4=$15M  (합계=$48M)
변환: + "The total for the reporting period was $40.80M"  (실제 합계와 불일치)</div>
          <p style="margin-top:6px;"><strong>리뷰 체크:</strong> 합산 불일치가 실제로 존재하는지, 사용된 숫자들이 같은 시리즈인지</p>
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
        <h3>모델 응답 읽는 법</h3>
        <p style="font-size:12px;color:var(--text2);margin-bottom:8px;">
          각 변환 탭에 gpt-4o-mini가 해당 변환된 문제를 풀려고 시도한 응답이 표시됩니다.
          FinanceReasoning 원논문 방식(POT: Program of Thought)으로 모델이 Python 코드를 생성하고,
          이 코드를 실제로 실행한 결과를 정답과 비교합니다.
        </p>
        <table>
          <tr><th>배지</th><th>의미</th><th>리뷰 시사점</th></tr>
          <tr>
            <td style="color:#22c55e;">거부 (C1)</td>
            <td>모델이 정보 부족/충돌을 인식하고 INSUFFICIENT_INFORMATION 반환</td>
            <td>변환이 유효할 가능성 높음 (모델이 unsolvable로 판단)</td>
          </tr>
          <tr>
            <td style="color:#f59e0b;">오답 (C2)</td>
            <td>모델이 답을 시도했지만 틀림</td>
            <td>변환이 혼란을 줬지만 모델이 명시적으로 거부하지는 않음</td>
          </tr>
          <tr>
            <td style="color:#ef4444;">정답 (C3)</td>
            <td>변환에도 불구하고 모델이 정답을 맞춤</td>
            <td>변환이 불충분할 가능성 높음 — 집중 확인 필요</td>
          </tr>
        </table>
        <p style="font-size:11px;color:var(--text2);margin-top:6px;">
          C3(정답)인데 변환이 올바르다면: 모델이 삭제된 데이터를 암기(memorization)했거나
          남은 데이터에서 역산한 것입니다. 이 경우는 리뷰에서 메모를 남겨주세요.
        </p>
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

      <div class="guide-section">
        <h3>프롬프트 전략 비교 (3개 응답)</h3>
        <p style="font-size:12px;color:var(--text2);margin-bottom:10px;">
          각 변환 탭에 3개 LLM 응답이 나란히 표시됩니다. 모델(gpt-4o-mini)에 동일한 POT(Program of Thought) 방식으로 코드 생성을 요청하되,
          시스템 프롬프트만 달리하여 모델 행동의 차이를 관찰합니다.
        </p>
        <div class="prompt-comparison">
          <div class="prompt-card" style="border-color:#3b82f6;">
            <h4 style="color:#3b82f6;">1. 원본 Context (Baseline)</h4>
            <p class="prompt-desc">원본 question + 원본 context를 Standard 프롬프트로 평가. 모델이 완전한 정보로 정답을 맞추는지 확인 (기준선).</p>
            <details class="prompt-details">
              <summary>System Prompt 전문 보기</summary>
              <pre class="prompt-pre">You are a financial expert, you are supposed to generate a Python program to answer the given question. The returned value of the program is supposed to be the answer. Here is an example of the Python program:
```python
def solution():
    # Define variables name and value
    revenue = 600000
    avg_account_receivable = 50000

    # Do math calculation to get the answer
    receivables_turnover = revenue / avg_account_receivable
    answer = 365 / receivables_turnover

    # return answer
    return answer
```</pre>
            </details>
          </div>

          <div class="prompt-card" style="border-color:#f59e0b;">
            <h4 style="color:#f59e0b;">2. 변환 Context + Standard</h4>
            <p class="prompt-desc">변환된 question/context를 Standard 프롬프트로 평가. 거부 옵션 없이 모델이 어떻게 반응하는지 (강제 풀이).</p>
            <details class="prompt-details">
              <summary>System Prompt 전문 보기</summary>
              <pre class="prompt-pre">(위와 동일한 Standard 프롬프트 — 거부 옵션 없음)</pre>
            </details>
          </div>

          <div class="prompt-card" style="border-color:#a78bfa;">
            <h4 style="color:#a78bfa;">3. 변환 Context + Metacognitive</h4>
            <p class="prompt-desc">변환된 question/context + 거부 옵션이 있는 프롬프트. 모델이 정보 부족/충돌을 인식하고 거부할 수 있는지 (메타인지 핵심 테스트).</p>
            <details class="prompt-details">
              <summary>System Prompt 전문 보기</summary>
              <pre class="prompt-pre">You are a financial expert, you are supposed to generate a Python program to answer the given question.

IMPORTANT: Before writing code, verify that ALL required data is present in the context.
If any critical information is missing, contradictory, or ambiguous, generate:
```python
def solution():
    # INSUFFICIENT_INFORMATION: [explain what data is missing]
    return "INSUFFICIENT_INFORMATION"
```
Do NOT guess, assume, or fabricate missing values.

If the data is sufficient, generate the solution program normally.</pre>
            </details>
          </div>
        </div>
        <p style="font-size:11px;color:var(--text2);margin-top:8px;">
          <strong>공통 User Prompt:</strong> "The following question context is provided for your reference.
          [context] Question: [question] Please generate a Python program... def solution(): ..."
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
  <kbd>1</kbd>~<kbd>8</kbd> 탭 전환 &nbsp;
  <kbd>A</kbd> 승인 &nbsp;
  <kbd>R</kbd> 부적절 &nbsp;
  <kbd>U</kbd> Unsolvable &nbsp;
  <kbd>S</kbd> Solvable
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
      if (data.judgment) {{
        highlightStage1(qid, ttype, data.judgment);
        if (data.judgment === 'approved') {{
          const s2 = document.getElementById(`stage2-${{qid}}-${{ttype}}`);
          if (s2) s2.style.display = 'block';
        }}
      }}
      if (data.solvability) {{
        highlightStage2(qid, ttype, data.solvability);
        if (data.solvability === 'solvable') {{
          const reason = document.getElementById(`solvable-reason-${{qid}}-${{ttype}}`);
          if (reason) {{ reason.style.display = 'block'; reason.value = data.solvable_reason || ''; }}
        }}
      }}
      if (data.note) {{
        const el = document.getElementById(`note-${{qid}}-${{ttype}}`);
        if (el) el.value = data.note;
      }}
      if (data.edited_context) {{
        const el = document.getElementById(`edit-ctx-${{qid}}-${{ttype}}`);
        if (el) el.value = data.edited_context;
      }}
    }}
    updateCardStatus(qid);
  }}
}}

// ===== Two-Stage Judgment =====
function setStage1(qid, ttype, judgment) {{
  if (!annotations[qid]) annotations[qid] = {{}};
  if (!annotations[qid][ttype]) annotations[qid][ttype] = {{}};
  annotations[qid][ttype].judgment = judgment;
  annotations[qid][ttype].assignee = currentAssignee;
  annotations[qid][ttype].timestamp = new Date().toISOString();
  // Clear stage 2 when stage 1 changes
  delete annotations[qid][ttype].solvability;
  delete annotations[qid][ttype].solvable_reason;
  highlightStage1(qid, ttype, judgment);
  // Show/hide stage 2
  const s2 = document.getElementById(`stage2-${{qid}}-${{ttype}}`);
  if (s2) {{
    s2.style.display = judgment === 'approved' ? 'block' : 'none';
    // Reset stage 2 highlight
    s2.querySelectorAll('.j-btn').forEach(b => b.classList.remove('selected'));
    const reason = document.getElementById(`solvable-reason-${{qid}}-${{ttype}}`);
    if (reason) {{ reason.style.display = 'none'; reason.value = ''; }}
  }}
  saveAnnotations();
  updateCardStatus(qid);
}}

function setStage2(qid, ttype, solvability) {{
  // Guard: Stage 1 must be approved before Stage 2
  const ann = annotations[qid]?.[ttype];
  if (!ann || ann.judgment !== 'approved') return;
  if (!annotations[qid]) annotations[qid] = {{}};
  if (!annotations[qid][ttype]) annotations[qid][ttype] = {{}};
  annotations[qid][ttype].solvability = solvability;
  annotations[qid][ttype].timestamp = new Date().toISOString();
  highlightStage2(qid, ttype, solvability);
  // Show reason input only for solvable
  const reason = document.getElementById(`solvable-reason-${{qid}}-${{ttype}}`);
  if (reason) {{
    reason.style.display = solvability === 'solvable' ? 'block' : 'none';
    if (solvability !== 'solvable') reason.value = '';
  }}
  saveAnnotations();
  updateCardStatus(qid);
}}

function highlightStage1(qid, ttype, judgment) {{
  const box = document.getElementById(`judgment-${{qid}}-${{ttype}}`);
  if (!box) return;
  const stage1 = box.querySelector('.stage-box:first-child');
  if (!stage1) return;
  stage1.querySelectorAll('.j-btn').forEach(b => b.classList.remove('selected'));
  const map = {{'approved':'j-approve','needs_modification':'j-modify','rejected':'j-reject'}};
  const cls = map[judgment];
  if (cls) stage1.querySelector(`.${{cls}}`)?.classList.add('selected');

  const status = document.getElementById(`status-${{qid}}-${{ttype}}`);
  const labels = {{'approved':'승인','needs_modification':'수정 필요','rejected':'부적절'}};
  const colors = {{'approved':'var(--green)','needs_modification':'var(--amber)','rejected':'var(--red)'}};
  let statusText = `<span style="color:${{colors[judgment]}}">${{labels[judgment]}}</span>`;
  const ann = annotations[qid]?.[ttype];
  if (ann && ann.solvability) {{
    const s2labels = {{'unsolvable':'Unsolvable','solvable':'Solvable','review_needed':'확인필요'}};
    const s2colors = {{'unsolvable':'var(--green)','solvable':'var(--amber)','review_needed':'var(--purple)'}};
    statusText += ` → <span style="color:${{s2colors[ann.solvability]}}">${{s2labels[ann.solvability]}}</span>`;
  }}
  if (status) status.innerHTML = statusText;
}}

function highlightStage2(qid, ttype, solvability) {{
  const s2 = document.getElementById(`stage2-${{qid}}-${{ttype}}`);
  if (!s2) return;
  s2.querySelectorAll('.j-btn').forEach(b => b.classList.remove('selected'));
  const map = {{'unsolvable':'j-unsolvable','solvable':'j-solvable','review_needed':'j-review'}};
  const cls = map[solvability];
  if (cls) s2.querySelector(`.${{cls}}`)?.classList.add('selected');
  // Update combined status display
  const ann = annotations[qid]?.[ttype];
  if (ann && ann.judgment) highlightStage1(qid, ttype, ann.judgment);
}}

function saveMeta(qid, ttype) {{
  if (!annotations[qid]) annotations[qid] = {{}};
  if (!annotations[qid][ttype]) annotations[qid][ttype] = {{}};
  const note = document.getElementById(`note-${{qid}}-${{ttype}}`);
  const solvableReason = document.getElementById(`solvable-reason-${{qid}}-${{ttype}}`);
  if (note) annotations[qid][ttype].note = note.value;
  if (solvableReason && solvableReason.style.display !== 'none') annotations[qid][ttype].solvable_reason = solvableReason.value;
  saveAnnotations();
}}

// ===== Edit =====
function toggleEdit(qid, ttype, btn) {{
  const view = document.getElementById(`view-ctx-${{qid}}-${{ttype}}`);
  const edit = document.getElementById(`edit-ctx-${{qid}}-${{ttype}}`);
  if (!btn) btn = event.target;
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
    case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': {{
      const tabIdx = parseInt(e.key) - 1;
      const types = ['EA-partial', 'EA-full', 'SA', 'IC-L1', 'IC-L2', 'IC-L3', 'IC-L4', 'TA'];
      if (tabIdx < types.length && currentCardIdx >= 0 && currentCardIdx < cards.length) {{
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
          setStage1(qid, ttype, 'approved');
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
          setStage1(qid, ttype, 'rejected');
        }}
      }}
      break;
    }}
    case 'u': {{
      if (currentCardIdx >= 0 && currentCardIdx < cards.length) {{
        const qid = cards[currentCardIdx].dataset.qid;
        const activeTab = cards[currentCardIdx].querySelector('.tab-btn.active');
        if (activeTab) {{
          const ttype = activeTab.id.replace(`tab-${{qid}}-`, '');
          setStage2(qid, ttype, 'unsolvable');
        }}
      }}
      break;
    }}
    case 's': {{
      if (currentCardIdx >= 0 && currentCardIdx < cards.length) {{
        const qid = cards[currentCardIdx].dataset.qid;
        const activeTab = cards[currentCardIdx].querySelector('.tab-btn.active');
        if (activeTab) {{
          const ttype = activeTab.id.replace(`tab-${{qid}}-`, '');
          setStage2(qid, ttype, 'solvable');
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
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Evaluation results JSON (adds model responses to review)",
    )
    parser.add_argument(
        "--enrich-reasons",
        type=str,
        default=None,
        help="JSON file with A1 reason categories (output of auto_tag_reason_categories.py)",
    )
    parser.add_argument(
        "--enrich-memorization",
        type=str,
        default=None,
        help="JSON file with F1 Memorization Score per (qid, transformation_type)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Slice start index of problems (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Slice end index of problems (exclusive)",
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

    # Optional --start/--end slicing of problem set
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(data["problems"])
        data["problems"] = data["problems"][start:end]
        # Update metadata range so output filename + coverage stats reflect slice
        data.setdefault("metadata", {})["range"] = [start, end]
        logger.info(f"범위 제한 적용: [{start}, {end}) → {len(data['problems'])}문제")

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

    # Load evaluation results (auto-detect if not specified)
    eval_map = {}  # {(question_id, transformation_type): eval_result}
    eval_path = None
    if args.eval:
        eval_path = results_dir / args.eval
    else:
        # Auto-detect largest batch_evaluation file
        candidates = sorted(
            results_dir.glob("batch_evaluation_*.json"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if candidates:
            eval_path = candidates[0]
            logger.info(f"Evaluation 자동 감지: {eval_path.name}")

    if eval_path and eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for r in eval_data.get("results", []):
            key = (r.get("question_id", ""), r.get("transformation_type", ""))
            eval_map[key] = r
        logger.info(f"Evaluation 로드: {eval_path} ({len(eval_map)}건)")

        # Inject eval results into problem transformations
        for problem in data["problems"]:
            qid = problem.get("question_id", "")
            for ttype, tdata in problem.get("transformations", {}).items():
                if not tdata.get("success"):
                    continue
                eval_result = eval_map.get((qid, ttype))
                if eval_result:
                    tdata["model_response"] = eval_result.get("raw_response", "")
                    tdata["model_name"] = eval_result.get("model", "")
                    tdata["response_type"] = eval_result.get("response_type", "")
                    tdata["is_correct"] = eval_result.get("is_correct", False)
                    tdata["case_type"] = eval_result.get("case_type", 0)
                    tdata["predicted_answer"] = eval_result.get("predicted_answer", "")
                    tdata["execution_error"] = eval_result.get("execution_error", "")
    elif eval_path:
        logger.warning(f"Evaluation 파일 없음: {eval_path}")

    # Load additional eval results (original context + metacognitive)
    for eval_name, field_prefix in [
        ("eval_original_context.json", "original"),
        ("eval_metacognitive.json", "metacognitive"),
    ]:
        extra_path = results_dir / eval_name
        if extra_path.exists():
            with open(extra_path, "r", encoding="utf-8") as f:
                extra_data = json.load(f)
            extra_map = {}
            for r in extra_data.get("results", []):
                key = (r.get("question_id", ""), r.get("transformation_type", ""))
                extra_map[key] = r

            injected = 0
            for problem in data["problems"]:
                qid = problem.get("question_id", "")
                for ttype, tdata in problem.get("transformations", {}).items():
                    if not tdata.get("success"):
                        continue
                    er = extra_map.get((qid, ttype))
                    if er:
                        tdata[f"{field_prefix}_response"] = er.get("raw_response", "")
                        tdata[f"{field_prefix}_predicted"] = er.get(
                            "predicted_answer", ""
                        )
                        tdata[f"{field_prefix}_is_correct"] = er.get(
                            "is_correct", False
                        )
                        tdata[f"{field_prefix}_response_type"] = er.get(
                            "response_type", ""
                        )
                        tdata[f"{field_prefix}_case_type"] = er.get("case_type", 0)
                        tdata[f"{field_prefix}_execution_error"] = er.get(
                            "execution_error", ""
                        )
                        injected += 1

            logger.info(f"추가 Eval 로드: {eval_name} ({injected}건)")

    # Load enrich overlays (A1 reason categories + F1 Memorization Score)
    enrich_reasons = None
    enrich_mems = None
    if args.enrich_reasons:
        er_path = project_root / args.enrich_reasons
        if not er_path.is_absolute() and not er_path.exists():
            er_path = results_dir / Path(args.enrich_reasons).name
        if er_path.exists():
            with open(er_path, "r", encoding="utf-8") as f:
                enrich_reasons = json.load(f)
            logger.info(
                f"Enrich(reasons) 로드: {er_path} "
                f"({sum(len(v) for v in enrich_reasons.values())} cells)"
            )
        else:
            logger.warning(f"Enrich(reasons) 파일 없음: {er_path}")
    if args.enrich_memorization:
        em_path = project_root / args.enrich_memorization
        if not em_path.is_absolute() and not em_path.exists():
            em_path = results_dir / Path(args.enrich_memorization).name
        if em_path.exists():
            with open(em_path, "r", encoding="utf-8") as f:
                enrich_mems = json.load(f)
            logger.info(
                f"Enrich(memorization) 로드: {em_path} "
                f"({sum(len(v) for v in enrich_mems.values())} cells)"
            )
        else:
            logger.warning(f"Enrich(memorization) 파일 없음: {em_path}")

    html_content = generate_html(
        data,
        preloaded_annotations=preloaded,
        enrich_reasons=enrich_reasons,
        enrich_mems=enrich_mems,
    )

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
