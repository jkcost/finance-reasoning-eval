"""
Validation Report Generator

Generates an interactive HTML report from reasoning trace validation results.

Sections:
1. Summary Cards - total, case distribution, verdicts
2. Heatmap - Model × Transformation Type → Case distribution (Chart.js)
3. Case 2 Detail - value provenance table with color coding
4. Case 3 Detail - context independence scores + LLM Judge analysis
5. Effectiveness Summary - effective/compromised/insufficient by type

Usage:
    python experiments/generate_validation_report.py
    python experiments/generate_validation_report.py --input path/to/validation_analysis.json
    python experiments/generate_validation_report.py --results-dir experiments/results/metacognitive/
"""

import argparse
import html as html_module
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "metacognitive"


# ============================================================================
# DATA LOADING
# ============================================================================


def find_latest_validation(results_dir: Path) -> Optional[Path]:
    """Find the most recent validation_analysis JSON file."""
    files = sorted(results_dir.glob("validation_analysis_*.json"), reverse=True)
    return files[0] if files else None


def load_validation_data(path: Path) -> Dict[str, Any]:
    """Load validation analysis JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# HTML BUILDERS
# ============================================================================


def _esc(text: Any) -> str:
    """HTML escape."""
    return html_module.escape(str(text))


def build_summary_cards(summary: Dict[str, Any]) -> str:
    """Build top-level summary cards."""
    cd = summary.get("case_distribution", {})
    total = summary.get("total_results", 0)
    vc = summary.get("verdict_counts", {})
    total_trans = summary.get("total_transformations", 0)
    src = summary.get("source_info", {})
    n_questions = src.get("unique_questions", "?")
    ds_total = src.get("dataset_total", "?")
    n_models = len(src.get("unique_models", []))
    n_strategies = len(src.get("unique_strategies", []))
    models_str = ", ".join(src.get("unique_models", []))
    strategies_str = ", ".join(src.get("unique_strategies", []))

    def pct(n: int, t: int) -> str:
        return f"{n / t * 100:.1f}%" if t else "0%"

    return f"""
    <div class="source-info" style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:16px 20px; margin-bottom:16px; font-size:14px; color:#ccc;">
        <div style="font-size:16px; font-weight:600; color:#fff; margin-bottom:8px;">Data Source</div>
        <div><strong>Dataset:</strong> hard.json ({ds_total}문제 중 {n_questions}개 사용)</div>
        <div><strong>Models ({n_models}):</strong> {models_str}</div>
        <div><strong>Strategies ({n_strategies}):</strong> {strategies_str}</div>
        <div style="margin-top:6px; color:#aaa;">
            {n_questions} questions × {total_trans} transformations × {n_models} models × {n_strategies} strategies = <strong style="color:#fff;">{total}</strong> results
        </div>
    </div>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-value">{total}</div>
            <div class="summary-label">Total Results</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color:#4caf50">{cd.get("1", cd.get(1, 0))}</div>
            <div class="summary-label">Case 1 (거부) {pct(cd.get("1", cd.get(1, 0)), total)}</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color:#ff9800">{cd.get("2", cd.get(2, 0))}</div>
            <div class="summary-label">Case 2 (오답) {pct(cd.get("2", cd.get(2, 0)), total)}</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color:#f44336">{cd.get("3", cd.get(3, 0))}</div>
            <div class="summary-label">Case 3 (정답) {pct(cd.get("3", cd.get(3, 0)), total)}</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color:#00d4ff">{total_trans}</div>
            <div class="summary-label">Unique Transformations</div>
        </div>
    </div>
    <div class="summary-grid" style="grid-template-columns: repeat(3, 1fr);">
        <div class="summary-card" style="border-left: 3px solid #4caf50;">
            <div class="summary-value" style="color:#4caf50">{vc.get("effective", 0)}</div>
            <div class="summary-label">Effective</div>
        </div>
        <div class="summary-card" style="border-left: 3px solid #ff9800;">
            <div class="summary-value" style="color:#ff9800">{vc.get("still_solvable", 0)}</div>
            <div class="summary-label">Still Solvable (변환 불충분)</div>
        </div>
        <div class="summary-card" style="border-left: 3px solid #f44336;">
            <div class="summary-value" style="color:#f44336">{vc.get("transformation_insufficient", 0)}</div>
            <div class="summary-label">Insufficient (변환 불충분)</div>
        </div>
    </div>
    """


def build_type_table(summary: Dict[str, Any]) -> str:
    """Build case distribution by transformation type table."""
    by_type = summary.get("case_by_type", {})
    if not by_type:
        return "<p>No data available.</p>"

    rows = ""
    for trans_type in sorted(by_type.keys()):
        counts = by_type[trans_type]
        # Handle both int and string keys
        c1 = counts.get(1, counts.get("1", 0))
        c2 = counts.get(2, counts.get("2", 0))
        c3 = counts.get(3, counts.get("3", 0))
        total = c1 + c2 + c3
        eff_rate = (c1 + c2) / total * 100 if total else 0

        # Bar visualization
        p1 = c1 / total * 100 if total else 0
        p2 = c2 / total * 100 if total else 0
        p3 = c3 / total * 100 if total else 0

        bar = f"""<div class="stacked-bar">
            <div style="width:{p1}%;background:#4caf50;" title="Case 1: {c1}"></div>
            <div style="width:{p2}%;background:#ff9800;" title="Case 2: {c2}"></div>
            <div style="width:{p3}%;background:#f44336;" title="Case 3: {c3}"></div>
        </div>"""

        rows += f"""<tr>
            <td>{_esc(trans_type)}</td>
            <td class="num">{c1}</td>
            <td class="num">{c2}</td>
            <td class="num">{c3}</td>
            <td class="num">{total}</td>
            <td class="num" style="color:{"#4caf50" if eff_rate >= 80 else "#ff9800" if eff_rate >= 60 else "#f44336"}">{eff_rate:.1f}%</td>
            <td>{bar}</td>
        </tr>"""

    return f"""
    <table>
        <thead><tr>
            <th>Transformation Type</th>
            <th>Case 1<br>(거부)</th>
            <th>Case 2<br>(오답)</th>
            <th>Case 3<br>(정답)</th>
            <th>Total</th>
            <th>Effective<br>Rate</th>
            <th>Distribution</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def build_model_table(summary: Dict[str, Any]) -> str:
    """Build case distribution by model table."""
    by_model = summary.get("case_by_model", {})
    if not by_model:
        return "<p>No data available.</p>"

    rows = ""
    for model in sorted(by_model.keys()):
        counts = by_model[model]
        c1 = counts.get(1, counts.get("1", 0))
        c2 = counts.get(2, counts.get("2", 0))
        c3 = counts.get(3, counts.get("3", 0))
        total = c1 + c2 + c3

        p1 = c1 / total * 100 if total else 0
        p2 = c2 / total * 100 if total else 0
        p3 = c3 / total * 100 if total else 0

        bar = f"""<div class="stacked-bar">
            <div style="width:{p1}%;background:#4caf50;" title="Case 1: {c1}"></div>
            <div style="width:{p2}%;background:#ff9800;" title="Case 2: {c2}"></div>
            <div style="width:{p3}%;background:#f44336;" title="Case 3: {c3}"></div>
        </div>"""

        rows += f"""<tr>
            <td>{_esc(model)}</td>
            <td class="num">{c1}</td>
            <td class="num">{c2}</td>
            <td class="num">{c3}</td>
            <td class="num">{total}</td>
            <td>{bar}</td>
        </tr>"""

    return f"""
    <table>
        <thead><tr>
            <th>Model</th>
            <th>Case 1<br>(거부)</th>
            <th>Case 2<br>(오답)</th>
            <th>Case 3<br>(정답)</th>
            <th>Total</th>
            <th>Distribution</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def build_source_distribution(summary: Dict[str, Any]) -> str:
    """Build value source distribution section."""
    sources = summary.get("value_source_counts", {})
    if not sources:
        return "<p>No value provenance data.</p>"

    total = sum(sources.values())
    source_colors = {
        "from_context": "#4caf50",
        "from_removed_data": "#f44336",
        "fabricated": "#ff5722",
        "common_constant": "#666",
        "derived": "#2196f3",
    }

    items = ""
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        color = source_colors.get(source, "#888")
        label_map = {
            "from_context": "Context에서 유래",
            "from_removed_data": "제거된 데이터 (변환 미제거)",
            "fabricated": "출처 불명 (환각?)",
            "common_constant": "일반 상수",
            "derived": "산술 파생값",
        }
        label = label_map.get(source, source)
        items += f"""
        <div class="source-item">
            <div class="source-bar-bg">
                <div class="source-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="source-label" style="color:{color}">{label}</span>
            <span class="source-count">{count} ({pct:.1f}%)</span>
        </div>"""

    return f'<div class="source-distribution">{items}</div>'


def build_heatmap_data(data: Dict[str, Any]) -> str:
    """Build Chart.js heatmap data for Model × Type → Case distribution."""
    detailed = data.get("detailed_results", [])
    if not detailed:
        return ""

    # Collect models and types
    models = sorted({r["model_name"] for r in detailed})
    types = sorted({r["transformation_type"] for r in detailed})

    # Build matrix: model × type → {1: n, 2: n, 3: n}
    matrix: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: {1: 0, 2: 0, 3: 0})
    )
    for r in detailed:
        case = r.get("analysis", {}).get("case_type", 0)
        if case in (1, 2, 3):
            matrix[r["model_name"]][r["transformation_type"]][case] += 1

    # Build HTML table heatmap (more flexible than Chart.js for 3-value cells)
    header = "<th>Model</th>" + "".join(f"<th>{_esc(t[:20])}</th>" for t in types)

    rows = ""
    for model in models:
        cells = ""
        for ttype in types:
            counts = matrix[model][ttype]
            total = sum(counts.values())
            if total == 0:
                cells += '<td class="heat-cell">-</td>'
                continue

            c1_pct = counts[1] / total
            c3_pct = counts[3] / total
            # Green = high Case 1, Red = high Case 3
            r_val = int(c3_pct * 200 + 40)
            g_val = int(c1_pct * 180 + 40)
            bg = f"rgba({r_val},{g_val},60,0.6)"

            cells += f"""<td class="heat-cell" style="background:{bg};">
                <div class="heat-nums">{counts[1]}/{counts[2]}/{counts[3]}</div>
                <div class="heat-legend">R/W/C</div>
            </td>"""

        rows += f"<tr><td class='model-name'>{_esc(model)}</td>{cells}</tr>"

    return f"""
    <table class="heatmap-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <div class="heat-legend-bar">
        <span style="color:#4caf50">R = 거부(Case 1)</span> /
        <span style="color:#ff9800">W = 오답(Case 2)</span> /
        <span style="color:#f44336">C = 정답(Case 3)</span>
    </div>"""


def _detect_response_format(raw_response: str) -> str:
    """Detect if response is POT (Python code) or COT (text reasoning)."""
    if "```python" in raw_response or "def solution" in raw_response:
        return "POT"
    return "COT"


def _build_context_diff(original: str, transformed: str, description: str) -> str:
    """Build a visual diff showing what was changed in the context."""
    if not original and not transformed:
        return ""

    desc_html = (
        f'<div class="diff-desc">{_esc(description)}</div>' if description else ""
    )

    # Truncate for display
    orig_display = original
    trans_display = transformed

    return f"""
    <div class="context-diff">
        {desc_html}
        <div class="diff-panels">
            <div class="diff-panel">
                <div class="diff-panel-title" style="color:#f44336;">원본 Context</div>
                <pre class="diff-content">{_esc(orig_display)}</pre>
            </div>
            <div class="diff-panel">
                <div class="diff-panel-title" style="color:#4caf50;">변환된 Context (모델이 받은 것)</div>
                <pre class="diff-content">{_esc(trans_display)}</pre>
            </div>
        </div>
    </div>"""


def _build_case3_card(r: Dict[str, Any]) -> str:
    """Build a single Case 3 detail card."""
    analysis = r.get("analysis", {})
    indep_score = analysis.get("context_independence_score", 0)
    llm_verdict = analysis.get("llm_judge_verdict")
    raw_response = r.get("raw_response", "")
    resp_format = _detect_response_format(raw_response)

    score_color = "#ff9800" if indep_score >= 0.5 else "#f44336"

    # Value provenance chips
    values_html = ""
    for v in analysis.get("values_used", []):
        source = v.get("source", "")
        source_colors = {
            "from_context": "#4caf50",
            "from_removed_data": "#f44336",
            "fabricated": "#ff5722",
            "derived": "#2196f3",
        }
        color = source_colors.get(source, "#888")
        source_label = {
            "from_context": "context",
            "from_removed_data": "제거됨",
            "fabricated": "환각",
            "derived": "파생",
        }.get(source, source)
        values_html += f'<span class="value-chip" style="border-color:{color};color:{color}">{v["value"]} ({source_label})</span> '

    # LLM Judge section
    llm_section = ""
    if llm_verdict:
        llm_section = f"""
        <div class="llm-verdict">
            <div class="llm-verdict-title">LLM Judge 분석</div>
            <pre class="llm-verdict-text">{_esc(llm_verdict)}</pre>
        </div>"""

    # Independence factors
    factors = analysis.get("independence_factors", [])
    factors_html = ""
    if factors:
        factor_items = "".join(f"<li>{_esc(f)}</li>" for f in factors)
        factors_html = f"""
        <div class="dep-factors">
            <div class="dep-factors-title">판정 근거</div>
            <ul class="dep-factors-list">{factor_items}</ul>
        </div>"""

    # Question text
    question = r.get("question", "")
    question_html = (
        f'<div class="card-question">{_esc(question)}</div>' if question else ""
    )

    # Answer comparison: GT vs Predicted
    gt = r.get("ground_truth", "N/A")
    pred = r.get("predicted_answer", "N/A")
    format_badge = (
        '<span class="format-badge pot">POT (코드→실행)</span>'
        if resp_format == "POT"
        else '<span class="format-badge cot">COT (텍스트)</span>'
    )
    answer_html = f"""
    <div class="answer-compare">
        <div class="answer-box">
            <span class="answer-label">정답 (GT)</span>
            <span class="answer-value">{_esc(str(gt))}</span>
        </div>
        <div class="answer-eq">=</div>
        <div class="answer-box" style="border-color:#f44336;">
            <span class="answer-label">모델 예측</span>
            <span class="answer-value" style="color:#f44336;">{_esc(str(pred))}</span>
        </div>
        {format_badge}
    </div>"""

    # Context diff (collapsible)
    context_diff = _build_context_diff(
        r.get("context_original", ""),
        r.get("context_transformed", ""),
        r.get("transformation_description", ""),
    )

    return f"""
    <div class="case3-card">
        <div class="case3-header">
            <span class="case3-id">{_esc(r.get("example_id", ""))}</span>
            <span class="case3-model">{_esc(r.get("model_name", ""))}</span>
            <span class="case3-type">{_esc(r.get("transformation_type", ""))}</span>
            <span class="case3-strategy">{_esc(r.get("prompt_strategy", ""))}</span>
            <span class="dep-score" style="color:{score_color}">
                {indep_score:.2f}
            </span>
        </div>
        {question_html}
        {answer_html}
        {factors_html}
        <div class="case3-values">
            <strong>응답 내 숫자 출처:</strong>
            {values_html if values_html else '<span style="color:#666">추출된 값 없음</span>'}
        </div>
        <details>
            <summary class="case3-response-toggle">변환 전/후 Context 비교</summary>
            {context_diff}
        </details>
        <details>
            <summary class="case3-response-toggle">모델 응답 원문</summary>
            <pre class="case3-response">{_esc(raw_response)}</pre>
        </details>
        {llm_section}
    </div>"""


def build_case3_details(data: Dict[str, Any]) -> str:
    """Build Case 3 section: why did the model answer correctly despite transformation?

    Structure:
        1. Narrative: 핵심 질문 설명
        2. Summary stats: context 비의존 vs 변환 불충분 비율, 모델별/타입별 분포
        3. Group A: context 비의존 (score >= 0.5) — 대표 사례
        4. Group B: 추론으로 해결 (score < 0.5) — 대표 사례
    """
    detailed = data.get("detailed_results", [])
    case3_results = [r for r in detailed if r.get("analysis", {}).get("case_type") == 3]

    if not case3_results:
        return "<p>Case 3 결과 없음 — 모든 변환이 유효합니다.</p>"

    total = len(case3_results)

    # Split into groups
    ctx_independent = [
        r
        for r in case3_results
        if r.get("analysis", {}).get("context_independence_score", 0) >= 0.5
    ]
    insufficient = [
        r
        for r in case3_results
        if r.get("analysis", {}).get("context_independence_score", 0) < 0.5
    ]
    ctx_independent.sort(
        key=lambda r: r.get("analysis", {}).get("context_independence_score", 0),
        reverse=True,
    )
    insufficient.sort(
        key=lambda r: r.get("analysis", {}).get("context_independence_score", 0),
        reverse=True,
    )

    # By-model stats
    model_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"ctx_independent": 0, "insufficient": 0}
    )
    for r in ctx_independent:
        model_stats[r.get("model_name", "")]["ctx_independent"] += 1
    for r in insufficient:
        model_stats[r.get("model_name", "")]["insufficient"] += 1

    # By-type stats
    type_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"ctx_independent": 0, "insufficient": 0}
    )
    for r in ctx_independent:
        type_stats[r.get("transformation_type", "")]["ctx_independent"] += 1
    for r in insufficient:
        type_stats[r.get("transformation_type", "")]["insufficient"] += 1

    # Value source summary for Case 3
    source_counts: Dict[str, int] = defaultdict(int)
    for r in case3_results:
        for v in r.get("analysis", {}).get("values_used", []):
            source_counts[v.get("source", "")] += 1

    removed_count = source_counts.get("from_removed_data", 0)
    fabricated_count = source_counts.get("fabricated", 0)

    # --- Build HTML ---

    # 1. Narrative
    narrative = f"""
    <div class="case3-narrative">
        <div class="narrative-question">정보를 제거/변환했는데 왜 정답을 맞췄는가?</div>
        <div class="narrative-explain">
            변환된 문제에서 정답을 맞춘 <strong>{total}건</strong>을 두 가지로 분류합니다:
        </div>
        <div class="narrative-two-col">
            <div class="narrative-box" style="border-color:#ff9800;">
                <div class="narrative-box-title" style="color:#ff9800;">Context 비의존 (Context Independent)</div>
                <div class="narrative-box-desc">
                    모델이 변환된 context를 제대로 참조하지 않고 답을 도출한 경우.
                    제거된 데이터를 사용하거나, context 키워드를 미참조하거나, 추론 과정이 없음.
                    <strong>변환 자체는 유효</strong>하나, 모델의 응답이 context에 의존하지 않음.
                </div>
            </div>
            <div class="narrative-box" style="border-color:#f44336;">
                <div class="narrative-box-title" style="color:#f44336;">변환 불충분 (Insufficient Transform)</div>
                <div class="narrative-box-desc">
                    남은 정보만으로 추론하여 정답에 도달한 경우.
                    <strong>변환이 불충분</strong>했을 가능성 — 핵심 정보가 아닌 것을 제거했거나,
                    다른 경로로 답을 구할 수 있음.
                </div>
            </div>
        </div>
    </div>"""

    # 2. Summary stats
    indep_pct = len(ctx_independent) / total * 100 if total else 0
    insuf_pct = len(insufficient) / total * 100 if total else 0

    model_rows = ""
    for model in sorted(model_stats.keys()):
        s = model_stats[model]
        m_total = s["ctx_independent"] + s["insufficient"]
        bar_w = s["ctx_independent"] / max(m_total, 1) * 100
        model_rows += f"""<tr>
            <td>{_esc(model)}</td>
            <td class="num">{s["ctx_independent"]}</td>
            <td class="num">{s["insufficient"]}</td>
            <td class="num">{m_total}</td>
            <td><div class="stacked-bar">
                <div style="width:{bar_w}%;background:#ff9800;"></div>
                <div style="width:{100 - bar_w}%;background:#f44336;"></div>
            </div></td>
        </tr>"""

    type_rows = ""
    for ttype in sorted(type_stats.keys()):
        s = type_stats[ttype]
        t_total = s["ctx_independent"] + s["insufficient"]
        bar_w = s["ctx_independent"] / max(t_total, 1) * 100
        type_rows += f"""<tr>
            <td>{_esc(ttype)}</td>
            <td class="num">{s["ctx_independent"]}</td>
            <td class="num">{s["insufficient"]}</td>
            <td class="num">{t_total}</td>
            <td><div class="stacked-bar">
                <div style="width:{bar_w}%;background:#ff9800;"></div>
                <div style="width:{100 - bar_w}%;background:#f44336;"></div>
            </div></td>
        </tr>"""

    stats_html = f"""
    <div class="case3-stats-grid">
        <div class="case3-stat-card" style="border-left:3px solid #ff9800;">
            <div class="summary-value" style="color:#ff9800;font-size:1.6em;">{len(ctx_independent)}</div>
            <div class="summary-label">Context 비의존 ({indep_pct:.0f}%)</div>
        </div>
        <div class="case3-stat-card" style="border-left:3px solid #f44336;">
            <div class="summary-value" style="color:#f44336;font-size:1.6em;">{len(insufficient)}</div>
            <div class="summary-label">변환 불충분 ({insuf_pct:.0f}%)</div>
        </div>
        <div class="case3-stat-card" style="border-left:3px solid #888;">
            <div class="summary-value" style="color:#ccc;font-size:1.6em;">{removed_count}</div>
            <div class="summary-label">제거된 값 사용 횟수</div>
        </div>
    </div>

    <div class="case3-tables-row">
        <div>
            <div class="chart-title">모델별 분포</div>
            <table>
                <thead><tr><th>Model</th><th style="color:#ff9800">비의존</th><th style="color:#f44336">불충분</th><th>합계</th><th>비율</th></tr></thead>
                <tbody>{model_rows}</tbody>
            </table>
        </div>
        <div>
            <div class="chart-title">변환 유형별 분포</div>
            <table>
                <thead><tr><th>Type</th><th style="color:#ff9800">비의존</th><th style="color:#f44336">불충분</th><th>합계</th><th>비율</th></tr></thead>
                <tbody>{type_rows}</tbody>
            </table>
        </div>
    </div>
    """

    # 3. Group A: Context Independent (show top 10)
    SHOW_LIMIT = 10
    indep_cards = ""
    for r in ctx_independent[:SHOW_LIMIT]:
        indep_cards += _build_case3_card(r)
    indep_remaining = len(ctx_independent) - SHOW_LIMIT
    indep_extra = ""
    if indep_remaining > 0:
        indep_extra_cards = ""
        for r in ctx_independent[SHOW_LIMIT:]:
            indep_extra_cards += _build_case3_card(r)
        indep_extra = f"""
        <details>
            <summary class="case3-response-toggle" style="margin:12px 0;">나머지 {indep_remaining}건 보기</summary>
            {indep_extra_cards}
        </details>"""

    group_a = (
        f"""
    <div class="case3-group">
        <div class="case3-group-header" style="border-color:#ff9800;">
            <span class="case3-group-icon" style="background:#ff9800;">A</span>
            <span class="case3-group-title">Context 비의존 — 변환은 유효하나 모델이 context를 미사용</span>
            <span class="case3-group-count">{len(ctx_independent)}건</span>
        </div>
        <p style="color:#888;font-size:0.85em;margin:8px 0 12px;">
            비의존 점수 ≥ 0.5 — 제거된 데이터 사용, 하드코딩 솔루션, context 미참조 등의 근거
        </p>
        {indep_cards}
        {indep_extra}
    </div>"""
        if ctx_independent
        else ""
    )

    # 4. Group B: Insufficient Transform (show top 10)
    insuf_cards = ""
    for r in insufficient[:SHOW_LIMIT]:
        insuf_cards += _build_case3_card(r)
    insuf_remaining = len(insufficient) - SHOW_LIMIT
    insuf_extra = ""
    if insuf_remaining > 0:
        insuf_extra_cards = ""
        for r in insufficient[SHOW_LIMIT:]:
            insuf_extra_cards += _build_case3_card(r)
        insuf_extra = f"""
        <details>
            <summary class="case3-response-toggle" style="margin:12px 0;">나머지 {insuf_remaining}건 보기</summary>
            {insuf_extra_cards}
        </details>"""

    group_b = (
        f"""
    <div class="case3-group">
        <div class="case3-group-header" style="border-color:#f44336;">
            <span class="case3-group-icon" style="background:#f44336;">B</span>
            <span class="case3-group-title">변환 불충분 — 남은 정보로 정답 도달 가능</span>
            <span class="case3-group-count">{len(insufficient)}건</span>
        </div>
        <p style="color:#888;font-size:0.85em;margin:8px 0 12px;">
            비의존 점수 &lt; 0.5 — context의 남은 정보로 정답을 유도할 수 있었음을 시사
        </p>
        {insuf_cards}
        {insuf_extra}
    </div>"""
        if insufficient
        else ""
    )

    return narrative + stats_html + group_a + group_b


def build_case2_details(data: Dict[str, Any]) -> str:
    """Build detailed view for interesting Case 2 results (with awareness or removed data usage)."""
    detailed = data.get("detailed_results", [])
    interesting = [
        r
        for r in detailed
        if r.get("analysis", {}).get("case_type") == 2
        and (
            r.get("analysis", {}).get("awareness_signals")
            or any(
                v.get("source") == "from_removed_data"
                for v in r.get("analysis", {}).get("values_used", [])
            )
        )
    ]

    if not interesting:
        return "<p>특이 Case 2 결과 없음.</p>"

    # Limit to 20 most interesting
    interesting = interesting[:20]

    items = ""
    for r in interesting:
        analysis = r.get("analysis", {})
        signals = analysis.get("awareness_signals", [])
        raw_response = r.get("raw_response", "")
        resp_format = _detect_response_format(raw_response)

        values_html = ""
        for v in analysis.get("values_used", []):
            source = v.get("source", "")
            source_colors = {
                "from_context": "#4caf50",
                "from_removed_data": "#f44336",
                "fabricated": "#ff5722",
                "derived": "#2196f3",
            }
            color = source_colors.get(source, "#888")
            source_label = {
                "from_context": "context",
                "from_removed_data": "제거됨",
                "fabricated": "환각",
                "derived": "파생",
            }.get(source, source)
            values_html += f'<span class="value-chip" style="border-color:{color};color:{color}">{v["value"]} ({source_label})</span> '

        signal_html = ""
        if signals:
            signal_html = (
                '<div class="awareness-signals"><strong>인식 신호:</strong> '
                + ", ".join(
                    f'<span class="signal-chip">{_esc(s)}</span>' for s in signals
                )
                + "</div>"
            )

        # Question
        question = r.get("question", "")
        question_html = (
            f'<div class="card-question">{_esc(question[:250])}'
            + ("..." if len(question) > 250 else "")
            + "</div>"
            if question
            else ""
        )

        # Answer comparison
        gt = r.get("ground_truth", "N/A")
        pred = r.get("predicted_answer", "N/A")
        format_badge = (
            '<span class="format-badge pot">POT (코드→실행)</span>'
            if resp_format == "POT"
            else '<span class="format-badge cot">COT (텍스트)</span>'
        )

        # Context diff
        context_diff = _build_context_diff(
            r.get("context_original", ""),
            r.get("context_transformed", ""),
            r.get("transformation_description", ""),
        )

        items += f"""
        <div class="case2-card">
            <div class="case2-header">
                <span class="case3-id">{_esc(r.get("example_id", ""))}</span>
                <span class="case3-model">{_esc(r.get("model_name", ""))}</span>
                <span class="case3-type">{_esc(r.get("transformation_type", ""))}</span>
                <span class="case3-strategy">{_esc(r.get("prompt_strategy", ""))}</span>
            </div>
            {question_html}
            <div class="answer-compare">
                <div class="answer-box">
                    <span class="answer-label">정답 (GT)</span>
                    <span class="answer-value">{_esc(str(gt))}</span>
                </div>
                <div class="answer-neq">≠</div>
                <div class="answer-box" style="border-color:#ff9800;">
                    <span class="answer-label">모델 예측</span>
                    <span class="answer-value" style="color:#ff9800;">{_esc(str(pred))}</span>
                </div>
                {format_badge}
            </div>
            {signal_html}
            <div class="case3-values">
                <strong>응답 내 숫자 출처:</strong> {values_html if values_html else '<span style="color:#666">-</span>'}
            </div>
            <details>
                <summary class="case3-response-toggle">변환 전/후 Context 비교</summary>
                {context_diff}
            </details>
            <details>
                <summary class="case3-response-toggle">모델 응답 원문</summary>
                <pre class="case3-response">{_esc(raw_response)}</pre>
            </details>
        </div>"""

    return items


def build_verdict_table(data: Dict[str, Any]) -> str:
    """Build transformation-level verdict table."""
    verdicts = data.get("aggregated_verdicts", [])
    if not verdicts:
        return "<p>No verdict data.</p>"

    verdict_colors = {
        "effective": "#4caf50",
        "still_solvable": "#ff9800",
        "transformation_insufficient": "#f44336",
    }
    verdict_labels = {
        "effective": "유효",
        "still_solvable": "여전히 풀림",
        "transformation_insufficient": "불충분",
    }

    # Group by type
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for v in verdicts:
        by_type[v.get("transformation_type", "")].append(v)

    rows = ""
    for trans_type in sorted(by_type.keys()):
        type_verdicts = by_type[trans_type]
        type_eff = sum(1 for v in type_verdicts if v["verdict"] == "effective")
        type_comp = sum(1 for v in type_verdicts if v["verdict"] == "still_solvable")
        type_insuf = sum(
            1 for v in type_verdicts if v["verdict"] == "transformation_insufficient"
        )
        type_total = len(type_verdicts)

        rows += f"""<tr>
            <td>{_esc(trans_type)}</td>
            <td class="num" style="color:#4caf50">{type_eff}</td>
            <td class="num" style="color:#ff9800">{type_comp}</td>
            <td class="num" style="color:#f44336">{type_insuf}</td>
            <td class="num">{type_total}</td>
            <td class="num" style="color:#4caf50">{type_eff / type_total * 100:.0f}%</td>
        </tr>"""

    return f"""
    <table>
        <thead><tr>
            <th>Transformation Type</th>
            <th>유효</th>
            <th>여전히 풀림</th>
            <th>불충분</th>
            <th>Total</th>
            <th>유효율</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# ============================================================================
# MAIN HTML ASSEMBLY
# ============================================================================


def generate_html(data: Dict[str, Any]) -> str:
    """Generate the complete HTML report."""
    summary = data.get("summary", {})
    timestamp = data.get("timestamp", "unknown")

    summary_cards = build_summary_cards(summary)
    type_table = build_type_table(summary)
    model_table = build_model_table(summary)
    source_dist = build_source_distribution(summary)
    heatmap = build_heatmap_data(data)
    case3_details = build_case3_details(data)
    case2_details = build_case2_details(data)
    verdict_table = build_verdict_table(data)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transformation Validation Report - {_esc(timestamp)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 5px; color: #00d4ff; font-size: 2em; }}
        h2 {{ color: #00d4ff; margin: 30px 0 15px; font-size: 1.3em; border-bottom: 1px solid rgba(0,212,255,0.2); padding-bottom: 8px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 25px; font-size: 0.95em; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 15px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }}
        .summary-value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
        .summary-label {{ color: #888; margin-top: 4px; font-size: 0.85em; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th {{
            background: rgba(0,212,255,0.1);
            padding: 12px 8px;
            text-align: left;
            font-size: 0.85em;
            color: #00d4ff;
        }}
        td {{
            padding: 10px 8px;
            border-top: 1px solid rgba(255,255,255,0.05);
            font-size: 0.88em;
        }}
        .num {{ text-align: center; font-family: 'Fira Code', monospace; }}

        .stacked-bar {{
            display: flex;
            height: 20px;
            border-radius: 4px;
            overflow: hidden;
            min-width: 100px;
        }}
        .stacked-bar > div {{ min-width: 1px; }}

        .heatmap-table td {{ text-align: center; padding: 8px; }}
        .heat-cell {{ border-radius: 6px; min-width: 80px; }}
        .heat-nums {{ font-weight: bold; font-size: 0.95em; }}
        .heat-legend {{ font-size: 0.65em; color: rgba(255,255,255,0.5); }}
        .heat-legend-bar {{ text-align: center; margin: 8px 0 20px; font-size: 0.85em; color: #888; }}
        .model-name {{ font-weight: bold; color: #ccc; white-space: nowrap; }}

        .source-distribution {{ margin: 15px 0; }}
        .source-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 8px 0;
        }}
        .source-bar-bg {{
            flex: 1;
            height: 22px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
            overflow: hidden;
        }}
        .source-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .source-label {{ width: 180px; font-size: 0.88em; }}
        .source-count {{ width: 100px; text-align: right; font-size: 0.85em; color: #888; }}

        .case3-card, .case2-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .case3-header, .case2-header {{
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 8px;
        }}
        .case3-id {{
            background: rgba(0,212,255,0.15);
            color: #00d4ff;
            padding: 2px 10px;
            border-radius: 8px;
            font-size: 0.85em;
        }}
        .case3-model {{
            background: rgba(255,255,255,0.1);
            padding: 2px 10px;
            border-radius: 8px;
            font-size: 0.85em;
        }}
        .case3-type {{
            color: #888;
            font-size: 0.85em;
        }}
        .dep-score {{
            font-weight: bold;
            font-size: 0.9em;
            margin-left: auto;
        }}
        .case3-summary {{
            color: #aaa;
            font-size: 0.9em;
            margin-bottom: 8px;
        }}
        .case3-values {{ margin: 8px 0; font-size: 0.88em; }}
        .value-chip {{
            display: inline-block;
            border: 1px solid;
            padding: 1px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            margin: 2px;
        }}
        .signal-chip {{
            display: inline-block;
            background: rgba(33,150,243,0.15);
            color: #2196f3;
            padding: 1px 8px;
            border-radius: 8px;
            font-size: 0.85em;
            margin: 2px;
        }}
        .awareness-signals {{ margin: 6px 0; font-size: 0.88em; }}
        .case3-response-toggle {{
            color: #00d4ff;
            cursor: pointer;
            font-size: 0.85em;
            margin-top: 6px;
        }}
        .case3-response {{
            background: rgba(0,0,0,0.3);
            padding: 12px;
            border-radius: 8px;
            font-size: 0.8em;
            margin-top: 8px;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .llm-verdict {{
            background: rgba(33,150,243,0.1);
            border-left: 3px solid #2196f3;
            padding: 12px;
            border-radius: 0 8px 8px 0;
            margin-top: 10px;
        }}
        .llm-verdict-title {{
            color: #2196f3;
            font-weight: bold;
            margin-bottom: 6px;
            font-size: 0.9em;
        }}
        .llm-verdict-text {{
            font-size: 0.85em;
            white-space: pre-wrap;
            color: #ccc;
        }}

        .section {{ margin-bottom: 30px; }}
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .chart-panel {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
        }}
        .chart-title {{ color: #00d4ff; margin-bottom: 12px; font-size: 1.05em; }}

        /* Case 3 narrative & grouping */
        .case3-narrative {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .narrative-question {{
            font-size: 1.2em;
            font-weight: bold;
            color: #fff;
            margin-bottom: 10px;
        }}
        .narrative-explain {{
            color: #aaa;
            font-size: 0.92em;
            margin-bottom: 14px;
        }}
        .narrative-two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }}
        .narrative-box {{
            border: 1px solid;
            border-radius: 10px;
            padding: 14px;
            background: rgba(0,0,0,0.2);
        }}
        .narrative-box-title {{
            font-weight: bold;
            margin-bottom: 6px;
            font-size: 1em;
        }}
        .narrative-box-desc {{
            color: #aaa;
            font-size: 0.85em;
            line-height: 1.5;
        }}
        .case3-stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }}
        .case3-stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .case3-tables-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }}
        .case3-group {{
            margin-bottom: 24px;
        }}
        .case3-group-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            border-left: 4px solid;
            padding: 10px 14px;
            background: rgba(255,255,255,0.03);
            border-radius: 0 10px 10px 0;
            margin-bottom: 4px;
        }}
        .case3-group-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            color: #fff;
            font-weight: bold;
            font-size: 0.85em;
            flex-shrink: 0;
        }}
        .case3-group-title {{
            font-weight: bold;
            font-size: 0.95em;
            color: #ddd;
        }}
        .case3-group-count {{
            margin-left: auto;
            color: #888;
            font-size: 0.88em;
        }}
        .dep-factors {{
            background: rgba(255,152,0,0.08);
            border-left: 3px solid #ff9800;
            border-radius: 0 8px 8px 0;
            padding: 8px 12px;
            margin: 6px 0 8px;
        }}
        .dep-factors-title {{
            color: #ff9800;
            font-weight: bold;
            font-size: 0.82em;
            margin-bottom: 4px;
        }}
        .dep-factors-list {{
            margin: 0;
            padding-left: 18px;
            font-size: 0.82em;
            color: #bbb;
            line-height: 1.6;
        }}
        .dep-factors-list li {{
            margin-bottom: 2px;
        }}

        /* Question display */
        .card-question {{
            background: rgba(0,212,255,0.06);
            border-left: 3px solid rgba(0,212,255,0.3);
            padding: 8px 12px;
            margin: 6px 0 10px;
            font-size: 0.88em;
            color: #ccc;
            line-height: 1.5;
            border-radius: 0 6px 6px 0;
        }}

        /* Answer comparison */
        .answer-compare {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0 10px;
            flex-wrap: wrap;
        }}
        .answer-box {{
            border: 1px solid #4caf50;
            border-radius: 8px;
            padding: 6px 14px;
            text-align: center;
            min-width: 100px;
        }}
        .answer-label {{
            display: block;
            font-size: 0.72em;
            color: #888;
            margin-bottom: 2px;
        }}
        .answer-value {{
            font-weight: bold;
            font-size: 1.1em;
            font-family: 'Fira Code', monospace;
            color: #4caf50;
        }}
        .answer-eq, .answer-neq {{
            font-size: 1.3em;
            font-weight: bold;
            color: #888;
        }}

        /* Response format badges */
        .format-badge {{
            font-size: 0.75em;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: bold;
        }}
        .format-badge.pot {{
            background: rgba(156,39,176,0.15);
            color: #ce93d8;
            border: 1px solid rgba(156,39,176,0.3);
        }}
        .format-badge.cot {{
            background: rgba(0,150,136,0.15);
            color: #80cbc4;
            border: 1px solid rgba(0,150,136,0.3);
        }}

        /* Strategy badge */
        .case3-strategy {{
            background: rgba(255,255,255,0.08);
            color: #aaa;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.78em;
        }}

        /* Context diff panels */
        .context-diff {{
            margin: 8px 0;
        }}
        .diff-desc {{
            background: rgba(255,152,0,0.1);
            color: #ff9800;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85em;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        .diff-panels {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .diff-panel {{
            background: rgba(0,0,0,0.25);
            border-radius: 8px;
            padding: 10px;
            overflow: hidden;
        }}
        .diff-panel-title {{
            font-size: 0.78em;
            font-weight: bold;
            margin-bottom: 6px;
        }}
        .diff-content {{
            font-size: 0.75em;
            white-space: pre-wrap;
            word-break: break-all;
            color: #bbb;
            line-height: 1.4;
        }}

        @media (max-width: 900px) {{
            .diff-panels {{ grid-template-columns: 1fr; }}
            .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .charts-row {{ grid-template-columns: 1fr; }}
            .narrative-two-col {{ grid-template-columns: 1fr; }}
            .case3-tables-row {{ grid-template-columns: 1fr; }}
            .case3-stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Transformation Validation Report</h1>
    <div class="subtitle">추론 추적 기반 변환 검증 · {_esc(timestamp)}</div>

    <div class="section">
        <h2>1. Summary</h2>
        {summary_cards}
    </div>

    <div class="section">
        <h2>2. Case Distribution by Transformation Type</h2>
        {type_table}
    </div>

    <div class="section">
        <h2>3. Case Distribution by Model</h2>
        {model_table}
    </div>

    <div class="section">
        <h2>4. Model × Type Heatmap</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:10px;">
            각 셀: 거부(R) / 오답(W) / 정답(C) — 녹색=거부 우세, 적색=정답 우세
        </p>
        {heatmap}
    </div>

    <div class="section">
        <h2>5. Value Source Distribution</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:10px;">
            LLM 응답에서 사용된 숫자값들의 출처 분류
        </p>
        {source_dist}
    </div>

    <div class="section">
        <h2>6. 왜 변환된 문제에서 정답을 맞췄는가?</h2>
        {case3_details}
    </div>

    <div class="section">
        <h2>7. Case 2 특이 사례 — 부분 인식/제거 데이터 사용</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:10px;">
            오답이지만 부분적 인식 신호 또는 제거된 데이터 값 사용이 감지된 경우
        </p>
        {case2_details}
    </div>

    <div class="section">
        <h2>8. Transformation Effectiveness Verdicts</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:10px;">
            문제×변환 단위 유효성 판정 (전 모델 종합)
        </p>
        {verdict_table}
    </div>

    <div style="text-align:center;color:#555;padding:30px 0;font-size:0.8em;">
        Generated by Reasoning Trace Validation Pipeline
    </div>
</div>
</body>
</html>"""


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Generate Validation HTML Report")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to validation_analysis JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to search for validation analysis files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: results-dir/validation_report.html)",
    )
    args = parser.parse_args()

    # Find input file
    input_path = args.input
    if input_path is None:
        input_path = find_latest_validation(args.results_dir)
        if input_path is None:
            logger.error(
                "No validation_analysis_*.json found in %s. Run run_validation_pipeline.py first.",
                args.results_dir,
            )
            sys.exit(1)

    logger.info("Loading validation data from %s", input_path)
    data = load_validation_data(input_path)

    # Generate HTML
    html_content = generate_html(data)

    # Save
    output_path = args.output
    if output_path is None:
        output_path = args.results_dir / "validation_report.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Report saved to %s", output_path)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
