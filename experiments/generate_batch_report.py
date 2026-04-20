"""
Batch Report Generator — Phase 2

변환 + 평가 결과를 통합 HTML 리포트로 생성.

Usage:
    python experiments/generate_batch_report.py
    python experiments/generate_batch_report.py \\
        --transformations batch_transformations_0_30.json \\
        --evaluation batch_evaluation_0_30.json
"""

import argparse
import json
import logging
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TYPE_KEYS = ["EA-partial", "EA-full", "SA", "IC", "TA"]
CASE_COLORS = {1: "#4caf50", 2: "#ff9800", 3: "#f44336"}
CASE_LABELS = {1: "C1: Refused", 2: "C2: Wrong", 3: "C3: Correct"}


def _esc(text: str, max_len: int = 0) -> str:
    """Escape HTML. If max_len > 0, truncate."""
    if not text:
        return ""
    t = escape(str(text))
    if max_len > 0 and len(t) > max_len:
        t = t[:max_len] + "..."
    return t


def _pct(num: int, total: int) -> str:
    """Format percentage."""
    if total == 0:
        return "0%"
    return f"{num / total * 100:.1f}%"


def generate_html(
    trans_data: Dict,
    eval_data: Optional[Dict],
    output_path: Path,
) -> None:
    """Generate the full HTML report."""
    problems = trans_data["problems"]
    coverage = trans_data["coverage_summary"]
    meta = trans_data["metadata"]

    # Index evaluation results by (question_id, transformation_type, model)
    eval_index: Dict[str, List[Dict]] = {}
    eval_summary = None
    models: List[str] = []
    if eval_data:
        eval_summary = eval_data.get("summary", {})
        models = eval_data["metadata"].get("models", [])
        for r in eval_data.get("results", []):
            key = f"{r['question_id']}_{r['transformation_type']}"
            if key not in eval_index:
                eval_index[key] = []
            eval_index[key].append(r)

    html_parts = [_build_head(meta, coverage, eval_summary, models)]
    html_parts.append(_build_coverage_section(problems, coverage))
    html_parts.append(_build_failure_analysis_section(problems, coverage))
    eval_meta = eval_data.get("metadata") if eval_data else None
    if eval_summary:
        html_parts.append(_build_eval_summary_section(eval_summary, models, eval_meta))
    eval_results = eval_data.get("results", []) if eval_data else []
    if eval_results:
        html_parts.append(_build_c3_analysis_section(problems, eval_results, models))
        html_parts.append(_build_intensity_section(problems, eval_results))
    html_parts.append(_build_problem_cards(problems, eval_index, models))
    html_parts.append(_build_footer())

    full_html = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info(f"리포트 생성: {output_path}")


def _build_head(
    meta: Dict, coverage: Dict, eval_summary: Optional[Dict], models: List[str]
) -> str:
    r = meta["range"]
    total = meta["dataset_total"]
    n = coverage["total_problems"]

    eval_info = ""
    if eval_summary:
        eval_info = f"""
        <div class="stat-card">
            <div class="stat-num">{eval_summary.get("total_evaluations", 0)}</div>
            <div class="stat-label">Total Evaluations</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">${eval_summary.get("total_cost", 0):.4f}</div>
            <div class="stat-label">Total Cost</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{len(models)}</div>
            <div class="stat-label">Models</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Report: hard.json [{r[0]}-{r[1]})</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #0f1117; color: #e0e0e0; padding: 20px; line-height: 1.5; }}
h1 {{ font-size: 1.6rem; margin-bottom: 8px; color: #fff; }}
h2 {{ font-size: 1.2rem; margin: 24px 0 12px; color: #82aaff; border-bottom: 1px solid #333; padding-bottom: 6px; }}
h3 {{ font-size: 1rem; margin: 16px 0 8px; color: #c3e88d; }}
.subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 16px; }}
.stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.stat-card {{ background: #1a1d2e; border: 1px solid #333; border-radius: 8px;
              padding: 12px 16px; min-width: 120px; text-align: center; }}
.stat-num {{ font-size: 1.4rem; font-weight: 700; color: #82aaff; }}
.stat-label {{ font-size: 0.75rem; color: #888; margin-top: 2px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85rem; }}
th, td {{ padding: 6px 10px; border: 1px solid #333; text-align: center; }}
th {{ background: #1a1d2e; color: #82aaff; font-weight: 600; }}
tr:hover {{ background: rgba(130,170,255,0.05); }}
.ok {{ background: rgba(76,175,80,0.15); color: #81c784; font-weight: 600; }}
.fail {{ background: rgba(244,67,54,0.08); color: #666; }}
.c1 {{ background: rgba(76,175,80,0.2); color: #81c784; }}
.c2 {{ background: rgba(255,152,0,0.2); color: #ffb74d; }}
.c3 {{ background: rgba(244,67,54,0.2); color: #ef5350; }}
.card {{ background: #1a1d2e; border: 1px solid #333; border-radius: 8px;
         margin: 12px 0; overflow: hidden; }}
.card-header {{ padding: 10px 16px; cursor: pointer; display: flex;
                justify-content: space-between; align-items: center; }}
.card-header:hover {{ background: rgba(130,170,255,0.05); }}
.card-body {{ padding: 0 16px 16px; display: none; }}
.card.open .card-body {{ display: block; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
          font-size: 0.7rem; font-weight: 600; margin: 0 2px; }}
.badge-ok {{ background: #1b5e20; color: #81c784; }}
.badge-fail {{ background: #333; color: #666; }}
.badge-text {{ background: #1a237e; color: #82aaff; }}
.badge-md {{ background: #4a148c; color: #ce93d8; }}
.badge-json {{ background: #e65100; color: #ffb74d; }}
.badge-none {{ background: #333; color: #888; }}
pre {{ background: #0d1117; padding: 10px; border-radius: 6px; overflow-x: auto;
       font-size: 0.8rem; line-height: 1.4; white-space: pre-wrap; word-break: break-word;
       max-height: 300px; overflow-y: auto; margin: 6px 0; }}
.model-result {{ margin: 8px 0; padding: 8px; border-left: 3px solid #333; background: #151822; border-radius: 4px; }}
.model-result.c1 {{ border-left-color: {CASE_COLORS[1]}; }}
.model-result.c2 {{ border-left-color: {CASE_COLORS[2]}; }}
.model-result.c3 {{ border-left-color: {CASE_COLORS[3]}; }}
.bar {{ height: 20px; border-radius: 3px; display: inline-block; min-width: 2px; }}
.bar-container {{ display: flex; gap: 1px; margin: 4px 0; }}
.legend {{ display: flex; gap: 16px; font-size: 0.8rem; margin: 8px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
.tab-bar {{ display: flex; gap: 4px; margin: 8px 0; }}
.tab {{ padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 0.8rem;
        background: #222; color: #888; border: 1px solid #333; }}
.tab.active {{ background: #1a237e; color: #82aaff; border-color: #82aaff; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>
<h1>Batch Transformation & Evaluation Report</h1>
<p class="subtitle">hard.json [{r[0]}, {r[1]}) — {n}문제 (전체 {total}문제 중) | {meta["timestamp"]}</p>
<div class="stats">
    <div class="stat-card">
        <div class="stat-num">{total}</div>
        <div class="stat-label">hard.json 전체</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{n}</div>
        <div class="stat-label">분석 대상</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{coverage["transformable"]}</div>
        <div class="stat-label">Transformable</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{coverage["not_transformable"]}</div>
        <div class="stat-label">Not Transformable</div>
    </div>
    {eval_info}
</div>
"""


def _build_coverage_section(problems: List[Dict], coverage: Dict) -> str:
    """Build the transformation coverage table."""
    rows = []
    for p in problems:
        qid = p["question_id"]
        fmt = p["context_format"]
        fmt_badge = f'<span class="badge badge-{fmt}">{fmt}</span>'
        hc = "Y" if p.get("is_hardcoded_solution") else "N"

        cells = []
        for key in TYPE_KEYS:
            t = p["transformations"][key]
            if t["success"]:
                desc = _esc(t.get("description", ""), 60)
                cells.append(f'<td class="ok" title="{desc}">O</td>')
            else:
                reason = _esc(t.get("reason", "?"), 60)
                cells.append(f'<td class="fail" title="{reason}">X</td>')

        total_ok = sum(1 for k in TYPE_KEYS if p["transformations"][k]["success"])
        rows.append(
            f"<tr><td>{p['index']}</td><td>{qid}</td><td>{fmt_badge}</td>"
            f"<td>{hc}</td>{''.join(cells)}<td><b>{total_ok}</b>/5</td></tr>"
        )

    # Type summary row
    type_summary = []
    total = coverage["total_problems"]
    for key in TYPE_KEYS:
        info = coverage["by_type"][key]
        type_summary.append(f'<td class="ok">{info["success"]} ({info["rate"]}%)</td>')

    total_transformations = sum(coverage["by_type"][k]["success"] for k in TYPE_KEYS)
    type_summary_row = (
        f'<tr style="font-weight:700;background:#1a1d2e"><td colspan="4">Total Success</td>'
        f"{''.join(type_summary)}<td>{total_transformations}</td></tr>"
    )

    return f"""
<h2>Transformation Coverage</h2>
<table>
<tr><th>#</th><th>Question ID</th><th>Format</th><th>HC</th>
{"".join(f"<th>{k}</th>" for k in TYPE_KEYS)}<th>Total</th></tr>
{"".join(rows)}
{type_summary_row}
</table>
<div style="font-size:0.75rem;color:#666;margin-top:4px;">
O = 변환 성공 (hover for description), X = 변환 실패 (hover for reason), HC = Hardcoded Solution
</div>

<h3>Type Success Rate</h3>
<canvas id="coverageChart" height="60"></canvas>
<script>
new Chart(document.getElementById('coverageChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(TYPE_KEYS)},
        datasets: [{{
            label: 'Success Rate (%)',
            data: [{",".join(str(coverage["by_type"][k]["rate"]) for k in TYPE_KEYS)}],
            backgroundColor: ['#4caf50','#2196f3','#ff9800','#f44336','#9c27b0'],
            borderRadius: 4,
        }}]
    }},
    options: {{
        indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ max: 100, grid: {{ color: '#333' }}, ticks: {{ color: '#888' }} }},
            y: {{ grid: {{ display: false }}, ticks: {{ color: '#e0e0e0' }} }}
        }}
    }}
}});
</script>
"""


def _build_failure_analysis_section(problems: List[Dict], coverage: Dict) -> str:
    """Build the transformation failure analysis section."""
    total = coverage["total_problems"]

    # --- Per-type failure reason descriptions ---
    REASON_DESCRIPTIONS = {
        "no_context": "context 필드가 비어있거나 없음 (question에만 정보 포함)",
        "no_match": "변환 대상 패턴을 찾지 못함 (키워드/숫자/연도 매칭 실패)",
        "not_structured_format": "해당 변환은 구조화된 포맷(JSON/Markdown)만 지원",
        "no_year_pattern": "question과 context 모두에 연도(19xx/20xx) 패턴이 없음",
    }

    # --- Per-type analysis cards ---
    type_cards = []
    for key in TYPE_KEYS:
        info = coverage["by_type"][key]
        fail_count = info["fail"]
        if fail_count == 0:
            type_cards.append(f"""
            <div class="card open">
                <div class="card-header" style="cursor:default">
                    <div><b>{key}</b> — <span style="color:#81c784">전체 성공 ({info["success"]}/{total})</span></div>
                </div>
            </div>""")
            continue

        # Failure reason breakdown
        reasons = info.get("fail_reasons", {})
        reason_rows = []
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            desc = REASON_DESCRIPTIONS.get(reason, reason)
            pct = f"{count / total * 100:.0f}%"
            bar_w = max(2, int(count / total * 200))
            reason_rows.append(
                f'<tr><td style="text-align:left"><code>{reason}</code></td>'
                f"<td>{count}</td><td>{pct}</td>"
                f'<td style="text-align:left"><div class="bar" style="width:{bar_w}px;background:#f44336"></div></td>'
                f'<td style="text-align:left;font-size:0.8rem;color:#aaa">{desc}</td></tr>'
            )

        # Per-problem failure list
        fail_problems = []
        for p in problems:
            t = p["transformations"][key]
            if t["success"]:
                continue
            reason = t.get("reason", "?")
            desc = REASON_DESCRIPTIONS.get(reason, reason)
            fmt = p["context_format"]
            q_short = _esc(p["question"][:120])
            fail_problems.append(
                f"<tr>"
                f"<td>{p['index']}</td>"
                f"<td>{p['question_id']}</td>"
                f'<td><span class="badge badge-{fmt}">{fmt}</span></td>'
                f"<td><code>{reason}</code></td>"
                f'<td style="text-align:left;font-size:0.8rem">{q_short}</td>'
                f"</tr>"
            )

        type_cards.append(f"""
        <div class="card" onclick="this.classList.toggle('open')">
            <div class="card-header">
                <div>
                    <b>{key}</b> —
                    <span style="color:#81c784">{info["success"]} 성공</span> /
                    <span style="color:#ef5350">{fail_count} 실패</span>
                    ({info["rate"]}%)
                </div>
                <span style="color:#666;font-size:0.8rem">▼</span>
            </div>
            <div class="card-body" onclick="event.stopPropagation()">
                <h3>실패 사유 분포</h3>
                <table>
                <tr><th>Reason</th><th>Count</th><th>%</th><th>Bar</th><th>설명</th></tr>
                {"".join(reason_rows)}
                </table>

                <h3 style="margin-top:12px">실패 문제 목록 ({fail_count}개)</h3>
                <table>
                <tr><th>#</th><th>ID</th><th>Format</th><th>Reason</th><th>Question</th></tr>
                {"".join(fail_problems)}
                </table>
            </div>
        </div>""")

    # --- Format × Type matrix (sankey-style) ---
    formats = sorted(set(p["context_format"] for p in problems))
    matrix_rows = []
    for fmt in formats:
        fmt_problems = [p for p in problems if p["context_format"] == fmt]
        fmt_total = len(fmt_problems)
        cells = []
        for key in TYPE_KEYS:
            ok = sum(1 for p in fmt_problems if p["transformations"][key]["success"])
            fail = fmt_total - ok
            if fmt_total == 0:
                cells.append("<td>-</td>")
            elif fail == 0:
                cells.append(f'<td class="ok">{ok}/{fmt_total}</td>')
            elif ok == 0:
                cells.append(f'<td class="fail">{ok}/{fmt_total}</td>')
            else:
                cells.append(
                    f'<td style="background:rgba(255,152,0,0.15);color:#ffb74d">'
                    f"{ok}/{fmt_total}</td>"
                )
        matrix_rows.append(
            f'<tr><td><span class="badge badge-{fmt}">{fmt}</span> ({fmt_total})</td>'
            f"{''.join(cells)}</tr>"
        )

    return f"""
<h2>Transformation Failure Analysis</h2>
<p style="font-size:0.85rem;color:#aaa;margin-bottom:12px">
각 변환 타입별로 왜 특정 문제에 적용이 불가능한지 분석합니다.
실패 사유는 context 포맷, 데이터 구조, 패턴 매칭 조건에 의해 결정됩니다.
</p>

<h3>Format × Type 호환성 매트릭스</h3>
<table>
<tr><th>Format</th>{"".join(f"<th>{k}</th>" for k in TYPE_KEYS)}</tr>
{"".join(matrix_rows)}
</table>
<div style="font-size:0.75rem;color:#666;margin:4px 0 16px">
<span class="ok" style="padding:1px 6px">green</span> = 전체 성공,
<span style="padding:1px 6px;background:rgba(255,152,0,0.15);color:#ffb74d">amber</span> = 부분 성공,
<span class="fail" style="padding:1px 6px">gray</span> = 전체 실패
</div>

<h3>타입별 상세 실패 분석</h3>
<p style="font-size:0.8rem;color:#666">카드를 클릭하면 실패 사유와 해당 문제 목록을 볼 수 있습니다.</p>
{"".join(type_cards)}
"""


def _build_eval_summary_section(
    summary: Dict, models: List[str], eval_meta: Optional[Dict] = None
) -> str:
    """Build the evaluation summary section with heatmap and charts."""
    by_type = summary.get("by_type", {})
    by_model = summary.get("by_model", {})
    case_dist = summary.get("case_distribution", {})

    # Model × Type heatmap
    heatmap_rows = []
    for model in models:
        cells = []
        for t_type in TYPE_KEYS:
            # Find results for this model × type
            key = f"{model}_{t_type}"
            c1 = c2 = c3 = 0
            # We need to compute from results; use summary by_type as fallback
            cells.append("<td>-</td>")
        heatmap_rows.append(f"<tr><td>{model}</td>{''.join(cells)}</tr>")

    # Model summary table
    model_rows = []
    for model in models:
        if model not in by_model:
            continue
        s = by_model[model]
        total = s["total"]
        model_rows.append(
            f"<tr><td>{model}</td><td>{total}</td>"
            f'<td class="c1">{s["case1"]} ({_pct(s["case1"], total)})</td>'
            f'<td class="c2">{s["case2"]} ({_pct(s["case2"], total)})</td>'
            f'<td class="c3">{s["case3"]} ({_pct(s["case3"], total)})</td>'
            f"<td>${s.get('cost', 0):.4f}</td></tr>"
        )

    # Type summary table
    type_rows = []
    for t_type in TYPE_KEYS:
        if t_type not in by_type:
            continue
        s = by_type[t_type]
        total = s["total"]
        type_rows.append(
            f"<tr><td>{t_type}</td><td>{total}</td>"
            f'<td class="c1">{s["case1"]} ({_pct(s["case1"], total)})</td>'
            f'<td class="c2">{s["case2"]} ({_pct(s["case2"], total)})</td>'
            f'<td class="c3">{s["case3"]} ({_pct(s["case3"], total)})</td></tr>'
        )

    # Prompt display
    prompt_html = ""
    if eval_meta:
        strategy = eval_meta.get("prompt_strategy", "unknown")
        sys_prompt = _esc(eval_meta.get("system_prompt", ""))
        user_tpl = _esc(eval_meta.get("user_prompt_template", ""))
        prompt_html = f"""
<div class="card open" style="margin-bottom:16px">
    <div class="card-header" style="cursor:default">
        <div><b>Prompt Strategy: {strategy}</b></div>
    </div>
    <div class="card-body" style="display:block">
        <h3>System Prompt</h3>
        <pre>{sys_prompt}</pre>
        <h3 style="margin-top:12px">User Prompt Template</h3>
        <pre>{user_tpl}</pre>
        <p style="font-size:0.75rem;color:#666;margin-top:4px;">
        {{context}}와 {{question}}은 각 문제별로 대체됩니다.
        </p>
    </div>
</div>"""

    return f"""
<h2>Evaluation Summary</h2>
{prompt_html}
<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:{CASE_COLORS[1]}"></div> C1: Refused (metacognitive success)</div>
    <div class="legend-item"><div class="legend-dot" style="background:{CASE_COLORS[2]}"></div> C2: Wrong answer (transformation effective)</div>
    <div class="legend-item"><div class="legend-dot" style="background:{CASE_COLORS[3]}"></div> C3: Correct (memorization/reasoning)</div>
</div>

<h3>By Model</h3>
<table>
<tr><th>Model</th><th>Total</th><th>C1: Refused</th><th>C2: Wrong</th><th>C3: Correct</th><th>Cost</th></tr>
{"".join(model_rows)}
</table>

<h3>By Transformation Type</h3>
<table>
<tr><th>Type</th><th>Total</th><th>C1: Refused</th><th>C2: Wrong</th><th>C3: Correct</th></tr>
{"".join(type_rows)}
</table>

<h3>Case Distribution</h3>
<canvas id="caseChart" height="50"></canvas>
<script>
new Chart(document.getElementById('caseChart'), {{
    type: 'doughnut',
    data: {{
        labels: ['C1: Refused','C2: Wrong','C3: Correct'],
        datasets: [{{
            data: [{case_dist.get(1, 0)},{case_dist.get(2, 0)},{case_dist.get(3, 0)}],
            backgroundColor: ['{CASE_COLORS[1]}','{CASE_COLORS[2]}','{CASE_COLORS[3]}'],
        }}]
    }},
    options: {{
        plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
        cutout: '50%'
    }}
}});
</script>

<h3>Model Refusal Rate</h3>
<canvas id="modelChart" height="60"></canvas>
<script>
new Chart(document.getElementById('modelChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(models)},
        datasets: [
            {{ label: 'C1', data: [{",".join(str(by_model.get(m, {}).get("case1", 0)) for m in models)}], backgroundColor: '{CASE_COLORS[1]}' }},
            {{ label: 'C2', data: [{",".join(str(by_model.get(m, {}).get("case2", 0)) for m in models)}], backgroundColor: '{CASE_COLORS[2]}' }},
            {{ label: 'C3', data: [{",".join(str(by_model.get(m, {}).get("case3", 0)) for m in models)}], backgroundColor: '{CASE_COLORS[3]}' }}
        ]
    }},
    options: {{
        plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
        scales: {{
            x: {{ stacked: true, ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }},
            y: {{ stacked: true, ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
        }}
    }}
}});
</script>
"""


def _extract_nums(text: str) -> set:
    """Extract meaningful numbers from text (ignore single digits)."""
    import re as _re

    return {
        n.replace(",", "")
        for n in _re.findall(r"\$?([\d,]+\.?\d*)", text)
        if len(n.replace(",", "").replace(".", "")) > 1
    }


def _classify_c3_cause(
    question: str,
    ttype: str,
    t_data: Dict,
    context_original: str,
) -> str:
    """Classify root cause of a C3 (correct answer despite transformation)."""
    import re as _re

    q_nums = _extract_nums(question)

    if ttype == "EA-partial":
        desc = t_data.get("description", "")
        removed_nums = _extract_nums(desc)
        if removed_nums & q_nums:
            return "question_leaks_data"
        if removed_nums and all(n in ("10", "20", "100", "1") for n in removed_nums):
            return "removed_generic_number"
        return "removed_irrelevant_column"

    if ttype == "SA":
        ctx_trans = t_data.get("context_transformed", "")
        orig_sents = set(_re.split(r"(?<=[.!?])\s+", context_original.strip()))
        trans_sents = set(_re.split(r"(?<=[.!?])\s+", ctx_trans.strip()))
        removed_text = " ".join(orig_sents - trans_sents)
        removed_nums = _extract_nums(removed_text)
        if not removed_nums:
            return "removed_narrative_only"
        if removed_nums & q_nums:
            return "question_leaks_data"
        return "possible_memorization"

    if ttype == "IC":
        return "ic_ignored_contradiction"

    if ttype == "EA-full":
        return "removed_irrelevant_column"

    if ttype == "TA":
        return "ta_year_in_data"

    return "unknown"


C3_CAUSE_LABELS = {
    "question_leaks_data": (
        "Question에 핵심 데이터 중복",
        "제거된 값이 question 텍스트에도 포함되어 있어 모델이 context 없이도 풀 수 있음",
        "#e65100",
    ),
    "removed_irrelevant_column": (
        "비핵심 데이터 제거",
        "제거된 컬럼/값이 문제 풀이에 실제로 필요하지 않은 데이터였음",
        "#1565c0",
    ),
    "ic_ignored_contradiction": (
        "모순 무시 (IC 고유)",
        "모델이 삽입된 모순 데이터를 무시하고 원본 값으로 풀이 — 모순 탐지 실패",
        "#c62828",
    ),
    "possible_memorization": (
        "암기 가능성",
        "제거된 데이터가 question에 없지만 모델이 정답을 산출 — 학습 데이터 암기 또는 도메인 지식 활용 의심",
        "#6a1b9a",
    ),
    "removed_narrative_only": (
        "서술문만 제거",
        "제거된 문장에 수치 데이터가 없어 풀이에 영향 없음",
        "#2e7d32",
    ),
    "removed_generic_number": (
        "범용 숫자 제거",
        "제거된 숫자(10, 100 등)가 question에도 있거나 도메인 상식으로 유추 가능",
        "#f57f17",
    ),
    "ta_year_in_data": (
        "연도 데이터 잔존 (TA 고유)",
        "연도를 모호하게 바꿨지만 데이터 자체에 연도가 남아있어 유추 가능",
        "#00695c",
    ),
    "unknown": (
        "미분류",
        "원인 미상",
        "#616161",
    ),
}


def _build_c3_analysis_section(
    problems: List[Dict], eval_results: List[Dict], models: List[str]
) -> str:
    """Build C3 root cause analysis section."""
    from collections import Counter

    c3_results = [r for r in eval_results if r["case_type"] == 3]
    if not c3_results:
        return ""

    total_evals = len(eval_results)
    trans_index = {p["question_id"]: p for p in problems}

    # Classify each C3
    classified = []
    for r in c3_results:
        qid = r["question_id"]
        ttype = r["transformation_type"]
        p = trans_index.get(qid)
        if not p:
            continue
        t_data = p["transformations"].get(ttype, {})
        cause = _classify_c3_cause(p["question"], ttype, t_data, p["context_original"])
        classified.append(
            {
                "qid": qid,
                "ttype": ttype,
                "model": r["model"],
                "cause": cause,
                "answer": r.get("predicted_answer", "N/A"),
                "ground_truth": r.get("ground_truth", "N/A"),
            }
        )

    # Summary stats
    cause_counts = Counter(c["cause"] for c in classified)
    total_c3 = len(classified)

    # Cause distribution chart data
    cause_labels = []
    cause_values = []
    cause_colors = []
    for cause, count in cause_counts.most_common():
        info = C3_CAUSE_LABELS.get(cause, C3_CAUSE_LABELS["unknown"])
        cause_labels.append(info[0])
        cause_values.append(count)
        cause_colors.append(info[2])

    # By type breakdown
    by_type_cause: Dict[str, Dict[str, int]] = {}
    for c in classified:
        tt = c["ttype"]
        ca = c["cause"]
        if tt not in by_type_cause:
            by_type_cause[tt] = Counter()
        by_type_cause[tt][ca] += 1

    type_rows = []
    for ttype in TYPE_KEYS:
        if ttype not in by_type_cause:
            continue
        causes = by_type_cause[ttype]
        ttype_total = sum(causes.values())
        cause_parts = []
        for cause, count in causes.most_common():
            info = C3_CAUSE_LABELS.get(cause, C3_CAUSE_LABELS["unknown"])
            cause_parts.append(
                f'<span style="color:{info[2]}">{info[0]}: {count}</span>'
            )
        type_rows.append(
            f"<tr><td>{ttype}</td><td>{ttype_total}</td>"
            f'<td style="text-align:left">{" / ".join(cause_parts)}</td></tr>'
        )

    # By model breakdown
    by_model_cause: Dict[str, Counter] = {}
    for c in classified:
        m = c["model"]
        if m not in by_model_cause:
            by_model_cause[m] = Counter()
        by_model_cause[m][c["cause"]] += 1

    model_rows = []
    for model in models:
        if model not in by_model_cause:
            continue
        causes = by_model_cause[model]
        model_total = sum(causes.values())
        model_rows.append(
            f"<tr><td>{model}</td><td>{model_total}</td>"
            f"<td>{causes.get('question_leaks_data', 0)}</td>"
            f"<td>{causes.get('removed_irrelevant_column', 0)}</td>"
            f"<td>{causes.get('ic_ignored_contradiction', 0)}</td>"
            f"<td>{causes.get('possible_memorization', 0)}</td>"
            f"<td>{causes.get('removed_narrative_only', 0) + causes.get('removed_generic_number', 0) + causes.get('ta_year_in_data', 0)}</td>"
            f"</tr>"
        )

    # Per-problem detail cards
    c3_by_qid: Dict[str, List[Dict]] = {}
    for c in classified:
        if c["qid"] not in c3_by_qid:
            c3_by_qid[c["qid"]] = []
        c3_by_qid[c["qid"]].append(c)

    problem_cards = []
    for qid in sorted(c3_by_qid.keys()):
        items = c3_by_qid[qid]
        p = trans_index[qid]
        q = _esc(p["question"])
        gt = p.get("ground_truth", "?")

        # Group by ttype
        by_tt: Dict[str, List[Dict]] = {}
        for item in items:
            if item["ttype"] not in by_tt:
                by_tt[item["ttype"]] = []
            by_tt[item["ttype"]].append(item)

        detail_rows = []
        for ttype, tt_items in sorted(by_tt.items()):
            cause = tt_items[0]["cause"]
            info = C3_CAUSE_LABELS.get(cause, C3_CAUSE_LABELS["unknown"])
            model_list = ", ".join(i["model"] for i in tt_items)
            t_data = p["transformations"].get(ttype, {})
            desc = _esc(t_data.get("description", "N/A"))
            detail_rows.append(
                f"<tr><td>{ttype}</td>"
                f'<td style="color:{info[2]}">{info[0]}</td>'
                f"<td>{model_list}</td>"
                f"<td>{desc}</td></tr>"
            )

        problem_cards.append(f"""
        <div class="card" onclick="this.classList.toggle('open')">
            <div class="card-header">
                <div>
                    <b>{qid}</b> (GT: {gt}) —
                    <span style="color:#ef5350">{len(items)}건 C3</span>
                </div>
                <span style="color:#666;font-size:0.8rem">▼</span>
            </div>
            <div class="card-body" onclick="event.stopPropagation()">
                <p style="font-size:0.85rem;margin-bottom:8px">{q}</p>
                <table>
                <tr><th>Type</th><th>원인</th><th>C3 Models</th><th>변환 설명</th></tr>
                {"".join(detail_rows)}
                </table>
            </div>
        </div>""")

    # Cause legend
    legend_items = []
    for cause, (label, desc, color) in C3_CAUSE_LABELS.items():
        if cause == "unknown":
            continue
        legend_items.append(
            f'<div style="margin:4px 0;padding:6px 10px;border-left:3px solid {color};background:#151822;border-radius:4px">'
            f'<b style="color:{color}">{label}</b>'
            f'<span style="color:#aaa;font-size:0.8rem;margin-left:8px">{desc}</span></div>'
        )

    return f"""
<h2>C3 Analysis: 변환에도 정답을 맞힌 케이스 분석</h2>
<p style="font-size:0.85rem;color:#aaa;margin-bottom:12px">
    전체 {total_evals}건 평가 중 <b style="color:#ef5350">{total_c3}건 ({total_c3 / total_evals * 100:.1f}%)</b>이
    변환된 문제에서도 정답을 산출했습니다. 이 섹션에서는 그 원인을 분류합니다.
</p>

<h3>원인 분류 기준</h3>
{"".join(legend_items)}

<h3 style="margin-top:16px">원인별 분포</h3>
<canvas id="c3CauseChart" height="70"></canvas>
<script>
new Chart(document.getElementById('c3CauseChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(cause_labels, ensure_ascii=False)},
        datasets: [{{
            label: 'C3 건수',
            data: {json.dumps(cause_values)},
            backgroundColor: {json.dumps(cause_colors)},
            borderRadius: 4,
        }}]
    }},
    options: {{
        indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#888' }} }},
            y: {{ grid: {{ display: false }}, ticks: {{ color: '#e0e0e0', font: {{ size: 11 }} }} }}
        }}
    }}
}});
</script>

<h3>변환 타입별 C3 원인</h3>
<table>
<tr><th>Type</th><th>C3 건수</th><th>원인 분포</th></tr>
{"".join(type_rows)}
</table>

<h3>모델별 C3 원인</h3>
<table>
<tr><th>Model</th><th>C3 Total</th><th>Q에 데이터 중복</th><th>비핵심 제거</th><th>모순 무시</th><th>암기 가능성</th><th>기타</th></tr>
{"".join(model_rows)}
</table>

<h3>문제별 C3 상세 ({len(c3_by_qid)}개 문제)</h3>
<p style="font-size:0.8rem;color:#666">카드를 클릭하면 변환 타입별 원인과 해당 모델을 볼 수 있습니다.</p>
{"".join(problem_cards)}
"""


def _build_intensity_section(problems: List[Dict], eval_results: List[Dict]) -> str:
    """Build transformation intensity analysis section."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
    from transformation_intensity import compute_all_metrics

    trans_index = {p["question_id"]: p for p in problems}

    # Compute intensity for each transformation, merge with eval
    merged = []
    for r in eval_results:
        qid = r["question_id"]
        ttype = r["transformation_type"]
        p = trans_index.get(qid)
        if not p:
            continue
        t = p["transformations"].get(ttype, {})
        if not t.get("success"):
            continue

        ctx_orig = p["context_original"]
        if ttype == "TA":
            ctx_trans = ctx_orig
        else:
            ctx_trans = t.get("context_transformed", ctx_orig)

        metrics = compute_all_metrics(
            ctx_orig, ctx_trans, p["question"], p.get("python_solution", "")
        )
        merged.append(
            {
                "case": r["case_type"],
                "cvrr": metrics["cvrr"],
                "irr": metrics["irr"],
                "leakage": metrics["leakage_category"],
                "ttype": ttype,
                "model": r["model"],
            }
        )

    if not merged:
        return ""

    # CVRR bins
    bins = [
        ("0.0", lambda x: x == 0),
        ("0.01-0.2", lambda x: 0 < x <= 0.2),
        ("0.21-0.5", lambda x: 0.2 < x <= 0.5),
        ("0.51+", lambda x: x > 0.5),
    ]
    cvrr_rows = []
    for label, pred in bins:
        items = [m for m in merged if pred(m["cvrr"])]
        if not items:
            continue
        n = len(items)
        c1 = sum(1 for m in items if m["case"] == 1)
        c2 = sum(1 for m in items if m["case"] == 2)
        c3 = sum(1 for m in items if m["case"] == 3)
        c1p = c1 / n * 100
        c3p = c3 / n * 100
        bar_c1 = f'<div class="bar" style="width:{c1p * 1.5:.0f}px;background:{CASE_COLORS[1]}"></div>'
        bar_c3 = f'<div class="bar" style="width:{c3p * 1.5:.0f}px;background:{CASE_COLORS[3]}"></div>'
        cvrr_rows.append(
            f"<tr><td>{label}</td><td>{n}</td>"
            f'<td class="c1">{c1} ({c1p:.1f}%)</td>'
            f'<td class="c2">{c2} ({c2 / n * 100:.1f}%)</td>'
            f'<td class="c3">{c3} ({c3p:.1f}%)</td>'
            f"<td>{bar_c1}{bar_c3}</td></tr>"
        )

    # Leakage breakdown
    leak_rows = []
    for cat in ["none", "partial", "full"]:
        items = [m for m in merged if m["leakage"] == cat]
        if not items:
            continue
        n = len(items)
        c1 = sum(1 for m in items if m["case"] == 1)
        c3 = sum(1 for m in items if m["case"] == 3)
        leak_rows.append(
            f"<tr><td>{cat}</td><td>{n}</td>"
            f'<td class="c1">{c1} ({c1 / n * 100:.1f}%)</td>'
            f'<td class="c3">{c3} ({c3 / n * 100:.1f}%)</td></tr>'
        )

    # IRR by type
    irr_by_type = {}
    for m in merged:
        tt = m["ttype"]
        if tt not in irr_by_type:
            irr_by_type[tt] = []
        irr_by_type[tt].append(m["irr"])

    irr_rows = []
    for tt in TYPE_KEYS:
        vals = irr_by_type.get(tt, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        irr_rows.append(
            f"<tr><td>{tt}</td><td>{len(vals)}</td>"
            f"<td>{avg:.2%}</td><td>{min(vals):.2%}</td><td>{max(vals):.2%}</td></tr>"
        )

    # Type x CVRR -> Case
    type_cvrr_rows = []
    for tt in TYPE_KEYS:
        items = [m for m in merged if m["ttype"] == tt]
        if not items:
            continue
        low = [m for m in items if m["cvrr"] <= 0.1]
        high = [m for m in items if m["cvrr"] > 0.1]
        if not low and not high:
            continue

        def _rate(lst, case):
            return f"{sum(1 for m in lst if m['case'] == case) / max(len(lst), 1) * 100:.1f}%"

        type_cvrr_rows.append(
            f"<tr><td rowspan='2'>{tt}</td>"
            f"<td>Low (n={len(low)})</td><td>{_rate(low, 1)}</td><td>{_rate(low, 3)}</td></tr>"
            f"<tr><td>High (n={len(high)})</td><td>{_rate(high, 1)}</td><td>{_rate(high, 3)}</td></tr>"
        )

    return f"""
<h2>Transformation Intensity Analysis</h2>
<p style="font-size:0.85rem;color:#aaa;margin-bottom:12px">
    변환의 강도를 수치화하여 모델 응답(C1/C2/C3)과의 상관관계를 분석합니다.
    CVRR(Critical Value Removal Ratio)이 높을수록 풀이에 필수적인 데이터가 많이 제거된 것입니다.
</p>

<h3>CVRR vs Case Distribution</h3>
<p style="font-size:0.8rem;color:#666">CVRR = 풀이 필수 값 중 제거된 비율. 높을수록 변환이 효과적.</p>
<table>
<tr><th>CVRR Range</th><th>Count</th><th>C1 (Refused)</th><th>C2 (Wrong)</th><th>C3 (Correct)</th><th>Visual</th></tr>
{"".join(cvrr_rows)}
</table>

<h3>Question Data Leakage vs Case</h3>
<p style="font-size:0.8rem;color:#666">
    Leakage = context에서 제거된 값이 question 텍스트에도 존재하는 비율.
    Full leakage = 변환 무의미 (question만으로 풀이 가능).
</p>
<table>
<tr><th>Leakage</th><th>Count</th><th>C1 (Refused)</th><th>C3 (Correct)</th></tr>
{"".join(leak_rows)}
</table>

<h3>IRR (Information Removal Ratio) by Type</h3>
<table>
<tr><th>Type</th><th>Count</th><th>Mean IRR</th><th>Min</th><th>Max</th></tr>
{"".join(irr_rows)}
</table>

<h3>Type x CVRR Intensity -> Case</h3>
<p style="font-size:0.8rem;color:#666">Low CVRR (<=0.1) vs High CVRR (>0.1). 높은 CVRR에서 C1이 높아야 변환이 유효.</p>
<table>
<tr><th>Type</th><th>CVRR Level</th><th>C1 Rate</th><th>C3 Rate</th></tr>
{"".join(type_cvrr_rows)}
</table>
"""


def _build_problem_cards(
    problems: List[Dict], eval_index: Dict[str, List[Dict]], models: List[str]
) -> str:
    """Build collapsible cards for each problem."""
    cards = []
    for p in problems:
        qid = p["question_id"]
        idx = p["index"]
        fmt = p["context_format"]
        question = p["question"]
        context = p["context_original"]

        success_types = [k for k in TYPE_KEYS if p["transformations"][k]["success"]]
        badges = " ".join(
            f'<span class="badge badge-ok">{k}</span>'
            if p["transformations"][k]["success"]
            else f'<span class="badge badge-fail">{k}</span>'
            for k in TYPE_KEYS
        )

        # Build transformation details
        trans_details = []
        for key in TYPE_KEYS:
            t = p["transformations"][key]
            if not t["success"]:
                continue

            # Show transformed context/question
            if key == "TA":
                transformed = t.get("question_transformed", "")
                label = "Question (transformed)"
            else:
                transformed = t.get("context_transformed", "")
                label = "Context (transformed)"

            # Evaluation results for this transformation
            eval_key = f"{qid}_{key}"
            eval_results_html = ""
            if eval_key in eval_index:
                for er in eval_index[eval_key]:
                    c = er["case_type"]
                    c_class = f"c{c}"
                    ans = er.get("predicted_answer", "N/A")
                    resp_type = er.get("response_type", "?")
                    raw = _esc(er.get("raw_response", ""))
                    eval_results_html += f"""
                    <div class="model-result {c_class}">
                        <b>{er["model"]}</b> — {CASE_LABELS.get(c, "?")} | Response: {resp_type} | Answer: {_esc(str(ans))}
                        <pre style="max-height:none">{raw}</pre>
                    </div>"""

            trans_details.append(f"""
            <div style="margin:8px 0;padding:8px;background:#151822;border-radius:6px;">
                <h3 style="margin:0 0 6px">{key}: {_esc(t.get("description", ""), 100)}</h3>
                <details><summary style="font-size:0.8rem;color:#666;cursor:pointer">{label}</summary>
                <pre>{_esc(str(transformed), 2000)}</pre></details>
                {eval_results_html}
            </div>""")

        cards.append(f"""
        <div class="card" onclick="this.classList.toggle('open')">
            <div class="card-header">
                <div>
                    <b>[{idx}] {qid}</b>
                    <span class="badge badge-{fmt}">{fmt}</span>
                    {badges}
                </div>
                <span style="color:#666;font-size:0.8rem">{len(success_types)}/5 ▼</span>
            </div>
            <div class="card-body" onclick="event.stopPropagation()">
                <h3>Question</h3>
                <pre>{_esc(question, 1000)}</pre>
                <details><summary style="font-size:0.8rem;color:#666;cursor:pointer">Original Context</summary>
                <pre>{_esc(context, 3000)}</pre></details>
                <h3>Transformations</h3>
                {"".join(trans_details) if trans_details else '<p style="color:#666">No successful transformations</p>'}
            </div>
        </div>""")

    return f"""
<h2>Problem Details ({len(problems)} problems)</h2>
<p style="font-size:0.8rem;color:#666">Click cards to expand/collapse</p>
{"".join(cards)}
"""


def _build_footer() -> str:
    return """
<div style="margin-top:40px;padding:16px 0;border-top:1px solid #333;color:#666;font-size:0.75rem;text-align:center;">
    Generated by generate_batch_report.py | Metacognitive Evaluation Framework
</div>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate batch report HTML")
    parser.add_argument(
        "--transformations",
        type=str,
        default=None,
        help="batch_transformations JSON file",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default=None,
        help="batch_evaluation JSON file (optional)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/metacognitive",
        help="Results directory to auto-discover files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir

    # Auto-discover files if not specified
    trans_path = None
    eval_path = None

    if args.transformations:
        trans_path = Path(args.transformations)
    else:
        candidates = sorted(results_dir.glob("batch_transformations_*.json"))
        if candidates:
            trans_path = candidates[-1]  # most recent
            logger.info(f"Auto-discovered transformations: {trans_path.name}")

    if args.evaluation:
        eval_path = Path(args.evaluation)
    else:
        candidates = sorted(results_dir.glob("batch_evaluation_*.json"))
        if candidates:
            eval_path = candidates[-1]
            logger.info(f"Auto-discovered evaluation: {eval_path.name}")

    if trans_path is None or not trans_path.exists():
        logger.error(
            "Transformation file not found. Run run_batch_transformation.py first."
        )
        sys.exit(1)

    with open(trans_path, "r", encoding="utf-8") as f:
        trans_data = json.load(f)

    eval_data = None
    if eval_path and eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    else:
        logger.info("No evaluation data — generating transformation-only report.")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / "batch_report.html"

    generate_html(trans_data, eval_data, output_path)


if __name__ == "__main__":
    main()
