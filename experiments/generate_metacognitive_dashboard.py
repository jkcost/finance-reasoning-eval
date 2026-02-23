"""
Metacognitive Evaluation Dashboard Generator

Generates an interactive HTML dashboard from metacognitive experiment results.

Sections:
1. Progress Tracker - completed phases, evaluations, cumulative cost
2. Core Metrics Comparison - Refusal Accuracy / False Confidence by model
3. Heatmap - Model x Transformation Type x Prompt Strategy
4. Interesting Failure Cases - all-model failures, cheap-model wins, RAG deltas
5. Week-over-Week Trends - delta vs previous run

Usage:
    python experiments/generate_metacognitive_dashboard.py --results-dir experiments/results/metacognitive/
"""

import json
import argparse
import html as html_module
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from metacognitive_metrics import (
    MetacognitiveResult,
    MetacognitiveMetrics,
    compute_metrics,
    compute_metrics_by_dimension,
    compute_cross_phase_metrics,
)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_all_results(results_dir: Path) -> List[MetacognitiveResult]:
    """Load all phase result files from directory"""
    all_results = []
    for fp in sorted(results_dir.glob("phase_*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            all_results.append(MetacognitiveResult.from_dict(r))
    return all_results


def load_previous_results(results_dir: Path, current_timestamp: str) -> List[MetacognitiveResult]:
    """Load results from a previous run for week-over-week comparison"""
    files = sorted(results_dir.glob("phase_*.json"))
    prev_results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", "")
        if ts < current_timestamp:
            for r in data.get("results", []):
                prev_results.append(MetacognitiveResult.from_dict(r))
    return prev_results


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================


def load_prompt_catalogs(results_dir: Path) -> Dict[str, Dict[str, str]]:
    """Collect prompt catalogs from result JSON files.

    Falls back to importing PROMPT_SYSTEMS directly if no catalog
    is found in the stored results (backwards compatibility).
    """
    merged: Dict[str, Dict[str, str]] = {}
    for fp in sorted(results_dir.glob("phase_*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for strategy, methods in data.get("prompt_catalog", {}).items():
            if strategy not in merged:
                merged[strategy] = {}
            merged[strategy].update(methods)

    # Fallback: import from experiment module when legacy JSONs lack catalog
    if not merged:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from run_metacognitive_experiment import PROMPT_SYSTEMS
            merged = {
                strategy: dict(methods)
                for strategy, methods in PROMPT_SYSTEMS.items()
            }
        except ImportError:
            pass

    return merged


def build_prompt_reference_html(prompt_catalogs: Dict[str, Dict[str, str]]) -> str:
    """Build collapsible HTML cards showing actual prompt texts per strategy."""
    if not prompt_catalogs:
        return '<div style="color:#666;text-align:center;">No prompt data available.</div>'

    cards = ""
    for strategy in sorted(prompt_catalogs):
        methods = prompt_catalogs[strategy]
        method_blocks = ""
        for method_name in sorted(methods):
            prompt_text = html_module.escape(methods[method_name])
            method_blocks += f"""
                <div class="prompt-method-block">
                    <div class="prompt-method-name">{html_module.escape(method_name)}</div>
                    <pre class="prompt-text-pre">{prompt_text}</pre>
                </div>"""

        cards += f"""
        <div class="prompt-ref-card">
            <details>
                <summary class="prompt-ref-summary">
                    <span class="prompt-ref-name">{html_module.escape(strategy)}</span>
                    <span class="prompt-ref-count">{len(methods)} method(s)</span>
                </summary>
                <div class="prompt-ref-body">
                    {method_blocks}
                </div>
            </details>
        </div>"""

    return cards


TRANSFORMATION_DESIGN_REFERENCE = [
    {
        "type": "Type 1: Information Removal",
        "short": "정보 제거 + 명시적 마커",
        "description": "컨텍스트에서 특정 데이터를 제거하고 [DATA MISSING] 또는 N/A 마커를 삽입합니다.",
        "domain_specificity": "높음",
        "detection_difficulty": "낮음",
        "before": '{"Revenue": {"2022": 500, "2023": 600}}',
        "after": '{"Revenue": {"2023": 600}}  (2022 entry removed)',
    },
    {
        "type": "Type 2: Table Column Removal",
        "short": "컬럼/키 단위 구조적 제거",
        "description": "재무제표의 전체 컬럼 또는 JSON 키를 삭제합니다. 마커 없이 구조 자체가 변경됩니다.",
        "domain_specificity": "높음",
        "detection_difficulty": "중간",
        "before": '{"Revenue": {...}, "Net Income": {...}}',
        "after": '{"Net Income": {...}}  (Revenue key deleted)',
    },
    {
        "type": "Type 3: Ambiguous Time Period",
        "short": "시간 참조 모호화",
        "description": "질문의 구체적 연도를 \"the end of the period\"로 대체하여 시간 참조를 모호하게 만듭니다.",
        "domain_specificity": "높음",
        "detection_difficulty": "중간",
        "before": "What was the revenue growth in 2023?",
        "after": "What was the revenue growth at the end of the period?",
    },
    {
        "type": "Type 4: Critical Data Removal",
        "short": "무표시 전면 제거",
        "description": "핵심 수치를 마커 없이 제거합니다. 컨텍스트 구조는 유지되지만 데이터가 비어 있습니다.",
        "domain_specificity": "중간",
        "detection_difficulty": "높음",
        "before": '{"Revenue": {"2022": 500, "2023": 600}}',
        "after": '{"Revenue": {}}  (all values silently removed)',
    },
    {
        "type": "Type 5: Contradictory Information",
        "short": "모순 정보 삽입 (1.5배)",
        "description": "동일 데이터 포인트에 1.5배 모순값을 삽입합니다. 누락이 아닌 충돌 탐지 능력을 테스트합니다.",
        "domain_specificity": "낮음",
        "detection_difficulty": "매우 높음",
        "before": '{"Revenue": {"2023": 600}}',
        "after": '{"Revenue": {"2023": 600, "2023_conflicting_report": 900}}',
    },
]


def compute_strategy_type_matrix(
    results: List[MetacognitiveResult],
) -> Dict[str, Dict[str, float]]:
    """Compute Strategy × Transformation Type refusal rate matrix.

    Returns:
        {strategy: {transformation_type: refusal_rate_pct, ...}, ...}
    """
    unsolvable = [r for r in results if r.transformation_type != "original"]
    matrix: Dict[str, Dict[str, Dict[str, int]]] = {}

    for r in unsolvable:
        strat = r.prompt_strategy
        ttype = r.transformation_type
        matrix.setdefault(strat, {}).setdefault(ttype, {"refused": 0, "total": 0})
        matrix[strat][ttype]["total"] += 1
        if r.response_type == "refused":
            matrix[strat][ttype]["refused"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for strat, types in matrix.items():
        result[strat] = {}
        for ttype, counts in types.items():
            rate = counts["refused"] / counts["total"] * 100 if counts["total"] > 0 else 0
            result[strat][ttype] = round(rate, 1)

    return result


def compute_type_difficulty_ranking(
    results: List[MetacognitiveResult],
) -> List[Dict[str, Any]]:
    """Compute overall detection difficulty ranking by transformation type.

    Returns list sorted by refusal rate (ascending = harder to detect):
        [{"type": ..., "refusal_rate": ..., "total": ..., "refused": ...}, ...]
    """
    unsolvable = [r for r in results if r.transformation_type != "original"]
    counts: Dict[str, Dict[str, int]] = {}

    for r in unsolvable:
        ttype = r.transformation_type
        counts.setdefault(ttype, {"refused": 0, "total": 0})
        counts[ttype]["total"] += 1
        if r.response_type == "refused":
            counts[ttype]["refused"] += 1

    ranking = []
    for ttype, c in counts.items():
        rate = c["refused"] / c["total"] * 100 if c["total"] > 0 else 0
        ranking.append({
            "type": ttype,
            "refusal_rate": round(rate, 1),
            "total": c["total"],
            "refused": c["refused"],
        })

    return sorted(ranking, key=lambda x: x["refusal_rate"])


def find_interesting_cases(results: List[MetacognitiveResult]) -> Dict[str, List[Dict]]:
    """Find interesting failure/success cases for the dashboard"""
    unsolvable = [r for r in results if r.transformation_type != "original"]
    cases: Dict[str, List[Dict]] = {
        "all_model_failures": [],
        "cheap_model_wins": [],
        "rag_deltas": [],
    }

    # Group by example_id + transformation_type
    groups: Dict[str, List[MetacognitiveResult]] = {}
    for r in unsolvable:
        key = f"{r.example_id}|{r.transformation_type}"
        groups.setdefault(key, []).append(r)

    for key, group in groups.items():
        all_confident = all(r.response_type == "confident" for r in group)
        if all_confident and len(group) >= 2:
            cases["all_model_failures"].append({
                "example_id": group[0].example_id,
                "transformation_type": group[0].transformation_type,
                "models": [r.model_name for r in group],
                "note": "All models answered confidently on unsolvable problem",
            })

        # cheap model wins: a cheaper model refused but expensive one didn't
        refused = [r for r in group if r.response_type == "refused"]
        confident = [r for r in group if r.response_type == "confident"]
        for ref_r in refused:
            for conf_r in confident:
                if ref_r.cost_usd < conf_r.cost_usd:
                    cases["cheap_model_wins"].append({
                        "example_id": ref_r.example_id,
                        "transformation_type": ref_r.transformation_type,
                        "cheap_model": ref_r.model_name,
                        "expensive_model": conf_r.model_name,
                    })

    # RAG deltas
    rag_on = [r for r in unsolvable if r.rag_enabled]
    rag_off = [r for r in unsolvable if not r.rag_enabled]
    if rag_on and rag_off:
        rag_on_metrics = compute_metrics(rag_on)
        rag_off_metrics = compute_metrics(rag_off)
        for model_name in set(rag_on_metrics) & set(rag_off_metrics):
            delta = (
                rag_on_metrics[model_name].refusal_accuracy
                - rag_off_metrics[model_name].refusal_accuracy
            )
            if abs(delta) > 0.05:
                cases["rag_deltas"].append({
                    "model": model_name,
                    "rag_on_refusal_acc": rag_on_metrics[model_name].refusal_accuracy,
                    "rag_off_refusal_acc": rag_off_metrics[model_name].refusal_accuracy,
                    "delta": delta,
                })

    return cases


def build_detailed_cases_html(all_results: List[MetacognitiveResult]) -> str:
    """Build HTML for detailed per-problem case analysis section.

    Groups results by (example_id, transformation_type) and shows:
    - Original question and context
    - What was transformed and how
    - Each model's response, classification, and reasoning
    """
    unsolvable = [r for r in all_results if r.transformation_type != "original"]
    if not unsolvable:
        return '<div style="color:#666;text-align:center;">Run Phase B to see detailed case analysis.</div>'

    # Group by (example_id, transformation_type)
    groups: Dict[str, List[MetacognitiveResult]] = {}
    for r in unsolvable:
        key = f"{r.example_id}|{r.transformation_type}"
        groups.setdefault(key, []).append(r)

    cards_html = ""
    for idx, (key, group) in enumerate(sorted(groups.items())):
        first = group[0]
        eid = first.example_id
        trans_type = first.transformation_type
        trans_desc = first.transformation_description or "N/A"
        question = first.question or "N/A"
        gt = first.ground_truth
        ctx_orig = first.context_original or ""
        ctx_trans = first.context_transformed or ""

        # Escape HTML
        question_safe = html_module.escape(question)
        trans_desc_safe = html_module.escape(trans_desc)
        ctx_orig_safe = html_module.escape(ctx_orig)
        ctx_trans_safe = html_module.escape(ctx_trans)

        # Build model response rows
        model_rows = ""
        for r in sorted(group, key=lambda x: (x.model_name, x.prompt_strategy)):
            response_color = {
                "refused": "#4caf50",
                "caveat": "#ff9800",
                "confident": "#f44336",
                "error": "#9e9e9e",
            }.get(r.response_type, "#666")

            response_label = {
                "refused": "REFUSED (correct)",
                "caveat": "CAVEAT",
                "confident": "CONFIDENT (wrong)",
                "error": "ERROR",
            }.get(r.response_type, r.response_type)

            # Truncate raw response for display
            raw_resp = r.raw_response or ""
            raw_resp_display = html_module.escape(raw_resp[:400])
            if len(raw_resp) > 400:
                raw_resp_display += "..."

            halluc_info = ""
            if r.hallucinated_values:
                halluc_info = (
                    f'<div class="halluc-tag">Hallucinated: '
                    f'{html_module.escape(", ".join(r.hallucinated_values[:5]))}</div>'
                )

            model_rows += f"""
            <div class="model-response">
                <div class="model-response-header">
                    <span class="model-name-tag">{html_module.escape(r.model_name)}</span>
                    <span class="strategy-tag">{html_module.escape(r.prompt_strategy)}</span>
                    <span class="response-badge" style="background:rgba({_hex_to_rgb(response_color)},0.2);color:{response_color};">
                        {response_label}
                    </span>
                    <span class="cost-tag">${r.cost_usd:.4f}</span>
                </div>
                <div class="model-answer">
                    Answer: <code>{html_module.escape(str(r.predicted_answer))}</code>
                </div>
                {halluc_info}
                <details class="response-details">
                    <summary>Raw Response</summary>
                    <pre class="response-pre">{raw_resp_display}</pre>
                </details>
            </div>"""

        # Context diff display
        context_section = ""
        if ctx_orig_safe or ctx_trans_safe:
            context_section = f"""
            <div class="context-diff">
                <div class="context-panel">
                    <div class="context-label">Original Context</div>
                    <pre class="context-pre">{ctx_orig_safe}</pre>
                </div>
                <div class="context-panel">
                    <div class="context-label">Transformed Context</div>
                    <pre class="context-pre">{ctx_trans_safe}</pre>
                </div>
            </div>"""

        cards_html += f"""
        <div class="detail-card">
            <details {'open' if idx < 3 else ''}>
                <summary class="detail-summary">
                    <span class="detail-id">{html_module.escape(eid)}</span>
                    <span class="detail-trans-type">{html_module.escape(trans_type)}</span>
                    <span class="detail-gt">GT: {html_module.escape(str(gt))}</span>
                </summary>
                <div class="detail-body">
                    <div class="detail-question">
                        <strong>Question:</strong> {question_safe}
                    </div>
                    <div class="detail-transform">
                        <strong>Transformation:</strong> {trans_desc_safe}
                    </div>
                    {context_section}
                    <div class="model-responses-section">
                        <strong>Model Responses:</strong>
                        {model_rows}
                    </div>
                </div>
            </details>
        </div>"""

    return cards_html


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to comma-separated RGB for rgba()"""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ============================================================================
# HTML GENERATION
# ============================================================================


def generate_dashboard(results_dir: Path, output_path: Optional[Path] = None) -> Path:
    """Generate the HTML dashboard"""
    all_results = load_all_results(results_dir)

    if not all_results:
        print("[ERROR] No results found.")
        return results_dir / "dashboard.html"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    current_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Split results
    baseline = [r for r in all_results if r.transformation_type == "original"]
    unsolvable = [r for r in all_results if r.transformation_type != "original"]

    # Metrics — use cross-phase when both baseline and unsolvable exist
    if baseline and unsolvable:
        model_metrics = compute_cross_phase_metrics(baseline, unsolvable)
    else:
        model_metrics = compute_metrics(unsolvable) if unsolvable else {}
    strategy_metrics = compute_metrics_by_dimension(unsolvable, "prompt_strategy") if unsolvable else {}
    transform_metrics = compute_metrics_by_dimension(unsolvable, "transformation_type") if unsolvable else {}

    # Strategy × Type matrix
    strategy_type_matrix = compute_strategy_type_matrix(all_results)
    type_difficulty = compute_type_difficulty_ranking(all_results)

    # Interesting cases
    cases = find_interesting_cases(all_results)

    # Build detailed case analysis HTML
    detailed_cases_html = build_detailed_cases_html(all_results)

    # Load prompt catalogs for reference section
    prompt_catalogs = load_prompt_catalogs(results_dir)
    prompt_reference_html = build_prompt_reference_html(prompt_catalogs)

    # Previous results for trends
    prev_results = load_previous_results(results_dir, current_ts)
    prev_metrics = compute_metrics(
        [r for r in prev_results if r.transformation_type != "original"]
    ) if prev_results else {}

    # Phase tracking
    phases_done = set()
    for r in all_results:
        if r.transformation_type == "original":
            phases_done.add("A")
        elif r.rag_enabled:
            phases_done.add("C")
        else:
            phases_done.add("B")

    total_cost = sum(r.cost_usd for r in all_results)
    total_evals = len(all_results)
    models = sorted({r.model_name for r in all_results})

    # Prepare chart data
    models_with_metrics = [m for m in models if m in model_metrics]
    model_names_js = json.dumps(models_with_metrics)
    refusal_accs_js = json.dumps(
        [round(model_metrics[m].refusal_recall * 100, 1) for m in models_with_metrics]
    )
    false_conf_js = json.dumps(
        [round(model_metrics[m].false_confidence_rate * 100, 1) for m in models_with_metrics]
    )
    mc_scores_js = json.dumps(
        [round(model_metrics[m].mc_score * 100, 1) for m in models_with_metrics]
    )
    refusal_prec_js = json.dumps(
        [round(model_metrics[m].refusal_precision * 100, 1) for m in models_with_metrics]
    )
    refusal_f1_js = json.dumps(
        [round(model_metrics[m].refusal_f1 * 100, 1) for m in models_with_metrics]
    )
    ocr_js = json.dumps(
        [round(model_metrics[m].over_conservatism_rate * 100, 1) for m in models_with_metrics]
    )

    # Baseline accuracy
    baseline_by_model: Dict[str, Dict] = {}
    for r in baseline:
        if r.model_name not in baseline_by_model:
            baseline_by_model[r.model_name] = {"correct": 0, "total": 0}
        baseline_by_model[r.model_name]["total"] += 1
        if r.is_original_correct:
            baseline_by_model[r.model_name]["correct"] += 1

    # Heatmap data: model x transformation_type
    heatmap_transforms = sorted({r.transformation_type for r in unsolvable})
    heatmap_data = []
    for m_name in models:
        row = []
        for t_type in heatmap_transforms:
            matching = [r for r in unsolvable if r.model_name == m_name and r.transformation_type == t_type]
            if matching:
                refused = sum(1 for r in matching if r.response_type == "refused")
                rate = refused / len(matching) * 100
                row.append(round(rate, 1))
            else:
                row.append(None)
        heatmap_data.append(row)

    heatmap_transforms_js = json.dumps([t[:30] for t in heatmap_transforms])
    heatmap_models_js = json.dumps(models)
    heatmap_data_js = json.dumps(heatmap_data)

    # Trends
    trend_rows = ""
    for m_name in models:
        curr = model_metrics.get(m_name)
        prev = prev_metrics.get(m_name)
        if curr:
            curr_mc = curr.mc_score
            if prev:
                delta = curr_mc - prev.mc_score
                delta_str = f'<span style="color:{"#4caf50" if delta >= 0 else "#f44336"}">{delta:+.3f}</span>'
            else:
                delta_str = '<span style="color:#888">N/A</span>'
            trend_rows += f"""
            <tr>
                <td>{html_module.escape(m_name)}</td>
                <td>{curr.refusal_recall:.1%}</td>
                <td>{curr.refusal_precision:.1%}</td>
                <td>{curr.refusal_f1:.3f}</td>
                <td>{curr.over_conservatism_rate:.1%}</td>
                <td>{curr.mc_score:.3f}</td>
                <td>{delta_str}</td>
                <td>${curr.total_cost_usd:.4f}</td>
            </tr>"""

    # Interesting cases HTML
    failure_cases_html = ""
    for case in cases.get("all_model_failures", [])[:5]:
        failure_cases_html += f"""
        <div class="case-card failure">
            <div class="case-badge">ALL FAILED</div>
            <div class="case-id">{html_module.escape(str(case['example_id']))}</div>
            <div class="case-detail">{html_module.escape(str(case['transformation_type']))}</div>
            <div class="case-models">Models: {html_module.escape(', '.join(case['models']))}</div>
        </div>"""

    for case in cases.get("cheap_model_wins", [])[:5]:
        failure_cases_html += f"""
        <div class="case-card success">
            <div class="case-badge">CHEAP WIN</div>
            <div class="case-id">{html_module.escape(str(case['example_id']))}</div>
            <div class="case-detail">{html_module.escape(str(case['cheap_model']))} beat {html_module.escape(str(case['expensive_model']))}</div>
        </div>"""

    for case in cases.get("rag_deltas", [])[:5]:
        direction = "positive" if case["delta"] > 0 else "negative"
        failure_cases_html += f"""
        <div class="case-card {'success' if case['delta'] > 0 else 'warning'}">
            <div class="case-badge">RAG {'+' if case['delta'] > 0 else '-'}</div>
            <div class="case-id">{html_module.escape(str(case['model']))}</div>
            <div class="case-detail">RefAcc: {case['rag_off_refusal_acc']:.1%} → {case['rag_on_refusal_acc']:.1%} ({case['delta']:+.1%})</div>
        </div>"""

    if not failure_cases_html:
        failure_cases_html = '<div class="case-card"><div class="case-detail">No interesting cases found yet. Run more experiments.</div></div>'

    # Strategy comparison table
    strategy_rows = ""
    for strat, strat_models in sorted(strategy_metrics.items()):
        for m_name, m in sorted(strat_models.items()):
            strategy_rows += f"""
            <tr>
                <td>{html_module.escape(strat)}</td>
                <td>{html_module.escape(m_name)}</td>
                <td>{m.refusal_recall:.1%}</td>
                <td>{m.refusal_f1:.3f}</td>
                <td>{m.mc_score:.3f}</td>
            </tr>"""

    # Baseline table
    baseline_rows = ""
    for m_name in models:
        bm = baseline_by_model.get(m_name, {"correct": 0, "total": 0})
        acc = bm["correct"] / bm["total"] * 100 if bm["total"] > 0 else 0
        baseline_rows += f"""
        <tr>
            <td>{html_module.escape(m_name)}</td>
            <td>{bm['correct']}/{bm['total']}</td>
            <td>{acc:.1f}%</td>
        </tr>"""

    # --- 3A: Strategy × Type Heatmap table (CSS gradient) ---
    strat_type_strategies = sorted(strategy_type_matrix.keys())
    all_trans_types = sorted({
        tt for strat_types in strategy_type_matrix.values()
        for tt in strat_types
    })
    heatmap_table_html = ""
    if strat_type_strategies and all_trans_types:
        header_cells = "".join(
            f"<th>{html_module.escape(s)}</th>" for s in strat_type_strategies
        )
        heatmap_table_html = f"""
        <table class="heatmap-table">
            <thead><tr><th>Transformation</th>{header_cells}</tr></thead>
            <tbody>"""
        for tt in all_trans_types:
            row_cells = ""
            for strat in strat_type_strategies:
                val = strategy_type_matrix.get(strat, {}).get(tt)
                if val is not None:
                    r_color = int(255 * (1 - val / 100))
                    g_color = int(255 * (val / 100))
                    bg = f"rgba({r_color},{g_color},0,0.25)"
                    row_cells += f'<td style="background:{bg};text-align:center;font-weight:bold;">{val:.0f}%</td>'
                else:
                    row_cells += '<td style="text-align:center;color:#555;">-</td>'
            tt_short = tt[:35] if len(tt) > 35 else tt
            heatmap_table_html += f"<tr><td>{html_module.escape(tt_short)}</td>{row_cells}</tr>"
        heatmap_table_html += "</tbody></table>"
    else:
        heatmap_table_html = '<div style="color:#666;text-align:center;">Run Phase B with multiple strategies to see heatmap.</div>'

    # --- 3B: Strategy MC Score chart data ---
    strat_mc_labels = []
    strat_mc_datasets: Dict[str, List[Optional[float]]] = {}
    for strat in strat_type_strategies:
        strat_mc_labels.append(strat)
    for m_name in models:
        strat_mc_datasets[m_name] = []
        for strat in strat_type_strategies:
            sm = strategy_metrics.get(strat, {}).get(m_name)
            strat_mc_datasets[m_name].append(
                round(sm.mc_score * 100, 1) if sm else 0
            )
    strat_mc_labels_js = json.dumps(strat_mc_labels)

    strat_mc_chart_datasets_js_parts = []
    chart_colors = [
        "rgba(0,212,255,0.7)", "rgba(76,175,80,0.7)", "rgba(255,152,0,0.7)",
        "rgba(156,39,176,0.7)", "rgba(244,67,54,0.7)", "rgba(255,235,59,0.7)",
    ]
    for idx, (m_name, vals) in enumerate(strat_mc_datasets.items()):
        color = chart_colors[idx % len(chart_colors)]
        strat_mc_chart_datasets_js_parts.append(
            f"{{label:'{m_name}',data:{json.dumps(vals)},backgroundColor:'{color}',borderRadius:4}}"
        )
    strat_mc_chart_datasets_js = "[" + ",".join(strat_mc_chart_datasets_js_parts) + "]"

    # --- 3C: Type difficulty ranking data ---
    diff_labels_js = json.dumps([d["type"][:30] for d in type_difficulty])
    diff_values_js = json.dumps([d["refusal_rate"] for d in type_difficulty])
    diff_counts_js = json.dumps([f"{d['refused']}/{d['total']}" for d in type_difficulty])

    # --- 3D: Transformation design reference cards ---
    design_ref_html = ""
    for ref in TRANSFORMATION_DESIGN_REFERENCE:
        domain_color = {"높음": "#4caf50", "중간": "#ff9800", "낮음": "#f44336"}.get(
            ref["domain_specificity"], "#888"
        )
        diff_color = {"낮음": "#4caf50", "중간": "#ff9800", "높음": "#f44336", "매우 높음": "#d32f2f"}.get(
            ref["detection_difficulty"], "#888"
        )
        design_ref_html += f"""
        <div class="prompt-ref-card">
            <details>
                <summary class="prompt-ref-summary">
                    <span class="prompt-ref-name">{html_module.escape(ref['type'])}</span>
                    <span style="color:{domain_color};font-size:0.8em;">도메인: {html_module.escape(ref['domain_specificity'])}</span>
                    <span style="color:{diff_color};font-size:0.8em;">탐지: {html_module.escape(ref['detection_difficulty'])}</span>
                </summary>
                <div class="prompt-ref-body">
                    <p style="margin-bottom:8px;">{html_module.escape(ref['description'])}</p>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                        <div class="context-panel">
                            <div class="context-label">Before</div>
                            <pre class="context-pre">{html_module.escape(ref['before'])}</pre>
                        </div>
                        <div class="context-panel">
                            <div class="context-label">After</div>
                            <pre class="context-pre">{html_module.escape(ref['after'])}</pre>
                        </div>
                    </div>
                </div>
            </details>
        </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metacognitive Evaluation Dashboard - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 5px; color: #00d4ff; font-size: 2em; }}
        h2 {{ color: #00d4ff; margin: 25px 0 15px; font-size: 1.3em; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 25px; font-size: 0.95em; }}

        /* Summary cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 25px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }}
        .summary-value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
        .summary-label {{ color: #888; margin-top: 4px; font-size: 0.85em; }}

        /* Phase tracker */
        .phase-tracker {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 25px;
        }}
        .phase-badge {{
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .phase-done {{ background: rgba(76,175,80,0.3); color: #4caf50; }}
        .phase-pending {{ background: rgba(255,255,255,0.05); color: #666; }}

        /* Charts */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
        }}
        .chart-title {{ color: #00d4ff; margin-bottom: 12px; font-size: 1.05em; }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        th {{ background: rgba(0,212,255,0.1); color: #00d4ff; font-size: 0.9em; }}
        td {{ font-size: 0.88em; }}

        /* Case cards */
        .cases-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
            margin-bottom: 25px;
        }}
        .case-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 14px;
            border-left: 3px solid #666;
        }}
        .case-card.failure {{ border-left-color: #f44336; }}
        .case-card.success {{ border-left-color: #4caf50; }}
        .case-card.warning {{ border-left-color: #ff9800; }}
        .case-badge {{
            font-size: 0.75em;
            font-weight: bold;
            color: #00d4ff;
            margin-bottom: 5px;
        }}
        .case-id {{ font-weight: bold; margin-bottom: 3px; }}
        .case-detail {{ color: #aaa; font-size: 0.85em; }}
        .case-models {{ color: #888; font-size: 0.8em; margin-top: 4px; }}

        /* Heatmap */
        .heatmap-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            overflow-x: auto;
        }}
        .heatmap {{ border-collapse: collapse; }}
        .heatmap td, .heatmap th {{ padding: 10px 15px; text-align: center; font-size: 0.85em; }}
        .heatmap th {{ color: #00d4ff; }}

        /* Prompt Reference */
        .prompt-ref-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        .prompt-ref-summary {{
            padding: 12px 18px;
            cursor: pointer;
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .prompt-ref-summary:hover {{ background: rgba(255,255,255,0.03); }}
        .prompt-ref-name {{
            font-weight: bold;
            color: #ce93d8;
            font-size: 1em;
        }}
        .prompt-ref-count {{
            color: #888;
            font-size: 0.8em;
        }}
        .prompt-ref-body {{
            padding: 0 18px 18px;
        }}
        .prompt-method-block {{
            margin-top: 10px;
        }}
        .prompt-method-name {{
            color: #00d4ff;
            font-size: 0.85em;
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .prompt-text-pre {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.78em;
            white-space: pre-wrap;
            word-break: break-word;
            color: #ccc;
            background: rgba(0,0,0,0.25);
            padding: 10px 12px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
            margin: 0;
        }}

        /* Heatmap table (Strategy x Type) */
        .heatmap-table {{ border-collapse: collapse; }}
        .heatmap-table td, .heatmap-table th {{
            padding: 10px 16px;
            text-align: center;
            font-size: 0.88em;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .heatmap-table th {{ color: #00d4ff; background: rgba(0,212,255,0.08); }}

        .section {{ margin-bottom: 30px; }}
        .footer {{ text-align: center; color: #555; font-size: 0.8em; margin-top: 30px; padding: 15px; }}

        /* Detailed Case Analysis */
        .detail-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            margin-bottom: 12px;
            overflow: hidden;
        }}
        .detail-summary {{
            padding: 14px 18px;
            cursor: pointer;
            display: flex;
            gap: 15px;
            align-items: center;
            font-size: 0.95em;
        }}
        .detail-summary:hover {{ background: rgba(255,255,255,0.03); }}
        .detail-id {{ color: #00d4ff; font-weight: bold; min-width: 100px; }}
        .detail-trans-type {{
            background: rgba(255,152,0,0.15);
            color: #ff9800;
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.85em;
        }}
        .detail-gt {{ color: #888; margin-left: auto; font-size: 0.85em; }}
        .detail-body {{ padding: 0 18px 18px; }}
        .detail-question {{ margin-bottom: 10px; line-height: 1.5; }}
        .detail-transform {{
            background: rgba(255,152,0,0.08);
            border-left: 3px solid #ff9800;
            padding: 8px 12px;
            margin-bottom: 12px;
            font-size: 0.9em;
            color: #ffcc80;
        }}
        .context-diff {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 14px;
        }}
        .context-panel {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 10px;
        }}
        .context-label {{
            color: #00d4ff;
            font-size: 0.8em;
            font-weight: bold;
            margin-bottom: 6px;
        }}
        .context-pre {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.78em;
            white-space: pre-wrap;
            word-break: break-all;
            color: #ccc;
            max-height: 200px;
            overflow-y: auto;
            margin: 0;
        }}
        .model-responses-section {{ margin-top: 10px; }}
        .model-response {{
            background: rgba(0,0,0,0.15);
            border-radius: 8px;
            padding: 10px 12px;
            margin-top: 8px;
        }}
        .model-response-header {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 6px;
        }}
        .model-name-tag {{
            font-weight: bold;
            color: #e0e0e0;
            min-width: 140px;
        }}
        .response-badge {{
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .strategy-tag {{
            background: rgba(156,39,176,0.15);
            color: #ce93d8;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.75em;
        }}
        .cost-tag {{ color: #666; font-size: 0.8em; margin-left: auto; }}
        .model-answer {{ font-size: 0.88em; margin-bottom: 4px; }}
        .model-answer code {{
            background: rgba(0,212,255,0.1);
            padding: 1px 6px;
            border-radius: 4px;
            color: #00d4ff;
        }}
        .halluc-tag {{
            color: #f44336;
            font-size: 0.8em;
            margin-top: 3px;
        }}
        .response-details {{ margin-top: 6px; }}
        .response-details summary {{
            color: #888;
            font-size: 0.8em;
            cursor: pointer;
        }}
        .response-pre {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.78em;
            white-space: pre-wrap;
            word-break: break-all;
            color: #aaa;
            background: rgba(0,0,0,0.2);
            padding: 8px;
            border-radius: 6px;
            max-height: 250px;
            overflow-y: auto;
            margin: 6px 0 0;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Metacognitive Financial LLM Evaluation</h1>
    <div class="subtitle">Do Financial LLMs Know What They Don't Know? | {timestamp}</div>

    <!-- Phase Tracker -->
    <div class="phase-tracker">
        <span class="phase-badge {'phase-done' if 'A' in phases_done else 'phase-pending'}">Phase A: Baseline</span>
        <span class="phase-badge {'phase-done' if 'B' in phases_done else 'phase-pending'}">Phase B: Metacognitive</span>
        <span class="phase-badge {'phase-done' if 'C' in phases_done else 'phase-pending'}">Phase C: RAG Impact</span>
        <span class="phase-badge {'phase-done' if len(phases_done) >= 3 else 'phase-pending'}">Phase D: Analysis</span>
    </div>

    <!-- Summary Cards -->
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-value">{total_evals}</div>
            <div class="summary-label">Total Evaluations</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{len(models)}</div>
            <div class="summary-label">Models Tested</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{len(unsolvable)}</div>
            <div class="summary-label">Unsolvable Tests</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{len(phases_done)}/4</div>
            <div class="summary-label">Phases Complete</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">${total_cost:.4f}</div>
            <div class="summary-label">Total Cost</div>
        </div>
    </div>

    <!-- Core Metrics Charts -->
    <div class="section">
        <h2>Core Metacognitive Metrics</h2>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Refusal Recall / Precision / F1</div>
                <canvas id="refusalChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">MC Score Comparison</div>
                <canvas id="mcCompareChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Model Comparison Table -->
    <div class="section">
        <h2>Model Performance (with Week-over-Week Delta)</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Ref Recall</th>
                    <th>Ref Prec</th>
                    <th>Ref F1</th>
                    <th>Over-Cons</th>
                    <th>MC Score</th>
                    <th>Delta</th>
                    <th>Cost</th>
                </tr>
            </thead>
            <tbody>{trend_rows if trend_rows else '<tr><td colspan="8" style="text-align:center;color:#666">Run Phase B to see metrics</td></tr>'}</tbody>
        </table>
    </div>

    <!-- Baseline Accuracy -->
    <div class="section">
        <h2>Phase A: Baseline Accuracy</h2>
        <table>
            <thead>
                <tr><th>Model</th><th>Correct/Total</th><th>Accuracy</th></tr>
            </thead>
            <tbody>{baseline_rows if baseline_rows else '<tr><td colspan="3" style="text-align:center;color:#666">Run Phase A to see baseline</td></tr>'}</tbody>
        </table>
    </div>

    <!-- Prompt Strategy Comparison -->
    <div class="section">
        <h2>Prompt Strategy Comparison</h2>
        <table>
            <thead>
                <tr><th>Strategy</th><th>Model</th><th>Ref Recall</th><th>Ref F1</th><th>MC Score</th></tr>
            </thead>
            <tbody>{strategy_rows if strategy_rows else '<tr><td colspan="5" style="text-align:center;color:#666">Run Phase B with multiple strategies to compare</td></tr>'}</tbody>
        </table>
    </div>

    <!-- 3A: Strategy × Type Heatmap -->
    <div class="section">
        <h2>Strategy × Transformation Type Refusal Heatmap</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            각 셀은 해당 전략×변환 조합의 거부율(%)을 나타냅니다.
            <span style="color:#4caf50;">초록</span> = 높은 거부율 (좋음),
            <span style="color:#f44336;">빨강</span> = 낮은 거부율 (나쁨).
        </p>
        <div class="heatmap-container">
            {heatmap_table_html}
        </div>
    </div>

    <!-- 3B: Strategy MC Score Comparison Chart -->
    <div class="section">
        <h2>Strategy MC Score Comparison</h2>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">전략별 MC Score (모델 비교)</div>
                <canvas id="strategyMcChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">변환 탐지 난이도 랭킹 (낮을수록 어려움)</div>
                <canvas id="difficultyChart"></canvas>
            </div>
        </div>
    </div>

    <!-- 3D: Transformation Design Reference -->
    <div class="section">
        <h2>Transformation Design Reference</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            각 변환 유형의 설계 근거, 도메인 특화도, 탐지 난이도, 그리고 Before/After 예시입니다.
        </p>
        {design_ref_html}
    </div>

    <!-- Prompt Strategy Reference -->
    <div class="section">
        <h2>Prompt Strategy Reference</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            모델에게 전달된 실제 시스템 프롬프트 전문입니다.
        </p>
        {prompt_reference_html}
    </div>

    <!-- Heatmap: Model x Transformation Type -->
    <div class="section">
        <h2>Refusal Rate Heatmap: Model x Transformation Type</h2>
        <div class="heatmap-container">
            <canvas id="heatmapChart" height="200"></canvas>
        </div>
    </div>

    <!-- Interesting Cases -->
    <div class="section">
        <h2>Interesting Cases</h2>
        <div class="cases-grid">
            {failure_cases_html}
        </div>
    </div>

    <!-- Detailed Case Analysis -->
    <div class="section">
        <h2>Detailed Case Analysis</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            Each card shows: original question, what data was changed, and how each model responded.
            <span style="color:#4caf50;">Green</span> = correctly refused,
            <span style="color:#f44336;">Red</span> = confidently answered (wrong),
            <span style="color:#ff9800;">Orange</span> = answered with caveats.
        </p>
        {detailed_cases_html}
    </div>

    <div class="footer">
        Generated by Metacognitive Evaluation Framework |
        {len(all_results)} evaluations | ${total_cost:.4f} total cost |
        {timestamp}
    </div>
</div>

<script>
// Refusal Recall / Precision / F1 Chart
const refusalCtx = document.getElementById('refusalChart').getContext('2d');
new Chart(refusalCtx, {{
    type: 'bar',
    data: {{
        labels: {model_names_js},
        datasets: [
            {{
                label: 'Refusal Recall (%)',
                data: {refusal_accs_js},
                backgroundColor: 'rgba(76,175,80,0.7)',
                borderRadius: 4,
            }},
            {{
                label: 'Refusal Precision (%)',
                data: {refusal_prec_js},
                backgroundColor: 'rgba(0,212,255,0.7)',
                borderRadius: 4,
            }},
            {{
                label: 'Refusal F1 (%)',
                data: {refusal_f1_js},
                backgroundColor: 'rgba(255,152,0,0.7)',
                borderRadius: 4,
            }}
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            y: {{
                beginAtZero: true,
                max: 100,
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.05)' }}
            }},
            x: {{ ticks: {{ color: '#888' }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#ccc' }} }}
        }}
    }}
}});

// MC Score Chart
const mcCompareCtx = document.getElementById('mcCompareChart').getContext('2d');
new Chart(mcCompareCtx, {{
    type: 'bar',
    data: {{
        labels: {model_names_js},
        datasets: [
            {{
                label: 'MC Score (F1 × (1-HR))',
                data: {mc_scores_js},
                backgroundColor: 'rgba(0,212,255,0.7)',
                borderRadius: 4,
            }}
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            y: {{
                beginAtZero: true,
                max: 100,
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.05)' }}
            }},
            x: {{ ticks: {{ color: '#888' }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#ccc' }} }}
        }}
    }}
}});

// Strategy MC Score Comparison Chart
const stratMcCtx = document.getElementById('strategyMcChart').getContext('2d');
new Chart(stratMcCtx, {{
    type: 'bar',
    data: {{
        labels: {strat_mc_labels_js},
        datasets: {strat_mc_chart_datasets_js}
    }},
    options: {{
        responsive: true,
        scales: {{
            y: {{
                beginAtZero: true,
                max: 100,
                title: {{ display: true, text: 'MC Score (%)', color: '#888' }},
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.05)' }}
            }},
            x: {{ ticks: {{ color: '#888' }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#ccc' }} }}
        }}
    }}
}});

// Transformation Difficulty Ranking (horizontal bar)
const diffCtx = document.getElementById('difficultyChart').getContext('2d');
new Chart(diffCtx, {{
    type: 'bar',
    data: {{
        labels: {diff_labels_js},
        datasets: [{{
            label: 'Refusal Rate (%)',
            data: {diff_values_js},
            backgroundColor: {diff_values_js}.map(v =>
                v > 70 ? 'rgba(76,175,80,0.7)' :
                v > 30 ? 'rgba(255,152,0,0.7)' :
                'rgba(244,67,54,0.7)'
            ),
            borderRadius: 4,
        }}]
    }},
    options: {{
        indexAxis: 'y',
        responsive: true,
        scales: {{
            x: {{
                beginAtZero: true,
                max: 100,
                title: {{ display: true, text: 'Refusal Rate (%)', color: '#888' }},
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.05)' }}
            }},
            y: {{ ticks: {{ color: '#888' }} }}
        }},
        plugins: {{
            legend: {{ display: false }},
            tooltip: {{
                callbacks: {{
                    afterLabel: function(context) {{
                        const counts = {diff_counts_js};
                        return 'Count: ' + counts[context.dataIndex];
                    }}
                }}
            }}
        }}
    }}
}});

// Heatmap (using Chart.js scatter with custom rendering)
const heatmapCtx = document.getElementById('heatmapChart').getContext('2d');
const heatmapModels = {heatmap_models_js};
const heatmapTransforms = {heatmap_transforms_js};
const heatmapData = {heatmap_data_js};

// Build datasets for grouped bar chart as heatmap approximation
const heatmapDatasets = heatmapModels.map((model, idx) => ({{
    label: model,
    data: heatmapData[idx].map(v => v !== null ? v : 0),
    backgroundColor: [
        'rgba(0,212,255,0.7)',
        'rgba(76,175,80,0.7)',
        'rgba(255,152,0,0.7)',
        'rgba(156,39,176,0.7)',
        'rgba(244,67,54,0.7)',
        'rgba(255,235,59,0.7)',
        'rgba(63,81,181,0.7)',
    ][idx % 7],
    borderRadius: 3,
}}));

new Chart(heatmapCtx, {{
    type: 'bar',
    data: {{
        labels: heatmapTransforms,
        datasets: heatmapDatasets
    }},
    options: {{
        responsive: true,
        scales: {{
            y: {{
                beginAtZero: true,
                max: 100,
                title: {{ display: true, text: 'Refusal Rate (%)', color: '#888' }},
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.05)' }}
            }},
            x: {{ ticks: {{ color: '#888', maxRotation: 45 }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#ccc' }} }}
        }}
    }}
}});
</script>
</body>
</html>"""

    if output_path is None:
        output_path = results_dir / "dashboard.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[SAVED] Dashboard: {output_path}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Phases: {', '.join(sorted(phases_done))}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate Metacognitive Evaluation Dashboard")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/metacognitive/",
        help="Directory with phase result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: results-dir/dashboard.html)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    output_path = Path(args.output) if args.output else None
    generate_dashboard(results_dir, output_path)


if __name__ == "__main__":
    main()
