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
sys.path.insert(0, str(Path(__file__).parent))

from metacognitive_metrics import (
    MetacognitiveResult,
    MetacognitiveMetrics,
    compute_metrics,
    compute_metrics_by_dimension,
    compute_cross_phase_metrics,
)
from hardcoded_solution_detector import (
    build_hardcoded_problem_ids,
    filter_contaminated_results,
    FilteringSummary,
)
from apply_transformations_full import (  # noqa: E402
    LABEL_EA_FULL,
    LABEL_EA_PARTIAL,
    LABEL_IC,
    LABEL_SA,
    LABEL_TA,
    normalize_transformation_label,
)


# ============================================================================
# MODEL TIERS
# ============================================================================

MODEL_TIERS: Dict[str, str] = {
    "gpt-4o-mini": "economic",
    "claude-haiku-4": "economic",
    "gemini-2.5-flash": "economic",
    "gpt-4o": "balanced",
    "claude-sonnet-4": "balanced",
    "gemini-2.5-pro": "balanced",
}

# Provider grouping for same-provider comparisons (RQ2)
MODEL_PROVIDERS: Dict[str, str] = {
    "gpt-4o-mini": "OpenAI",
    "gpt-4o": "OpenAI",
    "claude-haiku-4": "Anthropic",
    "claude-sonnet-4": "Anthropic",
    "gemini-2.5-flash": "Google",
    "gemini-2.5-pro": "Google",
}

TIER_PAIRS: List[tuple] = [
    ("gpt-4o-mini", "gpt-4o"),
    ("claude-haiku-4", "claude-sonnet-4"),
    ("gemini-2.5-flash", "gemini-2.5-pro"),
]

STRATEGY_ORDER: List[str] = [
    "standard",
    "metacognitive",
    "self_verification",
    "contradiction_aware",
]

MODEL_ORDER: List[str] = [
    # economic tier (provider순: Anthropic → Google → OpenAI)
    "claude-haiku-4",
    "gemini-2.5-flash",
    "gpt-4o-mini",
    # balanced tier
    "claude-sonnet-4",
    "gemini-2.5-pro",
    "gpt-4o",
]


def _ordered_strategies(keys) -> List[str]:
    """전략 키를 논리적 순서로 정렬"""
    ordered = [s for s in STRATEGY_ORDER if s in keys]
    ordered += [s for s in sorted(keys) if s not in STRATEGY_ORDER]
    return ordered


def _ordered_models(keys) -> List[str]:
    """모델 키를 tier→provider 논리적 순서로 정렬"""
    ordered = [m for m in MODEL_ORDER if m in keys]
    ordered += [m for m in sorted(keys) if m not in MODEL_ORDER]
    return ordered


# ============================================================================
# DATA LOADING
# ============================================================================


def load_all_results(results_dir: Path) -> List[MetacognitiveResult]:
    """Load all phase result files from directory.

    Automatically converts legacy 'Type N' labels to new taxonomy.
    """
    all_results = []
    for fp in sorted(results_dir.glob("phase_*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            result = MetacognitiveResult.from_dict(r)
            result.transformation_type = normalize_transformation_label(
                result.transformation_type
            )
            all_results.append(result)
    return all_results


def build_full_context_lookup(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load original hard.json and re-apply transformations to get full contexts.

    Returns:
        {"example_id|transformation_type": {"original": "...", "transformed": "..."}}
    """
    from apply_transformations_full import apply_transformations

    hard_path = (
        data_dir
        / "data"
        / "financereasoning"
        / "raw"
        / "FinanceReasoning"
        / "hard.json"
    )
    if not hard_path.exists():
        print(
            f"[WARN] hard.json not found at {hard_path}, skipping full context lookup"
        )
        return {}

    with open(hard_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    lookup: Dict[str, Dict[str, str]] = {}
    for example in examples:
        example_id = example.get("id", "")
        original_context = example.get("context", "")
        transformations = apply_transformations(example)

        for trans in transformations:
            trans_type = trans.get("transformation_type", "")
            key = f"{example_id}|{trans_type}"
            lookup[key] = {
                "original": original_context,
                "transformed": trans.get("context", ""),
            }

    return lookup


def load_previous_results(
    results_dir: Path, current_timestamp: str
) -> List[MetacognitiveResult]:
    """Load results from a previous run for week-over-week comparison.

    Automatically converts legacy 'Type N' labels to new taxonomy.
    """
    files = sorted(results_dir.glob("phase_*.json"))
    prev_results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", "")
        if ts < current_timestamp:
            for r in data.get("results", []):
                result = MetacognitiveResult.from_dict(r)
                result.transformation_type = normalize_transformation_label(
                    result.transformation_type
                )
                prev_results.append(result)
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
                strategy: dict(methods) for strategy, methods in PROMPT_SYSTEMS.items()
            }
        except ImportError:
            pass

    return merged


def build_prompt_reference_html(prompt_catalogs: Dict[str, Dict[str, str]]) -> str:
    """Build collapsible HTML cards showing actual prompt texts per strategy."""
    if not prompt_catalogs:
        return (
            '<div style="color:#666;text-align:center;">No prompt data available.</div>'
        )

    cards = ""
    for strategy in _ordered_strategies(prompt_catalogs.keys()):
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
        "type": LABEL_EA_PARTIAL,
        "short": "명시적 부재 (부분)",
        "description": "특정 값을 제거하고 [DATA MISSING]/N/A 마커를 삽입합니다. 가장 쉬운 탐지 과제 — 메타인지 능력의 하한선.",
        "domain_specificity": "높음",
        "detection_difficulty": "낮음",
        "before": '{"Revenue": {"2022": 500, "2023": 600}}',
        "after": '{"Revenue": {"2023": 600}}  (2022 entry removed)',
    },
    {
        "type": LABEL_EA_FULL,
        "short": "명시적 부재 (전체)",
        "description": "전체 키/컬럼을 삭제합니다. 마커 없이 데이터 차원 자체가 사라집니다. Markdown/JSON에서만 적용.",
        "domain_specificity": "높음",
        "detection_difficulty": "중간",
        "before": '{"Revenue": {...}, "Net Income": {...}}',
        "after": '{"Net Income": {...}}  (Revenue key deleted)',
    },
    {
        "type": LABEL_SA,
        "short": "무표지 부재",
        "description": "마커 없이 데이터를 제거합니다. JSON: 값 비움, Markdown: 행 삭제, Text: 핵심 문장 삭제. 맥락이 자연스러워 보이지만 필수 정보 부재.",
        "domain_specificity": "중간",
        "detection_difficulty": "높음",
        "before": '"Revenue was $500M. Expenses were $300M. Tax rate is 25%."',
        "after": '"Tax rate is 25%."  (revenue/expenses sentence removed)',
    },
    {
        "type": LABEL_IC,
        "short": "정보 충돌 (1.5배)",
        "description": "동일 데이터 포인트에 1.5배 모순값을 삽입합니다. 부재 탐지와 근본적으로 다른 인지 과정을 테스트.",
        "domain_specificity": "낮음",
        "detection_difficulty": "매우 높음",
        "before": '{"Revenue": {"2023": 600}}',
        "after": '{"Revenue": {"2023": 600, "2023_conflicting_report": 900}}',
    },
    {
        "type": LABEL_TA,
        "short": "시간 모호화 (보조)",
        "description": '질문의 연도를 "the end of the period"로 대체. ~11개 hard 문제에만 적용. 보조 분석으로 별도 보고.',
        "domain_specificity": "높음",
        "detection_difficulty": "중간",
        "before": "What was the revenue growth in 2023?",
        "after": "What was the revenue growth at the end of the period?",
    },
]


def group_metrics_by_tier(
    model_metrics: Dict[str, MetacognitiveMetrics],
) -> Dict[str, Dict[str, MetacognitiveMetrics]]:
    """Group model metrics by tier (economic/balanced).

    Returns:
        {"economic": {model_name: metrics, ...}, "balanced": {...}}
    """
    tiers: Dict[str, Dict[str, MetacognitiveMetrics]] = {}
    for model_name, metrics in model_metrics.items():
        tier = MODEL_TIERS.get(model_name, "unknown")
        tiers.setdefault(tier, {})[model_name] = metrics
    return tiers


def compute_tier_averages(
    tier_metrics: Dict[str, MetacognitiveMetrics],
) -> Dict[str, float]:
    """Compute average key metrics for a tier group.

    Returns dict with averaged mc_score, refusal_f1, hallucination_rate, etc.
    """
    if not tier_metrics:
        return {
            "mc_score": 0,
            "refusal_f1": 0,
            "refusal_recall": 0,
            "hallucination_rate": 0,
            "over_conservatism_rate": 0,
            "total_cost": 0,
        }
    n = len(tier_metrics)
    return {
        "mc_score": sum(m.mc_score for m in tier_metrics.values()) / n,
        "refusal_f1": sum(m.refusal_f1 for m in tier_metrics.values()) / n,
        "refusal_recall": sum(m.refusal_recall for m in tier_metrics.values()) / n,
        "hallucination_rate": sum(m.hallucination_rate for m in tier_metrics.values())
        / n,
        "over_conservatism_rate": sum(
            m.over_conservatism_rate for m in tier_metrics.values()
        )
        / n,
        "total_cost": sum(m.total_cost_usd for m in tier_metrics.values()) / n,
    }


def _build_rq1_findings(
    type_counts: Dict[str, Dict[str, int]],
    type_unique_problems: Dict[str, set],
) -> str:
    """Build dynamic RQ1 finding cards from actual data."""
    MIN_UNIQUE_PROBLEMS = 3  # 고유 문제 3개 미만이면 통계적으로 불충분
    findings = []

    # Low-N warning card: 고유 문제 수 기준으로 판정
    low_n_types = [
        tt
        for tt in type_counts
        if len(type_unique_problems.get(tt, set())) < MIN_UNIQUE_PROBLEMS
    ]
    if low_n_types:
        low_n_details = ", ".join(
            f"{tt} (n={len(type_unique_problems.get(tt, set()))}문제, {type_counts[tt]['total']}평가)"
            for tt in sorted(low_n_types)
        )
        findings.append(f"""
            <div class="finding-card" style="border-left-color:#ff9800;">
                <div class="finding-number" style="background:rgba(255,152,0,0.15);color:#ff9800;">N</div>
                <div class="finding-content">
                    <div class="finding-title">일부 변환 유형의 고유 문제 수가 통계적으로 불충분</div>
                    <div class="finding-detail">
                        다음 유형은 고유 문제 {MIN_UNIQUE_PROBLEMS}개 미만으로, 거부율의 통계적 신뢰도가 낮습니다: {low_n_details}.
                        N(총 평가 수)이 높아 보여도, 동일 문제에 대한 모델&times;전략 반복 평가이므로
                        실질적인 독립 표본은 고유 문제 수입니다.
                        이는 hard.json의 대부분(71%)이 text context이며, EA-full(테이블 구조 필요)과
                        TA(질문에 연도 필요)의 변환 적용 범위가 제한적이기 때문입니다.
                    </div>
                </div>
            </div>""")

    # IC (Information Conflict) finding (dynamic)
    ic_key = [
        tt
        for tt in type_counts
        if "conflict" in tt.lower() or "contradiction" in tt.lower()
    ]
    if ic_key:
        ic = ic_key[0]
        ic_c = type_counts[ic]
        ic_rate = ic_c["refused"] / ic_c["total"] * 100 if ic_c["total"] > 0 else 0
        ic_unique = len(type_unique_problems.get(ic, set()))
        findings.append(f"""
            <div class="finding-card">
                <div class="finding-number">!</div>
                <div class="finding-content">
                    <div class="finding-title">IC(정보 충돌) 탐지는 전략에 극단적으로 의존</div>
                    <div class="finding-detail">
                        IC의 전체 거부율은 {ic_rate:.1f}% (n={ic_unique}문제, {ic_c["total"]}평가)입니다.
                        standard 전략에서는 거부율이 매우 낮은 반면,
                        contradiction_aware 전략에서는 크게 상승합니다.
                        이는 LLM이 "정보 부재"와 "정보 충돌"을 서로 다른 메타인지 과제로 인식함을 시사합니다.
                    </div>
                </div>
            </div>""")

    if not findings:
        return ""
    return (
        '<div class="rq-findings" style="margin-top:20px;">'
        + "".join(findings)
        + "</div>"
    )


def build_rq1_section(
    results: List[MetacognitiveResult],
    type_difficulty: List[Dict[str, Any]],
) -> str:
    """Build RQ1: Can LLMs Detect Information Insufficiency?"""
    unsolvable = [r for r in results if r.transformation_type != "original"]
    if not unsolvable:
        return '<div style="color:#666;text-align:center;">Run Phase B to see RQ1 analysis.</div>'

    # Overall average refusal recall
    total = len(unsolvable)
    refused = sum(1 for r in unsolvable if r.response_type == "refused")
    avg_refusal_rate = refused / total * 100 if total else 0

    # Find hardest type
    hardest = type_difficulty[0] if type_difficulty else None
    easiest = type_difficulty[-1] if type_difficulty else None

    # Per-type refusal rates (all models averaged)
    type_counts: Dict[str, Dict[str, int]] = {}
    # Count unique problems per type
    type_unique_problems: Dict[str, set] = {}
    for r in unsolvable:
        tt = r.transformation_type
        type_counts.setdefault(tt, {"refused": 0, "total": 0})
        type_counts[tt]["total"] += 1
        if r.response_type == "refused":
            type_counts[tt]["refused"] += 1
        type_unique_problems.setdefault(tt, set()).add(r.example_id)

    sorted_types = sorted(type_counts.keys())
    type_chart_labels = json.dumps(
        [
            f"{tt} (n={len(type_unique_problems.get(tt, set()))}문제)"
            for tt in sorted_types
        ]
    )
    type_chart_values = json.dumps(
        [
            round(type_counts[tt]["refused"] / type_counts[tt]["total"] * 100, 1)
            if type_counts[tt]["total"] > 0
            else 0
            for tt in sorted_types
        ]
    )

    MIN_UNIQUE_PROBLEMS = 3

    hardest_html = ""
    if hardest:
        h_unique = len(type_unique_problems.get(hardest["type"], set()))
        h_low_n = h_unique < MIN_UNIQUE_PROBLEMS
        h_warn = (
            f' <span style="color:#f44336;font-size:0.75em;">(고유 문제 {h_unique}개 — 신뢰도 낮음)</span>'
            if h_low_n
            else ""
        )
        hardest_html = f"""
        <div class="rq-insight-card" style="border-left-color:#f44336;{"opacity:0.6;" if h_low_n else ""}">
            <div class="rq-insight-label">Hardest to Detect{h_warn}</div>
            <div class="rq-insight-value">{html_module.escape(hardest["type"])}</div>
            <div class="rq-insight-detail">{hardest["refusal_rate"]:.1f}% refusal rate (n={h_unique}문제, {hardest["total"]}평가)</div>
        </div>"""

    easiest_html = ""
    if easiest:
        e_unique = len(type_unique_problems.get(easiest["type"], set()))
        easiest_html = f"""
        <div class="rq-insight-card" style="border-left-color:#4caf50;">
            <div class="rq-insight-label">Easiest to Detect</div>
            <div class="rq-insight-value">{html_module.escape(easiest["type"])}</div>
            <div class="rq-insight-detail">{easiest["refusal_rate"]:.1f}% refusal rate (n={e_unique}문제, {easiest["total"]}평가)</div>
        </div>"""

    return f"""
    <div class="rq-section" id="rq1">
        <h2>RQ1: Can LLMs Detect Information Insufficiency?</h2>
        <p class="rq-description">금융 문제에서 정보가 부족할 때 LLM이 이를 탐지하고 적절히 거부할 수 있는가?</p>
        <div class="rq-summary-grid">
            <div class="rq-insight-card" style="border-left-color:#00d4ff;">
                <div class="rq-insight-label">Overall Refusal Rate</div>
                <div class="rq-insight-value" style="font-size:2em;">{avg_refusal_rate:.1f}%</div>
                <div class="rq-insight-detail">{refused}/{total} unsolvable problems correctly refused</div>
            </div>
            {hardest_html}
            {easiest_html}
        </div>
        <div class="chart-container" style="margin-top:15px;">
            <div class="chart-title">Refusal Rate by Transformation Type (All Models)</div>
            <canvas id="rq1TypeChart"></canvas>
        </div>
        {_build_rq1_findings(type_counts, type_unique_problems)}
    </div>
    <script>
    new Chart(document.getElementById('rq1TypeChart').getContext('2d'), {{
        type: 'bar',
        data: {{
            labels: {type_chart_labels},
            datasets: [{{
                label: 'Refusal Rate (%)',
                data: {type_chart_values},
                backgroundColor: {type_chart_values}.map(v =>
                    v > 70 ? 'rgba(76,175,80,0.7)' :
                    v > 30 ? 'rgba(255,152,0,0.7)' :
                    'rgba(244,67,54,0.7)'
                ),
                borderRadius: 4,
            }}]
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{ beginAtZero: true, max: 100, ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                x: {{ ticks: {{ color: '#888', maxRotation: 20 }} }}
            }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});
    </script>"""


def build_rq2_section(
    model_metrics: Dict[str, MetacognitiveMetrics],
    strategy_metrics: Dict[str, Dict[str, MetacognitiveMetrics]],
) -> str:
    """Build RQ2: Does Model Size Affect Metacognition?"""
    tier_groups = group_metrics_by_tier(model_metrics)
    eco_metrics = tier_groups.get("economic", {})
    bal_metrics = tier_groups.get("balanced", {})

    if not eco_metrics and not bal_metrics:
        return '<div style="color:#666;text-align:center;">Run experiments for both economic and balanced tiers to see RQ2 analysis.</div>'

    eco_avg = compute_tier_averages(eco_metrics)
    bal_avg = compute_tier_averages(bal_metrics)

    # Tier comparison table
    def _fmt(val: float, pct: bool = False) -> str:
        return f"{val:.1%}" if pct else f"{val:.3f}"

    tier_table = f"""
    <table>
        <thead><tr><th>Metric</th><th>Economic (avg)</th><th>Balanced (avg)</th><th>Delta</th></tr></thead>
        <tbody>
            <tr>
                <td>MC Score</td>
                <td>{_fmt(eco_avg["mc_score"])}</td>
                <td>{_fmt(bal_avg["mc_score"])}</td>
                <td style="color:{"#4caf50" if bal_avg["mc_score"] > eco_avg["mc_score"] else "#f44336"}">
                    {bal_avg["mc_score"] - eco_avg["mc_score"]:+.3f}</td>
            </tr>
            <tr>
                <td>Refusal F1</td>
                <td>{_fmt(eco_avg["refusal_f1"])}</td>
                <td>{_fmt(bal_avg["refusal_f1"])}</td>
                <td style="color:{"#4caf50" if bal_avg["refusal_f1"] > eco_avg["refusal_f1"] else "#f44336"}">
                    {bal_avg["refusal_f1"] - eco_avg["refusal_f1"]:+.3f}</td>
            </tr>
            <tr>
                <td>Hallucination Rate</td>
                <td>{_fmt(eco_avg["hallucination_rate"], True)}</td>
                <td>{_fmt(bal_avg["hallucination_rate"], True)}</td>
                <td style="color:{"#4caf50" if bal_avg["hallucination_rate"] < eco_avg["hallucination_rate"] else "#f44336"}">
                    {bal_avg["hallucination_rate"] - eco_avg["hallucination_rate"]:+.1%}</td>
            </tr>
            <tr>
                <td>Over-Conservatism</td>
                <td>{_fmt(eco_avg["over_conservatism_rate"], True)}</td>
                <td>{_fmt(bal_avg["over_conservatism_rate"], True)}</td>
                <td style="color:{"#4caf50" if bal_avg["over_conservatism_rate"] < eco_avg["over_conservatism_rate"] else "#f44336"}">
                    {bal_avg["over_conservatism_rate"] - eco_avg["over_conservatism_rate"]:+.1%}</td>
            </tr>
        </tbody>
    </table>"""

    # Provider pair comparison
    pair_rows = ""
    for eco_name, bal_name in TIER_PAIRS:
        eco_m = model_metrics.get(eco_name)
        bal_m = model_metrics.get(bal_name)
        provider = MODEL_PROVIDERS.get(eco_name, "")
        if eco_m and bal_m:
            delta = bal_m.mc_score - eco_m.mc_score
            pair_rows += f"""
            <tr>
                <td>{html_module.escape(provider)}</td>
                <td>{html_module.escape(eco_name)}</td><td>{eco_m.mc_score:.3f}</td>
                <td>{html_module.escape(bal_name)}</td><td>{bal_m.mc_score:.3f}</td>
                <td style="color:{"#4caf50" if delta > 0 else "#f44336"}">{delta:+.3f}</td>
            </tr>"""
        elif eco_m:
            pair_rows += f"""
            <tr>
                <td>{html_module.escape(provider)}</td>
                <td>{html_module.escape(eco_name)}</td><td>{eco_m.mc_score:.3f}</td>
                <td>{html_module.escape(bal_name)}</td><td style="color:#666">N/A</td>
                <td style="color:#666">-</td>
            </tr>"""

    pair_table = (
        f"""
    <table>
        <thead><tr><th>Provider</th><th>Economic</th><th>MC Score</th><th>Balanced</th><th>MC Score</th><th>Delta</th></tr></thead>
        <tbody>{pair_rows}</tbody>
    </table>"""
        if pair_rows
        else ""
    )

    # Chart data: tier MC score by strategy
    strat_labels = _ordered_strategies(strategy_metrics.keys())
    eco_strat_scores = []
    bal_strat_scores = []
    for strat in strat_labels:
        strat_models = strategy_metrics.get(strat, {})
        eco_vals = [
            m.mc_score * 100
            for name, m in strat_models.items()
            if MODEL_TIERS.get(name) == "economic"
        ]
        bal_vals = [
            m.mc_score * 100
            for name, m in strat_models.items()
            if MODEL_TIERS.get(name) == "balanced"
        ]
        eco_strat_scores.append(
            round(sum(eco_vals) / len(eco_vals), 1) if eco_vals else 0
        )
        bal_strat_scores.append(
            round(sum(bal_vals) / len(bal_vals), 1) if bal_vals else 0
        )

    strat_labels_js = json.dumps(strat_labels)
    eco_scores_js = json.dumps(eco_strat_scores)
    bal_scores_js = json.dumps(bal_strat_scores)

    return f"""
    <div class="rq-section" id="rq2">
        <h2>RQ2: Does Model Size Affect Metacognition?</h2>
        <p class="rq-description">모델 크기/유형이 정보 부족 탐지 능력에 어떤 영향을 미치는가?</p>

        <h3 style="color:#ce93d8;margin:15px 0 10px;">Economic vs Balanced Tier Comparison</h3>
        {tier_table}

        <h3 style="color:#ce93d8;margin:20px 0 10px;">Same-Provider Model Pair Comparison</h3>
        {pair_table}

        <div class="chart-container" style="margin-top:20px;">
            <div class="chart-title">Tier MC Score by Prompt Strategy</div>
            <canvas id="rq2TierChart"></canvas>
        </div>
        <div class="rq-findings" style="margin-top:20px;">
            <div class="finding-card">
                <div class="finding-number">1</div>
                <div class="finding-content">
                    <div class="finding-title">프롬프트 전략 > 모델 크기</div>
                    <div class="finding-detail">
                        프롬프트 전략이 모델 크기보다 메타인지에 훨씬 큰 영향을 미칩니다.
                        standard 전략의 MC Score는 0에 가까운 반면,
                        self_verification 전략은 0.85 이상을 달성합니다.
                        같은 모델이라도 프롬프트 전략에 따라 메타인지 능력이 극적으로 변합니다.
                    </div>
                </div>
            </div>
            <div class="finding-card">
                <div class="finding-number">2</div>
                <div class="finding-content">
                    <div class="finding-title">큰 모델의 역설: 높은 Hallucination Rate</div>
                    <div class="finding-detail">
                        balanced tier(gpt-4o, claude-sonnet-4, gemini-2.5-pro) 모델이
                        economic tier보다 hallucination rate가 높아 MC Score가 오히려 낮습니다.
                        큰 모델이 더 "자신 있게" 잘못된 답을 생성하는 경향이 있습니다.
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('rq2TierChart').getContext('2d'), {{
        type: 'bar',
        data: {{
            labels: {strat_labels_js},
            datasets: [
                {{ label: 'Economic (avg)', data: {eco_scores_js}, backgroundColor: 'rgba(255,152,0,0.7)', borderRadius: 4 }},
                {{ label: 'Balanced (avg)', data: {bal_scores_js}, backgroundColor: 'rgba(0,212,255,0.7)', borderRadius: 4 }}
            ]
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'MC Score (%)', color: '#888' }}, ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                x: {{ ticks: {{ color: '#888' }} }}
            }},
            plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }}
        }}
    }});
    </script>"""


def build_rq3_section(
    results: List[MetacognitiveResult],
    model_metrics: Dict[str, MetacognitiveMetrics],
) -> str:
    """Build RQ3: Does RAG Improve Metacognition?"""
    unsolvable = [r for r in results if r.transformation_type != "original"]
    rag_on = [r for r in unsolvable if r.rag_enabled]
    rag_off = [r for r in unsolvable if not r.rag_enabled]

    if not rag_on:
        return """
        <div class="rq-section" id="rq3">
            <h2>RQ3: Does RAG Improve Metacognition?</h2>
            <p class="rq-description">RAG(금융 함수 검색)가 정보 부족 인식을 개선하는가?</p>
            <div style="color:#666;text-align:center;padding:20px;">
                Run Phase C (RAG experiments) to see RQ3 analysis.<br>
                <code style="color:#888;">python experiments/run_metacognitive_experiment.py --phase C --budget economic --n 10 --rag</code>
            </div>
        </div>"""

    rag_on_metrics = compute_metrics(rag_on)
    rag_off_metrics = compute_metrics(rag_off)

    # Build model comparison table
    all_models = _ordered_models(
        set(list(rag_on_metrics.keys()) + list(rag_off_metrics.keys()))
    )
    comp_rows = ""
    delta_data_labels = []
    delta_data_values = []

    for m_name in all_models:
        on_m = rag_on_metrics.get(m_name)
        off_m = rag_off_metrics.get(m_name)
        if on_m and off_m:
            delta = on_m.mc_score - off_m.mc_score
            delta_data_labels.append(m_name)
            delta_data_values.append(round(delta * 100, 1))
            comp_rows += f"""
            <tr>
                <td>{html_module.escape(m_name)}</td>
                <td>{off_m.mc_score:.3f}</td>
                <td>{off_m.refusal_f1:.3f}</td>
                <td>{on_m.mc_score:.3f}</td>
                <td>{on_m.refusal_f1:.3f}</td>
                <td style="color:{"#4caf50" if delta > 0 else "#f44336"}">{delta:+.3f}</td>
            </tr>"""
        elif on_m:
            comp_rows += f"""
            <tr>
                <td>{html_module.escape(m_name)}</td>
                <td style="color:#666">N/A</td><td style="color:#666">N/A</td>
                <td>{on_m.mc_score:.3f}</td><td>{on_m.refusal_f1:.3f}</td>
                <td style="color:#666">-</td>
            </tr>"""

    delta_labels_js = json.dumps(delta_data_labels)
    delta_values_js = json.dumps(delta_data_values)

    # --- 4-2: Transformation type별 RAG 영향 분석 ---
    rag_type_on: Dict[str, Dict[str, Any]] = {}
    for r in rag_on:
        tt = r.transformation_type
        rag_type_on.setdefault(tt, {"refused": 0, "total": 0, "problems": set()})
        rag_type_on[tt]["total"] += 1
        if r.response_type == "refused":
            rag_type_on[tt]["refused"] += 1
        rag_type_on[tt]["problems"].add(r.example_id)

    rag_type_off: Dict[str, Dict[str, Any]] = {}
    for r in rag_off:
        tt = r.transformation_type
        rag_type_off.setdefault(tt, {"refused": 0, "total": 0, "problems": set()})
        rag_type_off[tt]["total"] += 1
        if r.response_type == "refused":
            rag_type_off[tt]["refused"] += 1
        rag_type_off[tt]["problems"].add(r.example_id)

    all_ttypes = sorted(set(list(rag_type_on.keys()) + list(rag_type_off.keys())))
    type_rows = ""
    for tt in all_ttypes:
        on_c = rag_type_on.get(tt, {"refused": 0, "total": 0, "problems": set()})
        off_c = rag_type_off.get(tt, {"refused": 0, "total": 0, "problems": set()})
        on_rate = on_c["refused"] / on_c["total"] * 100 if on_c["total"] > 0 else 0
        off_rate = off_c["refused"] / off_c["total"] * 100 if off_c["total"] > 0 else 0
        delta_rate = on_rate - off_rate
        # unique problems across both RAG conditions
        combined_problems = on_c["problems"] | off_c["problems"]
        type_rows += f"""
        <tr>
            <td>{html_module.escape(tt)} <span style="color:#888;font-size:0.8em;">(n={len(combined_problems)}문제)</span></td>
            <td>{off_c["refused"]}/{off_c["total"]} ({off_rate:.0f}%)</td>
            <td>{on_c["refused"]}/{on_c["total"]} ({on_rate:.0f}%)</td>
            <td style="color:{"#4caf50" if delta_rate > 0 else "#f44336" if delta_rate < 0 else "#888"}">{delta_rate:+.0f}pp</td>
        </tr>"""

    # --- 4-3: Data limitation notice ---
    rag_on_count = len(rag_on)
    rag_on_problems = len({r.example_id for r in rag_on})
    rag_strategies = sorted({r.prompt_strategy for r in rag_on})
    rag_models = sorted({r.model_name for r in rag_on})
    limitation_html = f"""
        <div style="background:rgba(255,152,0,0.08);border:1px solid rgba(255,152,0,0.25);border-radius:10px;padding:15px;margin-top:15px;">
            <strong style="color:#ff9800;">데이터 한계:</strong>
            <span style="color:#aaa;font-size:0.88em;">
                Phase C는 현재 {", ".join(rag_models)} ({len(rag_models)}종) ×
                {", ".join(rag_strategies)} 전략으로만 실행되었습니다
                (n={rag_on_problems}문제, {rag_on_count}평가).
                balanced 모델 및 다른 전략으로의 확장 실험이 필요합니다.
            </span>
        </div>"""

    # --- 4-4: Findings card ---
    avg_delta = (
        sum(delta_data_values) / len(delta_data_values) if delta_data_values else 0
    )
    rag_direction = (
        "개선" if avg_delta > 0 else "악화" if avg_delta < 0 else "변화 없음"
    )
    # --- 4-5: RAG 행동 변화 케이스 분석 ---
    # RAG ON은 전부 metacognitive 전략이므로, RAG OFF도 metacognitive만 비교
    rag_off_meta = [r for r in rag_off if r.prompt_strategy == "metacognitive"]

    # Group by (example_id, transformation_type, model_name) → single result
    rag_on_map: Dict[tuple, MetacognitiveResult] = {}
    for r in rag_on:
        key = (r.example_id, r.transformation_type, r.model_name)
        rag_on_map[key] = r

    rag_off_map: Dict[tuple, MetacognitiveResult] = {}
    for r in rag_off_meta:
        key = (r.example_id, r.transformation_type, r.model_name)
        rag_off_map.setdefault(key, r)  # 첫 번째만 사용

    common_keys = sorted(set(rag_on_map) & set(rag_off_map))

    # 행동 변화 분류
    changes: Dict[str, Any] = {
        "refused_to_confident": [],
        "confident_to_refused": [],
        "unchanged_refused": 0,
        "unchanged_confident": 0,
        "other": 0,
    }
    # 모델별 breakdown
    model_changes: Dict[str, Dict[str, int]] = {}

    for key in common_keys:
        on_r = rag_on_map[key]
        off_r = rag_off_map[key]
        off_t = "refused" if off_r.response_type == "refused" else "confident"
        on_t = "refused" if on_r.response_type == "refused" else "confident"

        m_name = key[2]
        if m_name not in model_changes:
            model_changes[m_name] = {
                "refused_to_confident": 0,
                "confident_to_refused": 0,
                "unchanged_refused": 0,
                "unchanged_confident": 0,
            }

        if off_t == "refused" and on_t == "confident":
            changes["refused_to_confident"].append(
                {
                    "example_id": key[0],
                    "transformation_type": key[1],
                    "model_name": m_name,
                    "off_response": off_r.raw_response[:150],
                    "on_response": on_r.raw_response[:150],
                    "off_type": off_r.response_type,
                    "on_type": on_r.response_type,
                }
            )
            model_changes[m_name]["refused_to_confident"] += 1
        elif off_t == "confident" and on_t == "refused":
            changes["confident_to_refused"].append(
                {
                    "example_id": key[0],
                    "transformation_type": key[1],
                    "model_name": m_name,
                    "off_response": off_r.raw_response[:150],
                    "on_response": on_r.raw_response[:150],
                    "off_type": off_r.response_type,
                    "on_type": on_r.response_type,
                }
            )
            model_changes[m_name]["confident_to_refused"] += 1
        elif off_t == on_t == "refused":
            changes["unchanged_refused"] += 1
            model_changes[m_name]["unchanged_refused"] += 1
        elif off_t == on_t == "confident":
            changes["unchanged_confident"] += 1
            model_changes[m_name]["unchanged_confident"] += 1
        else:
            changes["other"] += 1

    n_pairs = len(common_keys)
    n_r2c = len(changes["refused_to_confident"])
    n_c2r = len(changes["confident_to_refused"])
    r2c_pct = n_r2c / n_pairs * 100 if n_pairs > 0 else 0

    # 요약 카드 HTML
    case_summary_html = ""
    if n_pairs > 0:
        # 모델별 breakdown 테이블
        model_order = _ordered_models(set(model_changes.keys()))
        model_breakdown_rows = ""
        for m_name in model_order:
            mc = model_changes.get(m_name, {})
            model_breakdown_rows += f"""
                <tr>
                    <td>{html_module.escape(m_name)}</td>
                    <td style="color:#f44336;font-weight:600;">{mc.get("refused_to_confident", 0)}</td>
                    <td style="color:#4caf50;font-weight:600;">{mc.get("confident_to_refused", 0)}</td>
                    <td>{mc.get("unchanged_refused", 0)}</td>
                    <td>{mc.get("unchanged_confident", 0)}</td>
                </tr>"""

        case_summary_html = f"""
        <h3 style="color:#ce93d8;margin:25px 0 10px;">RAG 행동 변화 분석</h3>
        <div style="background:rgba(156,39,176,0.08);border:1px solid rgba(156,39,176,0.25);border-radius:10px;padding:18px;margin-bottom:15px;">
            <strong style="color:#ce93d8;">비교 가능 쌍: {n_pairs}건</strong>
            <span style="color:#aaa;font-size:0.88em;">(RAG ON/OFF 동일 문제·변환·모델)</span>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:12px;">
                <div style="background:rgba(244,67,54,0.1);border-radius:8px;padding:12px;">
                    <div style="color:#f44336;font-weight:700;font-size:1.3em;">{n_r2c}건</div>
                    <div style="color:#aaa;font-size:0.85em;">거부→환각 (메타인지 악화)</div>
                </div>
                <div style="background:rgba(76,175,80,0.1);border-radius:8px;padding:12px;">
                    <div style="color:#4caf50;font-weight:700;font-size:1.3em;">{n_c2r}건</div>
                    <div style="color:#aaa;font-size:0.85em;">환각→거부 (메타인지 개선)</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;">
                    <div style="color:#888;font-weight:700;font-size:1.3em;">{changes["unchanged_refused"]}건</div>
                    <div style="color:#aaa;font-size:0.85em;">변화 없음 (거부 유지)</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;">
                    <div style="color:#888;font-weight:700;font-size:1.3em;">{changes["unchanged_confident"]}건</div>
                    <div style="color:#aaa;font-size:0.85em;">변화 없음 (환각 유지)</div>
                </div>
            </div>
        </div>

        <table>
            <thead><tr>
                <th>Model</th>
                <th style="color:#f44336;">거부→환각</th>
                <th style="color:#4caf50;">환각→거부</th>
                <th>거부 유지</th>
                <th>환각 유지</th>
            </tr></thead>
            <tbody>{model_breakdown_rows}</tbody>
        </table>"""

    # 거부→환각 상세 케이스 (collapsible)
    case_detail_html = ""
    if changes["refused_to_confident"]:
        detail_rows = ""
        for case in changes["refused_to_confident"]:
            off_badge = '<span style="background:#4caf50;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.8em;">REFUSED</span>'
            on_badge = '<span style="background:#f44336;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.8em;">CONFIDENT</span>'
            detail_rows += f"""
                <tr>
                    <td style="font-family:monospace;font-size:0.85em;">{html_module.escape(case["example_id"])}</td>
                    <td>{html_module.escape(case["transformation_type"])}</td>
                    <td>{html_module.escape(case["model_name"])}</td>
                    <td>{off_badge}<br><span style="color:#aaa;font-size:0.8em;">{html_module.escape(case["off_response"])}…</span></td>
                    <td>{on_badge}<br><span style="color:#aaa;font-size:0.8em;">{html_module.escape(case["on_response"])}…</span></td>
                </tr>"""

        case_detail_html = f"""
        <details style="margin-top:12px;">
            <summary style="cursor:pointer;color:#f44336;font-weight:600;">
                거부→환각 상세 케이스 ({n_r2c}건) — RAG가 메타인지를 악화시킨 구체적 사례
            </summary>
            <table style="margin-top:10px;font-size:0.9em;">
                <thead><tr>
                    <th>문제 ID</th><th>변환 유형</th><th>모델</th>
                    <th>RAG OFF 응답</th><th>RAG ON 응답</th>
                </tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </details>"""

    # 환각→거부 상세 케이스 (collapsible)
    if changes["confident_to_refused"]:
        c2r_rows = ""
        for case in changes["confident_to_refused"]:
            off_badge = '<span style="background:#f44336;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.8em;">CONFIDENT</span>'
            on_badge = '<span style="background:#4caf50;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.8em;">REFUSED</span>'
            c2r_rows += f"""
                <tr>
                    <td style="font-family:monospace;font-size:0.85em;">{html_module.escape(case["example_id"])}</td>
                    <td>{html_module.escape(case["transformation_type"])}</td>
                    <td>{html_module.escape(case["model_name"])}</td>
                    <td>{off_badge}<br><span style="color:#aaa;font-size:0.8em;">{html_module.escape(case["off_response"])}…</span></td>
                    <td>{on_badge}<br><span style="color:#aaa;font-size:0.8em;">{html_module.escape(case["on_response"])}…</span></td>
                </tr>"""

        case_detail_html += f"""
        <details style="margin-top:12px;">
            <summary style="cursor:pointer;color:#4caf50;font-weight:600;">
                환각→거부 상세 케이스 ({n_c2r}건) — RAG가 메타인지를 개선한 사례
            </summary>
            <table style="margin-top:10px;font-size:0.9em;">
                <thead><tr>
                    <th>문제 ID</th><th>변환 유형</th><th>모델</th>
                    <th>RAG OFF 응답</th><th>RAG ON 응답</th>
                </tr></thead>
                <tbody>{c2r_rows}</tbody>
            </table>
        </details>"""

    case_analysis_html = case_summary_html + case_detail_html

    # --- Finding #3 card (보강) ---
    behavior_insight = ""
    if n_pairs > 0 and n_r2c > 0:
        behavior_insight = (
            f" 특히 비교 가능한 {n_pairs}쌍 중 {n_r2c}건({r2c_pct:.0f}%)에서 "
            f"거부→환각 전환이 발생했습니다. RAG가 함수 파라미터 정의(시그니처, docstring)를 "
            f'제공하면, 모델이 이를 "값 힌트"로 해석하여 누락 데이터를 함수 시그니처 기반으로 '
            f"날조하는 패턴이 관찰됩니다."
        )

    findings_html = f"""
        <div class="rq-findings" style="margin-top:20px;">
            <div class="finding-card">
                <div class="finding-number">3</div>
                <div class="finding-content">
                    <div class="finding-title">RAG의 메타인지 영향: 평균 {avg_delta:+.1f}pp ({rag_direction})</div>
                    <div class="finding-detail">
                        금융 함수 라이브러리(3,133개)를 Jaccard 유사도로 검색하여 top-5를 프롬프트에 삽입한 결과,
                        모델별로 MC Score가 평균 {avg_delta:+.1f}pp 변화했습니다.
                        RAG가 제공하는 함수 파라미터 정의가 "이 문제에 어떤 데이터가 필요한가"를 명확히 하여
                        {"정보 부족 인식을 돕는 것으로 보입니다." if avg_delta > 0 else "오히려 모델이 함수에 의존해 답변을 시도하는 경향이 있습니다."}{behavior_insight}
                    </div>
                </div>
            </div>
        </div>"""

    return f"""
    <div class="rq-section" id="rq3">
        <h2>RQ3: Does RAG Improve Metacognition?</h2>
        <p class="rq-description">RAG(금융 함수 검색)가 정보 부족 인식을 개선하는가?</p>

        <!-- 4-1: RAG 방법론 설명 -->
        <div class="methodology-subsection" style="margin-bottom:20px;">
            <h3>RAG 구현 방식</h3>
            <table class="reference-table">
                <tr><td style="min-width:120px;"><strong>문서 소스</strong></td>
                    <td>FinanceReasoning 금융 함수 라이브러리 (functions-article-all.json, 3,133개 함수)</td></tr>
                <tr><td><strong>검색 기법</strong></td>
                    <td>Jaccard 유사도 기반 키워드 매칭 — 질문+컨텍스트에서 추출한 키워드와
                        함수명+docstring 키워드의 교집합/합집합 비율로 점수 산출</td></tr>
                <tr><td><strong>검색 쿼리</strong></td>
                    <td>"{{question}} {{context[:200]}}" — 질문 전문 + 컨텍스트 앞 200자</td></tr>
                <tr><td><strong>Top-K</strong></td>
                    <td>상위 5개 함수를 검색하여 프롬프트에 삽입</td></tr>
                <tr><td><strong>프롬프트 강화</strong></td>
                    <td>원본 프롬프트 뒤에 "Available Financial Functions:" 블록을 추가하고,
                        "관련 함수를 참조하여 풀이하라"는 지시를 포함</td></tr>
                <tr><td><strong>실험 조건</strong></td>
                    <td>Phase C = Phase B와 동일한 unsolvable 문제에 RAG만 활성화.
                        전략: metacognitive / 모델: economic 3종 / 동일 변환 문제</td></tr>
            </table>
        </div>

        <!-- 모델별 RAG ON/OFF 비교 -->
        <h3 style="color:#ce93d8;margin:15px 0 10px;">모델별 RAG 영향 비교</h3>
        <table>
            <thead><tr>
                <th>Model</th>
                <th>No-RAG MC</th><th>No-RAG F1</th>
                <th>RAG MC</th><th>RAG F1</th>
                <th>Delta MC</th>
            </tr></thead>
            <tbody>{comp_rows}</tbody>
        </table>

        <div class="chart-container" style="margin-top:15px;">
            <div class="chart-title">RAG Impact on MC Score (delta percentage points)</div>
            <canvas id="rq3DeltaChart"></canvas>
        </div>

        <!-- 4-2: 변환 유형별 RAG 영향 -->
        <h3 style="color:#ce93d8;margin:20px 0 10px;">변환 유형별 RAG 영향</h3>
        <table>
            <thead><tr>
                <th>Transformation Type</th>
                <th>No-RAG Refusal</th>
                <th>RAG Refusal</th>
                <th>Delta</th>
            </tr></thead>
            <tbody>{type_rows}</tbody>
        </table>

        {limitation_html}
        {findings_html}
        {case_analysis_html}
    </div>
    <script>
    new Chart(document.getElementById('rq3DeltaChart').getContext('2d'), {{
        type: 'bar',
        data: {{
            labels: {delta_labels_js},
            datasets: [{{
                label: 'MC Score Delta (pp)',
                data: {delta_values_js},
                backgroundColor: {delta_values_js}.map(v => v >= 0 ? 'rgba(76,175,80,0.7)' : 'rgba(244,67,54,0.7)'),
                borderRadius: 4,
            }}]
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{ title: {{ display: true, text: 'Delta (percentage points)', color: '#888' }}, ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                x: {{ ticks: {{ color: '#888' }} }}
            }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});
    </script>"""


def build_rq4_section(
    model_metrics: Dict[str, MetacognitiveMetrics],
    strategy_metrics: Dict[str, Dict[str, MetacognitiveMetrics]],
) -> str:
    """Build RQ4: What is the Cost-Optimal Strategy?"""
    if not model_metrics:
        return '<div style="color:#666;text-align:center;">Run experiments to see RQ4 analysis.</div>'

    # Scatter data: each model point with cost and MC score
    scatter_data = []
    for m_name, m in model_metrics.items():
        tier = MODEL_TIERS.get(m_name, "unknown")
        mc_per_dollar = m.mc_score / m.total_cost_usd if m.total_cost_usd > 0 else 0
        scatter_data.append(
            {
                "name": m_name,
                "tier": tier,
                "cost": round(m.total_cost_usd, 5),
                "mc_score": round(m.mc_score * 100, 1),
                "mc_per_dollar": round(mc_per_dollar, 1),
            }
        )

    scatter_data.sort(key=lambda x: -x["mc_per_dollar"])

    # Efficiency ranking table
    rank_rows = ""
    for idx, sd in enumerate(scatter_data):
        tier_badge = f'<span style="color:{"#ff9800" if sd["tier"] == "economic" else "#00d4ff"}">{sd["tier"]}</span>'
        rank_rows += f"""
        <tr>
            <td>{idx + 1}</td>
            <td>{html_module.escape(sd["name"])}</td>
            <td>{tier_badge}</td>
            <td>{sd["mc_score"]:.1f}%</td>
            <td>${sd["cost"]:.4f}</td>
            <td style="font-weight:bold;color:#00d4ff;">{sd["mc_per_dollar"]:.1f}</td>
        </tr>"""

    # Scatter chart data — group by tier
    eco_points = [sd for sd in scatter_data if sd["tier"] == "economic"]
    bal_points = [sd for sd in scatter_data if sd["tier"] == "balanced"]

    eco_scatter_js = json.dumps(
        [{"x": p["cost"], "y": p["mc_score"], "label": p["name"]} for p in eco_points]
    )
    bal_scatter_js = json.dumps(
        [{"x": p["cost"], "y": p["mc_score"], "label": p["name"]} for p in bal_points]
    )

    # Strategy cost efficiency
    strat_rows = ""
    for strat in _ordered_strategies(strategy_metrics.keys()):
        strat_models = strategy_metrics[strat]
        for m_name in _ordered_models(strat_models.keys()):
            m = strat_models[m_name]
            mc_per_d = m.mc_score / m.total_cost_usd if m.total_cost_usd > 0 else 0
            strat_rows += f"""
            <tr>
                <td>{html_module.escape(strat)}</td>
                <td>{html_module.escape(m_name)}</td>
                <td>{m.mc_score:.3f}</td>
                <td>${m.total_cost_usd:.4f}</td>
                <td>{mc_per_d:.1f}</td>
            </tr>"""

    return f"""
    <div class="rq-section" id="rq4">
        <h2>RQ4: What is the Cost-Optimal Strategy?</h2>
        <p class="rq-description">메타인지 금융 추론의 비용-성능 최적 전략은 무엇인가?</p>

        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Cost vs MC Score (Scatter)</div>
                <canvas id="rq4ScatterChart"></canvas>
            </div>
            <div class="chart-container">
                <h3 style="color:#ce93d8;margin-bottom:10px;">MC/$ Efficiency Ranking</h3>
                <table>
                    <thead><tr><th>#</th><th>Model</th><th>Tier</th><th>MC Score</th><th>Cost</th><th>MC/$</th></tr></thead>
                    <tbody>{rank_rows}</tbody>
                </table>
            </div>
        </div>

        <h3 style="color:#ce93d8;margin:20px 0 10px;">Strategy Cost Efficiency</h3>
        <table>
            <thead><tr><th>Strategy</th><th>Model</th><th>MC Score</th><th>Cost</th><th>MC/$</th></tr></thead>
            <tbody>{strat_rows}</tbody>
        </table>
        <div class="rq-findings" style="margin-top:20px;">
            <div class="finding-card">
                <div class="finding-number">4</div>
                <div class="finding-content">
                    <div class="finding-title">비용 효율 1위: gemini-2.5-flash</div>
                    <div class="finding-detail">
                        gemini-2.5-flash가 MC/$ 효율 1위(22.0 MC/$)를 기록했습니다.
                        claude-sonnet-4는 MC Score는 높지만 비용 대비 효율은 최하위(0.2 MC/$)입니다.
                        경제적 모델 + 최적 프롬프트 전략이 가장 효율적인 조합입니다.
                    </div>
                </div>
            </div>
            <div class="finding-card">
                <div class="finding-number">5</div>
                <div class="finding-content">
                    <div class="finding-title">경제적 모델로 MC=0.925 달성</div>
                    <div class="finding-detail">
                        self_verification 전략과 gpt-4o-mini 조합이 MC Score 0.925를 달성했습니다.
                        비싼 모델 없이도 적절한 프롬프트 전략으로 높은 메타인지 능력을 이끌어낼 수 있음을
                        보여줍니다. 이는 실무 적용에서 비용을 크게 절감할 수 있는 발견입니다.
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('rq4ScatterChart').getContext('2d'), {{
        type: 'scatter',
        data: {{
            datasets: [
                {{
                    label: 'Economic',
                    data: {eco_scatter_js},
                    backgroundColor: 'rgba(255,152,0,0.8)',
                    pointRadius: 8,
                    pointHoverRadius: 12,
                }},
                {{
                    label: 'Balanced',
                    data: {bal_scatter_js},
                    backgroundColor: 'rgba(0,212,255,0.8)',
                    pointRadius: 8,
                    pointHoverRadius: 12,
                }}
            ]
        }},
        options: {{
            responsive: true,
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Total Cost ($)', color: '#888' }},
                    ticks: {{ color: '#888' }},
                    grid: {{ color: 'rgba(255,255,255,0.05)' }}
                }},
                y: {{
                    beginAtZero: true, max: 100,
                    title: {{ display: true, text: 'MC Score (%)', color: '#888' }},
                    ticks: {{ color: '#888' }},
                    grid: {{ color: 'rgba(255,255,255,0.05)' }}
                }}
            }},
            plugins: {{
                legend: {{ labels: {{ color: '#ccc' }} }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const pt = context.raw;
                            return pt.label + ': MC=' + pt.y.toFixed(1) + '%, Cost=$' + pt.x.toFixed(4);
                        }}
                    }}
                }}
            }}
        }}
    }});
    </script>"""


def compute_strategy_type_matrix(
    results: List[MetacognitiveResult],
) -> Dict[str, Dict[str, tuple]]:
    """Compute Strategy x Transformation Type refusal rate matrix.

    Returns:
        {strategy: {transformation_type: (refusal_rate_pct, total_evals, unique_problems), ...}, ...}
    """
    unsolvable = [r for r in results if r.transformation_type != "original"]
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for r in unsolvable:
        strat = r.prompt_strategy
        ttype = r.transformation_type
        matrix.setdefault(strat, {}).setdefault(
            ttype, {"refused": 0, "total": 0, "problems": set()}
        )
        matrix[strat][ttype]["total"] += 1
        if r.response_type == "refused":
            matrix[strat][ttype]["refused"] += 1
        matrix[strat][ttype]["problems"].add(r.example_id)

    result: Dict[str, Dict[str, tuple]] = {}
    for strat, types in matrix.items():
        result[strat] = {}
        for ttype, counts in types.items():
            rate = (
                counts["refused"] / counts["total"] * 100 if counts["total"] > 0 else 0
            )
            result[strat][ttype] = (
                round(rate, 1),
                counts["total"],
                len(counts["problems"]),
            )

    return result


def compute_type_difficulty_ranking(
    results: List[MetacognitiveResult],
) -> List[Dict[str, Any]]:
    """Compute overall detection difficulty ranking by transformation type.

    Returns list sorted by refusal rate (ascending = harder to detect):
        [{"type": ..., "refusal_rate": ..., "total": ..., "refused": ...,
          "unique_problems": ...}, ...]
    """
    unsolvable = [r for r in results if r.transformation_type != "original"]
    counts: Dict[str, Dict[str, int]] = {}
    unique_problems: Dict[str, set] = {}

    for r in unsolvable:
        ttype = r.transformation_type
        counts.setdefault(ttype, {"refused": 0, "total": 0})
        counts[ttype]["total"] += 1
        if r.response_type == "refused":
            counts[ttype]["refused"] += 1
        unique_problems.setdefault(ttype, set()).add(r.example_id)

    ranking = []
    for ttype, c in counts.items():
        rate = c["refused"] / c["total"] * 100 if c["total"] > 0 else 0
        ranking.append(
            {
                "type": ttype,
                "refusal_rate": round(rate, 1),
                "total": c["total"],
                "refused": c["refused"],
                "unique_problems": len(unique_problems.get(ttype, set())),
            }
        )

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
            cases["all_model_failures"].append(
                {
                    "example_id": group[0].example_id,
                    "transformation_type": group[0].transformation_type,
                    "models": [r.model_name for r in group],
                    "note": "All models answered confidently on unsolvable problem",
                }
            )

        # cheap model wins: a cheaper model refused but expensive one didn't
        refused = [r for r in group if r.response_type == "refused"]
        confident = [r for r in group if r.response_type == "confident"]
        for ref_r in refused:
            for conf_r in confident:
                if ref_r.cost_usd < conf_r.cost_usd:
                    cases["cheap_model_wins"].append(
                        {
                            "example_id": ref_r.example_id,
                            "transformation_type": ref_r.transformation_type,
                            "cheap_model": ref_r.model_name,
                            "expensive_model": conf_r.model_name,
                        }
                    )

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
                cases["rag_deltas"].append(
                    {
                        "model": model_name,
                        "rag_on_refusal_acc": rag_on_metrics[
                            model_name
                        ].refusal_accuracy,
                        "rag_off_refusal_acc": rag_off_metrics[
                            model_name
                        ].refusal_accuracy,
                        "delta": delta,
                    }
                )

    return cases


def build_detailed_cases_html(
    all_results: List[MetacognitiveResult],
    full_context_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Build HTML for detailed per-problem case analysis section.

    Groups results by (example_id, transformation_type) and shows:
    - Original question and context (full text from lookup when available)
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

    # Count applied types per example_id for N/5 badge
    types_per_example: Dict[str, set] = {}
    for r in unsolvable:
        types_per_example.setdefault(r.example_id, set()).add(r.transformation_type)

    cards_html = ""
    for idx, (key, group) in enumerate(sorted(groups.items())):
        first = group[0]
        eid = first.example_id
        trans_type = first.transformation_type
        trans_desc = first.transformation_description or "N/A"
        question = first.question or "N/A"
        gt = first.ground_truth

        # Use full context from lookup if available, fallback to truncated version
        if full_context_lookup and key in full_context_lookup:
            ctx_orig = full_context_lookup[key]["original"]
            ctx_trans = full_context_lookup[key]["transformed"]
        else:
            ctx_orig = first.context_original or ""
            ctx_trans = first.context_transformed or ""

        # Escape HTML
        question_safe = html_module.escape(question)
        trans_desc_safe = html_module.escape(trans_desc)
        ctx_orig_safe = html_module.escape(ctx_orig)
        ctx_trans_safe = html_module.escape(ctx_trans)

        # Build model response rows
        model_rows = ""
        model_order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
        for r in sorted(
            group,
            key=lambda x: (model_order_map.get(x.model_name, 999), x.prompt_strategy),
        ):
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

            # Show full raw response (inside collapsible <details>)
            raw_resp = r.raw_response or ""
            raw_resp_display = html_module.escape(raw_resp)

            halluc_info = ""
            if r.hallucinated_values:
                halluc_info = (
                    f'<div class="halluc-tag">Hallucinated: '
                    f"{html_module.escape(', '.join(r.hallucinated_values[:5]))}</div>"
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

        n_types = len(types_per_example.get(eid, set()))
        cards_html += f"""
        <div class="detail-card">
            <details {"open" if idx < 3 else ""}>
                <summary class="detail-summary">
                    <span class="detail-id">{html_module.escape(eid)}</span>
                    <span class="detail-trans-type">{html_module.escape(trans_type)}</span>
                    <span style="background:rgba(0,212,255,0.12);color:#00d4ff;padding:2px 8px;border-radius:8px;font-size:0.78em;">{n_types}/5 types</span>
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
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


def build_methodology_section(design_ref_html: str = "") -> str:
    """Build HTML section explaining MC Score, methodology, and key findings."""
    return f"""
    <div class="methodology-section" id="methodology">
        <h2 style="color:#00d4ff;margin-bottom:20px;">연구 방법론 및 핵심 발견</h2>

        <!-- 2-1: MC Score 의미 및 산출 근거 -->
        <div class="methodology-subsection">
            <h3>MC Score 의미 및 산출 근거</h3>
            <div class="formula-box">
                <div class="formula-main">MC Score = Refusal F1 × (1 − Hallucination Rate)</div>
            </div>
            <table class="reference-table">
                <thead>
                    <tr><th>구성 요소</th><th>정의</th><th>의미</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Refusal Recall (RR)</strong></td>
                        <td>correctly_refused / total_unsolvable</td>
                        <td>풀 수 없는 문제를 얼마나 잘 거부하는가</td>
                    </tr>
                    <tr>
                        <td><strong>Refusal Precision (RP)</strong></td>
                        <td>correctly_refused / total_refused</td>
                        <td>거부한 문제 중 실제로 풀 수 없는 비율</td>
                    </tr>
                    <tr>
                        <td><strong>Refusal F1</strong></td>
                        <td>2 × RP × RR / (RP + RR)</td>
                        <td>거부 정확도의 조화 평균 (과잉 거부 페널티 포함)</td>
                    </tr>
                    <tr>
                        <td><strong>Hallucination Rate (HR)</strong></td>
                        <td>hallucinated / answering_responses</td>
                        <td>응답 중 존재하지 않는 값을 사용한 비율</td>
                    </tr>
                    <tr>
                        <td><strong>Over-Conservatism Rate</strong></td>
                        <td>solvable_refused / total_solvable</td>
                        <td>풀 수 있는 문제를 과잉 거부한 비율 (RP에 반영)</td>
                    </tr>
                </tbody>
            </table>

            <h4 style="color:#ce93d8;margin:20px 0 10px;">학술 근거 참조</h4>
            <p style="color:#aaa;font-size:0.88em;margin-bottom:10px;">
                MC Score는 기존 연구에서 확립된 메트릭을 조합·개선한 것으로, 임의로 설계한 것이 아닙니다.
                아래 표는 각 논문의 원래 메트릭이 어떻게 사용되었고, 본 연구에서 무엇을 개선하여 MC Score에 반영했는지를 정리합니다.
            </p>
            <table class="reference-table">
                <thead>
                    <tr>
                        <th style="min-width:140px;">참조</th>
                        <th>원논문 메트릭</th>
                        <th>원논문에서의 사용</th>
                        <th>본 연구 개선점</th>
                        <th>MC Score 기여</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Feng et al. (2024)</strong></td>
                        <td>Abstention F1</td>
                        <td>QA 시스템에서 "답변 불가" 판단의 정확도를 Precision/Recall의 조화 평균으로 측정.
                            답변 가능 질문과 불가 질문이 혼합된 데이터셋에서 abstention 임계값을 최적화하는 데 사용</td>
                        <td>원논문은 confidence threshold 기반의 자동 abstention을 평가했지만,
                            본 연구는 LLM의 <strong>자발적 거부(free-form refusal)</strong>를 대상으로 확장.
                            또한 금융 도메인의 5가지 변환 유형별로 세분화 적용</td>
                        <td><strong>Refusal F1</strong> — MC Score의 주요 구성 요소.
                            Refusal Precision(거부의 정확성)과 Recall(탐지 완전성)의 균형을 직접 반영</td>
                    </tr>
                    <tr>
                        <td><strong>Yang et al. (2024)</strong></td>
                        <td>Honesty Score</td>
                        <td>LLM의 "정직한 응답"을 평가하기 위해 정답률에서 과잉 거부(over-refusal) 비율을
                            감산하는 방식. 풀 수 있는 문제를 거부하면 페널티를 부여하여
                            "무조건 거부" 전략을 방지</td>
                        <td>원논문은 단순 감산(정답률 - 과잉거부율)이지만,
                            본 연구는 Refusal Precision에 과잉 거부를 <strong>내재화</strong>하여
                            F1 계산 시 자동으로 페널티가 반영되도록 구조화.
                            별도 감산 없이 Precision 분모에 false refusal이 포함됨</td>
                        <td><strong>Over-Conservatism Rate</strong> → Refusal Precision에 반영.
                            풀 수 있는 문제를 거부하면 RP가 하락하고, 이것이 F1을 낮추어 MC Score를 감소시킴</td>
                    </tr>
                    <tr>
                        <td><strong>Cheng et al. (2024)</strong></td>
                        <td>Balanced IDK Metrics</td>
                        <td>"I Don't Know(IDK)" 응답의 적절성을 측정하기 위해
                            IDK Precision(IDK 중 실제 모르는 것)과 IDK Recall(모르는 것 중 IDK 응답 비율)을
                            균형 있게 평가. 과잉 IDK와 과소 IDK를 동시에 페널티</td>
                        <td>원논문은 factual knowledge 범위의 IDK를 대상으로 했지만,
                            본 연구는 <strong>금융 추론 문제의 정보 부족</strong>이라는
                            도메인 특화 상황으로 적용 범위를 확장.
                            IDK 대신 refused/caveat/confident의 3단계 응답 분류 체계를 사용</td>
                        <td><strong>설계 철학</strong> — "거부와 응답의 균형" 원칙을 차용.
                            Refusal F1이 Precision과 Recall의 조화 평균인 이유가
                            이 논문의 균형 평가 방법론에 근거</td>
                    </tr>
                    <tr>
                        <td><strong>Rajpurkar et al. (2018)</strong></td>
                        <td>SQuAD 2.0 No-Answer F1</td>
                        <td>기계독해에서 답변 불가 질문(unanswerable)을 포함한 최초의 대규모 벤치마크.
                            모델이 "답변 없음"을 올바르게 예측하면 정답 처리하는 평가 체계를 확립.
                            HasAns F1과 NoAns F1을 별도 보고</td>
                        <td>원논문은 span extraction 기반의 정해진 형식이지만,
                            본 연구는 LLM의 <strong>자유 형식 응답</strong>에서 거부 의사를 탐지하는
                            RefusalDetector를 설계하여 "INSUFFICIENT_INFORMATION",
                            caveat 표현 등 다양한 거부 패턴을 인식</td>
                        <td><strong>실험 패러다임</strong> — "풀 수 있는 문제 + 풀 수 없는 문제" 혼합 구성의 근거.
                            Phase A(solvable) + Phase B(unsolvable) 2단계 구조가 SQuAD 2.0의 설계를 계승</td>
                    </tr>
                    <tr>
                        <td><strong>Kadavath et al. (2022)</strong></td>
                        <td>P(True) Calibration</td>
                        <td>LLM에게 자신의 답이 맞는지 직접 물어보는 방식으로 자기 인식(self-knowledge)을 측정.
                            모델이 출력한 확률과 실제 정답률 간의 calibration을 분석하여
                            "자신이 아는 것과 모르는 것의 경계"를 정량화</td>
                        <td>원논문은 P(True) 확률을 직접 추출했지만,
                            본 연구는 확률 접근이 불가능한 API 모델 환경에서
                            <strong>행동 기반(behavioral) 메타인지</strong>를 측정.
                            "자신이 모른다고 말하는 행동" 자체를 평가 대상으로 전환</td>
                        <td><strong>연구 동기</strong> — "LLM이 자신의 지식 한계를 아는가?"라는
                            핵심 질문의 출발점. MC Score는 이 질문을 금융 도메인에서 정량화한 지표</td>
                    </tr>
                    <tr>
                        <td><strong>Min et al. (2023)</strong></td>
                        <td>FActScore</td>
                        <td>LLM 생성 텍스트를 atomic fact 단위로 분해한 후,
                            각 fact가 출처에 의해 지지되는지 검증하여 환각(hallucination) 비율을 산출.
                            long-form generation의 사실 정확성을 세밀하게 측정</td>
                        <td>원논문은 Wikipedia 기반 전기문 생성의 fact-level 검증이지만,
                            본 연구는 금융 컨텍스트에 <strong>존재하지 않는 수치를 사용했는지</strong>를
                            검증하는 방식으로 단순화. 변환으로 제거된 값이 응답에 나타나면 hallucination으로 판정</td>
                        <td><strong>Hallucination Rate</strong> — MC Score의 감산 요소.
                            (1 - HR)을 곱하여 "거부를 잘 하더라도 응답 시 환각이 심하면 감점"되는 구조를 형성</td>
                    </tr>
                </tbody>
            </table>

            <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);border-radius:10px;padding:15px;margin-top:15px;">
                <h4 style="color:#00d4ff;margin:0 0 8px;">MC Score 설계 요약</h4>
                <p style="color:#bbb;font-size:0.88em;line-height:1.7;margin:0;">
                    <strong>Feng+2024</strong>의 Abstention F1 프레임워크를 핵심 골격으로 채택하되,
                    <strong>Yang+2024</strong>의 과잉 거부 페널티를 Precision에 내재화하고,
                    <strong>Min+2023</strong>의 환각 측정 개념을 감산 항으로 결합했습니다.
                    <strong>Rajpurkar+2018</strong>의 solvable/unsolvable 혼합 평가 패러다임 위에서,
                    <strong>Kadavath+2022</strong>가 제기한 "LLM의 자기 지식 한계 인식"이라는 질문을
                    금융 도메인의 정보 조작 실험으로 구체화한 것이 MC Score입니다.
                </p>
            </div>
        </div>

        <!-- 2-2: 실험 방법론 요약 -->
        <div class="methodology-subsection">
            <h3>실험 방법론 요약</h3>

            <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);border-radius:10px;padding:15px;margin-bottom:15px;">
                <h4 style="color:#00d4ff;margin:0 0 8px;">실험 설계: 교차 구조</h4>
                <p style="color:#bbb;font-size:0.88em;line-height:1.7;margin:0;">
                    본 실험은 <strong>5가지 변환 유형 &times; 4가지 프롬프트 전략 &times; 6개 모델</strong>의 교차 설계입니다.
                    이론적 최대 조합은 5&times;4&times;6 = 120이지만, <strong>context 형식에 따라 모든 변환이 적용되지는 않습니다</strong>:
                </p>
                <ul style="color:#aaa;font-size:0.85em;line-height:1.8;margin:8px 0 0 16px;">
                    <li><strong>EA-full(구조적 삭제)</strong>: 테이블/JSON 구조가 필요 — text context(hard.json의 71%)에서는 적용 불가</li>
                    <li><strong>TA(시간 모호화)</strong>: question에 4자리 연도(20xx)가 필요 — 연도가 없는 질문에는 적용 불가</li>
                    <li><strong>Validation 필터</strong>: 변환 후에도 정답 도출 가능(still_solvable)하면 제외</li>
                </ul>
                <p style="color:#bbb;font-size:0.88em;line-height:1.7;margin:8px 0 0;">
                    따라서 각 테이블/차트의 <strong>N(샘플 수)</strong>은 변환 유형마다 크게 다릅니다.
                    N&lt;30인 결과는 시각적으로 흐리게 표시되며, 해석 시 주의가 필요합니다.
                </p>
            </div>

            <h4 style="color:#ce93d8;margin:15px 0 8px;">4단계 실험 파이프라인</h4>
            <table class="reference-table">
                <thead>
                    <tr><th>Phase</th><th>설명</th><th>목적</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Phase A</strong></td>
                        <td>원본 문제 Baseline 정확도 측정</td>
                        <td>모델의 기본 금융 추론 능력 + 과잉 거부(Over-Conservatism) 측정</td>
                    </tr>
                    <tr>
                        <td><strong>Phase B</strong></td>
                        <td>변환된 unsolvable 문제 → 거부/탐지 테스트</td>
                        <td>정보 부족 상황에서 거부 능력 측정 (Refusal Recall)</td>
                    </tr>
                    <tr>
                        <td><strong>Phase C</strong></td>
                        <td>RAG 활성화 상태에서 Phase B 반복</td>
                        <td>금융 함수 검색이 메타인지 능력에 미치는 영향 분석</td>
                    </tr>
                    <tr>
                        <td><strong>Phase D</strong></td>
                        <td>A~C 결과 종합 분석 (API 호출 없음)</td>
                        <td>비용-최적 전략 도출 및 교차 분석</td>
                    </tr>
                </tbody>
            </table>

            <h4 style="color:#ce93d8;margin:15px 0 8px;">5가지 변환 유형</h4>
            <table class="reference-table">
                <thead>
                    <tr><th>Type</th><th>이름</th><th>설명</th><th>탐지 난이도</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>Information Removal</td>
                        <td>[DATA MISSING] 마커로 교체</td>
                        <td style="color:#4caf50;">낮음</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>Table Column Removal</td>
                        <td>테이블 컬럼 전체 삭제</td>
                        <td style="color:#ff9800;">중간</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>Ambiguous Time Period</td>
                        <td>연도를 모호한 표현으로 대체</td>
                        <td style="color:#ff9800;">중간</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>Critical Data Removal</td>
                        <td>핵심 수치 전체 삭제 (마커 없이)</td>
                        <td style="color:#f44336;">높음</td>
                    </tr>
                    <tr>
                        <td>5</td>
                        <td>Contradictory Information</td>
                        <td>모순 문장 삽입 (1.5배 값)</td>
                        <td style="color:#d32f2f;">매우 높음</td>
                    </tr>
                </tbody>
            </table>

            <h4 style="color:#ce93d8;margin:15px 0 8px;">변환 유형 상세 설계 (Design Reference)</h4>
            <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
                각 변환 유형의 설계 근거, 도메인 특화도, 탐지 난이도, 그리고 Before/After 예시입니다.
            </p>
            {design_ref_html}

            <h4 style="color:#ce93d8;margin:15px 0 8px;">4가지 프롬프트 전략</h4>
            <table class="reference-table">
                <thead>
                    <tr><th>전략</th><th>설명</th><th>거부 지시</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>standard</td>
                        <td>FinanceReasoning 원본 COT/POT 프롬프트 (대조군)</td>
                        <td>없음</td>
                    </tr>
                    <tr>
                        <td>metacognitive</td>
                        <td>"정보 부족 시 INSUFFICIENT_INFORMATION" 지시 추가</td>
                        <td>명시적</td>
                    </tr>
                    <tr>
                        <td>self_verification</td>
                        <td>DATA AUDIT → SOLUTION 2단계 (충분할 때만 풀이)</td>
                        <td>구조적</td>
                    </tr>
                    <tr>
                        <td>contradiction_aware</td>
                        <td>모순 탐지 + 정보 부족 인식 강화</td>
                        <td>모순 특화</td>
                    </tr>
                </tbody>
            </table>

            <h4 style="color:#ce93d8;margin:15px 0 8px;">모델 구성 (2 Tier × 3 Provider)</h4>
            <table class="reference-table">
                <thead>
                    <tr><th>Tier</th><th>OpenAI</th><th>Anthropic</th><th>Google</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span style="color:#ff9800;">Economic</span></td>
                        <td>gpt-4o-mini</td>
                        <td>claude-haiku-4</td>
                        <td>gemini-2.5-flash</td>
                    </tr>
                    <tr>
                        <td><span style="color:#00d4ff;">Balanced</span></td>
                        <td>gpt-4o</td>
                        <td>claude-sonnet-4</td>
                        <td>gemini-2.5-pro</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>"""


# ============================================================================
# DATA INTEGRITY SECTION
# ============================================================================


def build_data_integrity_section(
    summary: FilteringSummary,
    raw_metrics: Dict[str, MetacognitiveMetrics],
    filtered_metrics: Dict[str, MetacognitiveMetrics],
) -> str:
    """Build Data Integrity section: before/after comparison of contamination filtering.

    Args:
        summary: FilteringSummary from filter_contaminated_results
        raw_metrics: Metrics computed from unfiltered Phase B results
        filtered_metrics: Metrics computed from filtered Phase B results

    Returns:
        HTML string for the Data Integrity section
    """
    if summary.excluded_count == 0:
        return """
    <div class="section" id="data-integrity">
        <h2>Data Integrity Filter</h2>
        <p style="color:#4caf50;">No contaminated results detected. All Phase B results are clean.</p>
    </div>"""

    # --- Warning banner ---
    banner_html = f"""
    <div style="background:rgba(255,152,0,0.15);border:1px solid #ff9800;border-radius:8px;padding:12px 16px;margin-bottom:16px;">
        <strong style="color:#ff9800;">Data Integrity Warning:</strong>
        Phase B {summary.total_phase_b}건 중 <strong>{summary.excluded_count}건 ({summary.excluded_pct:.1f}%)</strong> 제외됨.
        하드코딩 솔루션 문제에서 CONFIDENT+정답 응답은 변환 무효화로 인한 오염 데이터입니다.
    </div>"""

    # --- Before/After comparison table ---
    all_models = _ordered_models(
        set(list(raw_metrics.keys()) + list(filtered_metrics.keys()))
    )

    comparison_rows = ""
    for m_name in all_models:
        raw_m = raw_metrics.get(m_name)
        filt_m = filtered_metrics.get(m_name)
        if not raw_m and not filt_m:
            continue

        raw_mc = raw_m.mc_score if raw_m else 0
        filt_mc = filt_m.mc_score if filt_m else 0
        delta = filt_mc - raw_mc

        raw_rr = raw_m.refusal_recall if raw_m else 0
        filt_rr = filt_m.refusal_recall if filt_m else 0

        raw_fc = raw_m.false_confidence_rate if raw_m else 0
        filt_fc = filt_m.false_confidence_rate if filt_m else 0

        raw_n = raw_m.total_unsolvable if raw_m else 0
        filt_n = filt_m.total_unsolvable if filt_m else 0

        delta_color = "#4caf50" if delta >= 0 else "#f44336"
        delta_str = (
            f'<span style="color:{delta_color};font-weight:bold;">{delta:+.3f}</span>'
        )

        excluded_for_model = summary.excluded_by_model.get(m_name, 0)

        comparison_rows += f"""
        <tr>
            <td>{html_module.escape(m_name)}</td>
            <td style="text-align:center;">{raw_n}</td>
            <td style="text-align:center;">{filt_n}</td>
            <td style="text-align:center;">{excluded_for_model}</td>
            <td style="text-align:center;">{raw_rr:.1%}</td>
            <td style="text-align:center;">{filt_rr:.1%}</td>
            <td style="text-align:center;">{raw_fc:.1%}</td>
            <td style="text-align:center;">{filt_fc:.1%}</td>
            <td style="text-align:center;">{raw_mc:.3f}</td>
            <td style="text-align:center;">{filt_mc:.3f}</td>
            <td style="text-align:center;">{delta_str}</td>
        </tr>"""

    # --- Excluded by transformation type breakdown ---
    type_breakdown_rows = ""
    for t_type, count in sorted(summary.excluded_by_type.items(), key=lambda x: -x[1]):
        type_breakdown_rows += f"""
        <tr>
            <td>{html_module.escape(t_type)}</td>
            <td style="text-align:center;">{count}</td>
        </tr>"""

    # --- Technical explanation (collapsible) ---
    technical_html = f"""
    <details style="margin-top:12px;">
        <summary style="cursor:pointer;color:#888;font-size:0.85em;">
            Technical Details: Why filter hardcoded solutions?
        </summary>
        <div style="padding:8px 0;color:#aaa;font-size:0.85em;line-height:1.6;">
            <p>hard.json 238문제 중 <strong>{summary.hardcoded_problem_count}문제 ({summary.hardcoded_problem_count / summary.total_problems_in_dataset * 100:.0f}%)</strong>의
            <code>python_solution</code>이 context를 파싱하지 않고 숫자를 직접 하드코딩합니다.</p>
            <p>Phase B에서 context를 변환(정보 제거/모순 삽입)해도, 하드코딩 문제의 ground truth 솔루션은
            동일한 정답을 산출합니다. 따라서 모델이 CONFIDENT하게 정답을 맞추더라도,
            이는 메타인지 능력이 아니라 <strong>변환이 무효화</strong>된 것입니다.</p>
            <p><strong>필터링 규칙:</strong> 하드코딩 문제 AND unsolvable 변환 AND CONFIDENT 응답 AND
            predicted &asymp; ground_truth (0.2% 허용 오차) → 제외.
            REFUSED/CAVEAT/ERROR/오답은 유효한 데이터로 유지합니다.</p>
        </div>
    </details>"""

    return f"""
    <div class="section" id="data-integrity">
        <h2>Data Integrity Filter</h2>
        {banner_html}
        <h3 style="margin-top:16px;">Before / After Comparison</h3>
        <p style="color:#888;font-size:0.85em;margin-bottom:8px;">
            Raw: 필터링 전 전체 Phase B 결과. Filtered: 오염 데이터 제외 후.
            RQ1~4 섹션은 모두 Filtered 메트릭을 사용합니다.
        </p>
        <div style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Raw N</th>
                    <th>Filtered N</th>
                    <th>Excluded</th>
                    <th>Raw Ref Recall</th>
                    <th>Filt Ref Recall</th>
                    <th>Raw False Conf</th>
                    <th>Filt False Conf</th>
                    <th>Raw MC</th>
                    <th>Filt MC</th>
                    <th>Delta</th>
                </tr>
            </thead>
            <tbody>{comparison_rows}</tbody>
        </table>
        </div>

        <details style="margin-top:12px;">
            <summary style="cursor:pointer;color:#888;">
                Excluded by Transformation Type ({summary.excluded_count} total)
            </summary>
            <table style="margin-top:8px;max-width:400px;">
                <thead><tr><th>Transformation Type</th><th>Excluded</th></tr></thead>
                <tbody>{type_breakdown_rows}</tbody>
            </table>
        </details>

        {technical_html}
    </div>"""


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
    unsolvable_raw = [r for r in all_results if r.transformation_type != "original"]

    # --- Data Integrity: filter contaminated results ---
    project_root = Path(__file__).parent.parent
    hard_json_path = (
        project_root
        / "data"
        / "financereasoning"
        / "raw"
        / "FinanceReasoning"
        / "hard.json"
    )
    hardcoded_ids = build_hardcoded_problem_ids(hard_json_path)
    unsolvable, excluded_results, filtering_summary = filter_contaminated_results(
        unsolvable_raw, hardcoded_ids
    )
    print(
        f"  Data Integrity: {filtering_summary.excluded_count}/{filtering_summary.total_phase_b} "
        f"contaminated results excluded ({filtering_summary.excluded_pct:.1f}%)"
    )

    # Compute raw metrics (before filtering) for comparison
    if baseline and unsolvable_raw:
        raw_model_metrics = compute_cross_phase_metrics(baseline, unsolvable_raw)
    else:
        raw_model_metrics = compute_metrics(unsolvable_raw) if unsolvable_raw else {}

    # Filtered (primary) metrics
    if baseline and unsolvable:
        model_metrics = compute_cross_phase_metrics(baseline, unsolvable)
    else:
        model_metrics = compute_metrics(unsolvable) if unsolvable else {}
    strategy_metrics = (
        compute_metrics_by_dimension(unsolvable, "prompt_strategy")
        if unsolvable
        else {}
    )
    transform_metrics = (
        compute_metrics_by_dimension(unsolvable, "transformation_type")
        if unsolvable
        else {}
    )

    # Use filtered results for all downstream computations
    all_results_filtered = baseline + unsolvable

    # Strategy × Type matrix
    strategy_type_matrix = compute_strategy_type_matrix(all_results_filtered)
    type_difficulty = compute_type_difficulty_ranking(all_results_filtered)

    # Interesting cases (use filtered)
    cases = find_interesting_cases(all_results_filtered)

    # Build full context lookup from original data
    full_context_lookup = build_full_context_lookup(project_root)

    # Build detailed case analysis HTML (use filtered)
    detailed_cases_html = build_detailed_cases_html(
        all_results_filtered, full_context_lookup
    )

    # Load prompt catalogs for reference section
    prompt_catalogs = load_prompt_catalogs(results_dir)
    prompt_reference_html = build_prompt_reference_html(prompt_catalogs)

    # Build Data Integrity section
    data_integrity_html = build_data_integrity_section(
        filtering_summary, raw_model_metrics, model_metrics
    )

    # Build RQ sections (all use filtered metrics)
    rq1_html = build_rq1_section(all_results_filtered, type_difficulty)
    rq2_html = build_rq2_section(model_metrics, strategy_metrics)
    rq3_html = build_rq3_section(all_results_filtered, model_metrics)
    rq4_html = build_rq4_section(model_metrics, strategy_metrics)

    # Build transformation design reference cards (used inside methodology)
    design_ref_html = ""
    for ref in TRANSFORMATION_DESIGN_REFERENCE:
        domain_color = {"높음": "#4caf50", "중간": "#ff9800", "낮음": "#f44336"}.get(
            ref["domain_specificity"], "#888"
        )
        diff_color = {
            "낮음": "#4caf50",
            "중간": "#ff9800",
            "높음": "#f44336",
            "매우 높음": "#d32f2f",
        }.get(ref["detection_difficulty"], "#888")
        design_ref_html += f"""
        <div class="prompt-ref-card">
            <details>
                <summary class="prompt-ref-summary">
                    <span class="prompt-ref-name">{html_module.escape(ref["type"])}</span>
                    <span style="color:{domain_color};font-size:0.8em;">도메인: {html_module.escape(ref["domain_specificity"])}</span>
                    <span style="color:{diff_color};font-size:0.8em;">탐지: {html_module.escape(ref["detection_difficulty"])}</span>
                </summary>
                <div class="prompt-ref-body">
                    <p style="margin-bottom:8px;">{html_module.escape(ref["description"])}</p>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                        <div class="context-panel">
                            <div class="context-label">Before</div>
                            <pre class="context-pre">{html_module.escape(ref["before"])}</pre>
                        </div>
                        <div class="context-panel">
                            <div class="context-label">After</div>
                            <pre class="context-pre">{html_module.escape(ref["after"])}</pre>
                        </div>
                    </div>
                </div>
            </details>
        </div>"""

    # Build methodology section (includes design reference)
    methodology_html = build_methodology_section(design_ref_html)

    # Previous results for trends
    prev_results = load_previous_results(results_dir, current_ts)
    prev_metrics = (
        compute_metrics(
            [r for r in prev_results if r.transformation_type != "original"]
        )
        if prev_results
        else {}
    )

    # Phase tracking (use all_results for totals — cost/phases reflect all API calls)
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
    models = _ordered_models({r.model_name for r in all_results})

    # 데이터 파이프라인 요약: 고유 문제 수 / 유효 변환 쌍 수 / 총 평가 수
    unique_base_problems = len({r.example_id for r in all_results_filtered})
    unique_transform_pairs = len(
        {(r.example_id, r.transformation_type) for r in unsolvable}
    )

    # Prepare chart data
    models_with_metrics = [m for m in models if m in model_metrics]
    model_names_js = json.dumps(models_with_metrics)
    refusal_accs_js = json.dumps(
        [round(model_metrics[m].refusal_recall * 100, 1) for m in models_with_metrics]
    )
    false_conf_js = json.dumps(
        [
            round(model_metrics[m].false_confidence_rate * 100, 1)
            for m in models_with_metrics
        ]
    )
    mc_scores_js = json.dumps(
        [round(model_metrics[m].mc_score * 100, 1) for m in models_with_metrics]
    )
    refusal_prec_js = json.dumps(
        [
            round(model_metrics[m].refusal_precision * 100, 1)
            for m in models_with_metrics
        ]
    )
    refusal_f1_js = json.dumps(
        [round(model_metrics[m].refusal_f1 * 100, 1) for m in models_with_metrics]
    )
    ocr_js = json.dumps(
        [
            round(model_metrics[m].over_conservatism_rate * 100, 1)
            for m in models_with_metrics
        ]
    )

    # Baseline accuracy (model × strategy breakdown, normalized to 10)
    # Collect per (model, strategy) then cap at 10 for fair comparison
    BASELINE_CAP = 10
    _baseline_raw: Dict[str, Dict[str, List[bool]]] = {}
    for r in baseline:
        m = r.model_name
        s = r.prompt_strategy or "standard"
        _baseline_raw.setdefault(m, {}).setdefault(s, [])
        _baseline_raw[m][s].append(bool(r.is_original_correct))

    baseline_by_model_strategy: Dict[str, Dict[str, Dict]] = {}
    for m, strats in _baseline_raw.items():
        baseline_by_model_strategy[m] = {}
        for s, correctness_list in strats.items():
            capped = correctness_list[:BASELINE_CAP]
            baseline_by_model_strategy[m][s] = {
                "correct": sum(capped),
                "total": len(capped),
                "total_raw": len(correctness_list),
            }

    # Heatmap data: model x transformation_type
    heatmap_transforms = sorted({r.transformation_type for r in unsolvable})
    heatmap_data = []
    heatmap_n_data = []  # unique problem counts per model×type for tooltip
    for m_name in models:
        row = []
        n_row = []
        for t_type in heatmap_transforms:
            matching = [
                r
                for r in unsolvable
                if r.model_name == m_name and r.transformation_type == t_type
            ]
            if matching:
                refused = sum(1 for r in matching if r.response_type == "refused")
                rate = refused / len(matching) * 100
                row.append(round(rate, 1))
                n_row.append(len({r.example_id for r in matching}))
            else:
                row.append(None)
                n_row.append(0)
        heatmap_data.append(row)
        heatmap_n_data.append(n_row)

    # Include unique problem count in transform labels for Model×Type heatmap
    heatmap_type_unique = {}
    for t_type in heatmap_transforms:
        heatmap_type_unique[t_type] = len(
            {r.example_id for r in unsolvable if r.transformation_type == t_type}
        )
    heatmap_transforms_js = json.dumps(
        [f"{t[:25]} (n={heatmap_type_unique[t]}문제)" for t in heatmap_transforms]
    )
    heatmap_models_js = json.dumps(models)
    heatmap_data_js = json.dumps(heatmap_data)
    heatmap_n_data_js = json.dumps(heatmap_n_data)

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
            <div class="case-id">{html_module.escape(str(case["example_id"]))}</div>
            <div class="case-detail">{html_module.escape(str(case["transformation_type"]))}</div>
            <div class="case-models">Models: {html_module.escape(", ".join(case["models"]))}</div>
        </div>"""

    for case in cases.get("cheap_model_wins", [])[:5]:
        failure_cases_html += f"""
        <div class="case-card success">
            <div class="case-badge">CHEAP WIN</div>
            <div class="case-id">{html_module.escape(str(case["example_id"]))}</div>
            <div class="case-detail">{html_module.escape(str(case["cheap_model"]))} beat {html_module.escape(str(case["expensive_model"]))}</div>
        </div>"""

    for case in cases.get("rag_deltas", [])[:5]:
        direction = "positive" if case["delta"] > 0 else "negative"
        failure_cases_html += f"""
        <div class="case-card {"success" if case["delta"] > 0 else "warning"}">
            <div class="case-badge">RAG {"+" if case["delta"] > 0 else "-"}</div>
            <div class="case-id">{html_module.escape(str(case["model"]))}</div>
            <div class="case-detail">RefAcc: {case["rag_off_refusal_acc"]:.1%} → {case["rag_on_refusal_acc"]:.1%} ({case["delta"]:+.1%})</div>
        </div>"""

    if not failure_cases_html:
        failure_cases_html = '<div class="case-card"><div class="case-detail">No interesting cases found yet. Run more experiments.</div></div>'

    # Baseline table (model × strategy matrix, normalized to BASELINE_CAP)
    all_baseline_strategies = sorted(
        {s for strats in baseline_by_model_strategy.values() for s in strats}
    )
    all_baseline_strategies = _ordered_strategies(all_baseline_strategies)

    baseline_header_cells = "".join(
        f"<th>{html_module.escape(s)}</th>" for s in all_baseline_strategies
    )
    baseline_rows = ""
    for m_name in models:
        model_strats = baseline_by_model_strategy.get(m_name, {})
        if not model_strats:
            continue
        cells = ""
        for strat in all_baseline_strategies:
            bm = model_strats.get(strat)
            if bm:
                correct = bm["correct"]
                total = bm["total"]
                acc = correct / total * 100 if total > 0 else 0
                # Color by accuracy
                if acc >= 80:
                    color = "#4caf50"
                elif acc >= 50:
                    color = "#ff9800"
                else:
                    color = "#f44336"
                truncated = (
                    f' <span style="font-size:0.7em;color:#666;">(of {bm["total_raw"]})</span>'
                    if bm["total_raw"] > total
                    else ""
                )
                cells += f'<td style="text-align:center;"><span style="color:{color};font-weight:700;font-size:1.1em;">{correct}</span>/{total}{truncated}</td>'
            else:
                cells += '<td style="text-align:center;color:#555;">-</td>'
        baseline_rows += f"<tr><td>{html_module.escape(m_name)}</td>{cells}</tr>"

    # --- 3A: Strategy × Type Heatmap table (CSS gradient) ---
    strat_type_strategies = _ordered_strategies(strategy_type_matrix.keys())
    all_trans_types = sorted(
        {tt for strat_types in strategy_type_matrix.values() for tt in strat_types}
    )
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
                cell = strategy_type_matrix.get(strat, {}).get(tt)
                if cell is not None:
                    val, n_evals, n_problems = cell
                    r_color = int(255 * (1 - val / 100))
                    g_color = int(255 * (val / 100))
                    bg = f"rgba({r_color},{g_color},0,0.25)"
                    low_n_style = "opacity:0.5;" if n_problems < 3 else ""
                    n_label = f'<div style="font-size:0.7em;color:#888;font-weight:normal;">n={n_problems}문제</div>'
                    row_cells += f'<td style="background:{bg};text-align:center;font-weight:bold;{low_n_style}">{val:.0f}%{n_label}</td>'
                else:
                    row_cells += '<td style="text-align:center;color:#555;">-</td>'
            tt_short = tt[:35] if len(tt) > 35 else tt
            heatmap_table_html += (
                f"<tr><td>{html_module.escape(tt_short)}</td>{row_cells}</tr>"
            )
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
            strat_mc_datasets[m_name].append(round(sm.mc_score * 100, 1) if sm else 0)
    strat_mc_labels_js = json.dumps(strat_mc_labels)

    strat_mc_chart_datasets_js_parts = []
    chart_colors = [
        "rgba(0,212,255,0.7)",
        "rgba(76,175,80,0.7)",
        "rgba(255,152,0,0.7)",
        "rgba(156,39,176,0.7)",
        "rgba(244,67,54,0.7)",
        "rgba(255,235,59,0.7)",
    ]
    for idx, (m_name, vals) in enumerate(strat_mc_datasets.items()):
        color = chart_colors[idx % len(chart_colors)]
        strat_mc_chart_datasets_js_parts.append(
            f"{{label:'{m_name}',data:{json.dumps(vals)},backgroundColor:'{color}',borderRadius:4}}"
        )
    strat_mc_chart_datasets_js = "[" + ",".join(strat_mc_chart_datasets_js_parts) + "]"

    # --- 3C: Type difficulty ranking data ---
    diff_labels_js = json.dumps(
        [f"{d['type'][:25]} (n={d['unique_problems']}문제)" for d in type_difficulty]
    )
    diff_values_js = json.dumps([d["refusal_rate"] for d in type_difficulty])
    diff_counts_js = json.dumps(
        [
            f"{d['refused']}/{d['total']}평가 ({d['unique_problems']}문제)"
            for d in type_difficulty
        ]
    )

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

        /* RQ Sections */
        .rq-section {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(0,212,255,0.15);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .rq-section h2 {{ margin-top: 0; }}
        .rq-description {{ color: #aaa; font-size: 0.9em; margin-bottom: 15px; }}
        .rq-summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
        }}
        .rq-insight-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            border-left: 3px solid #00d4ff;
        }}
        .rq-insight-label {{ color: #888; font-size: 0.8em; margin-bottom: 4px; }}
        .rq-insight-value {{ color: #e0e0e0; font-weight: bold; font-size: 1.1em; }}
        .rq-insight-detail {{ color: #666; font-size: 0.8em; margin-top: 3px; }}

        /* RQ Nav */
        .rq-nav {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }}
        .rq-nav a {{
            padding: 8px 20px;
            border-radius: 20px;
            background: rgba(0,212,255,0.1);
            color: #00d4ff;
            text-decoration: none;
            font-weight: bold;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        .rq-nav a:hover {{ background: rgba(0,212,255,0.25); }}

        .footer {{ text-align: center; color: #555; font-size: 0.8em; margin-top: 30px; padding: 15px; }}

        /* Methodology Section */
        .methodology-section {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(0,212,255,0.15);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }}
        .methodology-subsection {{
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
        }}
        .methodology-subsection h3 {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.15em;
        }}
        .methodology-subsection h4 {{
            color: #ce93d8;
        }}
        .formula-box {{
            background: rgba(0,212,255,0.08);
            border: 1px solid rgba(0,212,255,0.25);
            border-radius: 10px;
            padding: 18px 24px;
            margin: 15px 0;
            text-align: center;
        }}
        .formula-main {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 1.2em;
            color: #00d4ff;
            font-weight: bold;
        }}
        .reference-table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }}
        .reference-table th {{
            background: rgba(0,212,255,0.08);
            color: #00d4ff;
            padding: 10px 14px;
            text-align: left;
            font-size: 0.88em;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .reference-table td {{
            padding: 10px 14px;
            font-size: 0.86em;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            color: #ccc;
        }}
        .findings-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }}
        .finding-card {{
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            padding: 18px;
            display: flex;
            gap: 16px;
            align-items: flex-start;
            border-left: 3px solid #00d4ff;
        }}
        .finding-number {{
            background: rgba(0,212,255,0.15);
            color: #00d4ff;
            font-weight: bold;
            font-size: 1.2em;
            min-width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .finding-content {{ flex: 1; }}
        .finding-title {{
            color: #e0e0e0;
            font-weight: bold;
            font-size: 1em;
            margin-bottom: 6px;
        }}
        .finding-detail {{
            color: #aaa;
            font-size: 0.88em;
            line-height: 1.6;
        }}

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
            max-height: 500px;
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
        <span class="phase-badge {"phase-done" if "A" in phases_done else "phase-pending"}">Phase A: Baseline</span>
        <span class="phase-badge {"phase-done" if "B" in phases_done else "phase-pending"}">Phase B: Metacognitive</span>
        <span class="phase-badge {"phase-done" if "C" in phases_done else "phase-pending"}">Phase C: RAG Impact</span>
        <span class="phase-badge {"phase-done" if len(phases_done) >= 3 else "phase-pending"}">Phase D: Analysis</span>
    </div>

    <!-- Summary Cards -->
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-value">{unique_base_problems}</div>
            <div class="summary-label">Base Problems</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{unique_transform_pairs}</div>
            <div class="summary-label">Transformation Pairs</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{total_evals}</div>
            <div class="summary-label">Total Evaluations</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{len(models)}</div>
            <div class="summary-label">Models Tested</div>
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
    <p style="color:#888;font-size:0.8em;text-align:center;margin:8px 0 0 0;">
        데이터 파이프라인: hard.json {unique_base_problems}문제 &rarr;
        5가지 변환 적용 &rarr; {unique_transform_pairs}개 유효 변환 쌍
        (context 형식·validation 필터 적용) &rarr;
        {len(models)}모델 &times; 4전략 = {total_evals}회 평가
    </p>

    <!-- RQ Navigation -->
    <div class="rq-nav">
        <a href="#methodology">Methodology</a>
        <a href="#data-integrity">Data Integrity</a>
        <a href="#rq1">RQ1: Information Detection</a>
        <a href="#rq2">RQ2: Model Size</a>
        <a href="#rq3">RQ3: RAG Impact</a>
        <a href="#rq4">RQ4: Cost-Optimal</a>
    </div>

    <!-- Methodology (moved to top for context before RQ sections) -->
    {methodology_html}

    <!-- Data Integrity Filter -->
    {data_integrity_html}

    <!-- RQ1: Information Insufficiency Detection -->
    {rq1_html}

    <!-- RQ2: Model Size Effect -->
    {rq2_html}

    <!-- RQ3: RAG Impact -->
    {rq3_html}

    <!-- RQ4: Cost-Optimal Strategy -->
    {rq4_html}

    <!-- Core Metrics Charts -->
    <div class="section">
        <h2>Core Metacognitive Metrics</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            교차 분석 (Cross-phase): Phase A(solvable) + Phase B(unsolvable) 결과를 통합하여
            Refusal Precision에 과잉 거부(Phase A에서의 false refusal)가 반영됩니다.
            모든 전략·tier의 결과가 모델별로 집계됩니다.
        </p>
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
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            모델별 종합 메트릭입니다. MC Score = Refusal F1 &times; (1 - Hallucination Rate).
            Delta는 이전 실험 대비 변화량입니다. 모든 전략·phase의 결과가 통합된 값입니다.
        </p>
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
        <h2>Phase A: Baseline Accuracy (Correct / {BASELINE_CAP})</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            Phase A는 원본 문제에 대한 baseline 정확도를 측정합니다.
            공정한 비교를 위해 각 (모델, 전략) 조합당 최대 <strong>{BASELINE_CAP}개</strong>로 통일했습니다.
            셀 값은 <strong>정답 수 / {BASELINE_CAP}</strong>이며,
            <span style="color:#4caf50;">초록</span>=80%+,
            <span style="color:#ff9800;">주황</span>=50%+,
            <span style="color:#f44336;">빨강</span>=50% 미만.
        </p>
        <table>
            <thead>
                <tr><th>Model</th>{baseline_header_cells}</tr>
            </thead>
            <tbody>{baseline_rows if baseline_rows else '<tr><td colspan="' + str(len(all_baseline_strategies) + 1) + '" style="text-align:center;color:#666">Run Phase A to see baseline</td></tr>'}</tbody>
        </table>
    </div>

    <!-- 3A: Strategy × Type Heatmap -->
    <div class="section">
        <h2>Strategy × Transformation Type Refusal Heatmap</h2>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
            실험 조건: Phase B 전체 데이터 (economic + balanced, 4가지 프롬프트 전략).
            각 셀은 해당 전략&times;변환 조합의 거부율(%)을 나타냅니다 (모든 모델 평균).
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
            각 문제에 대해 4+1가지 변환 유형을 시도하지만, 일부는 적용되지 않을 수 있습니다:
            <br>&bull; <strong>Context 형식 제약</strong>: TEXT 형식은 EA-full(구조적 삭제)을 지원하지 않습니다 (테이블 구조 필요)
            <br>&bull; <strong>변환 함수 한계</strong>: 질문에 연도가 없으면 TA(시간 모호화) 적용 불가
            <br>&bull; <strong>Validation 필터</strong>: 변환 후에도 정답 도출이 가능한 경우(still_solvable, &plusmn;1% 이내) 실험에서 제외
            <br>각 카드에서 적용된 변환 유형 수를 확인할 수 있습니다.
        </p>
        <p style="color:#888;font-size:0.85em;margin-bottom:12px;">
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
const heatmapNData = {heatmap_n_data_js};

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
            legend: {{ labels: {{ color: '#ccc' }} }},
            tooltip: {{
                callbacks: {{
                    afterLabel: function(context) {{
                        const modelIdx = context.datasetIndex;
                        const typeIdx = context.dataIndex;
                        const n = heatmapNData[modelIdx][typeIdx];
                        return n + '문제' + (n < 3 ? ' (표본 부족)' : '');
                    }}
                }}
            }}
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
    parser = argparse.ArgumentParser(
        description="Generate Metacognitive Evaluation Dashboard"
    )
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
