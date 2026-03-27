"""
Dataset Audit Dashboard Generator

FinanceReasoning 전체 데이터셋(easy 1000 + medium 1000 + hard 238 = 2,238문제)에 대해
5가지 변환의 적용 가능성과 validation 결과를 체계적으로 분석하는 독립 HTML 대시보드.

Usage:
    python experiments/generate_dataset_audit.py
    python experiments/generate_dataset_audit.py --levels hard
    python experiments/generate_dataset_audit.py --output path/to/output.html
"""

import json
import re
import sys
import html as html_module
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from apply_transformations_full import (
    apply_transformations,
    validate_transformation,
    detect_context_type,
)
from hardcoded_solution_detector import _solution_uses_hardcoded_values


# ============================================================================
# DATACLASSES
# ============================================================================

TRANSFORMATION_TYPES = [
    "Type 1: Information Removal",
    "Type 2: Table Column Removal",
    "Type 3: Ambiguous Time Period",
    "Type 4: Critical Data Removal",
    "Type 5: Contradictory Information",
]


@dataclass
class TransformationAudit:
    type_name: str
    is_applicable: bool
    inapplicable_reason: str
    is_valid: bool
    validation_reason: str
    description: str


@dataclass
class ProblemAudit:
    question_id: str
    level: str
    source: str
    context_type: str
    is_hardcoded: bool
    question_length: int
    context_length: int
    transformations: List[TransformationAudit]
    applicable_count: int
    valid_count: int


@dataclass
class DatasetAuditStats:
    total_problems: int = 0
    context_type_counts: Dict[str, int] = field(default_factory=dict)
    hardcoded_count: int = 0
    total_transformations: int = 0
    valid_transformations: int = 0
    zero_transform_count: int = 0
    # Per-type stats
    type_applicable: Dict[str, int] = field(default_factory=dict)
    type_valid: Dict[str, int] = field(default_factory=dict)
    type_inapplicable_reasons: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Validation reason breakdown
    validation_reasons: Dict[str, int] = field(default_factory=dict)
    # Context type × transformation type matrix
    context_type_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    context_type_valid_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def hardcoded_pct(self) -> float:
        if self.total_problems == 0:
            return 0.0
        return self.hardcoded_count / self.total_problems * 100

    @property
    def valid_pct(self) -> float:
        if self.total_transformations == 0:
            return 0.0
        return self.valid_transformations / self.total_transformations * 100


# ============================================================================
# DATA COLLECTION
# ============================================================================


def diagnose_inapplicability(
    example: Dict, context_type: str, applied_types: set
) -> Dict[str, str]:
    """각 미적용 변환 타입의 이유를 진단."""
    reasons = {}
    context = example.get("context", "")
    question = example.get("question", "")

    for type_name in TRANSFORMATION_TYPES:
        if type_name in applied_types:
            continue

        if type_name == "Type 3: Ambiguous Time Period":
            if not re.search(r"\b(19|20)\d{2}\b", question):
                reasons[type_name] = "question에 4자리 연도 없음"
            else:
                reasons[type_name] = "연도 치환 실패 (알 수 없는 원인)"
            continue

        if context_type == "none":
            reasons[type_name] = "context 없음"
            continue

        if context_type == "text":
            if type_name == "Type 2: Table Column Removal":
                reasons[type_name] = "text context에 미지원"
            elif type_name in (
                "Type 1: Information Removal",
                "Type 4: Critical Data Removal",
                "Type 5: Contradictory Information",
            ):
                number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
                if not re.findall(number_pattern, context):
                    reasons[type_name] = "숫자 없음"
                else:
                    reasons[type_name] = "패턴 매칭 실패"
            else:
                reasons[type_name] = "패턴 매칭 실패"
            continue

        if context_type == "markdown":
            if type_name == "Type 1: Information Removal":
                cell_pattern = r"\|\s*(\$?\d+(?:,\d{3})*(?:\.\d+)?)\s*\|"
                if not re.search(cell_pattern, context):
                    reasons[type_name] = "테이블 숫자 셀 없음"
                else:
                    reasons[type_name] = "셀 매칭 실패"
            elif type_name == "Type 2: Table Column Removal":
                lines = context.split("\n")
                has_header = any(
                    "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]
                    for i, line in enumerate(lines)
                )
                if not has_header:
                    reasons[type_name] = "테이블 헤더 없음"
                else:
                    header_idx = -1
                    for i, line in enumerate(lines):
                        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
                            header_idx = i
                            break
                    headers = [
                        h.strip() for h in lines[header_idx].split("|") if h.strip()
                    ]
                    if len(headers) <= 2:
                        reasons[type_name] = "컬럼 수 부족 (≤2)"
                    else:
                        reasons[type_name] = "컬럼 매칭 실패"
            elif type_name == "Type 4: Critical Data Removal":
                if not re.search(r"\d", context):
                    reasons[type_name] = "데이터 행 없음"
                else:
                    reasons[type_name] = "데이터 행 제거 실패"
            elif type_name == "Type 5: Contradictory Information":
                if not re.search(r"\d", context):
                    reasons[type_name] = "숫자 셀 없음"
                else:
                    reasons[type_name] = "숫자 행 매칭 실패"
            else:
                reasons[type_name] = "패턴 매칭 실패"
            continue

        if context_type == "json":
            question_lower = question.lower()
            try:
                context_dict = json.loads(context)
                has_key_match = any(
                    k.lower() in question_lower for k in context_dict.keys()
                )
            except (json.JSONDecodeError, AttributeError):
                has_key_match = False

            if not has_key_match:
                reasons[type_name] = "dict 키-질문 매칭 실패"
            else:
                reasons[type_name] = "데이터 구조 매칭 실패"
            continue

        reasons[type_name] = "알 수 없는 원인"

    return reasons


def audit_single_problem(example: Dict, level: str) -> ProblemAudit:
    """단일 문제를 감사하여 ProblemAudit 반환."""
    context = example.get("context", "")
    question = example.get("question", "")
    python_solution = example.get("python_solution", "")
    question_id = example.get("question_id", example.get("id", "unknown"))
    source = example.get("source", "")
    context_type = detect_context_type(context)
    is_hardcoded = bool(
        python_solution and _solution_uses_hardcoded_values(python_solution)
    )

    # Apply transformations
    raw_transformations = apply_transformations(example)

    # Build set of applied type names
    applied_types = set()
    for t in raw_transformations:
        applied_types.add(t.get("transformation_type", ""))

    # Diagnose inapplicability
    inapplicable_reasons = diagnose_inapplicability(
        example, context_type, applied_types
    )

    # Build TransformationAudit list for all 5 types
    transformation_audits = []
    for type_name in TRANSFORMATION_TYPES:
        if type_name in applied_types:
            # Find the matching transformation
            trans = next(
                t
                for t in raw_transformations
                if t.get("transformation_type") == type_name
            )
            # Validate
            validation = validate_transformation(trans, example)
            is_valid = validation.get("valid", False)
            validation_reason = validation.get("reason", "")
            description = trans.get("transformation_description", "")

            transformation_audits.append(
                TransformationAudit(
                    type_name=type_name,
                    is_applicable=True,
                    inapplicable_reason="",
                    is_valid=is_valid,
                    validation_reason=validation_reason,
                    description=description,
                )
            )
        else:
            reason = inapplicable_reasons.get(type_name, "알 수 없는 원인")
            transformation_audits.append(
                TransformationAudit(
                    type_name=type_name,
                    is_applicable=False,
                    inapplicable_reason=reason,
                    is_valid=False,
                    validation_reason="not_applicable",
                    description="",
                )
            )

    applicable_count = sum(1 for t in transformation_audits if t.is_applicable)
    valid_count = sum(1 for t in transformation_audits if t.is_valid)

    return ProblemAudit(
        question_id=question_id,
        level=level,
        source=source,
        context_type=context_type,
        is_hardcoded=is_hardcoded,
        question_length=len(question),
        context_length=len(context),
        transformations=transformation_audits,
        applicable_count=applicable_count,
        valid_count=valid_count,
    )


def compute_audit_statistics(audits: List[ProblemAudit]) -> DatasetAuditStats:
    """ProblemAudit 리스트로부터 집계 통계 계산."""
    stats = DatasetAuditStats()
    stats.total_problems = len(audits)

    for audit in audits:
        # Context type counts
        stats.context_type_counts[audit.context_type] = (
            stats.context_type_counts.get(audit.context_type, 0) + 1
        )
        # Hardcoded
        if audit.is_hardcoded:
            stats.hardcoded_count += 1
        # Zero-transform
        if audit.valid_count == 0:
            stats.zero_transform_count += 1

        for t in audit.transformations:
            if t.is_applicable:
                stats.total_transformations += 1
                stats.type_applicable[t.type_name] = (
                    stats.type_applicable.get(t.type_name, 0) + 1
                )

                # Context type matrix (applicable)
                if audit.context_type not in stats.context_type_matrix:
                    stats.context_type_matrix[audit.context_type] = {}
                stats.context_type_matrix[audit.context_type][t.type_name] = (
                    stats.context_type_matrix[audit.context_type].get(t.type_name, 0)
                    + 1
                )

                if t.is_valid:
                    stats.valid_transformations += 1
                    stats.type_valid[t.type_name] = (
                        stats.type_valid.get(t.type_name, 0) + 1
                    )

                    # Context type valid matrix
                    if audit.context_type not in stats.context_type_valid_matrix:
                        stats.context_type_valid_matrix[audit.context_type] = {}
                    stats.context_type_valid_matrix[audit.context_type][t.type_name] = (
                        stats.context_type_valid_matrix[audit.context_type].get(
                            t.type_name, 0
                        )
                        + 1
                    )

                # Validation reasons
                if t.validation_reason:
                    stats.validation_reasons[t.validation_reason] = (
                        stats.validation_reasons.get(t.validation_reason, 0) + 1
                    )
            else:
                # Inapplicable reasons
                if t.type_name not in stats.type_inapplicable_reasons:
                    stats.type_inapplicable_reasons[t.type_name] = {}
                reason = t.inapplicable_reason or "unknown"
                stats.type_inapplicable_reasons[t.type_name][reason] = (
                    stats.type_inapplicable_reasons[t.type_name].get(reason, 0) + 1
                )

    return stats


# ============================================================================
# HTML SECTION BUILDERS
# ============================================================================


def _esc(text: str) -> str:
    """HTML escape helper."""
    return html_module.escape(str(text))


def _pct(num: float, den: float) -> str:
    """Format percentage string."""
    if den == 0:
        return "0.0%"
    return f"{num / den * 100:.1f}%"


def _color_for_pct(pct: float) -> str:
    """Return CSS color based on percentage (green=high, red=low)."""
    if pct >= 70:
        return "#4caf50"
    if pct >= 40:
        return "#ff9800"
    return "#f44336"


def build_overview_section(
    all_stats: Dict[str, DatasetAuditStats],
    all_audits: Dict[str, List[ProblemAudit]],
) -> str:
    """Section 1: Overview summary cards + level comparison table."""
    # Aggregate totals
    total_problems = sum(s.total_problems for s in all_stats.values())
    total_hardcoded = sum(s.hardcoded_count for s in all_stats.values())
    total_transforms = sum(s.total_transformations for s in all_stats.values())
    total_valid = sum(s.valid_transformations for s in all_stats.values())
    total_zero = sum(s.zero_transform_count for s in all_stats.values())

    # Context type aggregate
    all_context_types = set()
    for s in all_stats.values():
        all_context_types.update(s.context_type_counts.keys())

    cards_html = f"""
    <div class="summary-grid" style="grid-template-columns: repeat(6, 1fr);">
        <div class="summary-card">
            <div class="summary-value">{total_problems:,}</div>
            <div class="summary-label">Total Problems</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{len(all_context_types)}</div>
            <div class="summary-label">Context Types</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color: #ff9800;">{_pct(total_hardcoded, total_problems)}</div>
            <div class="summary-label">Hardcoded Solutions</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{total_transforms:,}</div>
            <div class="summary-label">Total Transforms</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color: #4caf50;">{total_valid:,} ({_pct(total_valid, total_transforms)})</div>
            <div class="summary-label">Valid Transforms</div>
        </div>
        <div class="summary-card">
            <div class="summary-value" style="color: #f44336;">{total_zero:,}</div>
            <div class="summary-label">Zero-Transform Problems</div>
        </div>
    </div>
    """

    # Level comparison table
    level_order = ["easy", "medium", "hard"]
    levels = [l for l in level_order if l in all_stats]

    rows = ""
    for level in levels:
        s = all_stats[level]
        ctx_parts = ", ".join(
            f"{ct}: {cnt}" for ct, cnt in sorted(s.context_type_counts.items())
        )
        valid_color = _color_for_pct(s.valid_pct)
        rows += f"""
        <tr>
            <td><span class="level-badge level-{level}">{level}</span></td>
            <td>{s.total_problems:,}</td>
            <td style="font-size:0.82em;">{ctx_parts}</td>
            <td style="color:#ff9800;">{_pct(s.hardcoded_count, s.total_problems)}</td>
            <td>{s.total_transformations:,}</td>
            <td style="color:{valid_color};">{s.valid_transformations:,} ({_pct(s.valid_transformations, s.total_transformations)})</td>
            <td style="color:#f44336;">{s.zero_transform_count}</td>
        </tr>
        """

    table_html = f"""
    <table>
        <thead>
            <tr>
                <th>Level</th>
                <th>Problems</th>
                <th>Context Types</th>
                <th>Hardcoded %</th>
                <th>Transforms</th>
                <th>Valid</th>
                <th>Zero-Transform</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """

    return f"""
    <div class="rq-section" id="overview">
        <h2>1. Overview</h2>
        <p class="rq-description">전체 데이터셋 감사 요약 및 난이도별 비교</p>
        {cards_html}
        <h3 style="color:#00d4ff; margin:15px 0 10px;">난이도별 비교</h3>
        {table_html}
    </div>
    """


def build_context_type_section(
    all_stats: Dict[str, DatasetAuditStats],
    all_audits: Dict[str, List[ProblemAudit]],
) -> str:
    """Section 2: Context Type Analysis with pie chart and table."""
    # Aggregate context type counts
    agg_ctx = {}
    for s in all_stats.values():
        for ct, cnt in s.context_type_counts.items():
            agg_ctx[ct] = agg_ctx.get(ct, 0) + cnt

    # Average applicable/valid by context type
    ctx_applicable = {}
    ctx_valid = {}
    ctx_count = {}
    for audits in all_audits.values():
        for audit in audits:
            ct = audit.context_type
            ctx_count[ct] = ctx_count.get(ct, 0) + 1
            ctx_applicable[ct] = ctx_applicable.get(ct, 0) + audit.applicable_count
            ctx_valid[ct] = ctx_valid.get(ct, 0) + audit.valid_count

    # Pie chart data
    ctx_labels = sorted(agg_ctx.keys())
    ctx_values = [agg_ctx[ct] for ct in ctx_labels]
    ctx_colors = {
        "json": "#00d4ff",
        "text": "#4caf50",
        "markdown": "#ff9800",
        "none": "#f44336",
    }
    colors = [ctx_colors.get(ct, "#888") for ct in ctx_labels]

    # Table rows
    rows = ""
    for ct in ctx_labels:
        n = ctx_count.get(ct, 0)
        avg_app = ctx_applicable.get(ct, 0) / n if n > 0 else 0
        avg_val = ctx_valid.get(ct, 0) / n if n > 0 else 0
        rows += f"""
        <tr>
            <td><span style="color:{ctx_colors.get(ct, "#888")}; font-weight:bold;">{ct}</span></td>
            <td>{agg_ctx[ct]:,}</td>
            <td>{_pct(agg_ctx[ct], sum(agg_ctx.values()))}</td>
            <td>{avg_app:.1f}</td>
            <td>{avg_val:.1f}</td>
        </tr>
        """

    return f"""
    <div class="rq-section" id="context-type">
        <h2>2. Context Type Analysis</h2>
        <p class="rq-description">context 유형별 분포 및 변환 가능성 분석</p>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Context Type 분포</div>
                <canvas id="ctxPieChart" height="280"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Context Type별 변환 통계</div>
                <table>
                    <thead>
                        <tr><th>Type</th><th>Count</th><th>비율</th><th>평균 적용 가능</th><th>평균 유효</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                <p style="color:#888; font-size:0.82em; margin-top:8px;">
                    * "none" context는 Type 3 (연도 치환)만 적용 가능
                </p>
            </div>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('ctxPieChart'), {{
        type: 'pie',
        data: {{
            labels: {json.dumps(ctx_labels)},
            datasets: [{{
                data: {json.dumps(ctx_values)},
                backgroundColor: {json.dumps(colors)},
                borderWidth: 0
            }}]
        }},
        options: {{
            plugins: {{
                legend: {{ position: 'bottom', labels: {{ color: '#e0e0e0' }} }}
            }}
        }}
    }});
    </script>
    """


def build_applicability_section(
    all_stats: Dict[str, DatasetAuditStats],
    all_audits: Dict[str, List[ProblemAudit]],
) -> str:
    """Section 3: Transformation Applicability with stacked bar + matrix."""
    total_problems = sum(s.total_problems for s in all_stats.values())

    # Per-type: applicable+valid, applicable+invalid, inapplicable
    type_data = {}
    for tn in TRANSFORMATION_TYPES:
        total_applicable = sum(s.type_applicable.get(tn, 0) for s in all_stats.values())
        total_valid = sum(s.type_valid.get(tn, 0) for s in all_stats.values())
        inapplicable = total_problems - total_applicable
        type_data[tn] = {
            "valid": total_valid,
            "invalid": total_applicable - total_valid,
            "inapplicable": inapplicable,
        }

    short_names = [tn.split(": ")[1] for tn in TRANSFORMATION_TYPES]
    valid_vals = [type_data[tn]["valid"] for tn in TRANSFORMATION_TYPES]
    invalid_vals = [type_data[tn]["invalid"] for tn in TRANSFORMATION_TYPES]
    inapp_vals = [type_data[tn]["inapplicable"] for tn in TRANSFORMATION_TYPES]

    # Context type × transformation matrix
    all_ctx_types = sorted(
        set(ct for s in all_stats.values() for ct in s.context_type_counts)
    )
    matrix_rows = ""
    for ct in all_ctx_types:
        ct_color = {
            "json": "#00d4ff",
            "text": "#4caf50",
            "markdown": "#ff9800",
            "none": "#f44336",
        }.get(ct, "#888")
        cells = f"<td style='font-weight:bold; color:{ct_color};'>{ct}</td>"
        for tn in TRANSFORMATION_TYPES:
            app_count = sum(
                s.context_type_matrix.get(ct, {}).get(tn, 0) for s in all_stats.values()
            )
            val_count = sum(
                s.context_type_valid_matrix.get(ct, {}).get(tn, 0)
                for s in all_stats.values()
            )
            ct_total = sum(s.context_type_counts.get(ct, 0) for s in all_stats.values())
            if ct_total > 0 and app_count > 0:
                pct = app_count / ct_total * 100
                bg_alpha = min(pct / 100 * 0.5, 0.5)
                cells += f"<td style='background:rgba(0,212,255,{bg_alpha:.2f}); text-align:center;'>{app_count} ({val_count}v)</td>"
            else:
                cells += "<td style='text-align:center; color:#555;'>-</td>"
        matrix_rows += f"<tr>{cells}</tr>"

    matrix_headers = "".join(
        f"<th style='font-size:0.78em;'>{tn.split(': ')[1]}</th>"
        for tn in TRANSFORMATION_TYPES
    )

    # Top inapplicable reasons
    reason_rows = ""
    for tn in TRANSFORMATION_TYPES:
        agg_reasons = {}
        for s in all_stats.values():
            for reason, cnt in s.type_inapplicable_reasons.get(tn, {}).items():
                agg_reasons[reason] = agg_reasons.get(reason, 0) + cnt
        if agg_reasons:
            top_reason = max(agg_reasons, key=agg_reasons.get)
            reason_rows += f"<tr><td style='font-size:0.85em;'>{_esc(tn)}</td><td>{_esc(top_reason)}</td><td>{agg_reasons[top_reason]:,}</td></tr>"

    return f"""
    <div class="rq-section" id="applicability">
        <h2>3. Transformation Applicability</h2>
        <p class="rq-description">각 변환 타입별 적용 가능성과 context type × type 매트릭스</p>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">변환 타입별 분포 (Stacked Bar)</div>
                <canvas id="applicabilityChart" height="280"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">미적용 주요 원인</div>
                <table>
                    <thead><tr><th>Type</th><th>주요 원인</th><th>Count</th></tr></thead>
                    <tbody>{reason_rows}</tbody>
                </table>
            </div>
        </div>
        <h3 style="color:#00d4ff; margin:15px 0 10px;">Context Type × Transformation Type Matrix</h3>
        <div class="heatmap-container">
            <table class="heatmap-table">
                <thead><tr><th>Context</th>{matrix_headers}</tr></thead>
                <tbody>{matrix_rows}</tbody>
            </table>
            <p style="color:#666; font-size:0.8em; margin-top:8px;">숫자: 적용 가능 수 (유효 수v)</p>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('applicabilityChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(short_names)},
            datasets: [
                {{ label: 'Valid', data: {json.dumps(valid_vals)}, backgroundColor: '#4caf50' }},
                {{ label: 'Invalid', data: {json.dumps(invalid_vals)}, backgroundColor: '#ff9800' }},
                {{ label: 'Inapplicable', data: {json.dumps(inapp_vals)}, backgroundColor: 'rgba(255,255,255,0.1)' }}
            ]
        }},
        options: {{
            responsive: true,
            scales: {{
                x: {{ stacked: true, ticks: {{ color: '#aaa', font: {{ size: 10 }} }} }},
                y: {{ stacked: true, ticks: {{ color: '#aaa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
            }},
            plugins: {{
                legend: {{ labels: {{ color: '#e0e0e0' }} }}
            }}
        }}
    }});
    </script>
    """


def build_validation_section(
    all_stats: Dict[str, DatasetAuditStats],
) -> str:
    """Section 4: Validation Results with doughnut chart."""
    agg_reasons = {}
    for s in all_stats.values():
        for reason, cnt in s.validation_reasons.items():
            agg_reasons[reason] = agg_reasons.get(reason, 0) + cnt

    reason_labels = sorted(agg_reasons.keys())
    reason_values = [agg_reasons[r] for r in reason_labels]
    reason_colors = {
        "different_answer": "#4caf50",
        "execution_error": "#00d4ff",
        "execution_no_result": "#2196f3",
        "no_python_solution": "#9c27b0",
        "hardcoded_solution": "#f44336",
        "still_solvable": "#ff9800",
    }
    colors = [reason_colors.get(r, "#888") for r in reason_labels]

    total = sum(reason_values)
    rows = ""
    for r, v in sorted(agg_reasons.items(), key=lambda x: -x[1]):
        c = reason_colors.get(r, "#888")
        rows += f"<tr><td><span style='color:{c}; font-weight:bold;'>{_esc(r)}</span></td><td>{v:,}</td><td>{_pct(v, total)}</td></tr>"

    return f"""
    <div class="rq-section" id="validation">
        <h2>4. Validation Results</h2>
        <p class="rq-description">변환 적용 후 validation 결과 분류</p>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Validation Reason 분포</div>
                <canvas id="validationChart" height="280"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">상세 테이블</div>
                <table>
                    <thead><tr><th>Reason</th><th>Count</th><th>비율</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('validationChart'), {{
        type: 'doughnut',
        data: {{
            labels: {json.dumps(reason_labels)},
            datasets: [{{
                data: {json.dumps(reason_values)},
                backgroundColor: {json.dumps(colors)},
                borderWidth: 0
            }}]
        }},
        options: {{
            plugins: {{
                legend: {{ position: 'bottom', labels: {{ color: '#e0e0e0', font: {{ size: 11 }} }} }}
            }}
        }}
    }});
    </script>
    """


def build_hardcoded_section(
    all_stats: Dict[str, DatasetAuditStats],
    all_audits: Dict[str, List[ProblemAudit]],
) -> str:
    """Section 5: Hardcoded Solution Analysis."""
    # Per-context-type hardcoded rate
    ctx_hc = {}
    ctx_total = {}
    for audits in all_audits.values():
        for audit in audits:
            ct = audit.context_type
            ctx_total[ct] = ctx_total.get(ct, 0) + 1
            if audit.is_hardcoded:
                ctx_hc[ct] = ctx_hc.get(ct, 0) + 1

    # Hardcoded vs non-hardcoded valid transform rate
    hc_valid = 0
    hc_applicable = 0
    nhc_valid = 0
    nhc_applicable = 0
    for audits in all_audits.values():
        for audit in audits:
            for t in audit.transformations:
                if t.is_applicable:
                    if audit.is_hardcoded:
                        hc_applicable += 1
                        if t.is_valid:
                            hc_valid += 1
                    else:
                        nhc_applicable += 1
                        if t.is_valid:
                            nhc_valid += 1

    hc_valid_pct = hc_valid / hc_applicable * 100 if hc_applicable > 0 else 0
    nhc_valid_pct = nhc_valid / nhc_applicable * 100 if nhc_applicable > 0 else 0

    total_hc = sum(s.hardcoded_count for s in all_stats.values())
    total_problems = sum(s.total_problems for s in all_stats.values())

    ctx_labels = sorted(ctx_total.keys())
    hc_pcts = [round(ctx_hc.get(ct, 0) / ctx_total[ct] * 100, 1) for ct in ctx_labels]
    ctx_colors_list = [
        {
            "json": "#00d4ff",
            "text": "#4caf50",
            "markdown": "#ff9800",
            "none": "#f44336",
        }.get(ct, "#888")
        for ct in ctx_labels
    ]

    return f"""
    <div class="rq-section" id="hardcoded">
        <h2>5. Hardcoded Solution Analysis</h2>
        <p class="rq-description">하드코딩 솔루션이 validation에 미치는 영향</p>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Context Type별 하드코딩 비율</div>
                <canvas id="hardcodedChart" height="250"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">하드코딩 영향 분석</div>
                <div class="rq-summary-grid" style="grid-template-columns: 1fr;">
                    <div class="rq-insight-card">
                        <div class="rq-insight-label">전체 하드코딩 비율</div>
                        <div class="rq-insight-value" style="color:#ff9800;">{total_hc:,} / {total_problems:,} ({_pct(total_hc, total_problems)})</div>
                    </div>
                    <div class="rq-insight-card" style="border-left-color:#f44336;">
                        <div class="rq-insight-label">하드코딩 문제 변환 유효율</div>
                        <div class="rq-insight-value" style="color:#f44336;">{hc_valid_pct:.1f}%</div>
                        <div class="rq-insight-detail">{hc_valid:,} valid / {hc_applicable:,} applicable</div>
                    </div>
                    <div class="rq-insight-card" style="border-left-color:#4caf50;">
                        <div class="rq-insight-label">비하드코딩 문제 변환 유효율</div>
                        <div class="rq-insight-value" style="color:#4caf50;">{nhc_valid_pct:.1f}%</div>
                        <div class="rq-insight-detail">{nhc_valid:,} valid / {nhc_applicable:,} applicable</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('hardcodedChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(ctx_labels)},
            datasets: [{{
                label: 'Hardcoded %',
                data: {json.dumps(hc_pcts)},
                backgroundColor: {json.dumps(ctx_colors_list)}
            }}]
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{ beginAtZero: true, max: 100, ticks: {{ color: '#aaa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                x: {{ ticks: {{ color: '#aaa' }} }}
            }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});
    </script>
    """


def build_zero_transform_section(
    all_stats: Dict[str, DatasetAuditStats],
    all_audits: Dict[str, List[ProblemAudit]],
) -> str:
    """Section 6: Zero-Transform Problems."""
    zero_problems = []
    for level_audits in all_audits.values():
        for audit in level_audits:
            if audit.valid_count == 0:
                # Determine reason
                if audit.applicable_count == 0:
                    if audit.context_type == "none":
                        reason = "context 없음"
                    else:
                        reason = "모든 변환 패턴 매칭 실패"
                else:
                    reason = "적용 가능하나 모두 validation 실패"
                zero_problems.append((audit, reason))

    # Reason distribution
    reason_counts = {}
    for _, reason in zero_problems:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    reason_labels = sorted(reason_counts.keys())
    reason_values = [reason_counts[r] for r in reason_labels]
    reason_colors = ["#f44336", "#ff9800", "#00d4ff", "#4caf50"]

    # Problem list (collapsible)
    problems_html = ""
    for audit, reason in sorted(zero_problems, key=lambda x: x[0].question_id):
        problems_html += f"""
        <tr>
            <td>{_esc(audit.question_id)}</td>
            <td><span class="level-badge level-{audit.level}">{audit.level}</span></td>
            <td>{audit.context_type}</td>
            <td>{audit.is_hardcoded}</td>
            <td>{_esc(reason)}</td>
        </tr>
        """

    return f"""
    <div class="rq-section" id="zero-transform">
        <h2>6. Zero-Transform Problems</h2>
        <p class="rq-description">유효한 변환이 0개인 문제 분석 ({len(zero_problems)}개)</p>
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">이유별 분포</div>
                <canvas id="zeroChart" height="250"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">요약</div>
                <div class="rq-summary-grid" style="grid-template-columns: 1fr;">
                    {"".join(f'<div class="rq-insight-card"><div class="rq-insight-label">{_esc(r)}</div><div class="rq-insight-value">{reason_counts[r]:,}</div></div>' for r in reason_labels)}
                </div>
            </div>
        </div>
        <details style="margin-top:15px;">
            <summary style="cursor:pointer; color:#00d4ff; font-weight:bold;">문제 목록 ({len(zero_problems)}개)</summary>
            <table style="margin-top:10px;">
                <thead><tr><th>Question ID</th><th>Level</th><th>Context</th><th>Hardcoded</th><th>Reason</th></tr></thead>
                <tbody>{problems_html}</tbody>
            </table>
        </details>
    </div>
    <script>
    new Chart(document.getElementById('zeroChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(reason_labels)},
            datasets: [{{
                data: {json.dumps(reason_values)},
                backgroundColor: {json.dumps(reason_colors[: len(reason_labels)])}
            }}]
        }},
        options: {{
            indexAxis: 'y',
            responsive: true,
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                y: {{ ticks: {{ color: '#aaa' }} }}
            }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});
    </script>
    """


def build_drilldown_section(
    all_audits: Dict[str, List[ProblemAudit]],
) -> str:
    """Section 7: Per-Problem Drill-Down with filtering."""
    level_order = ["easy", "medium", "hard"]
    levels = [l for l in level_order if l in all_audits]

    total_count = sum(len(a) for a in all_audits.values())

    # Build problem cards as JS data
    problems_json = []
    for level in levels:
        for audit in all_audits[level]:
            trans_matrix = []
            for t in audit.transformations:
                short = (
                    t.type_name.split(": ")[1] if ": " in t.type_name else t.type_name
                )
                if t.is_valid:
                    status = "valid"
                elif t.is_applicable:
                    status = "invalid"
                else:
                    status = "na"
                trans_matrix.append(
                    {
                        "name": short,
                        "status": status,
                        "reason": t.inapplicable_reason or t.validation_reason,
                    }
                )
            problems_json.append(
                {
                    "id": audit.question_id,
                    "level": audit.level,
                    "ctx": audit.context_type,
                    "hc": audit.is_hardcoded,
                    "app": audit.applicable_count,
                    "val": audit.valid_count,
                    "src": audit.source[:50] if audit.source else "",
                    "qlen": audit.question_length,
                    "clen": audit.context_length,
                    "trans": trans_matrix,
                }
            )

    return f"""
    <div class="rq-section" id="drilldown">
        <h2>7. Per-Problem Drill-Down</h2>
        <p class="rq-description">전체 {total_count:,}문제 개별 감사 결과 (필터링 가능)</p>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:15px; align-items:center;">
            <input type="text" id="searchInput" placeholder="Question ID 검색..."
                style="padding:8px 14px; border-radius:8px; border:1px solid rgba(255,255,255,0.15);
                background:rgba(255,255,255,0.05); color:#e0e0e0; font-size:0.9em; width:200px;">
            <select id="levelFilter" style="padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15);
                background:rgba(255,255,255,0.05); color:#e0e0e0;">
                <option value="all">All Levels</option>
                {"".join(f'<option value="{l}">{l}</option>' for l in levels)}
            </select>
            <select id="ctxFilter" style="padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15);
                background:rgba(255,255,255,0.05); color:#e0e0e0;">
                <option value="all">All Context Types</option>
                <option value="json">json</option>
                <option value="text">text</option>
                <option value="markdown">markdown</option>
                <option value="none">none</option>
            </select>
            <select id="hcFilter" style="padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15);
                background:rgba(255,255,255,0.05); color:#e0e0e0;">
                <option value="all">All Solutions</option>
                <option value="true">Hardcoded</option>
                <option value="false">Non-hardcoded</option>
            </select>
            <select id="validFilter" style="padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15);
                background:rgba(255,255,255,0.05); color:#e0e0e0;">
                <option value="all">All Valid Counts</option>
                <option value="0">Zero Valid</option>
                <option value="1+">1+ Valid</option>
                <option value="3+">3+ Valid</option>
            </select>
            <span id="filterCount" style="color:#888; font-size:0.85em;"></span>
        </div>
        <div id="problemList"></div>
    </div>
    <script>
    const PROBLEMS = {json.dumps(problems_json, ensure_ascii=False)};
    const PAGE_SIZE = 50;
    let currentPage = 0;
    let filtered = PROBLEMS;

    function statusBadge(s) {{
        if (s === 'valid') return '<span style="color:#4caf50; font-weight:bold;">&#x2713;</span>';
        if (s === 'invalid') return '<span style="color:#ff9800;">&#x2717;</span>';
        return '<span style="color:#555;">-</span>';
    }}

    function renderProblems() {{
        const search = document.getElementById('searchInput').value.toLowerCase();
        const levelF = document.getElementById('levelFilter').value;
        const ctxF = document.getElementById('ctxFilter').value;
        const hcF = document.getElementById('hcFilter').value;
        const valF = document.getElementById('validFilter').value;

        filtered = PROBLEMS.filter(p => {{
            if (search && !p.id.toLowerCase().includes(search)) return false;
            if (levelF !== 'all' && p.level !== levelF) return false;
            if (ctxF !== 'all' && p.ctx !== ctxF) return false;
            if (hcF !== 'all' && String(p.hc) !== hcF) return false;
            if (valF === '0' && p.val !== 0) return false;
            if (valF === '1+' && p.val < 1) return false;
            if (valF === '3+' && p.val < 3) return false;
            return true;
        }});

        document.getElementById('filterCount').textContent = filtered.length + ' / ' + PROBLEMS.length + ' problems';

        const start = 0;
        const end = Math.min((currentPage + 1) * PAGE_SIZE, filtered.length);
        const showing = filtered.slice(start, end);

        let html = '<table><thead><tr><th>ID</th><th>Level</th><th>Context</th><th>HC</th>';
        html += '<th>Info Removal</th><th>Column Removal</th><th>Ambig Time</th><th>Data Removal</th><th>Contradict</th>';
        html += '<th>Valid</th></tr></thead><tbody>';

        for (const p of showing) {{
            html += '<tr>';
            html += '<td style="font-size:0.82em;">' + p.id + '</td>';
            html += '<td><span class="level-badge level-' + p.level + '">' + p.level + '</span></td>';
            html += '<td>' + p.ctx + '</td>';
            html += '<td>' + (p.hc ? '<span style="color:#ff9800;">Y</span>' : 'N') + '</td>';
            for (const t of p.trans) {{
                html += '<td style="text-align:center;" title="' + (t.reason || '') + '">' + statusBadge(t.status) + '</td>';
            }}
            html += '<td style="font-weight:bold; color:' + (p.val > 0 ? '#4caf50' : '#f44336') + ';">' + p.val + '</td>';
            html += '</tr>';
        }}
        html += '</tbody></table>';

        if (filtered.length > end) {{
            html += '<button onclick="currentPage++; renderProblems();" style="margin-top:10px; padding:8px 20px; border-radius:8px; border:1px solid rgba(0,212,255,0.3); background:rgba(0,212,255,0.1); color:#00d4ff; cursor:pointer;">Load More (' + (filtered.length - end) + ' remaining)</button>';
        }}

        document.getElementById('problemList').innerHTML = html;
    }}

    document.getElementById('searchInput').addEventListener('input', () => {{ currentPage = 0; renderProblems(); }});
    document.getElementById('levelFilter').addEventListener('change', () => {{ currentPage = 0; renderProblems(); }});
    document.getElementById('ctxFilter').addEventListener('change', () => {{ currentPage = 0; renderProblems(); }});
    document.getElementById('hcFilter').addEventListener('change', () => {{ currentPage = 0; renderProblems(); }});
    document.getElementById('validFilter').addEventListener('change', () => {{ currentPage = 0; renderProblems(); }});
    renderProblems();
    </script>
    """


# ============================================================================
# DASHBOARD ASSEMBLY
# ============================================================================


def generate_audit_dashboard(
    data_dir: Path,
    levels: List[str],
    output_path: Path,
) -> Path:
    """전체 감사 대시보드 생성."""
    all_audits: Dict[str, List[ProblemAudit]] = {}
    all_stats: Dict[str, DatasetAuditStats] = {}

    for level in levels:
        input_file = data_dir / f"{level}.json"
        if not input_file.exists():
            print(f"[SKIP] {input_file} not found")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Auditing {level}: {len(data)} problems...")
        audits = []
        for idx, example in enumerate(data):
            audit = audit_single_problem(example, level)
            audits.append(audit)
            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(data)}...")

        all_audits[level] = audits
        all_stats[level] = compute_audit_statistics(audits)
        s = all_stats[level]
        print(
            f"  Done: {s.total_transformations} transforms, "
            f"{s.valid_transformations} valid ({s.valid_pct:.1f}%), "
            f"{s.zero_transform_count} zero-transform"
        )

    if not all_audits:
        print("[ERROR] No data loaded")
        return output_path

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build sections
    sec1 = build_overview_section(all_stats, all_audits)
    sec2 = build_context_type_section(all_stats, all_audits)
    sec3 = build_applicability_section(all_stats, all_audits)
    sec4 = build_validation_section(all_stats)
    sec5 = build_hardcoded_section(all_stats, all_audits)
    sec6 = build_zero_transform_section(all_stats, all_audits)
    sec7 = build_drilldown_section(all_audits)

    nav_items = [
        ("overview", "Overview"),
        ("context-type", "Context Type"),
        ("applicability", "Applicability"),
        ("validation", "Validation"),
        ("hardcoded", "Hardcoded"),
        ("zero-transform", "Zero-Transform"),
        ("drilldown", "Drill-Down"),
    ]
    nav_html = "".join(f'<a href="#{id_}">{label}</a>' for id_, label in nav_items)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Audit Dashboard - {timestamp}</title>
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
        h3 {{ color: #00d4ff; font-size: 1.1em; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 25px; font-size: 0.95em; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
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

        .heatmap-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            overflow-x: auto;
        }}
        .heatmap-table {{ border-collapse: collapse; }}
        .heatmap-table td, .heatmap-table th {{
            padding: 10px 16px;
            text-align: center;
            font-size: 0.88em;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .heatmap-table th {{ color: #00d4ff; background: rgba(0,212,255,0.08); }}

        .level-badge {{
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .level-easy {{ background: rgba(76,175,80,0.2); color: #4caf50; }}
        .level-medium {{ background: rgba(255,152,0,0.2); color: #ff9800; }}
        .level-hard {{ background: rgba(244,67,54,0.2); color: #f44336; }}

        details > summary {{
            cursor: pointer;
            color: #00d4ff;
            font-weight: bold;
            padding: 8px 0;
        }}

        .footer {{ text-align: center; color: #555; font-size: 0.8em; margin-top: 30px; padding: 15px; }}

        @media (max-width: 900px) {{
            .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Dataset Audit Dashboard</h1>
    <p class="subtitle">FinanceReasoning Transformation Applicability &amp; Validation Analysis — {timestamp}</p>

    <div class="rq-nav">{nav_html}</div>

    {sec1}
    {sec2}
    {sec3}
    {sec4}
    {sec5}
    {sec6}
    {sec7}

    <div class="footer">
        Generated by generate_dataset_audit.py | {timestamp}
    </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[SAVED] Dashboard: {output_path}")

    # Save JSON results
    json_path = output_path.with_suffix(".json")
    json_data = {
        "timestamp": timestamp,
        "levels": levels,
        "summary": {},
        "problems": {},
    }
    for level in all_stats:
        s = all_stats[level]
        json_data["summary"][level] = {
            "total_problems": s.total_problems,
            "context_type_counts": s.context_type_counts,
            "hardcoded_count": s.hardcoded_count,
            "hardcoded_pct": round(s.hardcoded_pct, 1),
            "total_transformations": s.total_transformations,
            "valid_transformations": s.valid_transformations,
            "valid_pct": round(s.valid_pct, 1),
            "zero_transform_count": s.zero_transform_count,
            "type_applicable": s.type_applicable,
            "type_valid": s.type_valid,
            "validation_reasons": s.validation_reasons,
        }
    for level in all_audits:
        json_data["problems"][level] = [
            {
                "question_id": a.question_id,
                "level": a.level,
                "source": a.source,
                "context_type": a.context_type,
                "is_hardcoded": a.is_hardcoded,
                "applicable_count": a.applicable_count,
                "valid_count": a.valid_count,
                "transformations": [
                    {
                        "type_name": t.type_name,
                        "is_applicable": t.is_applicable,
                        "inapplicable_reason": t.inapplicable_reason,
                        "is_valid": t.is_valid,
                        "validation_reason": t.validation_reason,
                    }
                    for t in a.transformations
                ],
            }
            for a in all_audits[level]
        ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] JSON: {json_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Dataset Audit Dashboard Generator")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to audit (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/dataset_audit/audit_dashboard.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    data_dir = Path("data/financereasoning/raw/FinanceReasoning")
    output_path = Path(args.output)

    print("Dataset Audit Dashboard Generator")
    print(f"  Levels: {', '.join(args.levels)}")
    print(f"  Output: {output_path}")
    print()

    generate_audit_dashboard(data_dir, args.levels, output_path)

    print("\nDone! Open the dashboard in a browser.")


if __name__ == "__main__":
    main()
