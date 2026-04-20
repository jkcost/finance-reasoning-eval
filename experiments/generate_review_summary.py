"""
Review Summary HTML Generator (v2)

리뷰어별 판정을 문제×변환 매트릭스로 한눈에 비교하고,
불일치 항목은 원본/변환 context + LLM 응답까지 상세 표시.

기능:
  - 전체 문제의 리뷰 결과를 한눈에 (리뷰어별 색상 뱃지)
  - 불일치 항목: 원본 context, 변환 context, LLM 모델 응답 상세 표시
  - 변환 타입별 승인율 차트
  - 리뷰어별 통계
  - 필터링 (전체/불일치/미리뷰)

Usage:
    python experiments/generate_review_summary.py
    python experiments/generate_review_summary.py --annotations-dir annotations/
"""

import argparse
import html as html_module
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

JUDGMENT_COLORS = {
    "approved": "#22c55e",
    "needs_modification": "#f59e0b",
    "rejected": "#ef4444",
    "": "#555555",
}

JUDGMENT_LABELS = {
    "approved": "승인",
    "needs_modification": "수정필요",
    "rejected": "부적절",
    "": "미리뷰",
}

REVIEWER_COLORS = [
    "#60a5fa",  # blue
    "#f472b6",  # pink
    "#a78bfa",  # purple
    "#34d399",  # emerald
    "#fbbf24",  # amber
    "#fb923c",  # orange
]


def _esc(text: str) -> str:
    return html_module.escape(str(text)) if text else ""


def _extract_name_from_filename(path: Path) -> str:
    """Extract reviewer name from filename like review_김진수_0_238_2026-04-14.json."""
    stem = path.stem
    if stem.startswith("review_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return stem


def load_all_annotations(annotations_dir: Path) -> Tuple[Dict, List[str]]:
    """Load all reviewer annotations separately.

    Returns:
        (per_reviewer: {assignee: {qid: {ttype: annotation}}}, reviewer_names: [str])
    """
    per_reviewer: Dict[str, Dict[str, Dict]] = {}

    for f in sorted(annotations_dir.glob("*.json")):
        if f.name.startswith(".") or f.name in (
            "merged.json",
            "disagreements.json",
            "llm_judgments.json",
        ):
            continue

        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assignee = data.get("assignee") or _extract_name_from_filename(f)
        annotations = data.get("annotations", data)

        count = 0
        reviewer_anns: Dict[str, Dict] = {}
        for qid, types in annotations.items():
            if not isinstance(types, dict):
                continue
            reviewer_anns[qid] = {}
            for ttype, ann in types.items():
                if not isinstance(ann, dict):
                    continue
                if ann.get("judgment"):
                    reviewer_anns[qid][ttype] = ann
                    count += 1

        per_reviewer[assignee] = reviewer_anns
        logger.info(f"  로드: {f.name} ({assignee}, {count}건)")

    reviewer_names = sorted(per_reviewer.keys())
    return per_reviewer, reviewer_names


def load_problems(batch_path: Path) -> List[Dict]:
    with open(batch_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("problems", [])


def load_evaluations(eval_path: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """Load evaluation results, grouped by qid -> ttype -> [results].

    Returns: {qid: {ttype: [eval_result, ...]}}
    """
    if not eval_path.exists():
        logger.warning(f"평가 파일 없음: {eval_path}")
        return {}

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped: Dict[str, Dict[str, List[Dict]]] = {}
    for r in data.get("results", []):
        qid = r.get("question_id", "")
        ttype = r.get("transformation_type", "")
        if qid not in grouped:
            grouped[qid] = {}
        if ttype not in grouped[qid]:
            grouped[qid][ttype] = []
        grouped[qid][ttype].append(r)

    return grouped


def load_llm_judgments(judgments_path: Path) -> Dict[str, Dict[str, Dict]]:
    """Load LLM judgments, keyed by qid -> ttype -> judgment data."""
    if not judgments_path.exists():
        logger.info(f"LLM 판정 파일 없음 (생략): {judgments_path}")
        return {}

    with open(judgments_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, Dict[str, Dict]] = {}
    for j in data.get("judgments", []):
        qid = j.get("qid", "")
        ttype = j.get("ttype", "")
        if qid not in result:
            result[qid] = {}
        result[qid][ttype] = j

    logger.info(f"LLM 판정 로드: {len(data.get('judgments', []))}건")
    return result


def generate_summary_html(
    problems: List[Dict],
    per_reviewer: Dict[str, Dict[str, Dict]],
    reviewer_names: List[str],
    evaluations: Dict[str, Dict[str, List[Dict]]],
    llm_judgments: Optional[Dict[str, Dict[str, Dict]]] = None,
) -> str:
    """Generate enhanced review summary HTML with multi-reviewer comparison."""

    # Build problem lookup
    problem_map = {p["question_id"]: p for p in problems}

    # Collect all reviewed question IDs
    all_qids = set()
    for anns in per_reviewer.values():
        all_qids.update(anns.keys())
    all_qids = sorted(all_qids)

    # Stats
    total_annotations = sum(
        sum(1 for t in anns.values() for a in t.values())
        for anns in per_reviewer.values()
    )

    # Per-type stats
    type_stats: Dict[str, Dict[str, int]] = {
        t: {"approved": 0, "needs_modification": 0, "rejected": 0, "total": 0}
        for t in TYPE_KEYS
    }

    # Per-reviewer stats
    reviewer_stats: Dict[str, Dict[str, int]] = {
        name: {"total": 0, "approved": 0, "needs_modification": 0, "rejected": 0}
        for name in reviewer_names
    }

    # Collect disagreements and build matrix data
    matrix_data = []  # [(qid, {ttype: [(reviewer, judgment, note)]}, has_disagreement)]

    for qid in all_qids:
        type_reviews: Dict[str, List[Tuple[str, str, str]]] = {}
        has_disagreement = False

        for ttype in TYPE_KEYS:
            reviews = []
            for name in reviewer_names:
                ann = per_reviewer.get(name, {}).get(qid, {}).get(ttype, {})
                j = ann.get("judgment", "")
                note = ann.get("note", "")
                if j:
                    reviews.append((name, j, note))
                    type_stats[ttype]["total"] += 1
                    type_stats[ttype][j] += 1
                    reviewer_stats[name]["total"] += 1
                    reviewer_stats[name][j] += 1

            type_reviews[ttype] = reviews

            # Check disagreement: 2+ reviewers with different judgments
            judgments = set(r[1] for r in reviews)
            if len(reviews) >= 2 and len(judgments) > 1:
                has_disagreement = True

        matrix_data.append((qid, type_reviews, has_disagreement))

    disagreement_count = sum(1 for _, _, d in matrix_data if d)

    # Reviewer color mapping
    reviewer_color = {
        name: REVIEWER_COLORS[i % len(REVIEWER_COLORS)]
        for i, name in enumerate(reviewer_names)
    }

    # ===== Build HTML components =====

    # 1. Matrix table rows
    table_rows = ""
    for qid, type_reviews, has_disagreement in matrix_data:
        row_class = "disagreement-row" if has_disagreement else ""
        row = f'<tr class="{row_class}" data-qid="{qid}">'
        row += f'<td class="qid-cell">{_esc(qid)}</td>'

        for ttype in TYPE_KEYS:
            reviews = type_reviews.get(ttype, [])
            if not reviews:
                # Check if transformation exists
                p = problem_map.get(qid, {})
                t = p.get("transformations", {}).get(ttype, {})
                if t.get("success"):
                    row += '<td class="cell-empty">-</td>'
                else:
                    row += '<td class="cell-na"></td>'
                continue

            judgments = set(r[1] for r in reviews)
            is_conflict = len(reviews) >= 2 and len(judgments) > 1
            cell_class = "cell-conflict" if is_conflict else "cell-ok"

            badges = ""
            for name, j, note in reviews:
                color = JUDGMENT_COLORS[j]
                initial = name[0] if name else "?"
                rc = reviewer_color.get(name, "#888")
                tooltip = f"{name}: {JUDGMENT_LABELS[j]}"
                if note:
                    tooltip += f" — {note}"
                badges += (
                    f'<span class="badge" style="background:{color}30;color:{color};'
                    f'border:1px solid {color}50" title="{_esc(tooltip)}">'
                    f'<span class="badge-initial" style="color:{rc}">{_esc(initial)}</span>'
                    f"{JUDGMENT_LABELS[j][:1]}"
                    f"</span>"
                )

            row += f'<td class="{cell_class}">{badges}</td>'

        row += "</tr>"
        table_rows += row

    # 2. Type stats rows
    type_stats_rows = ""
    for t in TYPE_KEYS:
        s = type_stats[t]
        total = s["total"]
        if total == 0:
            type_stats_rows += f'<tr><td>{t}</td><td>0</td><td colspan="3">-</td></tr>'
            continue
        apr = s["approved"]
        mod = s["needs_modification"]
        rej = s["rejected"]
        apr_pct = apr / total * 100
        type_stats_rows += (
            f"<tr><td>{t}</td><td>{total}</td>"
            f'<td style="color:#22c55e">{apr} ({apr_pct:.0f}%)</td>'
            f'<td style="color:#f59e0b">{mod}</td>'
            f'<td style="color:#ef4444">{rej}</td></tr>'
        )

    # 3. Reviewer stats rows
    reviewer_rows = ""
    for name in reviewer_names:
        s = reviewer_stats[name]
        rc = reviewer_color.get(name, "#888")
        reviewer_rows += (
            f'<tr><td><span style="color:{rc};font-weight:700">{_esc(name)}</span></td>'
            f"<td>{s['total']}</td>"
            f'<td style="color:#22c55e">{s["approved"]}</td>'
            f'<td style="color:#f59e0b">{s["needs_modification"]}</td>'
            f'<td style="color:#ef4444">{s["rejected"]}</td></tr>'
        )

    # 4. Per-question detail cards (ALL reviewed, not only disagreements)
    if llm_judgments is None:
        llm_judgments = {}

    # Solvability verdict aggregation
    verdict_map: Dict[str, Dict[str, str]] = {}
    effective_count = 0
    compromised_count = 0
    type_verdict_stats: Dict[str, Dict[str, int]] = {
        t: {
            "effective": 0,
            "compromised": 0,
            "needs_modification": 0,
            "rejected": 0,
            "approved_total": 0,
        }
        for t in TYPE_KEYS
    }

    for qid, type_reviews, _ in matrix_data:
        verdict_map[qid] = {}
        for ttype in TYPE_KEYS:
            reviews = type_reviews.get(ttype, [])
            if not reviews:
                continue
            judgments_list = [r[1] for r in reviews]
            approved_n = judgments_list.count("approved")
            rejected_n = judgments_list.count("rejected")
            mod_n = judgments_list.count("needs_modification")
            if approved_n >= max(rejected_n, mod_n) and approved_n > 0:
                consensus = "approved"
            elif rejected_n >= mod_n and rejected_n > 0:
                consensus = "rejected"
            elif mod_n > 0:
                consensus = "needs_modification"
            else:
                consensus = ""

            if consensus == "approved":
                eval_results = evaluations.get(qid, {}).get(ttype, [])
                has_suspect = False
                for er in eval_results:
                    rt = er.get("response_type", "")
                    ok = bool(er.get("is_correct", False))
                    if ok and rt == "confident":
                        has_suspect = True
                        break
                if has_suspect:
                    verdict_map[qid][ttype] = "compromised"
                    compromised_count += 1
                    type_verdict_stats[ttype]["compromised"] += 1
                else:
                    verdict_map[qid][ttype] = "effective"
                    effective_count += 1
                    type_verdict_stats[ttype]["effective"] += 1
                type_verdict_stats[ttype]["approved_total"] += 1
            elif consensus == "rejected":
                verdict_map[qid][ttype] = "rejected"
                type_verdict_stats[ttype]["rejected"] += 1
            elif consensus == "needs_modification":
                verdict_map[qid][ttype] = "needs_modification"
                type_verdict_stats[ttype]["needs_modification"] += 1

    detail_cards = ""
    for qid, type_reviews, has_disagreement in matrix_data:
        p = problem_map.get(qid, {})
        question = p.get("question", "")
        context_original = p.get("context_original", "")
        gt = p.get("ground_truth", "")
        transformations = p.get("transformations", {})

        # ALL reviewed types for this qid
        reviewed_types = [
            t for t in TYPE_KEYS if type_reviews.get(t) and len(type_reviews[t]) > 0
        ]
        if not reviewed_types:
            continue

        # Card-level filter attributes
        card_verdicts = sorted({v for v in verdict_map.get(qid, {}).values() if v})
        card_reviewers = sorted({r[0] for t in reviewed_types for r in type_reviews[t]})

        # Consolidated comments (all reviewer notes across all ttypes)
        comments_html_parts = []
        for ttype in reviewed_types:
            for name, j, note in type_reviews[ttype]:
                if note and note.strip():
                    rc = reviewer_color.get(name, "#888")
                    tag_color = JUDGMENT_COLORS.get(j, "#555")
                    comments_html_parts.append(
                        f'<div class="comment-item" style="border-left:3px solid {rc}">'
                        f'<div class="comment-meta">'
                        f'<span class="comment-ttype">{_esc(ttype)}</span>'
                        f'<span class="comment-reviewer" style="color:{rc}">{_esc(name)}</span>'
                        f'<span class="judgment-tag" style="background:{tag_color}25;color:{tag_color}">{JUDGMENT_LABELS.get(j, "?")}</span>'
                        f"</div>"
                        f'<div class="comment-note">{_esc(note)}</div>'
                        f"</div>"
                    )
        consolidated_comments_html = (
            (
                '<div class="consolidated-comments"><div class="detail-label">리뷰어 코멘트 모음 (전체 변환타입)</div>'
                + "".join(comments_html_parts)
                + "</div>"
            )
            if comments_html_parts
            else ""
        )

        tabs_html = ""
        panels_html = ""
        for i, ttype in enumerate(reviewed_types):
            reviews = type_reviews[ttype]
            active = "active" if i == 0 else ""

            # LLM judgment badge for tab
            lj = llm_judgments.get(qid, {}).get(ttype, {})
            lj_judgment = lj.get("judgment", "")
            lj_color = JUDGMENT_COLORS.get(lj_judgment, "#555")
            lj_badge = ""
            if lj_judgment and lj_judgment not in ("parse_error", "error"):
                lj_badge = (
                    f' <span style="font-size:9px;color:{lj_color};'
                    f'background:{lj_color}20;padding:1px 4px;border-radius:3px;">'
                    f"AI:{JUDGMENT_LABELS.get(lj_judgment, '?')[:1]}</span>"
                )

            tabs_html += (
                f'<button class="detail-tab {active}" '
                f"onclick=\"switchDetailTab(this, '{qid}', '{ttype}')\">"
                f"{ttype}{lj_badge}</button>"
            )

            tdata = transformations.get(ttype, {})
            ctx_transformed = tdata.get("context_transformed", "")
            question_transformed = tdata.get("question_transformed", "")
            description = tdata.get("description", "")

            # Question comparison (original vs transformed)
            question_html = ""
            if question_transformed and question_transformed != question:
                question_html = f"""
                <div class="question-compare">
                  <div class="q-col">
                    <div class="detail-label">원본 Question</div>
                    <div class="q-text">{_esc(question)}</div>
                  </div>
                  <div class="q-col">
                    <div class="detail-label">변환 Question ({ttype})</div>
                    <div class="q-text q-transformed">{_esc(question_transformed)}</div>
                  </div>
                </div>"""
            else:
                question_html = f"""
                <div class="detail-section">
                  <div class="detail-label">Question (변경 없음)</div>
                  <div class="q-text">{_esc(question)}</div>
                </div>"""

            # Reviewer comparison
            reviewer_html = ""
            for name, j, note in reviews:
                color = JUDGMENT_COLORS[j]
                rc = reviewer_color.get(name, "#888")
                reviewer_html += (
                    f'<div class="reviewer-box" style="border-left:3px solid {rc}">'
                    f'<div class="reviewer-name" style="color:{rc}">{_esc(name)}</div>'
                    f'<span class="judgment-tag" style="background:{color}25;color:{color}">'
                    f"{JUDGMENT_LABELS[j]}</span>"
                )
                if note:
                    reviewer_html += f'<div class="reviewer-note">{_esc(note)}</div>'
                reviewer_html += "</div>"

            # LLM judgment section
            llm_judgment_html = ""
            if lj and lj_judgment not in ("parse_error", "error", ""):
                lj_conf = lj.get("confidence", "")
                lj_comment = lj.get("comment", "")
                lj_issues = lj.get("issues", [])
                lj_agrees = lj.get("agrees_with", "")
                lj_suggestion = lj.get("suggestion", "")
                lj_model = lj.get("model_used", "")

                conf_colors = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}
                conf_c = conf_colors.get(lj_conf, "#888")

                issues_html = ""
                if lj_issues:
                    issues_html = (
                        "<ul class='lj-issues'>"
                        + "".join(f"<li>{_esc(iss)}</li>" for iss in lj_issues)
                        + "</ul>"
                    )

                llm_judgment_html = f"""
                <div class="llm-judgment-box">
                  <div class="detail-label">LLM 분석 판정 ({_esc(lj_model)})</div>
                  <div class="lj-header">
                    <span class="judgment-tag" style="background:{lj_color}25;color:{lj_color};font-size:13px;padding:4px 12px;">
                      {JUDGMENT_LABELS.get(lj_judgment, lj_judgment)}</span>
                    <span class="lj-conf" style="color:{conf_c}">신뢰도: {lj_conf}</span>
                    {"<span class='lj-agrees'>동의: " + _esc(lj_agrees) + "</span>" if lj_agrees else ""}
                  </div>
                  <div class="lj-comment">{_esc(lj_comment)}</div>
                  {issues_html}
                  {"<div class='lj-suggestion'><strong>수정 제안:</strong> " + _esc(lj_suggestion) + "</div>" if lj_suggestion else ""}
                </div>"""

            # LLM experiment responses — per (model, prompt_strategy)
            eval_results = evaluations.get(qid, {}).get(ttype, [])
            eval_html = ""
            suspect_count = 0
            refused_count = 0
            total_evals = len(eval_results)
            if eval_results:

                def _sort_key(er):
                    rt = er.get("response_type", "")
                    ok = bool(er.get("is_correct", False))
                    if rt == "confident" and ok:
                        return (0, er.get("model", ""), er.get("prompt_strategy", ""))
                    if rt == "confident" and not ok:
                        return (1, er.get("model", ""), er.get("prompt_strategy", ""))
                    if rt == "caveat":
                        return (2, er.get("model", ""), er.get("prompt_strategy", ""))
                    if rt == "refused":
                        return (3, er.get("model", ""), er.get("prompt_strategy", ""))
                    return (4, er.get("model", ""), er.get("prompt_strategy", ""))

                for er in sorted(eval_results, key=_sort_key):
                    model = er.get("model", "?")
                    strategy = er.get("prompt_strategy", "")
                    resp_type = er.get("response_type", "")
                    is_correct = bool(er.get("is_correct", False))
                    predicted = str(er.get("predicted_answer", ""))[:200]
                    raw = str(er.get("raw_response", ""))[:1200]
                    exec_code = str(er.get("executed_code", "") or "")[:1200]

                    resp_color = "#22c55e" if is_correct else "#ef4444"
                    resp_label = "정답" if is_correct else "오답"
                    if resp_type == "refused":
                        resp_color = "#60a5fa"
                        resp_label = "거부"
                        refused_count += 1
                    elif resp_type == "caveat":
                        resp_color = "#a78bfa"
                        resp_label = "유보"
                    elif resp_type == "error":
                        resp_color = "#888"
                        resp_label = "에러"

                    suspect_badge = ""
                    if resp_type == "confident" and is_correct:
                        if verdict_map.get(qid, {}).get(ttype) == "compromised":
                            suspect_badge = (
                                '<span class="suspect-badge" '
                                'title="인간 승인(풀이 불가) 변환을 자신있게 정답 — 암기/환각 의심">'
                                "⚠ 의심</span>"
                            )
                            suspect_count += 1

                    exec_block = (
                        f'<details class="eval-raw"><summary>실행 코드 보기</summary>'
                        f"<pre>{_esc(exec_code)}</pre></details>"
                        if exec_code and exec_code != "None"
                        else ""
                    )

                    eval_html += (
                        f'<div class="eval-box">'
                        f'<div class="eval-header">'
                        f'<span class="eval-model">{_esc(model)}</span>'
                        f'<span class="eval-strategy">{_esc(strategy)}</span>'
                        f'<span class="eval-tag" style="background:{resp_color}25;color:{resp_color}">'
                        f"{resp_label}</span>"
                        f'<span class="eval-answer">답: {_esc(predicted)}</span>'
                        f"{suspect_badge}"
                        f"</div>"
                        f'<details class="eval-raw"><summary>전체 응답 보기</summary>'
                        f"<pre>{_esc(raw)}</pre></details>"
                        f"{exec_block}"
                        f"</div>"
                    )

            # Solvability verdict row
            verdict = verdict_map.get(qid, {}).get(ttype, "")
            if verdict == "compromised":
                verdict_color = "#fb923c"
                verdict_label = "⚠ 잠재적 암기 의심 (Potentially compromised)"
                verdict_desc = f"인간이 승인했으나 {suspect_count}개 모델이 자신있게 정답 — 변환 전 값 암기 가능성"
            elif verdict == "effective":
                verdict_color = "#22c55e"
                verdict_label = "✓ 유효한 변환 (Effective)"
                verdict_desc = f"인간 승인 + 전 모델 거부/오답 ({refused_count}/{total_evals} refused)"
            elif verdict == "rejected":
                verdict_color = "#ef4444"
                verdict_label = "✗ 리뷰어 부적절 판정"
                verdict_desc = "리뷰어가 변환을 부적절로 판정"
            elif verdict == "needs_modification":
                verdict_color = "#f59e0b"
                verdict_label = "✎ 수정 필요"
                verdict_desc = "리뷰어가 수정 필요로 판정"
            else:
                verdict_color = "#888"
                verdict_label = "—"
                verdict_desc = "판정 정보 부족"

            verdict_html = (
                f'<div class="verdict-row" data-verdict="{verdict}" '
                f'style="border-left:4px solid {verdict_color}">'
                f'<div class="verdict-label" style="color:{verdict_color}">{verdict_label}</div>'
                f'<div class="verdict-desc">{_esc(verdict_desc)}</div>'
                f"</div>"
            )

            # Context comparison (skip if both empty)
            ctx_html = ""
            if context_original or ctx_transformed:
                ctx_html = f"""
              <div class="context-compare">
                <div class="ctx-col">
                  <div class="detail-label">원본 Context</div>
                  <pre class="ctx-pre">{_esc(context_original[:2000]) if context_original else "(context 없음)"}</pre>
                </div>
                <div class="ctx-col">
                  <div class="detail-label">변환 Context ({ttype})</div>
                  <pre class="ctx-pre">{_esc(ctx_transformed[:2000]) if ctx_transformed else "(context 없음)"}</pre>
                </div>
              </div>"""

            display = "" if i == 0 else "none"
            panels_html += f"""
            <div class="detail-panel" id="panel-{qid}-{ttype}" style="display:{display}">
              {verdict_html}
              <div class="detail-section">
                <div class="detail-label">변환 설명</div>
                <div class="detail-desc">{_esc(description)}</div>
              </div>
              {question_html}
              <div class="reviewer-comparison">
                <div class="detail-label">리뷰어 판정 비교</div>
                {reviewer_html}
              </div>
              {llm_judgment_html}
              {ctx_html}
              {"<div class='eval-section'><div class='detail-label'>실험 LLM 응답 — 모델 × 프롬프트 전략</div>" + eval_html + "</div>" if eval_html else ""}
            </div>"""

        # Card-level filter metadata
        verdicts_attr = " ".join(card_verdicts) if card_verdicts else "none"
        reviewers_attr = "|" + "|".join(card_reviewers) + "|" if card_reviewers else ""
        types_attr = " ".join(reviewed_types)
        disagreement_flag = "1" if has_disagreement else "0"

        # Header badges summarizing verdicts
        header_badges = ""
        counts = {
            "effective": 0,
            "compromised": 0,
            "needs_modification": 0,
            "rejected": 0,
        }
        for v in verdict_map.get(qid, {}).values():
            if v in counts:
                counts[v] += 1
        if counts["effective"]:
            header_badges += (
                f'<span class="hdr-badge hb-effective">✓ {counts["effective"]}</span>'
            )
        if counts["compromised"]:
            header_badges += f'<span class="hdr-badge hb-compromised">⚠ {counts["compromised"]}</span>'
        if counts["needs_modification"]:
            header_badges += f'<span class="hdr-badge hb-mod">✎ {counts["needs_modification"]}</span>'
        if counts["rejected"]:
            header_badges += (
                f'<span class="hdr-badge hb-rejected">✗ {counts["rejected"]}</span>'
            )
        if has_disagreement:
            header_badges += '<span class="hdr-badge hb-disagree">불일치</span>'

        detail_cards += f"""
        <div class="detail-card" id="card-{qid}"
             data-verdicts="{verdicts_attr}"
             data-types="{types_attr}"
             data-reviewers="{reviewers_attr}"
             data-disagreement="{disagreement_flag}"
             data-qid="{qid}">
          <div class="detail-header" onclick="toggleCard('{qid}')">
            <span class="detail-qid">{_esc(qid)}</span>
            <span class="detail-gt">정답: {_esc(str(gt))}</span>
            {header_badges}
            <span class="toggle-icon" id="icon-{qid}">+</span>
          </div>
          <div class="detail-question">{_esc(question)}</div>
          <div class="detail-body" id="body-{qid}" style="display:none">
            <div class="detail-tabs">{tabs_html}</div>
            {panels_html}
            {consolidated_comments_html}
          </div>
        </div>"""

    # 5. Reviewer legend
    legend_html = " ".join(
        f'<span class="legend-item">'
        f'<span class="legend-dot" style="background:{reviewer_color[name]}"></span>'
        f"{_esc(name)}</span>"
        for name in reviewer_names
    )

    # 6. Per-type effective vs compromised table rows
    type_verdict_rows = ""
    for t in TYPE_KEYS:
        s = type_verdict_stats[t]
        approved_total = s["approved_total"]
        eff = s["effective"]
        comp = s["compromised"]
        mod = s["needs_modification"]
        rej = s["rejected"]
        if approved_total == 0 and mod == 0 and rej == 0:
            type_verdict_rows += f'<tr><td>{t}</td><td colspan="5" style="color:var(--text3)">-</td></tr>'
            continue
        eff_pct = (eff / approved_total * 100) if approved_total else 0
        comp_pct = (comp / approved_total * 100) if approved_total else 0
        type_verdict_rows += (
            f"<tr><td><strong>{t}</strong></td>"
            f"<td>{approved_total}</td>"
            f'<td style="color:#22c55e">{eff} ({eff_pct:.0f}%)</td>'
            f'<td style="color:#fb923c">{comp} ({comp_pct:.0f}%)</td>'
            f'<td style="color:#f59e0b">{mod}</td>'
            f'<td style="color:#ef4444">{rej}</td></tr>'
        )

    # 7. Filter controls — ttype checkboxes + reviewer checkboxes
    ttype_checkboxes = "".join(
        f'<label class="chk-label"><input type="checkbox" class="ttype-chk" value="{t}" checked onchange="applyDetailFilters()"> {t}</label>'
        for t in TYPE_KEYS
    )
    reviewer_checkboxes = "".join(
        f'<label class="chk-label"><input type="checkbox" class="reviewer-chk" value="{_esc(name)}" checked onchange="applyDetailFilters()">'
        f'<span style="color:{reviewer_color[name]};font-weight:600">{_esc(name)}</span></label>'
        for name in reviewer_names
    )

    total_cards = sum(
        1 for qid, tr, _ in matrix_data if any(tr.get(t) for t in TYPE_KEYS)
    )
    mod_count = sum(
        1 for qid in verdict_map if "needs_modification" in verdict_map[qid].values()
    )
    rej_count = sum(1 for qid in verdict_map if "rejected" in verdict_map[qid].values())
    eff_card_count = sum(
        1 for qid in verdict_map if "effective" in verdict_map[qid].values()
    )
    comp_card_count = sum(
        1 for qid in verdict_map if "compromised" in verdict_map[qid].values()
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Review Summary v2 — 리뷰어 비교 + 불일치 상세</title>
<style>
:root {{
  --bg:#0a0a0a; --surface:#141414; --surface2:#1c1c1c; --border:#2a2a2a;
  --text:#e5e5e5; --text2:#a0a0a0; --text3:#666;
}}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,'Segoe UI','Noto Sans KR',sans-serif; background:var(--bg); color:var(--text); font-size:14px; line-height:1.5; }}
.container {{ max-width:1600px; margin:0 auto; padding:20px; }}
h1 {{ font-size:22px; margin-bottom:4px; }}
h2 {{ font-size:16px; margin:28px 0 10px; color:var(--text2); border-bottom:1px solid var(--border); padding-bottom:6px; }}

/* Stats */
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:16px 0 24px; }}
.stat {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; text-align:center; }}
.stat .val {{ font-size:26px; font-weight:700; }}
.stat .label {{ font-size:11px; color:var(--text2); margin-top:2px; }}

/* Tables */
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; font-size:13px; }}
th {{ color:var(--text2); font-size:11px; text-transform:uppercase; position:sticky; top:0; background:var(--bg); z-index:2; }}
.qid-cell {{ font-family:monospace; font-size:12px; white-space:nowrap; min-width:90px; }}
.cell-na {{ color:#222; text-align:center; }}
.cell-empty {{ color:var(--text3); text-align:center; }}
.cell-ok {{ }}
.cell-conflict {{ background:#2a1a00; }}

/* Badges */
.badge {{
  display:inline-flex; align-items:center; gap:2px;
  font-size:11px; padding:2px 6px; border-radius:4px; margin:1px;
  white-space:nowrap; cursor:default;
}}
.badge-initial {{ font-weight:700; font-size:10px; margin-right:1px; }}

/* Legend */
.legend {{ display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 16px; }}
.legend-item {{ display:flex; align-items:center; gap:4px; font-size:13px; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; }}

/* Filter */
.filter-bar {{ display:flex; gap:6px; margin-bottom:12px; flex-wrap:wrap; }}
.filter-btn {{ padding:5px 14px; border:1px solid var(--border); border-radius:6px; background:var(--surface); color:var(--text); cursor:pointer; font-size:12px; transition:all .15s; }}
.filter-btn:hover,.filter-btn.active {{ background:var(--border); color:#fff; }}

/* Disagreement rows */
.disagreement-row {{ background:#1a1000; }}
.disagreement-row:hover {{ background:#2a2000; }}

/* Detail cards */
.detail-card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; margin-bottom:10px; overflow:hidden; }}
.detail-header {{
  display:flex; align-items:center; gap:12px; padding:12px 16px; cursor:pointer;
  border-bottom:1px solid transparent; transition:border-color .15s;
}}
.detail-header:hover {{ background:var(--surface2); }}
.detail-card.open .detail-header {{ border-bottom-color:var(--border); }}
.detail-qid {{ font-family:monospace; font-weight:700; font-size:14px; }}
.detail-gt {{ font-size:12px; color:var(--text2); }}
.detail-conflict-count {{ font-size:11px; color:#f59e0b; background:#f59e0b15; padding:2px 8px; border-radius:10px; }}
.toggle-icon {{ margin-left:auto; font-size:18px; color:var(--text3); font-weight:300; }}
.detail-question {{ font-size:12px; color:var(--text2); padding:0 16px 10px; }}
.detail-body {{ padding:0 16px 16px; }}

/* Detail tabs */
.detail-tabs {{ display:flex; gap:4px; margin-bottom:12px; }}
.detail-tab {{ padding:5px 14px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text2); cursor:pointer; font-size:12px; }}
.detail-tab.active {{ background:var(--border); color:#fff; }}

/* Detail sections */
.detail-section {{ margin-bottom:14px; }}
.detail-label {{ font-size:11px; color:var(--text3); text-transform:uppercase; margin-bottom:4px; font-weight:600; }}
.detail-desc {{ font-size:13px; color:var(--text); }}

/* Reviewer comparison */
.reviewer-comparison {{ margin-bottom:14px; }}
.reviewer-box {{ background:var(--bg); border-radius:6px; padding:8px 12px; margin:4px 0; }}
.reviewer-name {{ font-size:12px; font-weight:700; margin-bottom:2px; }}
.judgment-tag {{ font-size:11px; padding:2px 8px; border-radius:4px; display:inline-block; }}
.reviewer-note {{ font-size:12px; color:var(--text2); margin-top:4px; line-height:1.4; }}

/* Context compare */
.context-compare {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px; }}
.ctx-col {{ min-width:0; }}
.ctx-pre {{
  background:var(--bg); border:1px solid var(--border); border-radius:6px;
  padding:10px; font-size:11px; line-height:1.4; overflow-x:auto;
  max-height:400px; overflow-y:auto; white-space:pre-wrap; word-break:break-all;
}}

/* Eval section */
.eval-section {{ margin-bottom:14px; }}
.eval-box {{ background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:8px 12px; margin:4px 0; }}
.eval-header {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.eval-model {{ font-size:12px; font-weight:700; color:var(--text); }}
.eval-tag {{ font-size:11px; padding:2px 8px; border-radius:4px; }}
.eval-answer {{ font-size:12px; color:var(--text2); }}
.eval-raw {{ margin-top:6px; }}
.eval-raw summary {{ font-size:11px; color:var(--text3); cursor:pointer; }}
.eval-raw pre {{ font-size:11px; color:var(--text2); max-height:300px; overflow-y:auto; white-space:pre-wrap; word-break:break-all; margin-top:4px; }}

/* Question compare */
.question-compare {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px; }}
.q-col {{ min-width:0; }}
.q-text {{ font-size:13px; color:var(--text); padding:8px 12px; background:var(--bg); border:1px solid var(--border); border-radius:6px; line-height:1.5; }}
.q-transformed {{ border-color:#f59e0b40; background:#f59e0b08; }}

/* LLM judgment */
.llm-judgment-box {{
  background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border:1px solid #334155; border-radius:8px; padding:12px 16px; margin-bottom:14px;
}}
.lj-header {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap; }}
.lj-conf {{ font-size:11px; }}
.lj-agrees {{ font-size:11px; color:var(--text2); }}
.lj-comment {{ font-size:13px; color:var(--text); line-height:1.5; margin-bottom:6px; }}
.lj-issues {{ font-size:12px; color:#f59e0b; margin:6px 0 6px 16px; }}
.lj-issues li {{ margin:2px 0; }}
.lj-suggestion {{ font-size:12px; color:#60a5fa; margin-top:6px; padding:6px 10px; background:#60a5fa10; border-radius:4px; }}

.section {{ margin-bottom:30px; }}

/* Verdict row */
.verdict-row {{ background:var(--bg); padding:10px 14px; margin-bottom:14px; border-radius:6px; }}
.verdict-label {{ font-size:13px; font-weight:700; margin-bottom:2px; }}
.verdict-desc {{ font-size:11px; color:var(--text2); }}

/* Suspect badge (memorization) */
.suspect-badge {{
  background:linear-gradient(135deg,#fb923c,#ef4444);
  color:#fff; font-size:10px; font-weight:700;
  padding:3px 8px; border-radius:10px; margin-left:auto;
  box-shadow:0 0 8px rgba(251,146,60,0.4);
}}

/* Eval strategy */
.eval-strategy {{
  font-size:10px; color:var(--text3); font-family:monospace;
  background:var(--border); padding:1px 6px; border-radius:3px;
}}

/* Header badges */
.hdr-badge {{ font-size:10px; padding:2px 7px; border-radius:10px; font-weight:600; }}
.hb-effective {{ color:#22c55e; background:#22c55e18; }}
.hb-compromised {{ color:#fb923c; background:linear-gradient(135deg,#fb923c25,#ef444425); border:1px solid #fb923c55; }}
.hb-mod {{ color:#f59e0b; background:#f59e0b18; }}
.hb-rejected {{ color:#ef4444; background:#ef444418; }}
.hb-disagree {{ color:#fbbf24; background:#fbbf2415; }}

/* Consolidated comments */
.consolidated-comments {{
  background:var(--bg); border:1px solid var(--border); border-radius:6px;
  padding:10px 14px; margin-top:14px;
}}
.comment-item {{ padding:6px 10px; margin:4px 0; background:var(--surface2); border-radius:4px; }}
.comment-meta {{ display:flex; align-items:center; gap:8px; margin-bottom:3px; flex-wrap:wrap; }}
.comment-ttype {{ font-size:10px; font-family:monospace; background:var(--border); padding:1px 6px; border-radius:3px; color:var(--text2); }}
.comment-reviewer {{ font-size:11px; font-weight:700; }}
.comment-note {{ font-size:12px; color:var(--text); line-height:1.5; }}

/* Filter panel */
.detail-filters {{
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; margin-bottom:14px;
}}
.detail-filters .filter-row {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:5px 0; }}
.detail-filters .filter-label {{ font-size:11px; color:var(--text3); text-transform:uppercase; min-width:70px; font-weight:600; }}
.chk-label {{
  display:inline-flex; align-items:center; gap:4px; font-size:12px;
  padding:3px 8px; border:1px solid var(--border); border-radius:4px;
  background:var(--bg); cursor:pointer;
}}
.chk-label input {{ margin:0; cursor:pointer; }}
.search-input {{
  background:var(--bg); border:1px solid var(--border); border-radius:4px;
  padding:4px 8px; color:var(--text); font-size:12px; min-width:180px;
}}
.filter-summary {{ font-size:11px; color:var(--text3); margin-top:4px; }}

/* Compromised card highlight */
.detail-card[data-verdicts*="compromised"] {{ border-left:3px solid #fb923c; }}

@media (max-width:900px) {{
  .question-compare {{ grid-template-columns:1fr; }}
  .context-compare {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<div class="container">
  <h1>Review Summary v2</h1>
  <p style="color:var(--text2);margin-bottom:4px;">
    생성: {now} | {len(all_qids)}문제 리뷰됨 | {total_annotations}건 판정 | {len(reviewer_names)}명 리뷰어
  </p>

  <div class="legend">
    <span style="font-size:12px;color:var(--text3);margin-right:4px;">리뷰어:</span>
    {legend_html}
    <span style="margin-left:12px;font-size:12px;color:var(--text3);">|</span>
    <span class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>승인</span>
    <span class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>수정필요</span>
    <span class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>부적절</span>
  </div>

  <div class="stats-grid">
    <div class="stat"><div class="val">{len(all_qids)}</div><div class="label">리뷰된 문제</div></div>
    <div class="stat"><div class="val">{total_annotations}</div><div class="label">총 판정</div></div>
    <div class="stat"><div class="val" style="color:#22c55e">{sum(s["approved"] for s in type_stats.values())}</div><div class="label">승인</div></div>
    <div class="stat"><div class="val" style="color:#f59e0b">{sum(s["needs_modification"] for s in type_stats.values())}</div><div class="label">수정필요</div></div>
    <div class="stat"><div class="val" style="color:#ef4444">{sum(s["rejected"] for s in type_stats.values())}</div><div class="label">부적절</div></div>
    <div class="stat"><div class="val" style="color:#fb923c">{disagreement_count}</div><div class="label">불일치 문제</div></div>
  </div>

  <div class="section">
    <h2>변환 타입별 승인 현황</h2>
    <table>
      <tr><th>타입</th><th>판정 수</th><th>승인</th><th>수정필요</th><th>부적절</th></tr>
      {type_stats_rows}
    </table>
  </div>

  <div class="section">
    <h2>리뷰어별 통계</h2>
    <table>
      <tr><th>이름</th><th>리뷰 수</th><th>승인</th><th>수정필요</th><th>부적절</th></tr>
      {reviewer_rows}
    </table>
  </div>

  <div class="section">
    <h2>전체 문제 매트릭스 — 리뷰어별 판정 비교</h2>
    <p style="color:var(--text3);font-size:12px;margin-bottom:8px;">
      각 셀에 리뷰어 이니셜 + 판정이 표시됩니다. 노란 배경 = 리뷰어 간 불일치. 마우스 올리면 상세 메모.
    </p>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="filterRows('all')">전체 ({len(all_qids)})</button>
      <button class="filter-btn" onclick="filterRows('disagreement')">불일치만 ({disagreement_count})</button>
    </div>
    <div style="max-height:600px;overflow-y:auto;">
    <table id="matrix">
      <tr><th>문제</th>{"".join(f"<th>{t}</th>" for t in TYPE_KEYS)}</tr>
      {table_rows}
    </table>
    </div>
  </div>

  <div class="section">
    <h2>불일치 상세 ({disagreement_count}건)</h2>
    <p style="color:var(--text3);font-size:12px;margin-bottom:12px;">
      리뷰어 간 판정이 다른 문제입니다. 클릭하면 원본/변환 context + LLM 응답을 함께 볼 수 있습니다.
    </p>
    {detail_cards if detail_cards else '<p style="color:var(--text3);">불일치 없음</p>'}
  </div>
</div>

<script>
function filterRows(mode) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('#matrix tr').forEach((row, i) => {{
    if (i === 0) return;
    if (mode === 'all') row.style.display = '';
    else if (mode === 'disagreement') row.style.display = row.classList.contains('disagreement-row') ? '' : 'none';
  }});
}}

function toggleCard(qid) {{
  const body = document.getElementById('body-' + qid);
  const icon = document.getElementById('icon-' + qid);
  const card = document.getElementById('card-' + qid);
  if (body.style.display === 'none') {{
    body.style.display = '';
    icon.textContent = '−';
    card.classList.add('open');
  }} else {{
    body.style.display = 'none';
    icon.textContent = '+';
    card.classList.remove('open');
  }}
}}

function switchDetailTab(btn, qid, ttype) {{
  // Deactivate all tabs in this card
  const card = document.getElementById('card-' + qid);
  card.querySelectorAll('.detail-tab').forEach(t => t.classList.remove('active'));
  card.querySelectorAll('.detail-panel').forEach(p => p.style.display = 'none');
  btn.classList.add('active');
  document.getElementById('panel-' + qid + '-' + ttype).style.display = '';
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate review summary HTML (v2)")
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="experiments/results/metacognitive/annotations",
    )
    parser.add_argument(
        "--batch-input",
        type=str,
        default=None,
        help="Batch transformations JSON (auto-detect if not specified)",
    )
    parser.add_argument(
        "--eval-input",
        type=str,
        default=None,
        help="Batch evaluation JSON for LLM responses",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/metacognitive/review_summary.html",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    ann_dir = project_root / args.annotations_dir
    output_path = project_root / args.output

    # Auto-detect batch file
    if args.batch_input:
        batch_path = project_root / args.batch_input
    else:
        candidates = sorted(
            (project_root / "experiments/results/metacognitive").glob(
                "batch_transformations_*.json"
            ),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        batch_path = candidates[0] if candidates else None

    if not batch_path or not batch_path.exists():
        logger.error(f"배치 파일 없음: {batch_path}")
        sys.exit(1)

    # Auto-detect eval file
    if args.eval_input:
        eval_path = project_root / args.eval_input
    else:
        candidates = sorted(
            (project_root / "experiments/results/metacognitive").glob(
                "batch_evaluation_*.json"
            ),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        eval_path = candidates[0] if candidates else None

    problems = load_problems(batch_path)
    logger.info(f"문제 로드: {len(problems)}개 ({batch_path.name})")

    per_reviewer, reviewer_names = load_all_annotations(ann_dir)
    logger.info(f"리뷰어 {len(reviewer_names)}명: {reviewer_names}")

    evaluations = (
        load_evaluations(eval_path) if eval_path and eval_path.exists() else {}
    )
    if evaluations:
        logger.info(
            f"평가 데이터 로드: {sum(len(v) for t in evaluations.values() for v in t.values())}건 ({eval_path.name})"
        )

    # Load LLM judgments if available
    judgments_path = ann_dir / "llm_judgments.json"
    llm_judgments = load_llm_judgments(judgments_path)

    html_content = generate_summary_html(
        problems, per_reviewer, reviewer_names, evaluations, llm_judgments
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"요약 HTML 생성: {output_path}")


if __name__ == "__main__":
    main()
