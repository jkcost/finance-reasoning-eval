"""
IC Difficulty Ladder — Human Review HTML Generator

IC-L1~L4 변환 결과를 연구자들이 분담하여 리뷰할 수 있는 인터랙티브 HTML 생성.
기존 generate_human_review.py의 포맷을 재사용하되, IC Ladder 전용으로 특화.

Features:
  - IC-L1~L4 탭별 변환 결과 표시 (L5는 coverage 부족으로 제외)
  - 연구자 분배: URL ?assignee=이름&start=0&end=30
  - JSON 내보내기/가져오기 (기존 merge_annotations.py 호환)
  - 난이도별 색상 구분 + 예상 탐지 난이도 표시
  - 원본 context vs 변환 context 비교

Usage:
    python experiments/generate_ic_ladder_review.py
    python experiments/generate_ic_ladder_review.py --input ic_ladder_transformations_0_120.json

    # 4명 연구자 분배 (각 30문제)
    # ic_ladder_review.html?assignee=연구자A&start=0&end=30
    # ic_ladder_review.html?assignee=연구자B&start=30&end=60
    # ic_ladder_review.html?assignee=연구자C&start=60&end=90
    # ic_ladder_review.html?assignee=연구자D&start=90&end=120
"""

import argparse
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# IC Ladder type keys (L5 excluded due to <1% coverage)
IC_TYPE_KEYS = ["IC-L1", "IC-L2", "IC-L3", "IC-L4"]
IC_TYPE_DESCRIPTIONS = {
    "IC-L1": "10x 타이포 — 자릿수 오류 (가장 쉬움, baseline)",
    "IC-L2": "단위 불일치 — million/billion, %/bps 혼동",
    "IC-L3": "권위 충돌 — 1.5x 모순값 + 권위적 출처",
    "IC-L4": "기간 합산 불일치 — 분기합 ≠ 연간총계",
}
IC_TYPE_COLORS = {
    "IC-L1": "#22c55e",  # green (easy)
    "IC-L2": "#3b82f6",  # blue
    "IC-L3": "#ef4444",  # red (current IC)
    "IC-L4": "#f59e0b",  # amber
}
IC_DIFFICULTY_LABELS = {
    "IC-L1": "쉬움 ★☆☆☆",
    "IC-L2": "보통 ★★☆☆",
    "IC-L3": "어려움 ★★★☆",
    "IC-L4": "어려움 ★★★☆",
}


def _esc(text: str) -> str:
    return html.escape(str(text)) if text else ""


def _esc_js(s: str) -> str:
    """JavaScript 문자열 리터럴에 안전한 이스케이프."""
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _safe_json_for_html(data: Any) -> str:
    """JSON을 <script> 블록 안에 안전하게 삽입."""
    s = json.dumps(data, ensure_ascii=False)
    s = s.replace("</", "<\\/")
    s = s.replace("<!--", "<\\!--")
    return s


def _restructure_transformations(
    raw_transformations: List[Dict],
) -> Dict[str, Dict[str, Any]]:
    """Restructure flat transformation list into per-problem grouped format.

    Returns:
        {question_id: {
            "question": str,
            "context_original": str,
            "ground_truth": str,
            "python_solution": str,
            "transformations": {
                "IC-L1": {success, context_transformed, description, ...},
                "IC-L2": {...},
                ...
            }
        }}
    """
    problems: Dict[str, Dict[str, Any]] = {}

    for t in raw_transformations:
        qid = t.get("question_id", "unknown")
        ttype_full = t.get("transformation_type", "")

        # Extract short key (IC-L1, IC-L2, etc.)
        ttype_short = (
            ttype_full.split(":")[0].strip() if ":" in ttype_full else ttype_full
        )

        if qid not in problems:
            problems[qid] = {
                "question_id": qid,
                "question": t.get("question", ""),
                "context_original": t.get("context_original", t.get("context", "")),
                "ground_truth": str(t.get("ground_truth", "")),
                "python_solution": t.get("python_solution", ""),
                "transformations": {},
            }

        problems[qid]["transformations"][ttype_short] = {
            "success": True,
            "context_transformed": t.get("context", ""),
            "description": t.get("transformation_description", ""),
            "expected_behavior": t.get("expected_behavior", ""),
            "type_full": ttype_full,
        }

    return problems


def _render_problem_card(problem: Dict, idx: int) -> str:
    qid = problem["question_id"]
    question = problem["question"]
    context_orig = problem.get("context_original", "")
    ground_truth = problem.get("ground_truth", "")
    transformations = problem["transformations"]

    success_types = [k for k in IC_TYPE_KEYS if k in transformations]

    # Tabs
    tabs_html = ""
    panels_html = ""
    for ttype in IC_TYPE_KEYS:
        tdata = transformations.get(ttype)
        is_success = tdata is not None
        tab_class = "tab-success" if is_success else "tab-fail"
        color = IC_TYPE_COLORS.get(ttype, "#666")
        desc = IC_TYPE_DESCRIPTIONS.get(ttype, "")
        difficulty = IC_DIFFICULTY_LABELS.get(ttype, "")

        qid_js = _esc_js(qid)
        ttype_js = _esc_js(ttype)
        tabs_html += (
            f'<button class="tab-btn {tab_class}" '
            f"onclick=\"showTab('{qid_js}', '{ttype_js}')\" "
            f'id="tab-{qid}-{ttype}" '
            f'title="{_esc(desc)}" '
            f'style="--type-color:{color}">'
            f'{ttype} <span class="difficulty-badge">{difficulty}</span>'
            f"</button>"
        )

        if is_success:
            ctx_transformed = tdata.get("context_transformed", "")
            desc_text = tdata.get("description", "")
            expected = tdata.get("expected_behavior", "")

            panel_content = (
                f'<div class="transform-desc">'
                f'<span class="type-info">{_esc(IC_TYPE_DESCRIPTIONS.get(ttype, ""))}</span>'
                f"<br><strong>변환:</strong> {_esc(desc_text)}"
                f"<br><strong>기대 행동:</strong> {_esc(expected)}</div>"
                f'<div class="diff-container">'
                f'<div class="diff-panel">'
                f'<div class="diff-label">원본 Context</div>'
                f'<pre class="diff-content">{_esc(str(context_orig)[:2000])}</pre></div>'
                f'<div class="diff-panel diff-transformed">'
                f'<div class="diff-label">변환 후 Context ({ttype})</div>'
                f'<textarea class="diff-content editable" '
                f'id="edit-{qid}-{ttype}" rows="12"'
                f">{_esc(str(ctx_transformed)[:2000])}</textarea>"
                f"</div></div>"
            )

            # Review controls
            panel_content += (
                f'<div class="review-controls">'
                f'<div class="review-row">'
                f"<label>변환 품질:</label>"
                f"<select id=\"judgment-{qid}-{ttype}\" onchange=\"saveAnnotation('{qid_js}','{ttype_js}')\">"
                f'<option value="">미평가</option>'
                f'<option value="approved">✓ 승인 (자연스러운 변환)</option>'
                f'<option value="needs_modification">△ 수정 필요</option>'
                f'<option value="rejected">✗ 부적절 (변환 불가)</option>'
                f"</select></div>"
                f'<div class="review-row">'
                f"<label>모델이 탐지할 수 있을까?</label>"
                f"<select id=\"detectable-{qid}-{ttype}\" onchange=\"saveAnnotation('{qid_js}','{ttype_js}')\">"
                f'<option value="">미평가</option>'
                f'<option value="easy">쉬움 (대부분 탐지)</option>'
                f'<option value="medium">보통 (일부 탐지)</option>'
                f'<option value="hard">어려움 (대부분 실패)</option>'
                f'<option value="impossible">불가능 (탐지 기대 불가)</option>'
                f"</select></div>"
                f'<div class="review-row">'
                f"<label>메모:</label>"
                f'<input type="text" id="note-{qid}-{ttype}" placeholder="리뷰 메모..." '
                f"onchange=\"saveAnnotation('{qid_js}','{ttype_js}')\" style=\"flex:1\">"
                f"</div></div>"
            )
        else:
            panel_content = (
                f'<div class="transform-na">'
                f"이 문제에는 {ttype} 변환이 적용되지 않았습니다."
                f"</div>"
            )

        display = "block" if (success_types and ttype == success_types[0]) else "none"
        panels_html += (
            f'<div class="tab-panel" id="panel-{qid}-{ttype}" '
            f'style="display:{display}">{panel_content}</div>'
        )

    badge_html = " ".join(
        f'<span class="type-badge" style="background:{IC_TYPE_COLORS.get(t, "#666")}">{t}</span>'
        for t in success_types
    )

    return f"""
    <div class="problem-card" id="card-{qid}" data-index="{idx}">
      <div class="card-header">
        <div class="card-title">
          <span class="card-idx">#{idx}</span>
          <span class="card-qid">{qid}</span>
          {badge_html}
          <span class="card-count">{len(success_types)}/{len(IC_TYPE_KEYS)} levels</span>
        </div>
      </div>
      <div class="card-body">
        <div class="question-box">
          <div class="section-label">Question</div>
          <div class="question-text">{_esc(question)}</div>
        </div>
        <div class="gt-box">
          <span class="section-label">정답:</span> {_esc(ground_truth)}
        </div>
        <div class="tabs-row">{tabs_html}</div>
        <div class="panels-container">{panels_html}</div>
      </div>
    </div>"""


def generate_html(
    problems: Dict[str, Dict],
    metadata: Dict,
    level_counts: Dict[str, int],
) -> str:
    """Generate full IC Ladder review HTML."""
    total = metadata.get("total_problems", len(problems))

    # Sort by question_id index
    sorted_problems = sorted(
        problems.values(),
        key=lambda p: (
            int(p["question_id"].split("_")[-1]) if "_" in p["question_id"] else 0
        ),
    )

    # Assign index
    for i, p in enumerate(sorted_problems):
        p["_idx"] = i

    # Cards
    cards_html = ""
    for p in sorted_problems:
        cards_html += _render_problem_card(p, p["_idx"])

    # Coverage
    coverage_rows = ""
    for ttype in IC_TYPE_KEYS:
        label_key = f"IC-{ttype.split('-')[1]}: " if "-" in ttype else ttype
        # Find matching count
        count = 0
        for k, v in level_counts.items():
            if ttype in k:
                count = v
                break
        pct = count / total * 100 if total > 0 else 0
        color = IC_TYPE_COLORS.get(ttype, "#666")
        desc = IC_TYPE_DESCRIPTIONS.get(ttype, "")
        coverage_rows += (
            f'<tr><td><span class="type-badge" style="background:{color}">{ttype}</span></td>'
            f"<td>{desc}</td>"
            f"<td>{count}/{total}</td>"
            f'<td><div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div></td>'
            f"<td>{pct:.1f}%</td></tr>"
        )

    # Problem metadata for JS
    problems_meta = []
    for p in sorted_problems:
        problems_meta.append(
            {
                "question_id": p["question_id"],
                "index": p["_idx"],
                "success_types": [k for k in IC_TYPE_KEYS if k in p["transformations"]],
            }
        )
    problems_meta_json = _safe_json_for_html(problems_meta)

    range_start = metadata.get("range", [0, 120])[0]
    range_end = metadata.get("range", [0, 120])[1]

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IC Ladder Review [{range_start}-{range_end})</title>
<style>
:root {{
  --bg: #0a0a0a; --surface: #141414; --surface2: #1e1e1e;
  --border: #2a2a2a; --text: #e5e5e5; --text2: #a0a0a0;
  --green: #22c55e; --blue: #3b82f6; --red: #ef4444;
  --amber: #f59e0b; --gray: #6b7280;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; font-size: 14px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 22px; font-weight: 600; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 20px 0 10px; }}

.header-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }}
.assignee-bar {{ display: flex; align-items: center; gap: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 14px; }}
.assignee-bar label {{ font-size: 13px; color: var(--text2); }}
.assignee-bar input {{ background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 4px 8px; font-size: 13px; width: 120px; }}
.assignee-bar .range-info {{ font-size: 12px; color: var(--text2); }}

.btn {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 6px 14px; cursor: pointer; font-size: 13px; }}
.btn:hover {{ background: var(--border); }}
.btn-primary {{ background: var(--blue); border-color: var(--blue); }}
.btn-primary:hover {{ opacity: 0.9; }}

.coverage-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
.coverage-table th, .coverage-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
.coverage-table th {{ color: var(--text2); font-size: 12px; text-transform: uppercase; }}
.bar-bg {{ height: 6px; background: var(--surface2); border-radius: 3px; }}
.bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
.type-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; color: #fff; }}

.problem-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 16px; overflow: hidden; }}
.card-header {{ padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }}
.card-title {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.card-idx {{ font-weight: 700; color: var(--blue); }}
.card-qid {{ font-size: 12px; color: var(--text2); }}
.card-count {{ font-size: 12px; color: var(--text2); }}
.card-body {{ padding: 16px; }}
.question-box {{ background: var(--bg); border-radius: 6px; padding: 12px; margin-bottom: 10px; }}
.section-label {{ font-size: 11px; color: var(--text2); text-transform: uppercase; margin-bottom: 4px; }}
.question-text {{ font-size: 14px; }}
.gt-box {{ font-size: 13px; color: var(--text2); margin-bottom: 12px; }}

.tabs-row {{ display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }}
.tab-btn {{ padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface2); color: var(--text); cursor: pointer; font-size: 12px; display: flex; align-items: center; gap: 4px; }}
.tab-btn.tab-success {{ border-color: var(--type-color, var(--green)); }}
.tab-btn.tab-fail {{ opacity: 0.3; }}
.tab-btn:hover {{ background: var(--border); }}
.difficulty-badge {{ font-size: 10px; color: var(--text2); }}

.tab-panel {{ display: none; }}
.transform-desc {{ background: var(--bg); padding: 10px; border-radius: 6px; margin-bottom: 10px; font-size: 13px; }}
.type-info {{ color: var(--text2); font-size: 12px; }}
.transform-na {{ padding: 20px; text-align: center; color: var(--text2); }}

.diff-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }}
.diff-panel {{ background: var(--bg); border-radius: 6px; padding: 8px; }}
.diff-label {{ font-size: 11px; color: var(--text2); text-transform: uppercase; margin-bottom: 4px; }}
.diff-content {{ font-size: 12px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }}
.diff-transformed {{ border: 1px solid var(--red); border-opacity: 0.3; }}
textarea.editable {{ width: 100%; background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-family: inherit; resize: vertical; }}

.review-controls {{ background: var(--bg); border-radius: 6px; padding: 12px; }}
.review-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.review-row label {{ font-size: 13px; color: var(--text2); min-width: 140px; }}
.review-row select, .review-row input {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 4px 8px; font-size: 13px; }}

.progress-bar {{ position: fixed; top: 0; left: 0; right: 0; height: 3px; background: var(--surface2); z-index: 999; }}
.progress-fill {{ height: 100%; background: var(--green); transition: width 0.3s; }}

.stats-bar {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
.stat-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; text-align: center; }}
.stat-value {{ font-size: 20px; font-weight: 700; }}
.stat-label {{ font-size: 11px; color: var(--text2); }}

.filter-bar {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
</style>
</head>
<body>
<div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
<div class="container">
  <div class="header-row">
    <h1>IC Difficulty Ladder Review [{range_start}-{range_end})</h1>
    <div class="assignee-bar">
      <label>작업자:</label>
      <input type="text" id="assigneeInput" placeholder="이름">
      <span class="range-info" id="rangeInfo"></span>
      <button class="btn btn-primary" onclick="exportAnnotations()">내보내기</button>
      <button class="btn" onclick="importAnnotations()">가져오기</button>
    </div>
  </div>

  <div class="stats-bar">
    <div class="stat-item"><div class="stat-value">{total}</div><div class="stat-label">전체 문제</div></div>
    <div class="stat-item"><div class="stat-value" id="statReviewed">0</div><div class="stat-label">리뷰 완료</div></div>
    <div class="stat-item"><div class="stat-value" id="statApproved">0</div><div class="stat-label">승인</div></div>
    <div class="stat-item"><div class="stat-value" id="statRejected">0</div><div class="stat-label">부적절</div></div>
  </div>

  <h2>IC Level Coverage</h2>
  <table class="coverage-table">
    <tr><th>Level</th><th>설명</th><th>적용</th><th>비율</th><th>%</th></tr>
    {coverage_rows}
  </table>

  <div class="filter-bar">
    <button class="btn" onclick="filterCards('all')">전체</button>
    <button class="btn" onclick="filterCards('unreviewed')">미리뷰</button>
    <button class="btn" onclick="filterCards('approved')">승인</button>
    <button class="btn" onclick="filterCards('rejected')">부적절</button>
  </div>

  <div id="cardsContainer">
    {cards_html}
  </div>
</div>

<input type="file" id="importFile" style="display:none" accept=".json" onchange="handleImport(event)">

<script>
const TYPE_KEYS = {json.dumps(IC_TYPE_KEYS)};
const PROBLEMS_META = {problems_meta_json};
const RANGE_START = {range_start};
const RANGE_END = {range_end};
let annotations = {{}};
let assignee = '';
let visibleStart = RANGE_START;
let visibleEnd = RANGE_END;

function init() {{
  const params = new URLSearchParams(window.location.search);
  assignee = params.get('assignee') || '';
  visibleStart = parseInt(params.get('start') || RANGE_START);
  visibleEnd = parseInt(params.get('end') || RANGE_END);
  const start = visibleStart;
  const end = visibleEnd;

  document.getElementById('assigneeInput').value = assignee;
  document.getElementById('rangeInfo').textContent =
    assignee ? `${{assignee}}: #${{start}}~#${{end-1}}` : `전체: #${{RANGE_START}}~#${{RANGE_END-1}}`;

  // Filter visible cards by range
  document.querySelectorAll('.problem-card').forEach(card => {{
    const idx = parseInt(card.dataset.index);
    card.style.display = (idx >= start && idx < end) ? '' : 'none';
  }});

  // Load saved annotations from localStorage (keyed by actual visible range)
  const storageKey = `icl_${{start}}_${{end}}_${{assignee || 'default'}}`;
  const saved = localStorage.getItem(storageKey);
  if (saved) {{
    annotations = JSON.parse(saved);
    restoreAnnotations();
  }}

  updateStats();
}}

function showTab(qid, ttype) {{
  TYPE_KEYS.forEach(t => {{
    const panel = document.getElementById(`panel-${{qid}}-${{t}}`);
    if (panel) panel.style.display = t === ttype ? 'block' : 'none';
  }});
}}

function saveAnnotation(qid, ttype) {{
  const judgment = document.getElementById(`judgment-${{qid}}-${{ttype}}`);
  const detectable = document.getElementById(`detectable-${{qid}}-${{ttype}}`);
  const note = document.getElementById(`note-${{qid}}-${{ttype}}`);
  const editArea = document.getElementById(`edit-${{qid}}-${{ttype}}`);

  if (!annotations[qid]) annotations[qid] = {{}};
  annotations[qid][ttype] = {{
    judgment: judgment ? judgment.value : '',
    detectable: detectable ? detectable.value : '',
    note: note ? note.value : '',
    edited_context: editArea ? editArea.value : '',
    timestamp: new Date().toISOString(),
    assignee: assignee,
  }};

  // Save to localStorage (keyed by visible range)
  const storageKey = `icl_${{visibleStart}}_${{visibleEnd}}_${{assignee || 'default'}}`;
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateStats();
}}

function restoreAnnotations() {{
  for (const [qid, types] of Object.entries(annotations)) {{
    for (const [ttype, ann] of Object.entries(types)) {{
      const j = document.getElementById(`judgment-${{qid}}-${{ttype}}`);
      const d = document.getElementById(`detectable-${{qid}}-${{ttype}}`);
      const n = document.getElementById(`note-${{qid}}-${{ttype}}`);
      const e = document.getElementById(`edit-${{qid}}-${{ttype}}`);
      if (j && ann.judgment) j.value = ann.judgment;
      if (d && ann.detectable) d.value = ann.detectable;
      if (n && ann.note) n.value = ann.note;
      if (e && ann.edited_context) e.value = ann.edited_context;
    }}
  }}
}}

function updateStats() {{
  let reviewed = 0, approved = 0, rejected = 0;
  for (const types of Object.values(annotations)) {{
    for (const ann of Object.values(types)) {{
      if (ann.judgment) reviewed++;
      if (ann.judgment === 'approved') approved++;
      if (ann.judgment === 'rejected') rejected++;
    }}
  }}
  document.getElementById('statReviewed').textContent = reviewed;
  document.getElementById('statApproved').textContent = approved;
  document.getElementById('statRejected').textContent = rejected;

  // Progress bar
  const total = PROBLEMS_META.reduce((s, p) => s + p.success_types.length, 0);
  const pct = total > 0 ? (reviewed / total * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
}}

function exportAnnotations() {{
  assignee = document.getElementById('assigneeInput').value || 'unknown';
  const data = {{
    metadata: {{
      assignee: assignee,
      range: [RANGE_START, RANGE_END],
      exported_at: new Date().toISOString(),
      review_type: 'ic_ladder',
    }},
    annotations: annotations,
  }};
  const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ic_review_${{assignee}}_${{RANGE_START}}_${{RANGE_END}}_${{new Date().toISOString().slice(0,10)}}.json`;
  a.click();
  URL.revokeObjectURL(url);
}}

function importAnnotations() {{
  document.getElementById('importFile').click();
}}

function handleImport(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    let data;
    try {{
      data = JSON.parse(e.target.result);
    }} catch(err) {{
      alert('JSON 파싱 실패: ' + err.message);
      return;
    }}
    const imported = data.annotations || data;
    // Merge
    for (const [qid, types] of Object.entries(imported)) {{
      if (!annotations[qid]) annotations[qid] = {{}};
      Object.assign(annotations[qid], types);
    }}
    restoreAnnotations();
    updateStats();
    alert(`${{Object.keys(imported).length}}개 문제의 annotation을 가져왔습니다.`);
    event.target.value = '';
  }};
  reader.readAsText(file);
}}

function filterCards(mode) {{
  document.querySelectorAll('.problem-card').forEach(card => {{
    const idx = parseInt(card.dataset.index);
    // Always respect range filter
    if (idx < visibleStart || idx >= visibleEnd) {{
      card.style.display = 'none';
      return;
    }}
    const qid = card.id.replace('card-', '');
    const ann = annotations[qid] || {{}};
    const hasAny = Object.values(ann).some(a => a.judgment);
    const hasApproved = Object.values(ann).some(a => a.judgment === 'approved');
    const hasRejected = Object.values(ann).some(a => a.judgment === 'rejected');

    if (mode === 'all') card.style.display = '';
    else if (mode === 'unreviewed') card.style.display = hasAny ? 'none' : '';
    else if (mode === 'approved') card.style.display = hasApproved ? '' : 'none';
    else if (mode === 'rejected') card.style.display = hasRejected ? '' : 'none';
  }});
}}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate IC Ladder Review HTML")
    parser.add_argument(
        "--input",
        type=str,
        default="ic_ladder_transformations_0_120.json",
        help="Input IC ladder transformations JSON",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/metacognitive",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ic_ladder_review.html",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    input_path = results_dir / args.input

    if not input_path.exists():
        logger.error(f"입력 파일 없음: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    metadata = raw_data["metadata"]
    level_counts = metadata.get("level_counts", {})
    transformations = raw_data["transformations"]

    logger.info(f"로드: {input_path}")
    logger.info(f"변환 수: {len(transformations)}")

    # Restructure into per-problem format
    problems = _restructure_transformations(transformations)
    logger.info(f"문제 수: {len(problems)}")

    # We need to store original context — currently transformations only have
    # the transformed context. Read from hard.json for originals.
    hard_path = project_root / "data/financereasoning/raw/FinanceReasoning/hard.json"
    if hard_path.exists():
        with open(hard_path, "r", encoding="utf-8") as f:
            hard_data = json.load(f)
        for i, item in enumerate(hard_data[: metadata.get("total_problems", 120)]):
            qid = f"hard_{i}"
            if qid in problems:
                problems[qid]["context_original"] = str(item.get("context", ""))
        logger.info("원본 context 매핑 완료")

    html_content = generate_html(problems, metadata, level_counts)

    output_path = results_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML 생성 완료: {output_path}")
    logger.info("")
    logger.info("=== 연구자 분배 URL ===")
    logger.info(f"연구자A: {args.output}?assignee=연구자A&start=0&end=30")
    logger.info(f"연구자B: {args.output}?assignee=연구자B&start=30&end=60")
    logger.info(f"연구자C: {args.output}?assignee=연구자C&start=60&end=90")
    logger.info(f"연구자D: {args.output}?assignee=연구자D&start=90&end=120")


if __name__ == "__main__":
    main()
