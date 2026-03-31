"""
Review Summary HTML Generator

4명의 리뷰 결과를 합쳐서 팀이 함께 볼 수 있는 요약 대시보드 생성.

기능:
  - 전체 120문제의 리뷰 결과를 한눈에 (승인/수정필요/부적절/미리뷰)
  - 변환 타입별 승인율 차트
  - "수정필요" / "부적절" 문제만 필터링 → 팀 토론 대상
  - 각 문제 클릭 시 원본/변환 context + 리뷰어 메모 표시
  - 리뷰어별 통계 (몇 개 리뷰했는지, 승인/거절 비율)

Usage:
    # 1. annotations/ 폴더에 각 작업자의 JSON이 들어있는 상태에서
    python experiments/generate_review_summary.py

    # 2. 특정 폴더 지정
    python experiments/generate_review_summary.py --annotations-dir annotations/
"""

import argparse
import html as html_module
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TYPE_KEYS = [
    "EA-partial", "EA-full", "SA",
    "IC-L1", "IC-L2", "IC-L3", "IC-L4",
    "TA",
]

JUDGMENT_COLORS = {
    "approved": "#22c55e",
    "needs_modification": "#f59e0b",
    "rejected": "#ef4444",
    "": "#333333",
}

JUDGMENT_LABELS = {
    "approved": "승인",
    "needs_modification": "수정필요",
    "rejected": "부적절",
    "": "미리뷰",
}


def _esc(text: str) -> str:
    return html_module.escape(str(text)) if text else ""


def load_annotations(annotations_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Load all annotation JSON files from directory.

    Returns:
        {question_id: {type: {judgment, note, assignee, ...}}}
    """
    merged: Dict[str, Dict[str, Dict]] = {}

    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files in {annotations_dir}")
        return merged

    for f in json_files:
        if f.name.startswith(".") or f.name == "merged.json" or f.name == "disagreements.json":
            continue

        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Support both formats: {annotations: {...}} and raw dict
        annotations = data.get("annotations", data)
        assignee = data.get("metadata", {}).get("assignee", f.stem)

        for qid, types in annotations.items():
            if not isinstance(types, dict):
                continue
            if qid not in merged:
                merged[qid] = {}
            for ttype, ann in types.items():
                if not isinstance(ann, dict):
                    continue
                ann["assignee"] = ann.get("assignee", assignee)
                merged[qid][ttype] = ann

        logger.info(f"  로드: {f.name} ({assignee}, {sum(1 for t in annotations.values() if isinstance(t, dict) for a in t.values() if isinstance(a, dict) and a.get('judgment'))}건)")

    return merged


def load_problems(batch_path: Path) -> List[Dict]:
    """Load batch_transformations JSON."""
    with open(batch_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("problems", [])


def generate_summary_html(
    problems: List[Dict],
    annotations: Dict[str, Dict[str, Dict]],
) -> str:
    """Generate review summary HTML."""

    # Stats
    total_problems = len(problems)
    total_annotations = sum(
        1 for types in annotations.values()
        for ann in types.values()
        if isinstance(ann, dict) and ann.get("judgment")
    )

    # Per-type stats
    type_stats: Dict[str, Dict[str, int]] = {}
    for t in TYPE_KEYS:
        type_stats[t] = {"approved": 0, "needs_modification": 0, "rejected": 0, "unreviewed": 0, "total": 0}

    # Per-reviewer stats
    reviewer_stats: Dict[str, Dict[str, int]] = {}

    # Per-problem summary
    problem_rows = []
    needs_discussion = []

    for p in problems:
        qid = p.get("question_id", "")
        transformations = p.get("transformations", {})
        qid_annotations = annotations.get(qid, {})

        cells = []
        has_issue = False

        for t in TYPE_KEYS:
            tdata = transformations.get(t, {})
            is_success = tdata.get("success", False)

            if not is_success:
                cells.append(("na", "", "", ""))
                continue

            type_stats[t]["total"] += 1
            ann = qid_annotations.get(t, {})
            judgment = ann.get("judgment", "")
            note = ann.get("note", "")
            assignee = ann.get("assignee", "")

            if judgment:
                type_stats[t][judgment] = type_stats[t].get(judgment, 0) + 1
                if assignee not in reviewer_stats:
                    reviewer_stats[assignee] = {"total": 0, "approved": 0, "needs_modification": 0, "rejected": 0}
                reviewer_stats[assignee]["total"] += 1
                reviewer_stats[assignee][judgment] = reviewer_stats[assignee].get(judgment, 0) + 1
            else:
                type_stats[t]["unreviewed"] += 1

            if judgment in ("needs_modification", "rejected"):
                has_issue = True

            cells.append((judgment, note, assignee, tdata.get("description", "")))

        if has_issue:
            needs_discussion.append((qid, p, qid_annotations))

        problem_rows.append((qid, cells, has_issue))

    # Build HTML
    # Problem table rows
    table_rows = ""
    for qid, cells, has_issue in problem_rows:
        row_class = "issue-row" if has_issue else ""
        row = f'<tr class="{row_class}" data-qid="{qid}">'
        row += f'<td class="qid-cell">{_esc(qid)}</td>'

        for judgment, note, assignee, desc in cells:
            if judgment == "na":
                row += '<td class="cell-na">-</td>'
            else:
                color = JUDGMENT_COLORS.get(judgment, "#333")
                label = JUDGMENT_LABELS.get(judgment, "미리뷰")
                title = f"{assignee}: {note}" if note else assignee
                row += (
                    f'<td class="cell-judgment" style="background:{color}20;border-left:3px solid {color}" '
                    f'title="{_esc(title)}">'
                    f'<span style="color:{color}">{label}</span>'
                    f'</td>'
                )
        row += "</tr>"
        table_rows += row

    # Type stats rows
    type_stats_rows = ""
    for t in TYPE_KEYS:
        s = type_stats[t]
        total = s["total"]
        if total == 0:
            type_stats_rows += f'<tr><td>{t}</td><td>0</td><td colspan="4">해당 없음</td></tr>'
            continue
        apr = s["approved"]
        mod = s["needs_modification"]
        rej = s["rejected"]
        unr = s["unreviewed"]
        apr_pct = apr / total * 100 if total else 0
        type_stats_rows += (
            f'<tr><td>{t}</td><td>{total}</td>'
            f'<td style="color:#22c55e">{apr} ({apr_pct:.0f}%)</td>'
            f'<td style="color:#f59e0b">{mod}</td>'
            f'<td style="color:#ef4444">{rej}</td>'
            f'<td style="color:#666">{unr}</td></tr>'
        )

    # Reviewer stats rows
    reviewer_rows = ""
    for name, s in sorted(reviewer_stats.items()):
        reviewer_rows += (
            f'<tr><td>{_esc(name)}</td><td>{s["total"]}</td>'
            f'<td style="color:#22c55e">{s["approved"]}</td>'
            f'<td style="color:#f59e0b">{s["needs_modification"]}</td>'
            f'<td style="color:#ef4444">{s["rejected"]}</td></tr>'
        )

    # Discussion items (needs_modification + rejected)
    discussion_html = ""
    for qid, problem, qid_ann in needs_discussion:
        question = problem.get("question", "")[:150]
        items = []
        for t in TYPE_KEYS:
            ann = qid_ann.get(t, {})
            j = ann.get("judgment", "")
            if j in ("needs_modification", "rejected"):
                note = ann.get("note", "(메모 없음)")
                assignee = ann.get("assignee", "?")
                color = JUDGMENT_COLORS[j]
                items.append(
                    f'<div style="margin:4px 0;padding:6px 10px;background:{color}15;'
                    f'border-left:3px solid {color};border-radius:4px;font-size:13px;">'
                    f'<strong style="color:{color}">{t}</strong> — '
                    f'{JUDGMENT_LABELS[j]} ({_esc(assignee)})<br>'
                    f'<span style="color:#a0a0a0">{_esc(note)}</span></div>'
                )
        discussion_html += (
            f'<div class="discussion-card">'
            f'<div class="disc-header">{_esc(qid)}</div>'
            f'<div class="disc-question">{_esc(question)}...</div>'
            f'{"".join(items)}</div>'
        )

    if not discussion_html:
        discussion_html = '<p style="color:#a0a0a0;">토론 필요한 문제가 없습니다. 모든 변환이 승인되었습니다.</p>'

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Review Summary — 팀 토론용</title>
<style>
:root {{ --bg:#0a0a0a; --surface:#141414; --border:#2a2a2a; --text:#e5e5e5; --text2:#a0a0a0; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); font-size:14px; line-height:1.5; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}
h1 {{ font-size:22px; margin-bottom:16px; }}
h2 {{ font-size:16px; margin:24px 0 10px; color:var(--text2); }}

.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin-bottom:20px; }}
.stat {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; text-align:center; }}
.stat .val {{ font-size:28px; font-weight:700; }}
.stat .label {{ font-size:11px; color:var(--text2); }}

table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:6px 10px; border-bottom:1px solid var(--border); text-align:left; font-size:13px; }}
th {{ color:var(--text2); font-size:11px; text-transform:uppercase; position:sticky; top:0; background:var(--bg); }}
.qid-cell {{ font-family:monospace; font-size:12px; white-space:nowrap; }}
.cell-na {{ color:#333; text-align:center; }}
.cell-judgment {{ font-size:12px; cursor:default; }}
.issue-row {{ background:#1a1000; }}

.filter-bar {{ display:flex; gap:6px; margin-bottom:12px; }}
.filter-btn {{ padding:5px 12px; border:1px solid var(--border); border-radius:6px; background:var(--surface); color:var(--text); cursor:pointer; font-size:12px; }}
.filter-btn:hover,.filter-btn.active {{ background:var(--border); }}

.discussion-card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px; margin-bottom:10px; }}
.disc-header {{ font-weight:700; font-family:monospace; margin-bottom:4px; }}
.disc-question {{ font-size:12px; color:var(--text2); margin-bottom:8px; }}

.section {{ margin-bottom:30px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Review Summary — 팀 토론용</h1>
  <p style="color:var(--text2);margin-bottom:16px;">
    생성: {datetime.now().strftime("%Y-%m-%d %H:%M")} |
    총 {total_problems}문제 | {total_annotations}건 리뷰 완료
  </p>

  <div class="stats-grid">
    <div class="stat"><div class="val">{total_problems}</div><div class="label">전체 문제</div></div>
    <div class="stat"><div class="val">{total_annotations}</div><div class="label">리뷰 완료</div></div>
    <div class="stat"><div class="val" style="color:#22c55e">{sum(s['approved'] for s in type_stats.values())}</div><div class="label">승인</div></div>
    <div class="stat"><div class="val" style="color:#f59e0b">{sum(s['needs_modification'] for s in type_stats.values())}</div><div class="label">수정 필요</div></div>
    <div class="stat"><div class="val" style="color:#ef4444">{sum(s['rejected'] for s in type_stats.values())}</div><div class="label">부적절</div></div>
    <div class="stat"><div class="val">{len(needs_discussion)}</div><div class="label">토론 필요</div></div>
  </div>

  <div class="section">
    <h2>변환 타입별 승인 현황</h2>
    <table>
      <tr><th>타입</th><th>전체</th><th>승인</th><th>수정필요</th><th>부적절</th><th>미리뷰</th></tr>
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
    <h2>토론 필요 ({len(needs_discussion)}건)</h2>
    <p style="color:var(--text2);font-size:12px;margin-bottom:10px;">
      "수정필요" 또는 "부적절"로 판정된 변환 목록입니다. 팀이 함께 보면서 최종 결정하세요.
    </p>
    {discussion_html}
  </div>

  <div class="section">
    <h2>전체 문제 매트릭스</h2>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="filterRows('all')">전체</button>
      <button class="filter-btn" onclick="filterRows('issue')">토론 필요만</button>
      <button class="filter-btn" onclick="filterRows('unreviewed')">미리뷰만</button>
    </div>
    <div style="max-height:600px;overflow-y:auto;">
    <table id="matrix">
      <tr><th>문제</th>{"".join(f'<th>{t}</th>' for t in TYPE_KEYS)}</tr>
      {table_rows}
    </table>
    </div>
  </div>
</div>

<script>
function filterRows(mode) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('#matrix tr').forEach((row, i) => {{
    if (i === 0) return;
    if (mode === 'all') row.style.display = '';
    else if (mode === 'issue') row.style.display = row.classList.contains('issue-row') ? '' : 'none';
    else if (mode === 'unreviewed') {{
      const cells = row.querySelectorAll('.cell-judgment');
      const hasUnreviewed = Array.from(cells).some(c => c.textContent.includes('미리뷰'));
      row.style.display = hasUnreviewed ? '' : 'none';
    }}
  }});
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate review summary HTML")
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="experiments/results/metacognitive/annotations",
    )
    parser.add_argument(
        "--batch-input",
        type=str,
        default="experiments/results/metacognitive/batch_transformations_0_120.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/metacognitive/review_summary.html",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    ann_dir = project_root / args.annotations_dir
    batch_path = project_root / args.batch_input
    output_path = project_root / args.output

    if not batch_path.exists():
        logger.error(f"배치 파일 없음: {batch_path}")
        sys.exit(1)

    problems = load_problems(batch_path)
    logger.info(f"문제 로드: {len(problems)}개")

    annotations = load_annotations(ann_dir)
    logger.info(f"총 annotation: {sum(1 for t in annotations.values() for a in t.values() if isinstance(a, dict) and a.get('judgment'))}건")

    html_content = generate_summary_html(problems, annotations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"요약 HTML 생성: {output_path}")


if __name__ == "__main__":
    main()
