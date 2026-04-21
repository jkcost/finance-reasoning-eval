"""Reason Coding Workbook HTML Generator.

판정 이유(refusal_reason) 수동 카테고리 코딩용 인터랙티브 HTML.

입력:
    - run_batch_evaluation.py 결과 JSON (refusal_reason 필드 포함)

기능:
    - 리뷰어별 reason 표시 (URL 파라미터 ?coder=name&start=0&end=50)
    - 카테고리 드롭다운 선택 (v0 6개 + 확신도)
    - 메모 textarea
    - JSON 내보내기 → experiments/results/metacognitive/annotations/reason_coding_<coder>.json

Usage:
    python experiments/generate_reason_coding_workbook.py \\
        --input experiments/results/metacognitive/evaluation_results_*.json
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CATEGORIES: list[dict[str, str]] = [
    {
        "code": "MISSING_REQUIRED_VALUE",
        "label": "필수 수치 누락 (명시적)",
        "desc": "계산 필수 값이 명시적으로 비어있다고 인지 ([DATA MISSING] 등).",
        "color": "#3b82f6",
    },
    {
        "code": "MISSING_SILENT",
        "label": "무표지 부재 추론",
        "desc": "마커 없이 데이터가 없다고 암묵 추론.",
        "color": "#f59e0b",
    },
    {
        "code": "CONFLICTING_VALUES",
        "label": "값 충돌",
        "desc": "두 개 이상의 모순된 수치가 공존한다고 지적.",
        "color": "#ef4444",
    },
    {
        "code": "UNIT_AMBIGUITY",
        "label": "단위/규모 불일치",
        "desc": "단위(백만/십억) 또는 통화(yen/USD) 모순.",
        "color": "#8b5cf6",
    },
    {
        "code": "TEMPORAL_MISMATCH",
        "label": "기간·시점 불일치",
        "desc": "분기/연간/시점 불일치 (분기합 ≠ 연간 등).",
        "color": "#a855f7",
    },
    {
        "code": "UNDERSPECIFIED",
        "label": "일반적 정보 부족 (fallback)",
        "desc": "위 5개 어디에도 명확히 속하지 않는 일반 거부.",
        "color": "#6b7280",
    },
    {
        "code": "UNCATEGORIZABLE",
        "label": "분류 불가 (재검토 필요)",
        "desc": "reason이 애매하거나 v0 카테고리로 안 됨 → 회의에서 논의.",
        "color": "#111827",
    },
]

CONFIDENCE_LEVELS = [
    {"value": "high", "label": "확신 높음"},
    {"value": "medium", "label": "중간"},
    {"value": "low", "label": "낮음 (애매)"},
]


def _esc(text: Any) -> str:
    """HTML escape with None-safety."""
    if text is None:
        return ""
    return html.escape(str(text))


def load_evaluation_results(input_paths: list[Path]) -> list[dict[str, Any]]:
    """Load and merge evaluation result JSON files.

    Filters to records with non-empty refusal_reason or response_type == 'refused'.
    """
    records: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.exists():
            logger.warning(f"Input not found, skipping: {path}")
            continue
        logger.info(f"Loading {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            logger.warning(f"Unexpected JSON shape in {path}")
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            reason = item.get("refusal_reason")
            response_type = item.get("response_type")
            if reason or response_type == "refused":
                records.append(item)
    logger.info(f"Loaded {len(records)} coding-eligible records")
    return records


def render_workbook(records: list[dict[str, Any]]) -> str:
    """Render coding workbook HTML."""
    cats_json = json.dumps(CATEGORIES, ensure_ascii=False)
    confs_json = json.dumps(CONFIDENCE_LEVELS, ensure_ascii=False)
    records_json = json.dumps(records, ensure_ascii=False)

    category_options = "\n".join(
        f'<option value="{c["code"]}">{c["label"]}</option>' for c in CATEGORIES
    )
    confidence_options = "\n".join(
        f'<option value="{c["value"]}">{c["label"]}</option>' for c in CONFIDENCE_LEVELS
    )
    category_legend = "\n".join(
        f'<li><span class="chip" style="background:{c["color"]}">{_esc(c["code"])}</span> '
        f'<strong>{_esc(c["label"])}</strong> — {_esc(c["desc"])}</li>'
        for c in CATEGORIES
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Reason Coding Workbook</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; background: #f5f5f7; color: #111827; }}
  header {{ background: #111827; color: white; padding: 16px 24px; position: sticky; top: 0; z-index: 10; }}
  header h1 {{ margin: 0 0 4px; font-size: 18px; }}
  header .meta {{ font-size: 13px; color: #9ca3af; }}
  main {{ padding: 24px; max-width: 1200px; margin: 0 auto; }}
  .legend {{ background: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; }}
  .legend ul {{ margin: 8px 0 0; padding-left: 0; list-style: none; }}
  .legend li {{ margin: 6px 0; }}
  .chip {{ display: inline-block; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; }}
  .controls {{ background: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls button {{ background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .controls button:hover {{ background: #2563eb; }}
  .controls .stats {{ margin-left: auto; font-size: 13px; color: #6b7280; }}
  .record {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; border-left: 4px solid #e5e7eb; }}
  .record.coded {{ border-left-color: #10b981; }}
  .record h3 {{ margin: 0 0 8px; font-size: 14px; color: #374151; }}
  .record .meta-row {{ display: flex; gap: 12px; font-size: 12px; color: #6b7280; margin-bottom: 8px; flex-wrap: wrap; }}
  .record .meta-row span {{ background: #f3f4f6; padding: 2px 8px; border-radius: 4px; }}
  .record .reason {{ background: #fefce8; padding: 12px; border-radius: 6px; font-size: 13px; line-height: 1.6; margin: 8px 0; border-left: 3px solid #eab308; }}
  .record .coding-row {{ display: grid; grid-template-columns: 2fr 1fr 3fr; gap: 8px; margin-top: 8px; }}
  .record select, .record textarea {{ width: 100%; padding: 6px 8px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 13px; font-family: inherit; box-sizing: border-box; }}
  .record textarea {{ resize: vertical; min-height: 36px; }}
  details summary {{ cursor: pointer; color: #3b82f6; font-size: 12px; margin-top: 6px; }}
  details pre {{ background: #f9fafb; padding: 8px; border-radius: 4px; font-size: 11px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>
<header>
  <h1>Reason Coding Workbook v1</h1>
  <div class="meta">
    URL 파라미터: <code>?coder=name&amp;start=0&amp;end=50</code> | 카테고리 v0 (6개 + UNCATEGORIZABLE)
  </div>
</header>
<main>
  <div class="legend">
    <strong>카테고리 v0</strong> — 각 reason에 정확히 하나를 할당. 애매하면 <code>UNCATEGORIZABLE</code> + 메모.
    <ul>{category_legend}</ul>
  </div>
  <div class="controls">
    <label>코더: <input id="coder-name" type="text" placeholder="이름" style="padding:6px 8px; border:1px solid #d1d5db; border-radius:4px;"></label>
    <button onclick="exportJSON()">JSON 내보내기</button>
    <button onclick="importJSON()">불러오기</button>
    <input type="file" id="import-file" accept="application/json" style="display:none">
    <span class="stats" id="stats">0 / 0 코딩됨</span>
  </div>
  <div id="records-container"></div>
</main>

<template id="record-template">
  <div class="record" data-idx="">
    <h3></h3>
    <div class="meta-row">
      <span class="transformation-type"></span>
      <span class="model"></span>
      <span class="strategy"></span>
      <span class="response-type"></span>
    </div>
    <div class="reason"></div>
    <div class="coding-row">
      <select class="category-select">
        <option value="">— 카테고리 선택 —</option>
        {category_options}
      </select>
      <select class="confidence-select">
        <option value="">확신도</option>
        {confidence_options}
      </select>
      <textarea class="note" placeholder="메모 (선택)"></textarea>
    </div>
    <details><summary>원본 응답 보기</summary><pre class="raw-response"></pre></details>
  </div>
</template>

<script>
const CATEGORIES = {cats_json};
const CONFIDENCE_LEVELS = {confs_json};
const RECORDS = {records_json};

const params = new URLSearchParams(location.search);
const coder = params.get('coder') || '';
const start = parseInt(params.get('start') || '0', 10);
const end = parseInt(params.get('end') || String(RECORDS.length), 10);
const subset = RECORDS.slice(start, end);

document.getElementById('coder-name').value = coder;

function storageKey() {{
  const c = document.getElementById('coder-name').value.trim() || 'default';
  return `reason_coding_${{c}}_${{start}}_${{end}}`;
}}

function loadState() {{
  try {{
    return JSON.parse(localStorage.getItem(storageKey()) || '{{}}');
  }} catch (e) {{ return {{}}; }}
}}

function saveState(state) {{
  localStorage.setItem(storageKey(), JSON.stringify(state));
  updateStats(state);
}}

function updateStats(state) {{
  const coded = Object.values(state).filter(x => x && x.category).length;
  document.getElementById('stats').textContent = `${{coded}} / ${{subset.length}} 코딩됨`;
}}

function renderRecords() {{
  const container = document.getElementById('records-container');
  const tpl = document.getElementById('record-template');
  const state = loadState();

  subset.forEach((rec, i) => {{
    const node = tpl.content.cloneNode(true);
    const div = node.querySelector('.record');
    div.dataset.idx = String(i);
    node.querySelector('h3').textContent = `#${{start + i}} · ${{rec.question_id || '?'}}`;
    node.querySelector('.transformation-type').textContent = rec.transformation_type || '?';
    node.querySelector('.model').textContent = rec.model || '?';
    node.querySelector('.strategy').textContent = rec.prompt_strategy || '?';
    node.querySelector('.response-type').textContent = rec.response_type || '?';
    node.querySelector('.reason').textContent = rec.refusal_reason || '(reason 없음 — 거부로 분류되었으나 사유 미제공)';
    node.querySelector('.raw-response').textContent = rec.raw_response || '';

    const catSel = node.querySelector('.category-select');
    const confSel = node.querySelector('.confidence-select');
    const noteArea = node.querySelector('.note');

    const saved = state[i] || {{}};
    catSel.value = saved.category || '';
    confSel.value = saved.confidence || '';
    noteArea.value = saved.note || '';
    if (saved.category) div.classList.add('coded');

    const persist = () => {{
      const s = loadState();
      s[i] = {{
        category: catSel.value || null,
        confidence: confSel.value || null,
        note: noteArea.value || '',
      }};
      saveState(s);
      div.classList.toggle('coded', !!catSel.value);
    }};
    catSel.addEventListener('change', persist);
    confSel.addEventListener('change', persist);
    noteArea.addEventListener('input', persist);

    container.appendChild(node);
  }});
  updateStats(state);
}}

function exportJSON() {{
  const coderName = document.getElementById('coder-name').value.trim() || 'unknown';
  const state = loadState();
  const records = subset.map((rec, i) => {{
    const s = state[i] || {{}};
    return {{
      question_id: rec.question_id,
      transformation_type: rec.transformation_type,
      model: rec.model,
      prompt_strategy: rec.prompt_strategy,
      refusal_reason: rec.refusal_reason,
      category: s.category || null,
      confidence: s.confidence || null,
      note: s.note || '',
    }};
  }});
  const output = {{
    coder: coderName,
    range: {{ start, end }},
    completed_at: new Date().toISOString(),
    records,
  }};
  const blob = new Blob([JSON.stringify(output, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `reason_coding_${{coderName}}_${{start}}_${{end}}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}}

function importJSON() {{
  const input = document.getElementById('import-file');
  input.onchange = (e) => {{
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {{
      try {{
        const data = JSON.parse(ev.target.result);
        const state = {{}};
        (data.records || []).forEach((r, i) => {{
          state[i] = {{
            category: r.category || null,
            confidence: r.confidence || null,
            note: r.note || '',
          }};
        }});
        saveState(state);
        location.reload();
      }} catch (err) {{
        alert('JSON 파싱 실패: ' + err.message);
      }}
    }};
    reader.readAsText(f);
  }};
  input.click();
}}

document.getElementById('coder-name').addEventListener('change', () => {{
  location.reload();
}});

renderRecords();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate reason coding workbook HTML")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more evaluation result JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/reason_coding_workbook.html"),
        help="Output HTML path",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the workbook HTML."""
    args = parse_args()
    records = load_evaluation_results(args.input)
    if not records:
        logger.error("No coding-eligible records found (need refusal_reason or refused response_type).")
        return 1

    html_text = render_workbook(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    logger.info(f"Wrote workbook: {args.output} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
