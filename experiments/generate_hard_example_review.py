"""Hard Example Human Review HTML Generator.

validate_hard_example.py의 출력과 batch_transformations 원본을 결합하여
사람 검수용 HTML을 생성한다.

리뷰어가 확인할 것:
    1. original_answer ↔ new_answer 변화가 합리적인가? (delta_ratio)
    2. 치환된 값이 실제로 문제 풀이에서 쓰이는 값과 일치하는가?
    3. 변환된 context와 새 answer가 논리적으로 맞물리는가?
    4. hard example로 채택할지 결정 (adopt / revise / reject)

Usage:
    python experiments/generate_hard_example_review.py \\
        --validated experiments/results/metacognitive/hard_example_validated.json \\
        --transformations experiments/results/metacognitive/batch_transformations_0_238.json
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


def _esc(text: Any) -> str:
    """HTML escape with None-safety."""
    if text is None:
        return ""
    return html.escape(str(text))


def _index_problems(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build question_id → problem record index."""
    return {p.get("question_id", "?"): p for p in batch.get("problems", [])}


def _delta_ratio(original: str | None, new: str | None) -> str:
    """Compute original/new answer ratio for display."""
    try:
        o = float(original) if original else 0.0
        n = float(new) if new else 0.0
        if o == 0:
            return "?"
        return f"{n / o:.3f}×"
    except (TypeError, ValueError):
        return "?"


def render_review(
    validated: list[dict[str, Any]],
    problems_index: dict[str, dict[str, Any]],
) -> str:
    """Render a single-page review HTML."""
    cards: list[str] = []
    for i, v in enumerate(validated):
        qid = v.get("question_id", "?")
        ic_level = v.get("ic_level", "?")
        problem = problems_index.get(qid, {})
        tx = problem.get("transformations", {}).get(ic_level, {})

        original_answer = v.get("original_answer")
        new_answer = v.get("new_answer")
        ok = v.get("ok", False)
        ratio = _delta_ratio(original_answer, new_answer)
        badge_class = "ok" if ok else "fail"
        badge_label = "validated" if ok else (v.get("reason", "failed"))

        cards.append(f"""
<article class="card" data-idx="{i}" data-qid="{_esc(qid)}">
  <header>
    <h3>#{i} · {_esc(qid)}</h3>
    <span class="badge {_esc(ic_level)}">{_esc(ic_level)}</span>
    <span class="badge status {badge_class}">{_esc(badge_label)}</span>
  </header>

  <section class="answers">
    <div><label>원본 정답</label><code>{_esc(original_answer)}</code></div>
    <div><label>새 정답 (제안)</label><code>{_esc(new_answer)}</code></div>
    <div><label>Δ ratio</label><code>{_esc(ratio)}</code></div>
    <div><label>치환</label>
      <code>{_esc(v.get("matched_literal"))} → {_esc(v.get("proposed_new_value"))}</code>
      <small>({_esc(v.get("replacement_count"))}회 치환됨)</small>
    </div>
  </section>

  <section class="context-pair">
    <div>
      <label>원본 context</label>
      <pre>{_esc(problem.get("context_original", "")[:2000])}</pre>
    </div>
    <div>
      <label>변환된 content ({_esc(ic_level)})</label>
      <pre>{_esc(tx.get("transformed_content", "")[:2000])}</pre>
    </div>
  </section>

  <section>
    <label>LLM 설명 (removed_or_modified)</label>
    <pre>{_esc(tx.get("removed_or_modified", ""))}</pre>
  </section>

  <details>
    <summary>Python solution (원본)</summary>
    <pre class="code">{_esc(problem.get("python_solution", ""))}</pre>
  </details>

  <footer class="decision">
    <label>채택 결정</label>
    <select class="decision-select">
      <option value="">—</option>
      <option value="adopt">채택 (hard example 확정)</option>
      <option value="revise">수정 필요 (memo 참조)</option>
      <option value="reject">기각</option>
    </select>
    <textarea class="memo" placeholder="수정/기각 사유, 보정할 치환값 등"></textarea>
  </footer>
</article>
""")

    cards_html = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Hard Example Review</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; background: #f5f5f7; color: #111827; }}
  header.topbar {{ background: #111827; color: white; padding: 14px 24px; position: sticky; top: 0; z-index: 10; display: flex; gap: 16px; align-items: center; }}
  header.topbar h1 {{ margin: 0; font-size: 17px; }}
  header.topbar .stats {{ margin-left: auto; font-size: 13px; color: #9ca3af; }}
  header.topbar button {{ background: #3b82f6; color: white; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  header.topbar input {{ padding: 6px 10px; border: none; border-radius: 4px; font-size: 13px; }}
  main {{ padding: 24px; max-width: 1280px; margin: 0 auto; }}
  .card {{ background: white; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card header {{ display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }}
  .card h3 {{ margin: 0; font-size: 15px; flex: 1; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-family: monospace; color: white; background: #6b7280; }}
  .badge.IC-L1 {{ background: #22c55e; }}
  .badge.IC-L2 {{ background: #3b82f6; }}
  .badge.status.ok {{ background: #10b981; }}
  .badge.status.fail {{ background: #ef4444; }}
  .card section {{ margin: 10px 0; }}
  .card label {{ display: block; font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }}
  .answers {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #f9fafb; padding: 10px; border-radius: 6px; }}
  .answers code {{ font-size: 13px; font-weight: 600; color: #111827; }}
  .answers small {{ color: #6b7280; font-size: 11px; margin-left: 4px; }}
  .context-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  pre {{ background: #f9fafb; padding: 10px; border-radius: 4px; font-size: 12px; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; word-break: break-word; max-height: 240px; overflow-y: auto; }}
  pre.code {{ background: #1f2937; color: #e5e7eb; font-family: 'SF Mono', monospace; }}
  details summary {{ cursor: pointer; color: #3b82f6; font-size: 12px; }}
  footer.decision {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid #e5e7eb; display: grid; grid-template-columns: 200px 1fr; gap: 10px; align-items: start; }}
  select, textarea {{ padding: 7px 10px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 13px; font-family: inherit; width: 100%; box-sizing: border-box; }}
  textarea {{ min-height: 50px; resize: vertical; }}
  .card.adopt {{ border-left: 4px solid #10b981; }}
  .card.revise {{ border-left: 4px solid #f59e0b; }}
  .card.reject {{ border-left: 4px solid #ef4444; }}
</style>
</head>
<body>
<header class="topbar">
  <h1>Hard Example Review</h1>
  <input id="reviewer-name" type="text" placeholder="리뷰어 이름">
  <button onclick="exportJSON()">JSON 내보내기</button>
  <button onclick="importJSON()">불러오기</button>
  <input type="file" id="import-file" accept="application/json" style="display:none">
  <span class="stats" id="stats">0 / {len(validated)} 결정됨</span>
</header>
<main>{cards_html}</main>

<script>
const TOTAL = {len(validated)};

function storageKey() {{
  const n = document.getElementById('reviewer-name').value.trim() || 'default';
  return `hard_example_review_${{n}}`;
}}

function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(storageKey()) || '{{}}'); }} catch (e) {{ return {{}}; }}
}}

function saveState(state) {{
  localStorage.setItem(storageKey(), JSON.stringify(state));
  updateStats(state);
}}

function updateStats(state) {{
  const count = Object.values(state).filter(x => x && x.decision).length;
  document.getElementById('stats').textContent = `${{count}} / ${{TOTAL}} 결정됨`;
}}

function bindCards() {{
  const state = loadState();
  document.querySelectorAll('.card').forEach(card => {{
    const idx = card.dataset.idx;
    const sel = card.querySelector('.decision-select');
    const memo = card.querySelector('.memo');
    const saved = state[idx] || {{}};
    sel.value = saved.decision || '';
    memo.value = saved.memo || '';
    if (saved.decision) card.classList.add(saved.decision);

    const persist = () => {{
      const s = loadState();
      s[idx] = {{ decision: sel.value || null, memo: memo.value || '' }};
      saveState(s);
      card.classList.remove('adopt', 'revise', 'reject');
      if (sel.value) card.classList.add(sel.value);
    }};
    sel.addEventListener('change', persist);
    memo.addEventListener('input', persist);
  }});
  updateStats(state);
}}

function exportJSON() {{
  const name = document.getElementById('reviewer-name').value.trim() || 'unknown';
  const state = loadState();
  const output = {{
    reviewer: name,
    completed_at: new Date().toISOString(),
    decisions: state,
  }};
  const blob = new Blob([JSON.stringify(output, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `hard_example_review_${{name}}.json`;
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
        saveState(data.decisions || {{}});
        location.reload();
      }} catch (err) {{ alert('JSON 파싱 실패: ' + err.message); }}
    }};
    reader.readAsText(f);
  }};
  input.click();
}}

document.getElementById('reviewer-name').addEventListener('change', () => location.reload());
bindCards();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate hard example review HTML")
    parser.add_argument("--validated", type=Path, required=True, help="hard_example_validated.json")
    parser.add_argument("--transformations", type=Path, required=True, help="batch_transformations JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/hard_example_review.html"),
    )
    return parser.parse_args()


def main() -> int:
    """Generate review HTML."""
    args = parse_args()
    if not args.validated.exists() or not args.transformations.exists():
        logger.error("Required inputs missing")
        return 1

    val_data = json.loads(args.validated.read_text(encoding="utf-8"))
    batch = json.loads(args.transformations.read_text(encoding="utf-8"))
    validated = val_data.get("validated", [])
    problems_index = _index_problems(batch)
    logger.info(f"Rendering {len(validated)} validated records")

    html_text = render_review(validated, problems_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
