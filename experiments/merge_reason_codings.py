"""Merge two reason coding JSON files and compute Cohen's Kappa.

C1 of plan ``2026-04-21-001-feat-cikm-short-execution-plan.md``.

Inputs:
    Two JSON files produced by ``generate_reason_coding_workbook.py`` export:

    {
      "coder": "jkcost",
      "completed_at": "...",
      "records": [
        {"question_id": ..., "transformation_type": ...,
         "model": ..., "prompt_strategy": ..., "refusal_reason": ...,
         "category": "MISSING_REQUIRED_VALUE", "confidence": "high", "note": ""},
        ...
      ]
    }

Outputs:
    - ``reason_coding_merged_<range>.json`` — reconciled records (confidence-weighted,
      primary-wins on tie); unresolved conflicts retain both raters' decisions.
    - ``reason_coding_disagreement_<range>.html`` — reviewer pairs side by side
      for meeting triage.

Kappa interpretation (Landis & Koch 1977):
    ≥ 0.80 almost perfect | 0.60–0.79 substantial (project threshold) |
    0.40–0.59 moderate   | < 0.40 poor — categories need redesign

Usage:
    python experiments/merge_reason_codings.py \\
        --a experiments/results/metacognitive/annotations/reason_coding_jkcost_0_180.json \\
        --b experiments/results/metacognitive/annotations/reason_coding_wooik_0_180.json \\
        --primary jkcost \\
        --output-dir experiments/results/metacognitive/annotations
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CONFIDENCE_WEIGHT = {"high": 2, "medium": 1, "low": 0, None: 0, "": 0}


@dataclass
class MergedRecord:
    """A reconciled coding record."""

    key: tuple[str, str, str, str]
    question_id: str | None
    transformation_type: str | None
    model: str | None
    prompt_strategy: str | None
    refusal_reason: str | None
    category_a: str | None
    category_b: str | None
    confidence_a: str | None
    confidence_b: str | None
    note_a: str
    note_b: str
    agreed: bool
    merged_category: str | None
    resolution: str  # "agree" | "primary_wins" | "confidence_wins" | "unresolved"


def _load_coding(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load a coder export file, returning (coder_name, records)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    coder = data.get("coder", path.stem)
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid coding file shape: {path}")
    return coder, records


def _record_key(rec: dict[str, Any]) -> tuple[str, str, str, str]:
    """Unique key tuple for pairing two coders' records."""
    return (
        str(rec.get("question_id", "")),
        str(rec.get("transformation_type", "")),
        str(rec.get("model", "")),
        str(rec.get("prompt_strategy", "")),
    )


def compute_cohens_kappa(
    labels_a: list[str | None], labels_b: list[str | None]
) -> float:
    """Compute Cohen's Kappa without external dependencies.

    Only paired records where both sides assigned a non-null category count.
    Returns ``float('nan')`` if fewer than two distinct labels appear.
    """
    paired = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    if not paired:
        return float("nan")

    categories = sorted({a for a, _ in paired} | {b for _, b in paired})
    if len(categories) < 2:
        return float("nan")

    total = len(paired)
    po = sum(1 for a, b in paired if a == b) / total
    freq_a = Counter(a for a, _ in paired)
    freq_b = Counter(b for _, b in paired)
    pe = sum(
        (freq_a.get(c, 0) / total) * (freq_b.get(c, 0) / total)
        for c in categories
    )
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def reconcile_pair(
    rec_a: dict[str, Any] | None,
    rec_b: dict[str, Any] | None,
    primary: str,
    coder_a: str,
    coder_b: str,
) -> MergedRecord:
    """Merge a paired record from two coders using confidence + primary rules."""
    base = rec_a or rec_b or {}
    key = _record_key(base)

    category_a = (rec_a or {}).get("category")
    category_b = (rec_b or {}).get("category")
    conf_a = (rec_a or {}).get("confidence")
    conf_b = (rec_b or {}).get("confidence")

    agreed = category_a == category_b and category_a is not None

    if agreed:
        merged, resolution = category_a, "agree"
    elif category_a and not category_b:
        merged, resolution = category_a, "single_coder_a"
    elif category_b and not category_a:
        merged, resolution = category_b, "single_coder_b"
    else:
        w_a = CONFIDENCE_WEIGHT.get(conf_a, 0)
        w_b = CONFIDENCE_WEIGHT.get(conf_b, 0)
        if w_a > w_b:
            merged, resolution = category_a, "confidence_wins_a"
        elif w_b > w_a:
            merged, resolution = category_b, "confidence_wins_b"
        else:
            if primary == coder_a:
                merged, resolution = category_a, "primary_wins"
            elif primary == coder_b:
                merged, resolution = category_b, "primary_wins"
            else:
                merged, resolution = None, "unresolved"

    return MergedRecord(
        key=key,
        question_id=base.get("question_id"),
        transformation_type=base.get("transformation_type"),
        model=base.get("model"),
        prompt_strategy=base.get("prompt_strategy"),
        refusal_reason=base.get("refusal_reason"),
        category_a=category_a,
        category_b=category_b,
        confidence_a=conf_a,
        confidence_b=conf_b,
        note_a=(rec_a or {}).get("note", "") or "",
        note_b=(rec_b or {}).get("note", "") or "",
        agreed=agreed,
        merged_category=merged,
        resolution=resolution,
    )


def merge_codings(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    primary: str,
    coder_a: str,
    coder_b: str,
) -> list[MergedRecord]:
    """Pair records by identity tuple and reconcile."""
    map_a = {_record_key(r): r for r in records_a}
    map_b = {_record_key(r): r for r in records_b}
    all_keys = sorted(set(map_a) | set(map_b))
    return [
        reconcile_pair(map_a.get(k), map_b.get(k), primary, coder_a, coder_b)
        for k in all_keys
    ]


def _esc(value: Any) -> str:
    """HTML escape with None-safety."""
    return html.escape(str(value)) if value is not None else ""


def render_disagreement_html(
    merged: list[MergedRecord],
    coder_a: str,
    coder_b: str,
    kappa: float,
) -> str:
    """Build an HTML report surfacing only conflict rows for triage meetings."""
    conflicts = [m for m in merged if not m.agreed and m.category_a and m.category_b]

    rows: list[str] = []
    for m in conflicts:
        rows.append(f"""
<tr>
  <td><code>{_esc(m.question_id)}</code></td>
  <td><span class="chip">{_esc(m.transformation_type)}</span></td>
  <td class="reason">{_esc(m.refusal_reason)}</td>
  <td><strong>{_esc(m.category_a)}</strong><br><small>{_esc(m.confidence_a)}</small>
      <div class="note">{_esc(m.note_a)}</div></td>
  <td><strong>{_esc(m.category_b)}</strong><br><small>{_esc(m.confidence_b)}</small>
      <div class="note">{_esc(m.note_b)}</div></td>
  <td><span class="res">{_esc(m.resolution)}</span><br>
      <em>{_esc(m.merged_category)}</em></td>
</tr>
""")

    kappa_str = "N/A" if math.isnan(kappa) else f"{kappa:.3f}"
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Reason Coding Disagreements</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; background: #f5f5f7; color: #111827; }}
  header {{ background: #111827; color: white; padding: 16px 24px; }}
  header h1 {{ margin: 0; font-size: 18px; }}
  header .meta {{ color: #9ca3af; font-size: 13px; margin-top: 4px; }}
  main {{ padding: 24px; max-width: 1280px; margin: 0 auto; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid #e5e7eb; font-size: 13px; }}
  th {{ background: #f9fafb; color: #374151; font-size: 12px; text-transform: uppercase; }}
  .chip {{ display: inline-block; background: #6b7280; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; }}
  .reason {{ background: #fefce8; font-size: 12px; max-width: 320px; line-height: 1.5; }}
  .note {{ color: #6b7280; font-size: 11px; margin-top: 4px; font-style: italic; }}
  .res {{ color: #3b82f6; font-size: 11px; font-family: monospace; }}
  .summary {{ background: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; }}
  .kappa {{ font-size: 24px; font-weight: 700; color: #111827; }}
  .kappa.high {{ color: #10b981; }}
  .kappa.mid {{ color: #f59e0b; }}
  .kappa.low {{ color: #ef4444; }}
</style>
</head>
<body>
<header>
  <h1>Reason Coding Disagreements</h1>
  <div class="meta">{_esc(coder_a)} vs {_esc(coder_b)} · {len(conflicts)} / {len(merged)} rows in conflict</div>
</header>
<main>
  <div class="summary">
    Cohen's Kappa: <span class="kappa {'high' if kappa >= 0.6 else 'mid' if kappa >= 0.4 else 'low'}">{kappa_str}</span>
    <small>(Landis &amp; Koch 1977 — 기준선 ≥ 0.60 substantial agreement)</small>
  </div>
  <table>
    <thead>
      <tr><th>QID</th><th>Transform</th><th>Reason</th><th>{_esc(coder_a)}</th><th>{_esc(coder_b)}</th><th>Resolution</th></tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def build_output_record(m: MergedRecord) -> dict[str, Any]:
    """Convert MergedRecord → JSON-serializable dict."""
    return {
        "question_id": m.question_id,
        "transformation_type": m.transformation_type,
        "model": m.model,
        "prompt_strategy": m.prompt_strategy,
        "refusal_reason": m.refusal_reason,
        "category": m.merged_category,
        "category_a": m.category_a,
        "category_b": m.category_b,
        "confidence_a": m.confidence_a,
        "confidence_b": m.confidence_b,
        "note_a": m.note_a,
        "note_b": m.note_b,
        "agreed": m.agreed,
        "resolution": m.resolution,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge two reason coding JSONs + Cohen's Kappa")
    parser.add_argument("--a", type=Path, required=True, help="Coder A JSON")
    parser.add_argument("--b", type=Path, required=True, help="Coder B JSON")
    parser.add_argument(
        "--primary",
        type=str,
        default=None,
        help="Primary coder name (tie-break). Defaults to coder A.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/metacognitive/annotations"),
    )
    parser.add_argument("--tag", type=str, default="merged", help="Output filename tag")
    return parser.parse_args()


def main() -> int:
    """Run merge + Kappa + HTML."""
    args = parse_args()
    if not args.a.exists() or not args.b.exists():
        logger.error("Both coding files must exist")
        return 1

    coder_a, records_a = _load_coding(args.a)
    coder_b, records_b = _load_coding(args.b)
    primary = args.primary or coder_a
    logger.info(f"Coder A: {coder_a} ({len(records_a)} rows) | Coder B: {coder_b} ({len(records_b)} rows) | primary: {primary}")

    merged = merge_codings(records_a, records_b, primary, coder_a, coder_b)

    labels_a = [m.category_a for m in merged]
    labels_b = [m.category_b for m in merged]
    kappa = compute_cohens_kappa(labels_a, labels_b)

    agreement_rate = sum(1 for m in merged if m.agreed) / max(len(merged), 1)
    conflicts = sum(1 for m in merged if not m.agreed and m.category_a and m.category_b)
    unresolved = sum(1 for m in merged if m.resolution == "unresolved")

    logger.info(
        "Cohen's Kappa: N/A (insufficient variance)"
        if math.isnan(kappa)
        else f"Cohen's Kappa: {kappa:.3f}"
    )
    logger.info(f"  Agreement rate: {agreement_rate:.3f}")
    logger.info(f"  Conflicts: {conflicts}")
    logger.info(f"  Unresolved: {unresolved}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_json = args.output_dir / f"reason_coding_{args.tag}.json"
    html_path = args.output_dir / f"reason_coding_disagreement_{args.tag}.html"

    merged_json.write_text(
        json.dumps(
            {
                "coder_a": coder_a,
                "coder_b": coder_b,
                "primary": primary,
                "kappa": None if math.isnan(kappa) else kappa,
                "agreement_rate": agreement_rate,
                "total": len(merged),
                "conflicts": conflicts,
                "unresolved": unresolved,
                "records": [build_output_record(m) for m in merged],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote {merged_json}")

    html_path.write_text(
        render_disagreement_html(merged, coder_a, coder_b, kappa),
        encoding="utf-8",
    )
    logger.info(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
