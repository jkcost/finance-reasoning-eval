"""Heuristic reason category tagger for POT-era evaluation results.

v3 HTML 입력용. 각 평가 레코드의 POT reason 텍스트를 C2 sanity check에서
검증된 휴리스틱으로 자동 카테고리 태깅. Coder가 나중에 editable UI에서
수정 가능하므로 신뢰도는 "auto-preview" 수준.

Output JSON shape (enrich input 용):
    {
      "test-2009": {
        "EA-partial": {
          "reason": "shares_outstanding is missing",
          "category": "MISSING_REQUIRED_VALUE",
          "confidence": "heuristic"
        },
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_io import load_eval_records  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


INLINE_REASON_PATTERN = re.compile(
    r"INSUFFICIENT_INFORMATION\s*:\s*(.+?)(?:\n\n|\Z)", re.DOTALL
)


def extract_reason(raw_response: Any) -> str | None:
    """Parse INSUFFICIENT_INFORMATION: <reason> from a response payload."""
    if raw_response in (None, "", [], {}):
        return None
    text = raw_response if isinstance(raw_response, str) else str(raw_response)
    match = INLINE_REASON_PATTERN.search(text)
    if not match:
        return None
    reason = match.group(1).strip().strip("[]\"' ")
    return reason or None


def tag_category(reason: str) -> str:
    """Map a reason string to one of the v0 categories via keyword heuristics.

    Order matters: the first matching pattern wins. Conservative default
    ``UNDERSPECIFIED`` catches anything that does not match a specific pattern.
    Human reviewers override this in the workbook.
    """
    r = reason.lower()

    # Conflict signals override everything (two-value phrasing)
    if re.search(r"(contradict|conflict|however|but\s+the|two\s+values|different\s+values|"
                 r"inconsisten|doesn'?t\s+match|does\s+not\s+match|discrepan)", r):
        return "CONFLICTING_VALUES"

    # Unit / currency conflict (requires both unit keyword AND conflict signal)
    if re.search(r"(million|billion|thousand|basis\s+point|per\s+share|yen|dollar|usd|eur|%)", r) \
            and re.search(r"(different|mismatch|conflict|inconsisten)", r):
        return "UNIT_AMBIGUITY"

    # Temporal mismatch — requires explicit period + inconsistency phrasing
    if re.search(r"(quarter|annual|year|fiscal|period|q[1-4])", r) \
            and re.search(r"(sum|total|mismatch|inconsisten|doesn'?t\s+match)", r):
        return "TEMPORAL_MISMATCH"

    # Missing-data phrasing (most common pattern in POT reasons)
    if re.search(r"(not\s+provided|not\s+given|not\s+specified|not\s+available|"
                 r"\bmissing\b|\bn/a\b|not\s+mention|\bunknown\b|not\s+clear|\babsent\b)", r):
        return "MISSING_REQUIRED_VALUE"

    # Fallback
    return "UNDERSPECIFIED"


def build_reason_index(
    eval_records: list[dict[str, Any]],
    qid_subset: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build qid → transformation_type → reason metadata mapping."""
    index: dict[str, dict[str, dict[str, Any]]] = {}
    stats = {"with_reason": 0, "no_reason": 0, "not_refused": 0}

    for rec in eval_records:
        qid = str(rec.get("question_id", ""))
        if qid_subset is not None and qid not in qid_subset:
            continue

        ttype = rec.get("transformation_type")
        response_type = rec.get("response_type")
        if not ttype:
            continue

        if response_type != "refused":
            stats["not_refused"] += 1
            continue

        reason = extract_reason(rec.get("raw_response"))
        if not reason:
            stats["no_reason"] += 1
            continue

        category = tag_category(reason)
        stats["with_reason"] += 1

        index.setdefault(qid, {})[ttype] = {
            "reason": reason[:500],
            "category": category,
            "confidence": "heuristic",
            "source_model": rec.get("model"),
            "source_strategy": rec.get("prompt_strategy"),
        }

    logger.info(f"Indexed reasons: {stats}")
    return index


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Auto-tag reason categories for enrich input")
    parser.add_argument(
        "--eval-results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more evaluation result JSON files",
    )
    parser.add_argument(
        "--qid-range",
        type=str,
        default=None,
        help="Filter to qid range like 'test-2000:test-2019' (inclusive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (enrich input)",
    )
    return parser.parse_args()


def _resolve_qid_subset(spec: str | None) -> set[str] | None:
    """Expand 'test-2000:test-2019' to set of strings."""
    if not spec:
        return None
    try:
        start_s, end_s = spec.split(":")
        start_n = int(start_s.rsplit("-", 1)[-1])
        end_n = int(end_s.rsplit("-", 1)[-1])
        prefix = start_s.rsplit("-", 1)[0]
        return {f"{prefix}-{i}" for i in range(start_n, end_n + 1)}
    except (ValueError, IndexError):
        logger.error(f"Invalid --qid-range: {spec}")
        return None


def main() -> int:
    """Run auto-tagger."""
    args = parse_args()
    qid_subset = _resolve_qid_subset(args.qid_range)
    if qid_subset:
        logger.info(f"Filter to {len(qid_subset)} qids")

    all_records: list[dict[str, Any]] = []
    for path in args.eval_results:
        if not path.exists():
            logger.warning(f"skip {path}")
            continue
        all_records.extend(load_eval_records(path))

    index = build_reason_index(all_records, qid_subset)

    # Quick category distribution
    from collections import Counter
    cat_counter: Counter = Counter()
    for per_type in index.values():
        for meta in per_type.values():
            cat_counter[meta["category"]] += 1
    logger.info("Category distribution:")
    for c, n in cat_counter.most_common():
        logger.info(f"  {c}: {n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
