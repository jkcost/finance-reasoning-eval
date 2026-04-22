"""Extract inline INSUFFICIENT_INFORMATION reasons from existing evaluation results.

구 프롬프트 시대 평가 결과(raw_response에 ``INSUFFICIENT_INFORMATION: <reason>`` 패턴)를
후처리하여 refusal_reason 필드가 채워진 새 JSON을 생성한다.
generate_reason_coding_workbook.py와 호환되는 출력 포맷.

R4 (solvable 모집단) 필터 옵션:
    --original-context <path> 로 원본 컨텍스트 평가 결과를 받으면
    {qid | is_correct == True} 집합 계산 후 해당 qid에 속한 reason만 추출.

Usage:
    # eval_metacognitive.json에서 reason 추출 (전체)
    python experiments/extract_existing_reasons.py \\
        --input experiments/results/metacognitive/eval_metacognitive.json \\
        --output experiments/results/metacognitive/reasons_existing.json

    # R4 필터 적용 — 원본에서 맞춘 문제의 reason만
    python experiments/extract_existing_reasons.py \\
        --input experiments/results/metacognitive/eval_metacognitive.json \\
        --original-context experiments/results/metacognitive/eval_original_context.json \\
        --output experiments/results/metacognitive/reasons_solvable.json
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

from _eval_io import load_eval_records, qid_in_solvable, solvable_qids  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


INLINE_REASON_PATTERN = re.compile(
    r"INSUFFICIENT_INFORMATION\s*:\s*(.+?)(?:\n\n|\Z)", re.DOTALL
)


def _extract_inline_reason(raw_response: Any) -> str | None:
    """Parse INSUFFICIENT_INFORMATION: <reason> from a raw response payload.

    Tolerates dict/list/None raw_response by only operating on a stringified form;
    returns None when nothing reasonable matches so the caller can skip the record.
    """
    if raw_response in (None, "", [], {}):
        return None
    text = raw_response if isinstance(raw_response, str) else str(raw_response)
    match = INLINE_REASON_PATTERN.search(text)
    if not match:
        return None
    reason = match.group(1).strip()
    reason = reason.strip("[]\"' ")
    return reason or None


def extract_reasons(
    eval_records: list[dict[str, Any]],
    solvable_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build workbook-compatible records with refusal_reason populated.

    Records are kept when:
      1. response_type == "refused"   (always try to parse reason, even if pattern absent)
         OR raw_response text contains the INSUFFICIENT_INFORMATION marker
      2. an inline reason is successfully parsed (non-empty after stripping)
      3. question_id ∈ solvable_filter (when provided; str-normalized comparison)
    """
    output: list[dict[str, Any]] = []
    skipped_no_reason = 0
    skipped_not_solvable = 0

    for rec in eval_records:
        raw_response = rec.get("raw_response", "")
        response_type = rec.get("response_type")
        raw_text = raw_response if isinstance(raw_response, str) else str(raw_response or "")

        if response_type != "refused" and "INSUFFICIENT_INFORMATION" not in raw_text:
            continue

        reason = _extract_inline_reason(raw_response)
        if not reason:
            skipped_no_reason += 1
            continue

        qid = rec.get("question_id")
        if not qid_in_solvable(qid, solvable_filter):
            skipped_not_solvable += 1
            continue

        output.append({
            "question_id": qid,
            "transformation_type": rec.get("transformation_type"),
            "model": rec.get("model"),
            "prompt_strategy": rec.get("prompt_strategy"),
            "response_type": response_type,
            "refusal_reason": reason,
            "raw_response": raw_text,
            "is_correct": rec.get("is_correct", False),
        })

    logger.info(f"Extracted: {len(output)} reasons")
    if skipped_no_reason:
        logger.info(f"  skipped (no inline reason): {skipped_no_reason}")
    if skipped_not_solvable:
        logger.info(f"  skipped (not in solvable set): {skipped_not_solvable}")
    return output


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Build a coverage summary by model/strategy/transformation."""
    from collections import Counter

    summary: dict[str, dict[str, int]] = {}
    for field in ("model", "prompt_strategy", "transformation_type"):
        counter: Counter = Counter(r.get(field) for r in records)
        summary[field] = dict(counter)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract inline reasons from legacy eval results")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more evaluation result JSON files",
    )
    parser.add_argument(
        "--original-context",
        type=Path,
        default=None,
        help="Optional original-context eval result for R4 solvable filter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (workbook-compatible)",
    )
    return parser.parse_args()


def main() -> int:
    """Extract reasons and write workbook-compatible JSON."""
    args = parse_args()

    solvable: set[str] | None = None
    if args.original_context:
        if not args.original_context.exists():
            logger.error(f"Original-context file not found: {args.original_context}")
            return 1
        solvable = solvable_qids(args.original_context)
        logger.info(f"Solvable QIDs (original correct): {len(solvable or set())}")

    all_records: list[dict[str, Any]] = []
    for path in args.input:
        if not path.exists():
            logger.warning(f"Input missing, skipping: {path}")
            continue
        logger.info(f"Loading {path}")
        eval_records = load_eval_records(path)
        logger.info(f"  {len(eval_records)} eval records")
        extracted = extract_reasons(eval_records, solvable_filter=solvable)
        all_records.extend(extracted)

    if not all_records:
        logger.error("No reasons extracted.")
        return 1

    summary = summarize(all_records)
    logger.info("Coverage summary:")
    for field, counter in summary.items():
        logger.info(f"  {field}: {counter}")

    output = {
        "metadata": {
            "inputs": [str(p) for p in args.input],
            "original_context": str(args.original_context) if args.original_context else None,
            "solvable_only": solvable is not None,
            "total_reasons": len(all_records),
            "summary": summary,
        },
        "results": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
