"""Build v4 enrich bundle — consolidates CoT trace + re-tagged reasons + error tags.

v3에서 A1(POT reason)+F1(Memorization)만 있었던 오버레이를 v4에서는:
  - POT original / POT standard / POT metacognitive 응답 3개 (기존)
  - ⭐ CoT trace 응답 (신규, 4번째 카드)
  - A1 reason category — POT 기반 + CoT 기반 둘 다 태깅
  - F1 Memorization Score (기존 이전 세션 생성)
  - ⭐ Error Category (원본 오답만, 신규)

로 확장. 이 스크립트는 여러 JSON을 읽어 generate_human_review.py가 기대하는
enrich 포맷으로 묶어 출력한다.

Outputs:
  - reason_categories_v4_0_20.json  (POT+CoT 모두 태깅)
  - cot_responses_0_20.json  (CoT trace 응답을 변형별로 정리)
  - error_tags (이미 존재하는 것 재사용)
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
from auto_tag_reason_categories import extract_reason, tag_category  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _extract_cot_reason(raw_response: Any) -> str | None:
    """Parse the MISSING/CROSS-CHECK/INSUFFICIENT line from a CoT trace response."""
    if not raw_response:
        return None
    text = str(raw_response)
    # Preferred: final INSUFFICIENT_INFORMATION token
    m = re.search(r"INSUFFICIENT_INFORMATION\s*:\s*(.+?)(?:\"\s*\)?\s*\n|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip().strip("[]\"' ")[:300] or None
    # Fallback: MISSING section text
    m2 = re.search(r"#\s*MISSING\s*:\s*(.+?)(?:\n|$)", text)
    if m2:
        return m2.group(1).strip()[:300]
    # Fallback: CROSS-CHECK conflict
    m3 = re.search(r"#\s*CROSS-CHECK\s*:\s*(.+?)(?:\n|$)", text)
    if m3:
        return m3.group(1).strip()[:300]
    return None


def build_reason_index(
    pot_records: list[dict[str, Any]],
    cot_records_by_ttype: dict[str, list[dict[str, Any]]],
    qid_subset: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build qid → ttype → {pot_cat, cot_cat, cot_reason} index."""
    index: dict[str, dict[str, dict[str, Any]]] = {}

    # POT-based reason (from eval_metacognitive records)
    for rec in pot_records:
        qid = str(rec.get("question_id", ""))
        if qid not in qid_subset:
            continue
        if rec.get("response_type") != "refused":
            continue
        ttype = rec.get("transformation_type")
        reason = extract_reason(rec.get("raw_response"))
        if not reason:
            continue
        cat = tag_category(reason)
        index.setdefault(qid, {}).setdefault(ttype, {})
        index[qid][ttype]["reason"] = reason[:500]
        index[qid][ttype]["category"] = cat
        index[qid][ttype]["confidence"] = "heuristic"
        index[qid][ttype]["source"] = "POT"

    # CoT-based reason — override when CoT provides richer text
    for ttype, cot_recs in cot_records_by_ttype.items():
        for rec in cot_recs:
            qid = str(rec.get("question_id", ""))
            if qid not in qid_subset:
                continue
            reason = _extract_cot_reason(rec.get("raw_response", ""))
            if not reason:
                continue
            cat = tag_category(reason)
            index.setdefault(qid, {}).setdefault(ttype, {})
            # Only overwrite if CoT reason provides more substance
            existing = index[qid][ttype].get("reason", "")
            if len(reason) > len(existing):
                index[qid][ttype]["reason"] = reason
                index[qid][ttype]["category"] = cat
                index[qid][ttype]["confidence"] = "heuristic_cot"
                index[qid][ttype]["source"] = "CoT"

    return index


def build_cot_responses(
    cot_batch_records: list[dict[str, Any]],
    cot_original_records: list[dict[str, Any]],
    qid_subset: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build qid → ttype → CoT response payload (for injection into problem.transformations)."""
    cot_index: dict[str, dict[str, dict[str, Any]]] = {}

    for rec in cot_batch_records:
        qid = str(rec.get("question_id", ""))
        ttype = rec.get("transformation_type")
        if qid not in qid_subset or not ttype:
            continue
        cot_index.setdefault(qid, {})[ttype] = {
            "cot_trace_response": rec.get("raw_response", ""),
            "cot_trace_predicted": rec.get("predicted_answer", ""),
            "cot_trace_is_correct": rec.get("is_correct", False),
            "cot_trace_response_type": rec.get("response_type", ""),
            "cot_trace_case_type": rec.get("case_type", 0),
            "cot_trace_execution_error": rec.get("execution_error", ""),
        }

    # Original context is not tied to a transformation — store under synthetic "original"
    for rec in cot_original_records:
        qid = str(rec.get("question_id", ""))
        if qid not in qid_subset:
            continue
        cot_index.setdefault(qid, {})["_original"] = {
            "cot_trace_response": rec.get("raw_response", ""),
            "cot_trace_predicted": rec.get("predicted_answer", ""),
            "cot_trace_is_correct": rec.get("is_correct", False),
            "cot_trace_response_type": rec.get("response_type", ""),
        }
    return cot_index


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build v4 enrich bundle")
    parser.add_argument(
        "--pot-meta",
        type=Path,
        default=Path("experiments/results/metacognitive/eval_metacognitive.json"),
        help="POT-era metacognitive eval (for POT-based reason tagging)",
    )
    parser.add_argument(
        "--cot-batch",
        type=Path,
        default=Path("experiments/results/metacognitive/batch_evaluation_0_20.json"),
        help="CoT trace on transformations (139 records)",
    )
    parser.add_argument(
        "--cot-original",
        type=Path,
        default=Path("experiments/results/metacognitive/eval_original_ic_cot_trace_0_20.json"),
        help="CoT trace on original context (20 records)",
    )
    parser.add_argument(
        "--output-reasons",
        type=Path,
        default=Path("experiments/results/metacognitive/reason_categories_v4_0_20.json"),
    )
    parser.add_argument(
        "--output-cot",
        type=Path,
        default=Path("experiments/results/metacognitive/cot_responses_0_20.json"),
    )
    parser.add_argument(
        "--qid-range",
        type=str,
        default="test-2000:test-2019",
    )
    return parser.parse_args()


def _resolve_qid_subset(spec: str) -> set[str]:
    """Expand 'test-2000:test-2019' to set."""
    start_s, end_s = spec.split(":")
    start_n = int(start_s.rsplit("-", 1)[-1])
    end_n = int(end_s.rsplit("-", 1)[-1])
    prefix = start_s.rsplit("-", 1)[0]
    return {f"{prefix}-{i}" for i in range(start_n, end_n + 1)}


def main() -> int:
    """Build bundle."""
    args = parse_args()
    qid_subset = _resolve_qid_subset(args.qid_range)

    pot_records = load_eval_records(args.pot_meta) if args.pot_meta.exists() else []
    cot_batch_records = (
        load_eval_records(args.cot_batch) if args.cot_batch.exists() else []
    )
    cot_original_records = (
        load_eval_records(args.cot_original) if args.cot_original.exists() else []
    )

    # Group CoT batch by transformation_type (per qid → ttype → records)
    cot_by_ttype: dict[str, list[dict[str, Any]]] = {}
    for rec in cot_batch_records:
        ttype = rec.get("transformation_type")
        if ttype:
            cot_by_ttype.setdefault(ttype, []).append(rec)

    reason_index = build_reason_index(pot_records, cot_by_ttype, qid_subset)
    cot_index = build_cot_responses(cot_batch_records, cot_original_records, qid_subset)

    from collections import Counter
    cat_counter: Counter = Counter()
    for per_type in reason_index.values():
        for meta in per_type.values():
            cat_counter[meta.get("category", "?")] += 1
    logger.info(f"Reason category distribution (POT+CoT combined): {dict(cat_counter)}")

    cot_total = sum(len(v) for v in cot_index.values())
    logger.info(f"CoT responses indexed: {cot_total} across {len(cot_index)} qids")

    args.output_reasons.parent.mkdir(parents=True, exist_ok=True)
    args.output_reasons.write_text(
        json.dumps(reason_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_cot.write_text(
        json.dumps(cot_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output_reasons}")
    logger.info(f"Wrote {args.output_cot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
