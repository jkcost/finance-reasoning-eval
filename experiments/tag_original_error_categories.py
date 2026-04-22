"""Tag Error Categories for original context (POT vs CoT) — B1 확장.

Input:
    - eval_original_context.json  (POT + "original" strategy, 기존)
    - eval_original_ic_cot_trace_0_20.json  (CoT trace, 신규)

Output:
    - experiments/results/metacognitive/original_error_tags_0_20.json
      {
        "test-2001": {
          "pot_is_correct": false,
          "cot_response_type": "refused",
          "cot_refusal_reason": "...",
          "cot_trace_snippet": "...",
          "error_category": "FORMULA_ERROR",
          "failure_stage": "COMPUTE",
          "note": "POT applied wrong formula, CoT correctly refused (over-refusal trade-off)"
        },
        ...
      }

휴리스틱 분류 — 리뷰어 수정 가능한 preview 수준.
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


# error_taxonomy의 13개 중 본 실험 맥락에서 의미 있는 7개만 활용
ERROR_CATEGORIES = [
    "FORMULA_ERROR",       # 공식 자체 잘못 적용
    "EXTRACTION_ERROR",    # context에서 값 추출 실수
    "CALCULATION_ERROR",   # 산술 실수
    "MISUNDERSTANDING",    # 문제를 잘못 이해
    "UNIT_ERROR",          # 단위 혼동
    "OVER_REFUSAL",        # CoT가 정상 문제도 거부 (신규 — 본 실험 finding)
    "HALLUCINATION",       # context에 없는 값 사용
    "OK",                  # 오답 아님
]

FAILURE_STAGES = ["IDENTIFY", "LOCATE", "COMPUTE", "NONE"]


def _extract_trace_sections(cot_raw: str) -> dict[str, str]:
    """Parse IDENTIFY / LOCATE / MISSING / CROSS-CHECK / COMPUTE sections from CoT trace."""
    sections: dict[str, str] = {}
    for label in ["IDENTIFY", "LOCATE", "MISSING", "CROSS-CHECK", "COMPUTE"]:
        matches = re.findall(
            rf"#\s*{label}[:\s]+(.+?)(?=#\s*(?:IDENTIFY|LOCATE|MISSING|CROSS-CHECK|COMPUTE)|return|$)",
            cot_raw,
            re.DOTALL | re.IGNORECASE,
        )
        if matches:
            sections[label] = " | ".join(m.strip()[:150] for m in matches)
    return sections


def _classify_error(
    pot_correct: bool,
    cot_response_type: str,
    cot_refusal_reason: str | None,
    cot_trace_sections: dict[str, str],
    pot_predicted: str | None,
    ground_truth: str,
) -> tuple[str, str, str]:
    """Return (category, failure_stage, note) heuristically.

    Logic (priority order):
      1. POT correct → OK
      2. POT incorrect + CoT refused with "formula/method not found" → OVER_REFUSAL
         (frequent pattern: CoT refuses because it wants the method spelled out in
         context, but the model is expected to apply it from training knowledge)
      3. POT incorrect + CoT refused with missing required data → true MISSING but
         classified as MISUNDERSTANDING if POT still ventured a guess
      4. POT incorrect + CoT gave an answer but wrong → compute/formula/extraction
    """
    if pot_correct:
        return "OK", "NONE", "POT answered correctly; CoT behavior for reference only"

    reason_lower = (cot_refusal_reason or "").lower()
    trace_lower = " ".join(cot_trace_sections.values()).lower()

    # OVER_REFUSAL signature: CoT refuses but POT at least tried
    if cot_response_type == "refused":
        if any(kw in reason_lower for kw in [
            "method not found", "formula not", "calculation method", "not provided",
            "not specified in the context", "cannot determine"
        ]):
            if any(kw in trace_lower for kw in [
                "loan_amount", "payment", "revenue", "equity", "cost_of_"
            ]) and "identify" in trace_lower and "locate" in trace_lower:
                return (
                    "OVER_REFUSAL",
                    "COMPUTE",
                    "CoT located values but refused at COMPUTE — wanted method spelled out",
                )
        return (
            "MISUNDERSTANDING",
            "IDENTIFY",
            f"Both POT and CoT failed. CoT cites: {(cot_refusal_reason or '')[:120]}",
        )

    # POT wrong + CoT confident (but still likely wrong)
    if cot_response_type == "confident":
        # Inspect trace for LOCATE issues
        if "missing" in trace_lower or "not found" in trace_lower:
            return ("EXTRACTION_ERROR", "LOCATE", "Both failed; CoT trace admits missing LOCATE")
        return ("FORMULA_ERROR", "COMPUTE", "Both failed; likely wrong formula/computation path")

    return ("MISUNDERSTANDING", "IDENTIFY", "Error type uncertain")


def build_tags(
    pot_records: list[dict[str, Any]],
    cot_records: list[dict[str, Any]],
    qid_subset: set[str],
) -> dict[str, dict[str, Any]]:
    """Cross-reference POT + CoT records per qid and emit error tags."""
    pot_by_qid = {r["question_id"]: r for r in pot_records if r.get("question_id") in qid_subset}
    cot_by_qid = {r["question_id"]: r for r in cot_records if r.get("question_id") in qid_subset}

    tags: dict[str, dict[str, Any]] = {}
    for qid in sorted(qid_subset):
        pot = pot_by_qid.get(qid, {})
        cot = cot_by_qid.get(qid, {})
        pot_correct = bool(pot.get("is_correct"))
        cot_response_type = cot.get("response_type", "unknown")
        cot_reason = cot.get("refusal_reason")
        cot_raw = cot.get("raw_response", "")
        sections = _extract_trace_sections(cot_raw)

        category, stage, note = _classify_error(
            pot_correct,
            cot_response_type,
            cot_reason,
            sections,
            pot.get("predicted_answer"),
            pot.get("ground_truth", ""),
        )

        tags[qid] = {
            "pot_is_correct": pot_correct,
            "pot_predicted": pot.get("predicted_answer"),
            "pot_response_type": pot.get("response_type"),
            "cot_response_type": cot_response_type,
            "cot_is_correct": bool(cot.get("is_correct")),
            "cot_refusal_reason": cot_reason,
            "cot_predicted": cot.get("predicted_answer"),
            "trace_sections": sections,
            "ground_truth": pot.get("ground_truth") or cot.get("ground_truth"),
            "error_category": category,
            "failure_stage": stage,
            "note": note,
            "confidence": "heuristic",
        }
    return tags


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Tag error categories for original context (B1)")
    parser.add_argument("--pot", type=Path, required=True, help="eval_original_context.json")
    parser.add_argument("--cot", type=Path, required=True, help="eval_original_ic_cot_trace_*.json")
    parser.add_argument(
        "--qid-range",
        type=str,
        default="test-2000:test-2019",
        help="QID range (inclusive), default test-2000:test-2019",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/original_error_tags_0_20.json"),
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
    """Run tagger + write summary."""
    args = parse_args()
    qid_subset = _resolve_qid_subset(args.qid_range)

    pot_records = load_eval_records(args.pot)
    cot_records = load_eval_records(args.cot)
    logger.info(f"POT records: {len(pot_records)} | CoT records: {len(cot_records)}")

    tags = build_tags(pot_records, cot_records, qid_subset)

    from collections import Counter
    cat_counter = Counter(t["error_category"] for t in tags.values())
    stage_counter = Counter(t["failure_stage"] for t in tags.values())
    logger.info(f"\nError Category distribution ({len(tags)} qids):")
    for c, n in cat_counter.most_common():
        logger.info(f"  {c}: {n}")
    logger.info("\nFailure stage distribution:")
    for s, n in stage_counter.most_common():
        logger.info(f"  {s}: {n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(tags, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
