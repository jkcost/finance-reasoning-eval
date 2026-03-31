"""
Batch Transformation Generator (규칙 기반)

hard.json에서 5타입(EA-partial, EA-full, SA, IC, TA) 변환을 규칙 기반으로 생성하고
batch_transformations 포맷의 JSON을 출력.

이 스크립트는 API 호출 없이 즉시 실행 가능.
생성된 JSON은 generate_human_review.py의 입력으로 사용.

Usage:
    python experiments/generate_batch_transformations.py
    python experiments/generate_batch_transformations.py --start 0 --end 120
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from apply_transformations_full import (
    detect_context_type,
    normalize_context,
    transform_type1_json,
    transform_type1_markdown,
    transform_type1_text,
    transform_type2_json,
    transform_type2_markdown,
    transform_type2_text,
    transform_type3_context,
    transform_type3_question,
    transform_type4_json,
    transform_type4_markdown,
    transform_sa_text,
)
from hardcoded_solution_detector import _solution_uses_hardcoded_values  # noqa: E402
from reverse_calc_detector import analyze_batch_reversibility  # noqa: E402
from ic_difficulty_ladder import (  # noqa: E402
    transform_ic_l1_json,
    transform_ic_l1_text,
    transform_ic_l2_text,
    transform_ic_l3_json,
    transform_ic_l3_markdown,
    transform_ic_l3_text,
    transform_ic_l4_json,
    transform_ic_l4_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TYPE_KEYS = ["EA-partial", "EA-full", "SA", "IC-L1", "IC-L2", "IC-L3", "IC-L4", "TA"]


def _try_transform(fn, *args) -> Dict[str, Any]:
    """Execute a transformation function safely, return batch format dict."""
    try:
        result = fn(*args)
        if result[0] is not None:
            return {
                "success": True,
                "context_transformed": (
                    json.dumps(result[0])
                    if isinstance(result[0], dict)
                    else str(result[0])
                ),
                "description": result[1] or "",
                "expected_behavior": result[2] or "",
            }
    except Exception as e:
        return {"success": False, "reason": f"error: {e}"}
    return {"success": False, "reason": "not_applicable"}


def transform_problem(example: Dict) -> Dict[str, Dict[str, Any]]:
    """Apply all 5 transformation types to a single problem.

    Returns dict in batch_transformations format:
        {
            "EA-partial": {"success": bool, "context_transformed": str, ...},
            "EA-full": {...},
            "SA": {...},
            "IC": {...},
            "TA": {...},
        }
    """
    context = normalize_context(str(example.get("context", "")))
    question = example.get("question", "")
    python_solution = example.get("python_solution", "")
    context_type = detect_context_type(context)

    transformations: Dict[str, Dict[str, Any]] = {}

    if context_type == "json":
        try:
            ctx_dict = json.loads(context)
            transformations["EA-partial"] = _try_transform(
                transform_type1_json, ctx_dict, question
            )
            transformations["EA-full"] = _try_transform(
                transform_type2_json, ctx_dict, question
            )
            transformations["SA"] = _try_transform(
                transform_type4_json, ctx_dict, question, python_solution
            )
            # IC L1-L4
            transformations["IC-L1"] = _try_transform(
                transform_ic_l1_json, ctx_dict, question
            )
            transformations["IC-L2"] = {
                "success": False,
                "reason": "json_not_supported",
            }
            transformations["IC-L3"] = _try_transform(
                transform_ic_l3_json, ctx_dict, question
            )
            transformations["IC-L4"] = _try_transform(
                transform_ic_l4_json, ctx_dict, question
            )
        except (json.JSONDecodeError, TypeError):
            for t in [
                "EA-partial",
                "EA-full",
                "SA",
                "IC-L1",
                "IC-L2",
                "IC-L3",
                "IC-L4",
            ]:
                transformations[t] = {"success": False, "reason": "json_parse_error"}

    elif context_type == "text":
        transformations["EA-partial"] = _try_transform(
            transform_type1_text, context, question
        )
        transformations["EA-full"] = _try_transform(
            transform_type2_text, context, question
        )
        transformations["SA"] = _try_transform(transform_sa_text, context, question)
        # IC L1-L4
        transformations["IC-L1"] = _try_transform(
            transform_ic_l1_text, context, question, python_solution
        )
        transformations["IC-L2"] = _try_transform(
            transform_ic_l2_text, context, question, python_solution
        )
        transformations["IC-L3"] = _try_transform(
            transform_ic_l3_text, context, question, python_solution
        )
        transformations["IC-L4"] = _try_transform(
            transform_ic_l4_text, context, question, python_solution
        )

    elif context_type == "markdown":
        transformations["EA-partial"] = _try_transform(
            transform_type1_markdown, context, question
        )
        transformations["EA-full"] = _try_transform(
            transform_type2_markdown, context, question
        )
        transformations["SA"] = _try_transform(
            transform_type4_markdown, context, question, python_solution
        )
        # IC L1-L4 for markdown
        transformations["IC-L1"] = _try_transform(
            transform_ic_l1_text, context, question, python_solution
        )
        transformations["IC-L2"] = _try_transform(
            transform_ic_l2_text, context, question, python_solution
        )
        transformations["IC-L3"] = _try_transform(
            transform_ic_l3_markdown, context, question, python_solution
        )
        transformations["IC-L4"] = _try_transform(
            transform_ic_l4_text, context, question, python_solution
        )

    else:
        for t in ["EA-partial", "EA-full", "SA", "IC-L1", "IC-L2", "IC-L3", "IC-L4"]:
            transformations[t] = {"success": False, "reason": "no_context"}

    # TA: question-based transformation (applies to all context types)
    ta_result = _try_transform(transform_type3_question, question)
    if ta_result.get("success"):
        # TA transforms the question, not the context
        transformations["TA"] = {
            "success": True,
            "question_transformed": ta_result["context_transformed"],
            "description": ta_result["description"],
            "expected_behavior": ta_result["expected_behavior"],
        }
    else:
        ta_ctx = _try_transform(transform_type3_context, context, question)
        if ta_ctx.get("success"):
            transformations["TA"] = ta_ctx
        else:
            transformations["TA"] = {"success": False, "reason": "not_applicable"}

    return transformations


def run_batch(dataset: List[Dict], start: int, end: int) -> Dict[str, Any]:
    """Run batch transformation on dataset range.

    Returns batch_transformations format compatible with generate_human_review.py.
    """
    subset = dataset[start:end]
    logger.info(f"hard.json: {len(dataset)}문제, 처리 범위: [{start}, {end})")

    problems = []
    for i, example in enumerate(subset):
        idx = start + i
        context = normalize_context(str(example.get("context", "")))
        ctx_type = detect_context_type(context)
        python_solution = example.get("python_solution", "")
        is_hardcoded = (
            _solution_uses_hardcoded_values(python_solution)
            if python_solution
            else False
        )

        transformations = transform_problem(example)
        success_types = [
            k for k in TYPE_KEYS if transformations.get(k, {}).get("success")
        ]

        problem_record = {
            "question_id": example.get("question_id", f"hard_{idx}"),
            "index": idx,
            "question": example.get("question", ""),
            "context_original": context,
            "context_format": ctx_type,
            "ground_truth": example.get("ground_truth", example.get("answer")),
            "python_solution": python_solution,
            "is_hardcoded_solution": is_hardcoded,
            "transformations": transformations,
        }
        problems.append(problem_record)

        if (i + 1) % 20 == 0 or (i + 1) == len(subset):
            logger.info(f"  진행: {i + 1}/{len(subset)}")

    # Reverse calculation detection for EA-full
    logger.info("EA-full 역산 가능성 분석 중...")
    reversible = analyze_batch_reversibility(problems)
    for qid, rinfo in reversible.items():
        for p in problems:
            if p["question_id"] == qid:
                ea_full = p["transformations"].get("EA-full", {})
                if ea_full.get("success"):
                    ea_full["reverse_calculable"] = True
                    ea_full["reverse_explanation"] = rinfo["details"]["explanation"]
                    ea_full["reverse_formula"] = rinfo["details"]["formula"]
                break
    logger.info(f"  역산 가능: {len(reversible)}건 탐지")

    # Coverage summary
    total = len(problems)
    by_type = {}
    for t in TYPE_KEYS:
        successes = sum(
            1 for p in problems if p["transformations"].get(t, {}).get("success")
        )
        failures = total - successes
        fail_reasons: Dict[str, int] = {}
        for p in problems:
            reason = p["transformations"].get(t, {}).get("reason", "")
            if reason:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        by_type[t] = {
            "success": successes,
            "fail": failures,
            "rate": round(successes / total * 100, 1) if total > 0 else 0,
            "fail_reasons": fail_reasons,
        }

    transformable = sum(
        1
        for p in problems
        if any(p["transformations"].get(t, {}).get("success") for t in TYPE_KEYS)
    )

    coverage = {
        "total_problems": total,
        "transformable": transformable,
        "not_transformable": total - transformable,
        "by_type": by_type,
    }

    return {
        "metadata": {
            "dataset": "hard.json",
            "dataset_total": len(dataset),
            "range": [start, end],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "transformation_types": TYPE_KEYS,
            "method": "rule-based",
        },
        "coverage_summary": coverage,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate batch transformations (rule-based, no API)"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=120)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/metacognitive",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hard_path = project_root / "data/financereasoning/raw/FinanceReasoning/hard.json"

    with open(hard_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    output = run_batch(dataset, args.start, args.end)

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"batch_transformations_{args.start}_{args.end}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"변환 완료: {output_file}")
    logger.info(f"{'=' * 60}")
    cov = output["coverage_summary"]
    logger.info(
        f"전체: {cov['total_problems']}문제, 변환가능: {cov['transformable']}문제"
    )
    for t in TYPE_KEYS:
        info = cov["by_type"][t]
        logger.info(f"  {t:12s}: {info['success']:3d} 성공 ({info['rate']}%)")


if __name__ == "__main__":
    main()
