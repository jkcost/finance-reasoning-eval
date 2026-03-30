"""
Batch Transformation Pipeline — Phase 0

hard.json의 지정 범위 문제에 대해 5가지 변환을 모두 시도하고,
성공/실패를 기록하는 전수 변환 파이프라인.

Usage:
    python experiments/run_batch_transformation.py --start 0 --end 30
    python experiments/run_batch_transformation.py --start 30 --end 238
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from hardcoded_solution_detector import _solution_uses_hardcoded_values

sys.path.insert(0, str(Path(__file__).parent))
from apply_transformations_full import (  # noqa: E402
    LABEL_EA_FULL,
    LABEL_EA_PARTIAL,
    LABEL_IC,
    LABEL_SA,
    LABEL_TA,
    detect_context_type,
    normalize_context,
    transform_sa_text,
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
    transform_type5_json,
    transform_type5_markdown,
    transform_type5_text,
    transform_question_ea_partial,
    transform_question_ic,
    transform_question_sa,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Short labels for display and keys
TYPE_KEYS = ["EA-partial", "EA-full", "SA", "IC", "TA"]
LABEL_MAP = {
    "EA-partial": LABEL_EA_PARTIAL,
    "EA-full": LABEL_EA_FULL,
    "SA": LABEL_SA,
    "IC": LABEL_IC,
    "TA": LABEL_TA,
}


def _try_transform(
    transform_fn, *args
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Call a transform function and return (success, result, description, fail_reason)."""
    try:
        result, desc, _expected = transform_fn(*args)
        if result is not None:
            return True, result, desc, None
        return False, None, None, "no_match"
    except Exception as e:
        return False, None, None, f"error:{str(e)[:80]}"


def _transform_question_only(
    question: str, python_solution: str
) -> Dict[str, Dict[str, Any]]:
    """Apply question-level transformations for context-less problems.

    Returns results with Q-EA-partial, Q-SA, Q-IC keys plus
    empty EA-full and TA entries.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # Q-EA-partial
    ok, new_q, desc, reason = _try_transform(
        transform_question_ea_partial, question, python_solution
    )
    if ok:
        results["EA-partial"] = {
            "success": True,
            "question_transformed": new_q,
            "description": f"[Q] {desc}",
            "is_question_transform": True,
        }
    else:
        results["EA-partial"] = {"success": False, "reason": reason or "no_match"}

    # EA-full: not applicable for question-only
    results["EA-full"] = {"success": False, "reason": "question_only"}

    # Q-SA
    ok, new_q, desc, reason = _try_transform(
        transform_question_sa, question, python_solution
    )
    if ok:
        results["SA"] = {
            "success": True,
            "question_transformed": new_q,
            "description": f"[Q] {desc}",
            "is_question_transform": True,
        }
    else:
        results["SA"] = {"success": False, "reason": reason or "no_match"}

    # Q-IC
    ok, new_q, desc, reason = _try_transform(
        transform_question_ic, question, python_solution
    )
    if ok:
        results["IC"] = {
            "success": True,
            "question_transformed": new_q,
            "description": f"[Q] {desc}",
            "is_question_transform": True,
        }
    else:
        results["IC"] = {"success": False, "reason": reason or "no_match"}

    # TA: not applicable for question-only (already has no context)
    results["TA"] = {"success": False, "reason": "question_only"}

    return results


def transform_single_problem(example: Dict) -> Dict[str, Any]:
    """Apply all 5 transformation types to a single problem.

    Returns a dict with transformation results for each type.
    """
    context = normalize_context(example.get("context", ""))
    question = example.get("question", "")
    python_solution = example.get("python_solution", "")
    ctx_type = detect_context_type(context)

    results: Dict[str, Dict[str, Any]] = {}

    if ctx_type == "none":
        # Question-only problem: apply question transformations
        return _transform_question_only(question, python_solution)

    # --- EA-partial ---
    if ctx_type == "json":
        ctx_dict = json.loads(context)
        ok, res, desc, reason = _try_transform(transform_type1_json, ctx_dict, question)
        ctx_out = json.dumps(res) if ok else None
    elif ctx_type == "markdown":
        ok, ctx_out, desc, reason = _try_transform(
            transform_type1_markdown, context, question, python_solution
        )
    else:  # text
        ok, ctx_out, desc, reason = _try_transform(
            transform_type1_text, context, question
        )

    if ok:
        results["EA-partial"] = {
            "success": True,
            "context_transformed": ctx_out,
            "description": desc,
        }
    else:
        results["EA-partial"] = {"success": False, "reason": reason or "no_match"}

    # --- EA-full ---
    if ctx_type == "json":
        ctx_dict = json.loads(context)
        ok, res, desc, reason = _try_transform(transform_type2_json, ctx_dict, question)
        if ok:
            results["EA-full"] = {
                "success": True,
                "context_transformed": json.dumps(res),
                "description": desc,
            }
        else:
            results["EA-full"] = {"success": False, "reason": reason or "no_match"}
    elif ctx_type == "markdown":
        ok, ctx_out, desc, reason = _try_transform(
            transform_type2_markdown, context, question, python_solution
        )
        if ok:
            results["EA-full"] = {
                "success": True,
                "context_transformed": ctx_out,
                "description": desc,
            }
        else:
            results["EA-full"] = {"success": False, "reason": reason or "no_match"}
    else:  # text — remove all data sentences
        ok, ctx_out, desc, reason = _try_transform(
            transform_type2_text, context, question
        )
        if ok:
            results["EA-full"] = {
                "success": True,
                "context_transformed": ctx_out,
                "description": desc,
            }
        else:
            results["EA-full"] = {"success": False, "reason": reason or "no_match"}

    # --- SA ---
    if ctx_type == "json":
        ctx_dict = json.loads(context)
        ok, res, desc, reason = _try_transform(
            transform_type4_json, ctx_dict, question, python_solution
        )
        ctx_out = json.dumps(res) if ok else None
    elif ctx_type == "markdown":
        ok, ctx_out, desc, reason = _try_transform(
            transform_type4_markdown, context, question, python_solution
        )
    else:  # text — use sentence deletion
        ok, ctx_out, desc, reason = _try_transform(transform_sa_text, context, question)

    if ok:
        results["SA"] = {
            "success": True,
            "context_transformed": ctx_out,
            "description": desc,
        }
    else:
        results["SA"] = {"success": False, "reason": reason or "no_match"}

    # --- IC ---
    if ctx_type == "json":
        ctx_dict = json.loads(context)
        ok, res, desc, reason = _try_transform(transform_type5_json, ctx_dict, question)
        ctx_out = json.dumps(res) if ok else None
    elif ctx_type == "markdown":
        ok, ctx_out, desc, reason = _try_transform(
            transform_type5_markdown, context, question, python_solution
        )
    else:
        ok, ctx_out, desc, reason = _try_transform(
            transform_type5_text, context, question, python_solution
        )

    if ok:
        results["IC"] = {
            "success": True,
            "context_transformed": ctx_out,
            "description": desc,
        }
    else:
        results["IC"] = {"success": False, "reason": reason or "no_match"}

    # --- TA: question first, then context fallback ---
    ok, new_q, desc, reason = _try_transform(transform_type3_question, question)
    if ok:
        results["TA"] = {
            "success": True,
            "question_transformed": new_q,
            "description": desc,
        }
    else:
        # Fallback: replace year in context
        ok, ctx_out, desc, reason = _try_transform(
            transform_type3_context, context, question
        )
        if ok:
            results["TA"] = {
                "success": True,
                "context_transformed": ctx_out,
                "description": desc,
            }
        else:
            results["TA"] = {"success": False, "reason": reason or "no_year_pattern"}

    return results


def build_coverage_summary(problems: List[Dict]) -> Dict[str, Any]:
    """Build aggregated coverage summary from problem list."""
    total = len(problems)
    transformable = sum(
        1
        for p in problems
        if any(p["transformations"][k]["success"] for k in TYPE_KEYS)
    )

    by_type: Dict[str, Dict[str, Any]] = {}
    for key in TYPE_KEYS:
        successes = sum(1 for p in problems if p["transformations"][key]["success"])
        failures = total - successes
        # Aggregate failure reasons
        reasons: Dict[str, int] = {}
        for p in problems:
            t = p["transformations"][key]
            if not t["success"]:
                r = t.get("reason", "unknown")
                reasons[r] = reasons.get(r, 0) + 1
        by_type[key] = {
            "success": successes,
            "fail": failures,
            "rate": round(successes / total * 100, 1) if total > 0 else 0,
            "fail_reasons": reasons,
        }

    return {
        "total_problems": total,
        "transformable": transformable,
        "not_transformable": total - transformable,
        "by_type": by_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch transformation pipeline")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=30, help="End index (exclusive)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/metacognitive",
        help="Output directory",
    )
    args = parser.parse_args()

    # Load hard.json
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data/financereasoning/raw/FinanceReasoning/hard.json"

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_total = len(dataset)
    start = max(0, args.start)
    end = min(dataset_total, args.end)
    subset = dataset[start:end]

    logger.info(f"hard.json: {dataset_total}문제, 처리 범위: [{start}, {end})")
    logger.info(f"변환 대상: {len(subset)}문제")

    # Process each problem
    problems = []
    for i, example in enumerate(subset):
        idx = start + i
        qid = example.get("question_id", f"unknown-{idx}")
        context = normalize_context(example.get("context", ""))
        ctx_type = detect_context_type(context)
        python_solution = example.get("python_solution", "")
        is_hardcoded = (
            _solution_uses_hardcoded_values(python_solution)
            if python_solution
            else False
        )

        transformations = transform_single_problem(example)

        success_types = [k for k in TYPE_KEYS if transformations[k]["success"]]

        problem_record = {
            "question_id": qid,
            "index": idx,
            "question": example.get("question", ""),
            "context_original": context,
            "context_format": ctx_type,
            "ground_truth": example.get("ground_truth"),
            "python_solution": python_solution,
            "is_hardcoded_solution": is_hardcoded,
            "transformations": transformations,
        }
        problems.append(problem_record)

        logger.info(
            f"  [{idx:3d}] {qid} | {ctx_type:8s} | "
            f"hardcoded={is_hardcoded} | "
            f"성공: {', '.join(success_types) if success_types else '없음'}"
        )

    # Build output
    coverage = build_coverage_summary(problems)
    output = {
        "metadata": {
            "dataset": "hard.json",
            "dataset_total": dataset_total,
            "range": [start, end],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "transformation_types": TYPE_KEYS,
        },
        "coverage_summary": coverage,
        "problems": problems,
    }

    # Save
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"batch_transformations_{start}_{end}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"변환 완료: {output_file}")
    logger.info(f"{'=' * 60}")
    logger.info(f"전체: {coverage['total_problems']}문제")
    logger.info(f"변환 가능: {coverage['transformable']}문제")
    logger.info(f"변환 불가: {coverage['not_transformable']}문제")
    logger.info("")
    for key in TYPE_KEYS:
        info = coverage["by_type"][key]
        logger.info(
            f"  {key:12s}: {info['success']:3d} 성공 / {info['fail']:3d} 실패 ({info['rate']}%)"
        )
        if info["fail_reasons"]:
            for reason, count in info["fail_reasons"].items():
                logger.info(f"    └ {reason}: {count}")


if __name__ == "__main__":
    main()
