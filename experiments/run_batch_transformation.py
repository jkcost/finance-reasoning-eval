"""
Batch Transformation Pipeline (LLM-based)

hard.json의 지정 범위 문제에 대해 LLM이 금융 맥락을 분석하여
4가지 변환(EA-partial, EA-full, SA, IC)을 생성.

Usage:
    # 기본 (gemini-2.5-flash, 경제적)
    python experiments/run_batch_transformation.py --start 0 --end 120

    # 모델 지정
    python experiments/run_batch_transformation.py --start 0 --end 10 --model gpt-4o-mini

    # 동시 처리 수 조절
    python experiments/run_batch_transformation.py --start 0 --end 120 --concurrency 3
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from apply_transformations_full import detect_context_type, normalize_context
from hardcoded_solution_detector import _solution_uses_hardcoded_values
from llm_transform import LLMTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TYPE_KEYS = ["EA-partial", "EA-full", "SA", "IC", "TA"]


def build_coverage_summary(problems: List[Dict]) -> Dict[str, Any]:
    """Build aggregated coverage summary from problem list."""
    total = len(problems)
    transformable = sum(
        1
        for p in problems
        if any(p["transformations"].get(k, {}).get("success") for k in TYPE_KEYS)
    )

    by_type: Dict[str, Dict[str, Any]] = {}
    for key in TYPE_KEYS:
        successes = sum(
            1 for p in problems if p["transformations"].get(key, {}).get("success")
        )
        failures = total - successes
        reasons: Dict[str, int] = {}
        for p in problems:
            t = p["transformations"].get(key, {})
            if not t.get("success"):
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


async def run_batch(
    dataset: List[Dict],
    start: int,
    end: int,
    model: str,
    concurrency: int,
    output_dir: Path,
) -> None:
    """Run LLM-based batch transformation."""
    subset = dataset[start:end]
    logger.info(f"hard.json: {len(dataset)}문제, 처리 범위: [{start}, {end})")
    logger.info(f"변환 대상: {len(subset)}문제, 모델: {model}")

    transformer = LLMTransformer(model=model, concurrency=concurrency)

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

        # Build problem dict for transformer
        problem_input = {
            "question": example.get("question", ""),
            "context_original": context,
            "context_format": ctx_type,
            "ground_truth": example.get("ground_truth"),
            "python_solution": python_solution,
        }

        # LLM-based transformation
        transformations = await transformer.transform_problem(problem_input)

        # Add TA as skipped (not LLM-based for now)
        if "TA" not in transformations:
            transformations["TA"] = {"success": False, "reason": "skipped"}

        success_types = [
            k for k in TYPE_KEYS if transformations.get(k, {}).get("success")
        ]

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
            f"성공: {', '.join(success_types) if success_types else '없음'} | "
            f"비용: ${transformer.total_cost:.4f}"
        )

    await transformer.close()

    # Build output
    coverage = build_coverage_summary(problems)
    output = {
        "metadata": {
            "dataset": "hard.json",
            "dataset_total": len(dataset),
            "range": [start, end],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "transformation_types": TYPE_KEYS,
            "model": model,
            "method": "llm-based",
            "total_cost_usd": round(transformer.total_cost, 4),
            "total_api_calls": transformer.total_calls,
        },
        "coverage_summary": coverage,
        "problems": problems,
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"batch_transformations_{start}_{end}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"변환 완료: {output_file}")
    logger.info(f"{'=' * 60}")
    logger.info(f"모델: {model}")
    logger.info(f"총 API 호출: {transformer.total_calls}회")
    logger.info(f"총 비용: ${transformer.total_cost:.4f}")
    logger.info(f"전체: {coverage['total_problems']}문제")
    logger.info(f"변환 가능: {coverage['transformable']}문제")
    logger.info(f"변환 불가: {coverage['not_transformable']}문제")
    logger.info("")
    for key in TYPE_KEYS:
        info = coverage["by_type"][key]
        logger.info(
            f"  {key:12s}: {info['success']:3d} 성공 / "
            f"{info['fail']:3d} 실패 ({info['rate']}%)"
        )
        if info["fail_reasons"]:
            for reason, count in info["fail_reasons"].items():
                logger.info(f"    └ {reason}: {count}")


def main():
    # Load .env
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(
        description="LLM-based batch transformation pipeline"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index (inclusive)"
    )
    parser.add_argument(
        "--end", type=int, default=30, help="End index (exclusive)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model (gemini-2.5-flash, gpt-4o-mini, claude-haiku-4)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrent API calls",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/metacognitive",
        help="Output directory",
    )
    args = parser.parse_args()

    # Load hard.json
    data_path = (
        project_root / "data/financereasoning/raw/FinanceReasoning/hard.json"
    )

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    start = max(0, args.start)
    end = min(len(dataset), args.end)
    output_dir = project_root / args.output_dir

    asyncio.run(run_batch(dataset, start, end, args.model, args.concurrency, output_dir))


if __name__ == "__main__":
    main()
