"""
Batch Evaluation Pipeline — Phase 1

batch_transformations JSON의 성공한 변환을 모델에 투입하고
응답을 수집 + Case 분류하는 파이프라인.

Usage:
    python experiments/run_batch_evaluation.py \\
        --input experiments/results/metacognitive/batch_transformations_0_30.json \\
        --budget balanced

    python experiments/run_batch_evaluation.py \\
        --input experiments/results/metacognitive/batch_transformations_0_30.json \\
        --models gpt-4o-mini
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ModelConfig
from error_analysis import MODEL_REGISTRY
from error_analysis.error_taxonomy import BUDGET_MODEL_SETS
from metacognitive_metrics import ResponseType
from model_runner import AnthropicProvider, GoogleProvider, OpenAIProvider
from refusal_detector import RefusalDetector

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import prompt infrastructure from existing experiment
from run_metacognitive_experiment import (  # noqa: E402
    POT_PREFIX,
    PROMPT_SYSTEMS,
)


# ============================================================================
# PROVIDER INITIALIZATION
# ============================================================================


def init_providers(model_names: List[str]) -> Dict[str, Any]:
    """Initialize API providers for each model."""
    load_dotenv()
    providers = {}
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}")
            continue
        model_info = MODEL_REGISTRY[model_name]
        api_key = os.environ.get(model_info.api_key_env)
        if not api_key:
            logger.warning(f"{model_name}: No API key ({model_info.api_key_env})")
            continue
        model_config = ModelConfig(
            id=model_name,
            name=model_info.display_name,
            provider=model_info.provider,
            model_id=model_info.model_id,
            api_key_env_var=model_info.api_key_env,
            max_tokens=4096,
            temperature=0.0,
            cost_per_million_tokens=model_info.cost_per_million_output,
        )
        if model_info.provider == "openai":
            providers[model_name] = OpenAIProvider(model_config)
        elif model_info.provider == "anthropic":
            providers[model_name] = AnthropicProvider(model_config)
        elif model_info.provider == "google":
            providers[model_name] = GoogleProvider(model_config)
        logger.info(f"[OK] {model_name}")
    return providers


# ============================================================================
# PROMPT BUILDING
# ============================================================================

METHOD = "POT"


def build_prompt(
    question: str,
    context: str,
    prompt_strategy: str,
) -> str:
    """Build full prompt (system + user) for a given question/context."""
    if context and context != "[]":
        question_input = (
            f"The following question context is provided for your reference.\n"
            f"{context}\n\nQuestion: {question}\n"
        )
    else:
        question_input = f"Question: {question}\n"

    system_prompt = PROMPT_SYSTEMS[prompt_strategy][METHOD]
    user_prompt = question_input + "\n" + POT_PREFIX
    return f"{system_prompt}\n\n{user_prompt}"


# ============================================================================
# ANSWER EXTRACTION & CLASSIFICATION
# ============================================================================


def extract_answer(response: str) -> Tuple[Any, Optional[str], Optional[str]]:
    """Extract answer from POT response."""
    if re.search(r"(?i)INSUFFICIENT[_ ]INFORMATION", response):
        return "INSUFFICIENT_INFORMATION", None, None
    if re.search(r"(?i)CONTRADICTION[_ ]DETECTED", response):
        return "INSUFFICIENT_INFORMATION", None, None

    code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if not code_match:
        code_match = re.search(
            r"(def solution\(\):.*?)(?:\n\n|\Z)", response, re.DOTALL
        )

    if not code_match:
        # Fallback: extract number from text
        patterns = [
            r"[Tt]herefore,?\s*the\s+answer\s+is\s*[:\s]*([+-]?\d+\.?\d*%?)",
            r"[Tt]he\s+answer\s+is\s*[:\s]*([+-]?\d+\.?\d*%?)",
        ]
        for pattern in patterns:
            m = re.search(pattern, response)
            if m:
                try:
                    return (
                        float(m.group(1).replace("%", "").replace(",", "")),
                        None,
                        None,
                    )
                except ValueError:
                    continue
        return None, None, "no_code_block"

    code = code_match.group(1).strip()
    try:
        local_vars: Dict[str, Any] = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        if "solution" in local_vars and callable(local_vars["solution"]):
            result = local_vars["solution"]()
            if isinstance(result, str) and "INSUFFICIENT" in result.upper():
                return "INSUFFICIENT_INFORMATION", code, None
            return result, code, None
        elif "answer" in local_vars:
            return local_vars["answer"], code, None
        return None, code, "no_result_variable"
    except Exception as e:
        return None, code, str(e)[:200]


def check_answer(pred: Any, truth: Any, tolerance: float = 0.002) -> bool:
    """Check if prediction matches ground truth."""
    if pred is None or pred == "INSUFFICIENT_INFORMATION":
        return False
    try:
        pred_num = float(str(pred).replace(",", "").replace("%", ""))
        truth_num = float(str(truth).replace(",", "").replace("%", ""))
        if truth_num == 0:
            return abs(pred_num) < tolerance
        return abs(pred_num - truth_num) / abs(truth_num) <= tolerance
    except (ValueError, TypeError):
        return str(pred).strip().lower() == str(truth).strip().lower()


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """Calculate API cost."""
    model_info = MODEL_REGISTRY[model_name]
    input_cost = (input_tokens / 1_000_000) * model_info.cost_per_million_input
    output_cost = (output_tokens / 1_000_000) * model_info.cost_per_million_output
    return round(input_cost + output_cost, 6)


def classify_case(
    response_type: str, is_correct: bool, transformation_type: str
) -> int:
    """Classify into Case 1 (refused), Case 2 (wrong), Case 3 (correct)."""
    if response_type in (ResponseType.REFUSED.value, ResponseType.ERROR.value):
        return 1
    if is_correct:
        return 3
    return 2


# ============================================================================
# EVALUATION CORE
# ============================================================================


async def evaluate_one(
    provider,
    model_name: str,
    question: str,
    context: str,
    ground_truth: Any,
    question_id: str,
    transformation_type: str,
    prompt_strategy: str,
    refusal_detector: RefusalDetector,
) -> Dict[str, Any]:
    """Evaluate a single transformed problem with one model."""

    class MinimalExample:
        def __init__(self, qid, q, c, gt):
            self.id = qid
            self.question = q
            self.context = c
            self.ground_truth = gt
            self.python_solution = ""

    mini_ex = MinimalExample(question_id, question, context, ground_truth)
    full_prompt = build_prompt(question, context, prompt_strategy)

    try:
        response = await provider.call_model(mini_ex, full_prompt)
        raw_response = response.raw_response
        cost = calculate_cost(
            response.prompt_tokens, response.completion_tokens, model_name
        )
        latency = response.response_time_seconds

        answer, executed_code, execution_error = extract_answer(raw_response)

        detection = refusal_detector.detect(
            raw_response,
            context=context,
            execution_error=execution_error,
            executed_code=executed_code,
        )

        if answer == "INSUFFICIENT_INFORMATION":
            response_type = ResponseType.REFUSED.value
        else:
            response_type = detection.response_type.value

        is_correct = check_answer(answer, ground_truth)
        case_type = classify_case(response_type, is_correct, transformation_type)

        return {
            "question_id": question_id,
            "transformation_type": transformation_type,
            "model": model_name,
            "prompt_strategy": prompt_strategy,
            "case_type": case_type,
            "response_type": response_type,
            "predicted_answer": str(answer) if answer is not None else None,
            "ground_truth": str(ground_truth),
            "is_correct": is_correct,
            "cost_usd": cost,
            "latency_seconds": latency,
            "raw_response": raw_response,
            "executed_code": executed_code,
            "execution_error": execution_error,
        }

    except Exception as e:
        logger.error(f"  [ERROR] {model_name}/{question_id}/{transformation_type}: {e}")
        return {
            "question_id": question_id,
            "transformation_type": transformation_type,
            "model": model_name,
            "prompt_strategy": prompt_strategy,
            "case_type": 1,
            "response_type": ResponseType.ERROR.value,
            "predicted_answer": None,
            "ground_truth": str(ground_truth),
            "is_correct": False,
            "cost_usd": 0.0,
            "latency_seconds": 0.0,
            "raw_response": f"ERROR: {str(e)[:200]}",
            "executed_code": None,
            "execution_error": str(e)[:200],
        }


async def run_evaluation(
    input_data: Dict,
    providers: Dict[str, Any],
    prompt_strategy: str,
    concurrency: int = 5,
    checkpoint_path: Optional[Path] = None,
) -> List[Dict]:
    """Run all evaluations with concurrency control and checkpoint/resume.

    Args:
        checkpoint_path: If provided, saves progress after each batch of 10
            evaluations. On restart, skips already-completed (qid, model, type) combos.
    """
    refusal_detector = RefusalDetector()
    semaphore = asyncio.Semaphore(concurrency)

    # Load checkpoint if exists
    completed_keys: set = set()
    checkpoint_results: List[Dict] = []
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            checkpoint_results = checkpoint_data.get("results", [])
            for r in checkpoint_results:
                key = (r["question_id"], r["model"], r["transformation_type"])
                completed_keys.add(key)
        logger.info(
            f"Checkpoint loaded: {len(completed_keys)} evaluations already done"
        )

    # Build task list from successful transformations
    tasks = []
    for problem in input_data["problems"]:
        qid = problem["question_id"]
        ground_truth = problem.get("ground_truth")
        question = problem["question"]
        context_original = problem["context_original"]

        for t_type, t_data in problem["transformations"].items():
            if not t_data.get("success"):
                continue

            # Determine context and question to use
            if t_type == "TA":
                ctx = context_original
                q = t_data.get("question_transformed", question)
            else:
                ctx = t_data.get("context_transformed", context_original)
                q = t_data.get("question_transformed", question)

            for model_name, provider in providers.items():
                # Skip already-completed evaluations
                if (qid, model_name, t_type) in completed_keys:
                    continue
                tasks.append((provider, model_name, q, ctx, ground_truth, qid, t_type))

    skipped = len(completed_keys)
    logger.info(
        f"총 평가 작업: {len(tasks)}개 신규 "
        f"({skipped}개 checkpoint에서 복원, {len(providers)}모델 × 변환)"
    )

    async def bounded_eval(task_args):
        async with semaphore:
            return await evaluate_one(
                *task_args,
                prompt_strategy=prompt_strategy,
                refusal_detector=refusal_detector,
            )

    # Run with progress logging + periodic checkpoint saves
    results = list(checkpoint_results)
    coros = [bounded_eval(t) for t in tasks]
    for i, coro in enumerate(asyncio.as_completed(coros)):
        result = await coro
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(coros):
            logger.info(f"  진행: {i + 1}/{len(coros)} (총 {len(results)}건)")
            # Save checkpoint every 10 evaluations
            if checkpoint_path and (i + 1) % 10 == 0:
                _save_checkpoint(checkpoint_path, results)

    # Final checkpoint save
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, results)

    return results


def _save_checkpoint(checkpoint_path: Path, results: List[Dict]) -> None:
    """Save evaluation progress to checkpoint file."""
    checkpoint_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total": len(results),
        "results": results,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  Checkpoint saved: {len(results)} results")


def build_summary(results: List[Dict]) -> Dict[str, Any]:
    """Build summary statistics from evaluation results."""
    by_type: Dict[str, Dict[str, int]] = {}
    by_model: Dict[str, Dict[str, int]] = {}
    case_dist = {1: 0, 2: 0, 3: 0}

    for r in results:
        t = r["transformation_type"]
        m = r["model"]
        c = r["case_type"]

        case_dist[c] = case_dist.get(c, 0) + 1

        if t not in by_type:
            by_type[t] = {"total": 0, "case1": 0, "case2": 0, "case3": 0}
        by_type[t]["total"] += 1
        by_type[t][f"case{c}"] += 1

        if m not in by_model:
            by_model[m] = {"total": 0, "case1": 0, "case2": 0, "case3": 0, "cost": 0.0}
        by_model[m]["total"] += 1
        by_model[m][f"case{c}"] += 1
        by_model[m]["cost"] += r.get("cost_usd", 0)

    # Compute rates
    for stats in by_type.values():
        total = stats["total"]
        if total > 0:
            stats["refusal_rate"] = round(
                (stats["case1"] + stats["case2"]) / total * 100, 1
            )

    for stats in by_model.values():
        total = stats["total"]
        if total > 0:
            stats["refusal_rate"] = round(stats["case1"] / total * 100, 1)
            stats["cost"] = round(stats["cost"], 4)

    return {
        "by_type": by_type,
        "by_model": by_model,
        "case_distribution": case_dist,
        "total_evaluations": len(results),
        "total_cost": round(sum(r.get("cost_usd", 0) for r in results), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation pipeline")
    parser.add_argument(
        "--input", type=str, required=True, help="batch_transformations JSON file"
    )
    parser.add_argument(
        "--budget",
        type=str,
        default=None,
        choices=["economic", "balanced", "full"],
        help="Model budget set",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None, help="Specific model names"
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="metacognitive",
        choices=list(PROMPT_SYSTEMS.keys()),
        help="Prompt strategy",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Max concurrent API calls"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    args = parser.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    logger.info(f"Loaded: {input_path.name}")
    logger.info(f"문제 수: {len(input_data['problems'])}")

    # Determine models
    if args.models:
        model_names = args.models
    elif args.budget:
        model_names = BUDGET_MODEL_SETS[args.budget]
    else:
        model_names = BUDGET_MODEL_SETS["balanced"]

    logger.info(f"모델: {model_names}")
    logger.info(f"전략: {args.prompt_strategy}")

    # Init providers
    providers = init_providers(model_names)
    if not providers:
        logger.error("사용 가능한 모델이 없습니다.")
        sys.exit(1)

    # Checkpoint path for resume support
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_range = input_data["metadata"]["range"]
    checkpoint_path = (
        output_dir / f"batch_eval_checkpoint_{meta_range[0]}_{meta_range[1]}.json"
    )

    # Run evaluation with checkpoint/resume
    results = asyncio.run(
        run_evaluation(
            input_data,
            providers,
            args.prompt_strategy,
            args.concurrency,
            checkpoint_path=checkpoint_path,
        )
    )

    # Build output
    summary = build_summary(results)

    # Store prompts used for reproducibility
    system_prompt = PROMPT_SYSTEMS[args.prompt_strategy][METHOD]

    output = {
        "metadata": {
            "input_file": str(input_path.name),
            "models": list(providers.keys()),
            "prompt_strategy": args.prompt_strategy,
            "total_evaluations": len(results),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "range": meta_range,
            "system_prompt": system_prompt,
            "user_prompt_template": (
                "The following question context is provided for your reference.\n"
                "{context}\n\nQuestion: {question}\n\n" + POT_PREFIX
            ),
        },
        "summary": summary,
        "results": results,
    }

    # Save
    output_file = output_dir / f"batch_evaluation_{meta_range[0]}_{meta_range[1]}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"평가 완료: {output_file}")
    logger.info(f"{'=' * 60}")
    logger.info(f"총 평가: {summary['total_evaluations']}건")
    logger.info(f"총 비용: ${summary['total_cost']}")
    logger.info(
        f"Case 분포: C1={summary['case_distribution'][1]}, "
        f"C2={summary['case_distribution'][2]}, C3={summary['case_distribution'][3]}"
    )
    logger.info("")
    logger.info("모델별:")
    for model, stats in summary["by_model"].items():
        logger.info(
            f"  {model:20s}: C1={stats['case1']} C2={stats['case2']} C3={stats['case3']} "
            f"(거부율 {stats.get('refusal_rate', 0)}%) ${stats['cost']}"
        )
    logger.info("")
    logger.info("변환타입별:")
    for t_type, stats in summary["by_type"].items():
        logger.info(
            f"  {t_type:12s}: C1={stats['case1']} C2={stats['case2']} C3={stats['case3']} "
            f"({stats.get('refusal_rate', 0)}% 비정답)"
        )


if __name__ == "__main__":
    main()
