"""
Run evaluation on transformed samples to test model robustness.

This script:
1. Loads transformed samples from transformation_samples_v2.json
2. Runs COT and POT evaluation on each transformed problem
3. Saves full results including prompts, reasoning, and answers
4. Compares with original problem results
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import FinanceEvaluator
from src.evaluation.prompt_templates import COT_PROMPT_TEMPLATE, POT_PROMPT_TEMPLATE

# Paths
SAMPLES_FILE = Path("experiments/transformation_samples/transformation_samples_v2.json")
OUTPUT_DIR = Path("experiments/transformation_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_transformed_samples() -> Dict[str, List[Dict]]:
    """Load transformed samples."""
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def create_problem_for_evaluation(problem: Dict) -> Dict:
    """
    Convert problem format to match evaluator expectations.

    Evaluator expects:
    - question
    - context
    - ground_truth
    - question_id
    """
    return {
        "question": problem["question"],
        "context": problem["context"],
        "ground_truth": problem["ground_truth"],
        "question_id": problem.get("question_id", "unknown"),
        "level": problem.get("level", "unknown"),
        "python_solution": problem.get("python_solution", ""),
    }


def evaluate_single_problem(
    evaluator: FinanceEvaluator, problem: Dict, method: str, model_name: str
) -> Dict:
    """
    Evaluate a single problem and return full results.

    Returns:
        {
            'question_id': str,
            'method': 'COT' or 'POT',
            'model': str,
            'prompt_template': str,  # The template used
            'full_prompt': str,      # Template + Question + Context
            'raw_response': str,     # Model's complete response
            'extracted_answer': float or None,
            'ground_truth': float,
            'is_correct': bool,
            'error': str or None
        }
    """
    result = {
        "question_id": problem["question_id"],
        "method": method,
        "model": model_name,
        "ground_truth": problem["ground_truth"],
        "error": None,
    }

    try:
        # Get prompt template
        if method == "COT":
            template = COT_PROMPT_TEMPLATE
        else:
            template = POT_PROMPT_TEMPLATE

        result["prompt_template"] = template

        # Construct full prompt
        full_prompt = template.format(
            question=problem["question"], context=problem["context"]
        )
        result["full_prompt"] = full_prompt

        # Run evaluation
        if method == "COT":
            eval_result = evaluator.evaluate_cot([problem], model_name)
        else:
            eval_result = evaluator.evaluate_pot([problem], model_name)

        # Extract results
        if eval_result and len(eval_result) > 0:
            problem_result = eval_result[0]
            result["raw_response"] = problem_result.get("raw_response", "")
            result["extracted_answer"] = problem_result.get("extracted_answer")
            result["is_correct"] = problem_result.get("is_correct", False)

            # For POT, also save executed code
            if method == "POT":
                result["executed_code"] = problem_result.get("executed_code", "")
                result["execution_error"] = problem_result.get("execution_error")
        else:
            result["error"] = "No evaluation result returned"

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    """Run transformation experiment."""

    print("Loading transformed samples...")
    samples = load_transformed_samples()

    # Models to test
    models = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-haiku",
        "claude-sonnet-4",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]

    methods = ["COT", "POT"]

    # Initialize evaluator
    evaluator = FinanceEvaluator()

    # Results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "methods": methods,
            "total_problems": 0,
            "total_transformations": 0,
        },
        "problems": [],
    }

    print("\nRunning evaluations...\n")

    # Process each difficulty level
    for level in ["easy", "medium", "hard"]:
        print(f"=== {level.upper()} Problems ===\n")

        for problem_data in samples[level]:
            original = problem_data["original"]
            transformations = problem_data["transformations"]

            results["metadata"]["total_problems"] += 1
            results["metadata"]["total_transformations"] += len(transformations)

            problem_results = {"original": original, "transformations": []}

            # Evaluate each transformation
            for trans_idx, transformation in enumerate(transformations):
                print(f"Problem: {original['question_id']}")
                print(
                    f"Transformation {trans_idx + 1}/{len(transformations)}: {transformation['transformation_type']}"
                )

                trans_results = {
                    "transformation_info": {
                        "type": transformation["transformation_type"],
                        "description": transformation["transformation_description"],
                        "expected_behavior": transformation["expected_behavior"],
                    },
                    "transformed_problem": transformation,
                    "model_results": [],
                }

                # Prepare problem for evaluation
                eval_problem = create_problem_for_evaluation(transformation)

                # Test each model with each method
                for model in models:
                    for method in methods:
                        print(f"  Testing {model} ({method})...", end=" ")

                        result = evaluate_single_problem(
                            evaluator, eval_problem, method, model
                        )

                        trans_results["model_results"].append(result)

                        if result.get("error"):
                            print(f"ERROR: {result['error']}")
                        else:
                            status = "✓" if result.get("is_correct") else "✗"
                            print(f"{status}")

                problem_results["transformations"].append(trans_results)
                print()

            results["problems"].append(problem_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"transformation_experiment_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Results saved to: {output_file}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total problems: {results['metadata']['total_problems']}")
    print(f"Total transformations: {results['metadata']['total_transformations']}")
    print(
        f"Total evaluations: {len(models) * len(methods) * results['metadata']['total_transformations']}"
    )

    print("\nNext step: Generate visualization from results")


if __name__ == "__main__":
    main()
