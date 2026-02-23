"""
Quick Evaluation - FinanceReasoning Paper Methodology

Evaluates 5 selected hard examples using both COT and POT approaches.
Follows the paper's evaluation methodology:
1. Final Answer Accuracy
2. Step Completeness
3. Step Order Correctness
4. Reasoning Similarity (Jaccard Index)
5. Hallucination Rate
6. Overall Reasoning Score

Usage:
    python experiments/quick_evaluation.py
"""

import asyncio
import json
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, EvaluationConfig
from dataset_loader import DatasetLoader, Example
from model_runner import ModelRunner, LLMResponse
from prompt_builder import PromptBuilder, PROMPT_TEMPLATES
from response_parser import ResponseParser
from metrics_evaluator import MetricsEvaluator, EvaluationMetrics
from pot_executor import evaluate_pot_response, CodeExecutionSandbox


@dataclass
class StepDetail:
    """Detailed step information for visualization"""

    step_number: int
    step_type: str
    description: str
    code_snippet: str
    matches_ground_truth: bool = False


@dataclass
class ExampleEvaluation:
    """Complete evaluation result for a single example"""

    example_id: str
    question: str
    context_preview: str
    ground_truth: Any
    ground_truth_solution: str

    # COT Results
    cot_answer: Optional[Any] = None
    cot_correct: bool = False
    cot_steps: List[StepDetail] = field(default_factory=list)
    cot_metrics: Optional[Dict[str, float]] = None
    cot_raw_response: str = ""
    cot_cost_usd: float = 0.0
    cot_time_seconds: float = 0.0

    # POT Results
    pot_answer: Optional[Any] = None
    pot_correct: bool = False
    pot_code: str = ""
    pot_execution_success: bool = False
    pot_execution_output: Any = None
    pot_metrics: Optional[Dict[str, float]] = None
    pot_raw_response: str = ""
    pot_cost_usd: float = 0.0
    pot_time_seconds: float = 0.0


@dataclass
class QuickEvaluationResult:
    """Complete result from quick evaluation"""

    timestamp: str
    model_id: str
    num_examples: int
    examples: List[ExampleEvaluation]

    # Aggregate metrics
    cot_accuracy: float = 0.0
    pot_accuracy: float = 0.0
    cot_avg_overall_score: float = 0.0
    pot_avg_overall_score: float = 0.0
    cot_avg_step_completeness: float = 0.0
    pot_avg_step_completeness: float = 0.0
    cot_avg_hallucination_rate: float = 0.0
    pot_avg_hallucination_rate: float = 0.0

    total_cost_usd: float = 0.0
    total_time_seconds: float = 0.0


def select_diverse_samples(
    examples: List[Example], num_samples: int = 5
) -> List[Example]:
    """
    Select diverse samples from hard.json covering different question types.

    Criteria for diversity:
    1. Different source types (CodeFinQA, FinanceReasoning, FinanceMath)
    2. Different complexity levels (based on difficulty score)
    3. Different operation types (calculation, conditional, etc.)
    """
    if len(examples) <= num_samples:
        return examples

    # Group by source type
    by_source = {}
    for ex in examples:
        source_type = (
            ex.source.split("-")[0]
            if hasattr(ex, "source") and ex.source
            else "unknown"
        )
        if source_type not in by_source:
            by_source[source_type] = []
        by_source[source_type].append(ex)

    # Select from each source type
    selected = []
    sources = list(by_source.keys())

    # Prioritize diversity across sources
    for i in range(num_samples):
        source = sources[i % len(sources)]
        if by_source[source]:
            # Sort by difficulty and pick from different difficulty levels
            sorted_examples = sorted(
                by_source[source], key=lambda x: getattr(x, "difficulty", 0)
            )
            idx = min(i // len(sources), len(sorted_examples) - 1)
            selected.append(sorted_examples[idx])
            by_source[source].remove(sorted_examples[idx])

    # If still need more, fill randomly
    remaining = [ex for exs in by_source.values() for ex in exs]
    while len(selected) < num_samples and remaining:
        ex = random.choice(remaining)
        selected.append(ex)
        remaining.remove(ex)

    return selected[:num_samples]


class QuickEvaluator:
    """
    Evaluates examples using FinanceReasoning paper methodology.

    Implements both COT and POT evaluation with detailed step tracking.
    """

    def __init__(self, model_id: Optional[str] = None):
        # Load environment
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # Initialize components
        self.config_manager = ConfigManager()
        self.eval_config = self.config_manager.get_evaluation_config()

        # Find available model with API key
        import os

        self.available_models = [
            m
            for m in self.eval_config.models
            if m.api_key_env_var and os.environ.get(m.api_key_env_var)
        ]

        if not self.available_models:
            raise ValueError(
                "No models with API keys found. Please configure .env file."
            )

        # Select model
        if model_id:
            self.model = next(
                (m for m in self.available_models if m.id == model_id), None
            )
            if not self.model:
                raise ValueError(
                    f"Model {model_id} not found or no API key configured."
                )
        else:
            # Default to first available model
            self.model = self.available_models[0]

        print(f"[INFO] Using model: {self.model.id}")

        # Initialize components
        self.cot_prompt_builder = PromptBuilder("finance_reasoning_compliant")
        self.pot_prompt_builder = PromptBuilder("pot")
        self.response_parser = ResponseParser()
        self.metrics_evaluator = MetricsEvaluator()
        self.pot_executor = CodeExecutionSandbox(timeout_seconds=10)

        # Results storage
        self.results_dir = project_root / "experiments" / "quick_results"
        self.results_dir.mkdir(exist_ok=True)

    async def evaluate_cot(
        self, example: Example, model_runner: ModelRunner
    ) -> Dict[str, Any]:
        """Evaluate using Chain-of-Thought methodology"""
        print(f"    [COT] Evaluating...")

        responses = await model_runner.evaluate_example(
            example=example,
            template_type="finance_reasoning_compliant",
        )

        if not responses:
            return {"error": "No response received"}

        response = list(responses.values())[0]

        # Parse response
        parsed = self.response_parser.parse_response(
            response.parsed_data,
            provider_type="anthropic"
            if "claude" in self.model.id.lower()
            else "openai",
        )

        # Update response with parsed final_answer
        response.final_answer = parsed.final_answer

        # Calculate metrics
        metrics = self.metrics_evaluator.evaluate_example(
            example=example,
            llm_responses=responses,
        )

        model_metrics = list(metrics.values())[0] if metrics else None

        return {
            "answer": parsed.final_answer,
            "correct": model_metrics.final_answer_correct if model_metrics else False,
            "steps": [
                StepDetail(
                    step_number=s.step_number,
                    step_type=s.step_type,
                    description=s.description,
                    code_snippet=s.code_snippet,
                )
                for s in parsed.reasoning_steps
            ],
            "metrics": {
                "final_answer_correct": model_metrics.final_answer_correct
                if model_metrics
                else False,
                "step_completeness": model_metrics.step_completeness
                if model_metrics
                else 0.0,
                "step_order_score": model_metrics.step_order_score
                if model_metrics
                else 0.0,
                "reasoning_similarity": model_metrics.reasoning_similarity
                if model_metrics
                else 0.0,
                "hallucination_rate": model_metrics.hallucination_rate
                if model_metrics
                else 0.0,
                "overall_score": model_metrics.overall_reasoning_score
                if model_metrics
                else 0.0,
            },
            "raw_response": response.raw_response,
            "cost_usd": response.cost_usd,
            "time_seconds": response.response_time_seconds,
        }

    async def evaluate_pot(
        self, example: Example, model_runner: ModelRunner
    ) -> Dict[str, Any]:
        """Evaluate using Program-of-Thought methodology"""
        print(f"    [POT] Evaluating...")

        responses = await model_runner.evaluate_example(
            example=example,
            template_type="pot",
        )

        if not responses:
            return {"error": "No response received"}

        response = list(responses.values())[0]

        # Extract and execute code
        exec_result = evaluate_pot_response(
            response.raw_response,
            timeout_seconds=10,
        )

        # Determine correctness
        is_correct = False
        if exec_result.success and exec_result.output is not None:
            try:
                gt_num = float(str(example.ground_truth_final).replace(",", ""))
                llm_num = float(exec_result.output)
                is_correct = abs(gt_num - llm_num) < 0.01  # Allow small tolerance
            except (ValueError, TypeError):
                pass

        return {
            "answer": exec_result.output if exec_result.success else None,
            "correct": is_correct,
            "code": exec_result.code if hasattr(exec_result, "code") else "",
            "execution_success": exec_result.success,
            "execution_output": exec_result.output,
            "execution_error": exec_result.error,
            "metrics": {
                "final_answer_correct": is_correct,
                "execution_success": exec_result.success,
            },
            "raw_response": response.raw_response,
            "cost_usd": response.cost_usd,
            "time_seconds": response.response_time_seconds,
        }

    async def evaluate_examples(self, examples: List[Example]) -> QuickEvaluationResult:
        """Evaluate all examples with both COT and POT"""

        print(f"\n{'=' * 70}")
        print(f"Quick Evaluation - {len(examples)} Examples")
        print(f"Model: {self.model.id}")
        print(f"{'=' * 70}")

        # Initialize model runner
        test_config = EvaluationConfig(
            models=[self.model],
            output_dir=str(self.results_dir),
            max_cost_usd=5.0,
            concurrency_per_provider=1,
        )
        model_runner = ModelRunner(self.config_manager, test_config)

        evaluations = []
        total_cost = 0.0
        total_time = 0.0

        for i, example in enumerate(examples, 1):
            print(f"\n[{i}/{len(examples)}] Example: {example.id}")
            print(f"  Question: {example.question[:80]}...")
            print(f"  Ground Truth: {example.ground_truth_final}")

            eval_result = ExampleEvaluation(
                example_id=example.id,
                question=example.question,
                context_preview=example.context[:200] + "..."
                if len(example.context) > 200
                else example.context,
                ground_truth=example.ground_truth_final,
                ground_truth_solution=example.python_solution or "",
            )

            # COT Evaluation
            try:
                cot_result = await self.evaluate_cot(example, model_runner)
                eval_result.cot_answer = cot_result.get("answer")
                eval_result.cot_correct = cot_result.get("correct", False)
                eval_result.cot_steps = cot_result.get("steps", [])
                eval_result.cot_metrics = cot_result.get("metrics")
                eval_result.cot_raw_response = cot_result.get("raw_response", "")
                eval_result.cot_cost_usd = cot_result.get("cost_usd", 0.0)
                eval_result.cot_time_seconds = cot_result.get("time_seconds", 0.0)

                total_cost += eval_result.cot_cost_usd
                total_time += eval_result.cot_time_seconds

                status = "[OK]" if eval_result.cot_correct else "[X]"
                print(
                    f"    COT: {status} Answer={eval_result.cot_answer} (Cost: ${eval_result.cot_cost_usd:.4f})"
                )
            except Exception as e:
                print(f"    COT: Error - {e}")

            # POT Evaluation
            try:
                pot_result = await self.evaluate_pot(example, model_runner)
                eval_result.pot_answer = pot_result.get("answer")
                eval_result.pot_correct = pot_result.get("correct", False)
                eval_result.pot_code = pot_result.get("code", "")
                eval_result.pot_execution_success = pot_result.get(
                    "execution_success", False
                )
                eval_result.pot_execution_output = pot_result.get("execution_output")
                eval_result.pot_metrics = pot_result.get("metrics")
                eval_result.pot_raw_response = pot_result.get("raw_response", "")
                eval_result.pot_cost_usd = pot_result.get("cost_usd", 0.0)
                eval_result.pot_time_seconds = pot_result.get("time_seconds", 0.0)

                total_cost += eval_result.pot_cost_usd
                total_time += eval_result.pot_time_seconds

                status = "[OK]" if eval_result.pot_correct else "[X]"
                exec_status = (
                    "executed" if eval_result.pot_execution_success else "failed"
                )
                print(
                    f"    POT: {status} Answer={eval_result.pot_answer} ({exec_status}, Cost: ${eval_result.pot_cost_usd:.4f})"
                )
            except Exception as e:
                print(f"    POT: Error - {e}")

            evaluations.append(eval_result)

        # Calculate aggregate metrics
        cot_correct = sum(1 for e in evaluations if e.cot_correct)
        pot_correct = sum(1 for e in evaluations if e.pot_correct)

        cot_scores = [
            e.cot_metrics.get("overall_score", 0) for e in evaluations if e.cot_metrics
        ]
        pot_scores = [
            e.pot_metrics.get("final_answer_correct", 0)
            for e in evaluations
            if e.pot_metrics
        ]

        cot_completeness = [
            e.cot_metrics.get("step_completeness", 0)
            for e in evaluations
            if e.cot_metrics
        ]
        cot_hallucination = [
            e.cot_metrics.get("hallucination_rate", 0)
            for e in evaluations
            if e.cot_metrics
        ]

        result = QuickEvaluationResult(
            timestamp=datetime.now().isoformat(),
            model_id=self.model.id,
            num_examples=len(examples),
            examples=evaluations,
            cot_accuracy=cot_correct / len(examples) if examples else 0,
            pot_accuracy=pot_correct / len(examples) if examples else 0,
            cot_avg_overall_score=sum(cot_scores) / len(cot_scores)
            if cot_scores
            else 0,
            pot_avg_overall_score=sum(pot_scores) / len(pot_scores)
            if pot_scores
            else 0,
            cot_avg_step_completeness=sum(cot_completeness) / len(cot_completeness)
            if cot_completeness
            else 0,
            cot_avg_hallucination_rate=sum(cot_hallucination) / len(cot_hallucination)
            if cot_hallucination
            else 0,
            total_cost_usd=total_cost,
            total_time_seconds=total_time,
        )

        return result

    def save_results(self, result: QuickEvaluationResult) -> Path:
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_eval_{result.model_id}_{timestamp}.json"
        filepath = self.results_dir / filename

        # Convert to dict for JSON serialization
        result_dict = {
            "timestamp": result.timestamp,
            "model_id": result.model_id,
            "num_examples": result.num_examples,
            "aggregate_metrics": {
                "cot_accuracy": result.cot_accuracy,
                "pot_accuracy": result.pot_accuracy,
                "cot_avg_overall_score": result.cot_avg_overall_score,
                "pot_avg_overall_score": result.pot_avg_overall_score,
                "cot_avg_step_completeness": result.cot_avg_step_completeness,
                "cot_avg_hallucination_rate": result.cot_avg_hallucination_rate,
                "total_cost_usd": result.total_cost_usd,
                "total_time_seconds": result.total_time_seconds,
            },
            "examples": [],
        }

        for ex in result.examples:
            ex_dict = {
                "example_id": ex.example_id,
                "question": ex.question,
                "context_preview": ex.context_preview,
                "ground_truth": ex.ground_truth,
                "ground_truth_solution": ex.ground_truth_solution,
                "cot": {
                    "answer": ex.cot_answer,
                    "correct": ex.cot_correct,
                    "steps": [
                        asdict(s) if hasattr(s, "__dict__") else s for s in ex.cot_steps
                    ]
                    if ex.cot_steps
                    else [],
                    "metrics": ex.cot_metrics,
                    "raw_response": ex.cot_raw_response,
                    "cost_usd": ex.cot_cost_usd,
                    "time_seconds": ex.cot_time_seconds,
                },
                "pot": {
                    "answer": ex.pot_answer,
                    "correct": ex.pot_correct,
                    "code": ex.pot_code,
                    "execution_success": ex.pot_execution_success,
                    "execution_output": ex.pot_execution_output,
                    "metrics": ex.pot_metrics,
                    "raw_response": ex.pot_raw_response,
                    "cost_usd": ex.pot_cost_usd,
                    "time_seconds": ex.pot_time_seconds,
                },
            }
            result_dict["examples"].append(ex_dict)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[OK] Results saved to: {filepath}")
        return filepath

    def print_summary(self, result: QuickEvaluationResult):
        """Print evaluation summary"""
        print(f"\n{'=' * 70}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Model: {result.model_id}")
        print(f"Examples: {result.num_examples}")
        print(f"Total Cost: ${result.total_cost_usd:.4f}")
        print(f"Total Time: {result.total_time_seconds:.2f}s")

        print(f"\n{'-' * 40}")
        print("ACCURACY COMPARISON")
        print(f"{'-' * 40}")
        print(
            f"  COT Accuracy: {result.cot_accuracy * 100:.1f}% ({int(result.cot_accuracy * result.num_examples)}/{result.num_examples})"
        )
        print(
            f"  POT Accuracy: {result.pot_accuracy * 100:.1f}% ({int(result.pot_accuracy * result.num_examples)}/{result.num_examples})"
        )

        print(f"\n{'-' * 40}")
        print("COT DETAILED METRICS (Paper Methodology)")
        print(f"{'-' * 40}")
        print(f"  Overall Score: {result.cot_avg_overall_score:.3f}")
        print(f"  Step Completeness: {result.cot_avg_step_completeness * 100:.1f}%")
        print(f"  Hallucination Rate: {result.cot_avg_hallucination_rate * 100:.1f}%")

        print(f"\n{'-' * 40}")
        print("PER-EXAMPLE RESULTS")
        print(f"{'-' * 40}")
        for ex in result.examples:
            cot_status = "[OK]" if ex.cot_correct else "[X]"
            pot_status = "[OK]" if ex.pot_correct else "[X]"
            print(
                f"  {ex.example_id}: COT={cot_status} POT={pot_status} (GT={ex.ground_truth})"
            )


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Quick FinanceReasoning Evaluation")
    parser.add_argument("--model", "-m", type=str, help="Model ID to use")
    parser.add_argument(
        "--num", "-n", type=int, default=5, help="Number of examples (default: 5)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load dataset
    print("[1] Loading hard.json dataset...")
    data_root = Path(__file__).parent.parent / "data" / "financereasoning"
    dataset_loader = DatasetLoader(data_root)
    all_examples = dataset_loader.load_dataset(
        dataset_name="financereasoning", split="hard"
    )

    print(f"    Loaded {len(all_examples)} hard examples")

    # Select diverse samples
    print(f"[2] Selecting {args.num} diverse samples...")
    selected_examples = select_diverse_samples(all_examples, args.num)

    for i, ex in enumerate(selected_examples, 1):
        source = getattr(ex, "source", "unknown")
        print(f"    {i}. {ex.id} (source: {source})")

    # Initialize evaluator
    print("[3] Initializing evaluator...")
    evaluator = QuickEvaluator(model_id=args.model)

    # Run evaluation
    print("[4] Running evaluation...")
    result = await evaluator.evaluate_examples(selected_examples)

    # Save results
    print("[5] Saving results...")
    results_file = evaluator.save_results(result)

    # Print summary
    evaluator.print_summary(result)

    print(f"\n{'=' * 70}")
    print("NEXT STEPS")
    print(f"{'=' * 70}")
    print(f"1. View detailed results: {results_file}")
    print(
        f"2. Generate visualization: python experiments/visualization.py {results_file}"
    )
    print(f"3. Run full evaluation: python evaluation/main.py run <experiment_name>")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
