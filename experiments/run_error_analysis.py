"""
Run Error Analysis Experiment

This script runs the error analysis experiment on FinanceReasoning dataset,
classifying errors and comparing model types (general vs reasoning).

Usage:
    python run_error_analysis.py
    python run_error_analysis.py --models gpt-4o-mini,claude-sonnet-4
    python run_error_analysis.py --level hard --limit 50
    python run_error_analysis.py --budget economic
"""

import asyncio
import json
import sys
import os
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, ModelConfig
from model_runner import OpenAIProvider, AnthropicProvider, GoogleProvider

from error_analysis import (
    ErrorCategory,
    ErrorClassification,
    ModelType,
    MODEL_REGISTRY,
)
from error_analysis.error_classifier import ErrorClassifier, ClassificationResult
from error_analysis.error_report_generator import (
    ErrorReportGenerator,
    ErrorAnalysisResult,
)
from error_analysis.error_taxonomy import BUDGET_MODEL_SETS


# ============================================================================
# PROMPTS (from paper)
# ============================================================================

COT_SYSTEM_INPUT = """You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, identifying the exact variables and values, and documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. The final answer should be a numeric value."""

COT_PROGRAM_PREFIX_INPUT = """Let's think step by step to answer the given question.\n"""

POT_SYSTEM_INPUT = """You are a financial expert, you are supposed to generate a Python program to answer the given question. The returned value of the program is supposed to be the answer. Here is an example of the Python program:
```python
def solution():
    # Define variables name and value
    revenue = 600000
    avg_account_receivable = 50000

    # Do math calculation to get the answer
    receivables_turnover = revenue / avg_account_receivable
    answer = 365 / receivables_turnover

    # return answer
    return answer
```
"""

POT_PROGRAM_PREFIX_INPUT = """Please generate a Python program to answer the given question. The format of the program should be the following:
```python
def solution():
    # Define variables name and value

    # Do math calculation to get the answer

    # return answer
```

Continue your output:
```python
def solution():
    # Define variables name and value
"""


@dataclass
class ExperimentConfig:
    """Configuration for error analysis experiment"""

    models: List[str]
    methods: List[str]
    level: str  # easy, medium, hard
    limit: Optional[int]
    output_dir: Path


def load_dataset(level: str, limit: Optional[int] = None) -> List[Dict]:
    """Load FinanceReasoning dataset by level"""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data/financereasoning/raw/FinanceReasoning"
    filepath = data_dir / f"{level}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for example in data:
        example["level"] = level

    if limit:
        data = data[:limit]

    return data


def get_available_models(requested_models: List[str]) -> List[str]:
    """Filter models based on available API keys"""

    load_dotenv()
    available = []

    for model_name in requested_models:
        if model_name not in MODEL_REGISTRY:
            print(f"[WARN] Unknown model: {model_name}")
            continue

        model_info = MODEL_REGISTRY[model_name]
        api_key = os.environ.get(model_info.api_key_env)

        if api_key:
            available.append(model_name)
            print(f"[OK] {model_name}: API key found")
        else:
            print(f"[SKIP] {model_name}: No API key ({model_info.api_key_env})")

    return available


class ErrorAnalysisExperiment:
    """Run error analysis experiment"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.classifier = ErrorClassifier()
        self.report_generator = ErrorReportGenerator()

        # Initialize providers
        self.providers = {}
        for model_name in config.models:
            model_info = MODEL_REGISTRY[model_name]

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
                self.providers[model_name] = OpenAIProvider(model_config)
            elif model_info.provider == "anthropic":
                self.providers[model_name] = AnthropicProvider(model_config)
            elif model_info.provider == "google":
                self.providers[model_name] = GoogleProvider(model_config)

    def _build_prompt(self, example: Dict, method: str) -> Tuple[str, str]:
        """Build prompt for example"""

        context = example.get("context", "")
        if context and context != "[]":
            question_input = f"The following question context is provided for your reference.\n{context}\n\nQuestion: {example['question']}\n"
        else:
            question_input = f"Question: {example['question']}\n"

        if method == "COT":
            system_prompt = COT_SYSTEM_INPUT
            user_prompt = question_input + "\n" + COT_PROGRAM_PREFIX_INPUT
        else:  # POT
            system_prompt = POT_SYSTEM_INPUT
            user_prompt = question_input + "\n" + POT_PROGRAM_PREFIX_INPUT

        return system_prompt, user_prompt

    def _extract_cot_answer(self, response: str) -> Any:
        """Extract answer from COT response"""

        patterns = [
            r"[Tt]herefore,?\s*the\s+answer\s+is\s*[:\s]*([+-]?\d+\.?\d*%?)",
            r"[Tt]he\s+answer\s+is\s*[:\s]*([+-]?\d+\.?\d*%?)",
            r"[Ff]inal\s+answer[:\s]*([+-]?\d+\.?\d*%?)",
            r"[Aa]nswer[:\s]*([+-]?\d+\.?\d*%?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                answer_str = match.group(1).replace("%", "").replace(",", "")
                try:
                    return float(answer_str)
                except ValueError:
                    continue

        # Fallback: last number in response
        numbers = re.findall(r"[+-]?\d+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _execute_pot_code(self, response: str) -> Tuple[Any, Optional[str], Optional[str]]:
        """Execute POT code and return (answer, code, error)"""

        code_patterns = [
            r"```python\s*(.*?)```",
            r"```\s*(.*?)```",
        ]

        code = None
        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                break

        if not code:
            code = response.strip()

        # Ensure solution function exists
        if "def solution():" not in code:
            code = "def solution():\n    # Define variables name and value\n" + code

        # Ensure solution is called
        if "solution()" not in code or not re.search(
            r"^\s*solution\(\)", code, re.MULTILINE
        ):
            code = code + "\n\nresult = solution()"

        try:
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)

            if "result" in local_vars:
                return local_vars["result"], code, None
            elif "answer" in local_vars:
                return local_vars["answer"], code, None
            else:
                return None, code, "No 'result' or 'answer' variable found"
        except Exception as e:
            return None, code, str(e)

    def _check_answer(self, pred: Any, truth: Any, tolerance: float = 0.002) -> bool:
        """Check if prediction matches ground truth"""

        if pred is None:
            return False

        try:
            pred_num = float(str(pred).replace(",", "").replace("%", ""))
            truth_num = float(str(truth).replace(",", "").replace("%", ""))

            if truth_num == 0:
                return abs(pred_num) < tolerance

            return abs(pred_num - truth_num) / abs(truth_num) <= tolerance
        except (ValueError, TypeError):
            return str(pred).strip().lower() == str(truth).strip().lower()

    def _calculate_cost(
        self, input_tokens: int, output_tokens: int, model_name: str
    ) -> float:
        """Calculate API cost"""

        model_info = MODEL_REGISTRY[model_name]
        input_cost = (input_tokens / 1_000_000) * model_info.cost_per_million_input
        output_cost = (output_tokens / 1_000_000) * model_info.cost_per_million_output
        return round(input_cost + output_cost, 6)

    async def evaluate_single(
        self, example: Dict, model_name: str, method: str
    ) -> ErrorAnalysisResult:
        """Evaluate single example"""

        provider = self.providers[model_name]
        system_prompt, user_prompt = self._build_prompt(example, method)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Create minimal example object for provider
        class MinimalExample:
            def __init__(self, ex):
                self.id = ex.get("question_id", "unknown")
                self.question = ex.get("question", "")
                self.context = ex.get("context", "")
                self.ground_truth = ex.get("ground_truth", "")
                self.python_solution = ex.get("python_solution", "")

        mini_ex = MinimalExample(example)

        start_time = time.time()
        execution_error = None
        executed_code = None

        try:
            response = await provider.call_model(mini_ex, full_prompt)
            latency = response.response_time_seconds

            raw_response = response.raw_response
            input_tokens = response.prompt_tokens
            output_tokens = response.completion_tokens

            if method == "COT":
                final_answer = self._extract_cot_answer(raw_response)
                parse_status = "success" if final_answer is not None else "failed"
            else:  # POT
                final_answer, executed_code, execution_error = self._execute_pot_code(
                    raw_response
                )
                parse_status = "success" if final_answer is not None else "failed"

            is_correct = self._check_answer(final_answer, example.get("ground_truth"))
            cost = self._calculate_cost(input_tokens, output_tokens, model_name)

            # Classify error if incorrect
            classification = None
            if not is_correct:
                classification_result = ClassificationResult(
                    example_id=example.get("question_id", "unknown"),
                    model_name=model_name,
                    method=method,
                    ground_truth=example.get("ground_truth"),
                    predicted_answer=final_answer,
                    is_correct=is_correct,
                    raw_response=raw_response,
                    context=example.get("context", ""),
                    execution_error=execution_error,
                    executed_code=executed_code,
                    parse_status=parse_status,
                )

                classification = self.classifier.classify(
                    classification_result,
                    example.get("ground_truth"),
                    example.get("context", ""),
                )

            return ErrorAnalysisResult(
                example_id=example.get("question_id", "unknown"),
                model_name=model_name,
                method=method,
                ground_truth=example.get("ground_truth"),
                predicted_answer=final_answer,
                is_correct=is_correct,
                classification=classification,
                raw_response=raw_response,
                cost_usd=cost,
                latency_seconds=latency,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return ErrorAnalysisResult(
                example_id=example.get("question_id", "unknown"),
                model_name=model_name,
                method=method,
                ground_truth=example.get("ground_truth"),
                predicted_answer=None,
                is_correct=False,
                classification=ErrorClassification(
                    category=ErrorCategory.EXECUTION_ERROR,
                    confidence=1.0,
                    evidence=f"API call failed: {str(e)}",
                ),
                raw_response=f"ERROR: {str(e)}",
                cost_usd=0.0,
                latency_seconds=time.time() - start_time,
            )

    async def run(self, examples: List[Dict]) -> List[ErrorAnalysisResult]:
        """Run experiment on all examples"""

        results = []
        total = len(examples) * len(self.config.models) * len(self.config.methods)
        current = 0

        for example in examples:
            example_id = example.get("question_id", "unknown")
            gt = example.get("ground_truth", "")

            print(f"\n{'=' * 80}")
            print(f"[{example_id}] GT={gt} | Level={example.get('level', 'unknown')}")
            print(f"Q: {example.get('question', '')[:100]}...")
            print(f"{'=' * 80}")

            for model_name in self.config.models:
                for method in self.config.methods:
                    current += 1
                    model_info = MODEL_REGISTRY[model_name]
                    print(
                        f"  [{current}/{total}] {model_info.display_name} - {method}...",
                        end=" ",
                        flush=True,
                    )

                    result = await self.evaluate_single(example, model_name, method)
                    results.append(result)

                    status = "[OK]" if result.is_correct else "[X]"
                    error_type = ""
                    if not result.is_correct and result.classification:
                        error_type = f" ({result.classification.category.value})"
                    print(
                        f"{status} Ans={result.predicted_answer}{error_type} (${result.cost_usd:.4f})"
                    )

        return results


def print_summary(results: List[ErrorAnalysisResult]):
    """Print experiment summary"""

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)

    print("\n" + "=" * 90)
    print("ERROR ANALYSIS EXPERIMENT SUMMARY")
    print("=" * 90)

    print(f"\nOverall: {correct}/{total} correct ({correct/total*100:.1f}%)")

    # By model type
    general_results = [
        r
        for r in results
        if r.model_name in MODEL_REGISTRY
        and MODEL_REGISTRY[r.model_name].model_type == ModelType.GENERAL
    ]
    reasoning_results = [
        r
        for r in results
        if r.model_name in MODEL_REGISTRY
        and MODEL_REGISTRY[r.model_name].model_type == ModelType.REASONING
    ]

    if general_results:
        general_correct = sum(1 for r in general_results if r.is_correct)
        print(
            f"General Models: {general_correct}/{len(general_results)} ({general_correct/len(general_results)*100:.1f}%)"
        )

    if reasoning_results:
        reasoning_correct = sum(1 for r in reasoning_results if r.is_correct)
        print(
            f"Reasoning Models: {reasoning_correct}/{len(reasoning_results)} ({reasoning_correct/len(reasoning_results)*100:.1f}%)"
        )

    # Error distribution
    error_dist = {}
    for r in results:
        if not r.is_correct and r.classification:
            cat = r.classification.category.value
            error_dist[cat] = error_dist.get(cat, 0) + 1

    print("\nError Distribution:")
    for cat, count in sorted(error_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Total cost
    total_cost = sum(r.cost_usd for r in results)
    print(f"\nTotal Cost: ${total_cost:.4f}")


async def main():
    parser = argparse.ArgumentParser(description="Run Error Analysis Experiment")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["economic", "balanced", "full"],
        default=None,
        help="Predefined model budget set",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="COT,POT",
        help="Methods to test (COT,POT)",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["easy", "medium", "hard"],
        default="hard",
        help="Dataset difficulty level",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum examples to evaluate",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ERROR ANALYSIS EXPERIMENT")
    print("FinanceReasoning LLM Error Classification")
    print("=" * 80)

    # Determine models to use
    if args.models:
        requested_models = args.models.split(",")
    elif args.budget:
        requested_models = BUDGET_MODEL_SETS[args.budget]
    else:
        requested_models = BUDGET_MODEL_SETS["economic"]

    # Load environment
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    # Get available models
    print("\n[1] Checking API keys...")
    available_models = get_available_models(requested_models)

    if not available_models:
        print("ERROR: No models available. Check your API keys.")
        return

    # Load dataset
    print(f"\n[2] Loading {args.level} dataset (limit={args.limit})...")
    examples = load_dataset(args.level, args.limit)
    print(f"    Loaded {len(examples)} examples")

    # Create config
    output_dir = project_root / "experiments" / "results" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        models=available_models,
        methods=args.methods.split(","),
        level=args.level,
        limit=args.limit,
        output_dir=output_dir,
    )

    print(f"\n[3] Experiment Configuration:")
    print(f"    Models: {config.models}")
    print(f"    Methods: {config.methods}")
    print(f"    Level: {config.level}")
    print(f"    Examples: {len(examples)}")

    # Run experiment
    print(f"\n[4] Running experiment...")
    experiment = ErrorAnalysisExperiment(config)
    results = await experiment.run(examples)

    # Generate reports
    print(f"\n[5] Generating reports...")
    report_generator = ErrorReportGenerator()
    report_paths = report_generator.generate_reports(
        results, output_dir, f"error_analysis_{args.level}"
    )

    print(f"    JSON: {report_paths['json']}")
    print(f"    HTML: {report_paths['html']}")

    # Print summary
    print_summary(results)

    print(f"\n[OK] Experiment complete!")
    print(f"    Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
