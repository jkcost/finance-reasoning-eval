"""
Failure Case Experiment - Test on hardest problems from FinanceReasoning

Based on paper's error analysis (Appendix B), models struggle with:
1. Incorrect Formula Application - wrong formulas for financial calculations
2. Imprecise Calculations - numerical precision errors
3. Complex multi-step reasoning

This script:
1. Identifies the hardest problems from hard.json (highest difficulty scores)
2. Includes problems known to fail across multiple models (from our experiments)
3. Runs COT and POT evaluation
4. Generates detailed failure analysis
"""

import asyncio
import json
import sys
import os
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, ModelConfig
from model_runner import OpenAIProvider, AnthropicProvider, GoogleProvider


# ============================================================================
# OFFICIAL PAPER PROMPTS (from GitHub: BUPT-Reasoning/FinanceReasoning)
# ============================================================================

COT_SYSTEM_INPUT = """You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, identifying the exact variables and values, and documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. The final answer should be a numeric value."""

COT_PROGRAM_PREFIX_INPUT = (
    """Let's think step by step to answer the given question.\n"""
)

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


# ============================================================================
# MODEL CONFIGURATIONS (Budget-friendly models)
# ============================================================================

BUDGET_MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "api_key_env": "OPENAI_API_KEY",
        "cost_per_million_input": 0.15,
        "cost_per_million_output": 0.60,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "api_key_env": "OPENAI_API_KEY",
        "cost_per_million_input": 2.50,
        "cost_per_million_output": 10.00,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "display_name": "Claude 3 Haiku",
        "api_key_env": "ANTHROPIC_API_KEY",
        "cost_per_million_input": 0.25,
        "cost_per_million_output": 1.25,
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "api_key_env": "ANTHROPIC_API_KEY",
        "cost_per_million_input": 3.00,
        "cost_per_million_output": 15.00,
    },
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "model_id": "gemini-2.0-flash-lite",
        "display_name": "Gemini 2.0 Flash Lite",
        "api_key_env": "GOOGLE_API_KEY",
        "cost_per_million_input": 0.075,
        "cost_per_million_output": 0.30,
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "model_id": "gemini-2.0-flash-exp",
        "display_name": "Gemini 2.0 Flash",
        "api_key_env": "GOOGLE_API_KEY",
        "cost_per_million_input": 0.10,
        "cost_per_million_output": 0.40,
    },
}


@dataclass
class ExperimentResult:
    """Result from a single model evaluation"""

    model_name: str
    provider: str
    method: str
    example_id: str
    question: str
    ground_truth: Any
    final_answer: Any
    is_correct: bool
    raw_response: str
    system_prompt: str
    user_prompt: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    executed_code: Optional[str] = None
    execution_error: Optional[str] = None
    difficulty: Optional[float] = None
    error_type: Optional[str] = None  # For failure analysis


def select_failure_cases(data: List[Dict], num_samples: int = 10) -> List[Dict]:
    """
    Select failure case examples from hard.json:
    1. Highest difficulty scores (complex multi-step reasoning)
    2. Known problematic question types based on paper's error analysis
    """
    # Sort by difficulty (descending)
    sorted_data = sorted(data, key=lambda x: x.get("difficulty", 0), reverse=True)

    # Take top N most difficult
    selected = sorted_data[:num_samples]

    # Ensure we include test-2001 (annualized return) - known failure case
    test_2001 = next((x for x in data if x.get("question_id") == "test-2001"), None)
    if test_2001 and test_2001 not in selected:
        selected[-1] = test_2001

    return selected


def categorize_error(pred: Any, truth: Any, question: str, response: str) -> str:
    """Categorize the type of error based on paper's error taxonomy"""
    if pred is None:
        return "Execution Error"

    try:
        pred_num = float(str(pred).replace(",", "").replace("%", ""))
        truth_num = float(str(truth).replace(",", "").replace("%", ""))

        # Check if close but not within tolerance (precision error)
        if truth_num != 0:
            error_ratio = abs(pred_num - truth_num) / abs(truth_num)
            if error_ratio <= 0.1:  # Within 10% - likely precision error
                return "Imprecise Calculation"
            elif error_ratio <= 0.5:  # Within 50% - might be formula variant
                return "Partial Formula Error"
            else:
                return "Incorrect Formula Application"
        else:
            if abs(pred_num) < 1:
                return "Imprecise Calculation"
            else:
                return "Incorrect Formula Application"
    except:
        return "Unknown Error"


class FailureCaseExperiment:
    """Run experiments on failure case examples"""

    def __init__(self, models: List[str] = None):
        project_root = Path(__file__).parent.parent
        load_dotenv(project_root / ".env")

        available = []
        for model_name, config in BUDGET_MODELS.items():
            api_key = os.environ.get(config["api_key_env"])
            if api_key:
                available.append(model_name)
                print(f"[OK] {model_name}: API key found")
            else:
                print(f"[SKIP] {model_name}: No API key")

        self.models = models if models else available
        self.models = [m for m in self.models if m in available]

        if not self.models:
            raise ValueError("No models available")

        self.providers = {}
        for model_name in self.models:
            config = BUDGET_MODELS[model_name]
            model_config = ModelConfig(
                id=model_name,
                name=config["display_name"],
                provider=config["provider"],
                model_id=config["model_id"],
                api_key_env_var=config["api_key_env"],
                max_tokens=4096,
                temperature=0.0,
                cost_per_million_tokens=config["cost_per_million_output"],
            )

            if config["provider"] == "openai":
                self.providers[model_name] = OpenAIProvider(model_config)
            elif config["provider"] == "anthropic":
                self.providers[model_name] = AnthropicProvider(model_config)
            elif config["provider"] == "google":
                self.providers[model_name] = GoogleProvider(model_config)

    def _build_prompt(self, example: Dict, method: str) -> tuple:
        if example.get("context"):
            context = example["context"]
            if context != "[]":
                question_input = f"The following question context is provided for your reference.\n{context}\n\nQuestion: {example['question']}\n"
            else:
                question_input = f"Question: {example['question']}\n"
        else:
            question_input = f"Question: {example['question']}\n"

        if method == "COT":
            system_prompt = COT_SYSTEM_INPUT
            user_prompt = question_input + "\n" + COT_PROGRAM_PREFIX_INPUT
        else:
            system_prompt = POT_SYSTEM_INPUT
            user_prompt = question_input + "\n" + POT_PROGRAM_PREFIX_INPUT

        return system_prompt, user_prompt

    def _extract_cot_answer(self, response: str) -> Any:
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

        numbers = re.findall(r"[+-]?\d+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _execute_pot_code(self, response: str) -> tuple:
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

        if "def solution():" not in code:
            code = "def solution():\n    # Define variables name and value\n" + code

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
        config = BUDGET_MODELS[model_name]
        input_cost = (input_tokens / 1_000_000) * config["cost_per_million_input"]
        output_cost = (output_tokens / 1_000_000) * config["cost_per_million_output"]
        return round(input_cost + output_cost, 6)

    async def evaluate_single(
        self, example: Dict, model_name: str, method: str
    ) -> ExperimentResult:
        config = BUDGET_MODELS[model_name]
        provider = self.providers[model_name]

        system_prompt, user_prompt = self._build_prompt(example, method)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        class MinimalExample:
            def __init__(self, ex):
                self.id = ex.get("question_id", "unknown")
                self.question = ex.get("question", "")
                self.context = ex.get("context", "")
                self.ground_truth = ex.get("ground_truth", "")
                self.python_solution = ex.get("python_solution", "")

        mini_ex = MinimalExample(example)

        start_time = time.time()
        try:
            response = await provider.call_model(mini_ex, full_prompt)
            latency = response.response_time_seconds

            raw_response = response.raw_response
            input_tokens = response.prompt_tokens
            output_tokens = response.completion_tokens

            if method == "COT":
                final_answer = self._extract_cot_answer(raw_response)
                executed_code = None
                execution_error = None
            else:
                final_answer, executed_code, execution_error = self._execute_pot_code(
                    raw_response
                )

            is_correct = self._check_answer(final_answer, example.get("ground_truth"))
            cost = self._calculate_cost(input_tokens, output_tokens, model_name)

            # Categorize error if incorrect
            error_type = None
            if not is_correct:
                error_type = categorize_error(
                    final_answer,
                    example.get("ground_truth"),
                    example.get("question", ""),
                    raw_response,
                )

            return ExperimentResult(
                model_name=model_name,
                provider=config["provider"],
                method=method,
                example_id=example.get("question_id", "unknown"),
                question=example.get("question", ""),
                ground_truth=example.get("ground_truth"),
                final_answer=final_answer,
                is_correct=is_correct,
                raw_response=raw_response,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_seconds=latency,
                executed_code=executed_code,
                execution_error=execution_error,
                difficulty=example.get("difficulty"),
                error_type=error_type,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return ExperimentResult(
                model_name=model_name,
                provider=config["provider"],
                method=method,
                example_id=example.get("question_id", "unknown"),
                question=example.get("question", ""),
                ground_truth=example.get("ground_truth"),
                final_answer=None,
                is_correct=False,
                raw_response=f"ERROR: {str(e)}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_seconds=time.time() - start_time,
                executed_code=None,
                execution_error=str(e),
                difficulty=example.get("difficulty"),
                error_type="API Error",
            )

    async def run_experiment(
        self, examples: List[Dict], methods: List[str] = None
    ) -> List[ExperimentResult]:
        methods = methods or ["COT", "POT"]
        results = []

        total = len(examples) * len(self.models) * len(methods)
        current = 0

        for example in examples:
            example_id = example.get("question_id", "unknown")
            gt = example.get("ground_truth", "")
            difficulty = example.get("difficulty", 0)

            print(f"\n{'=' * 70}")
            print(f"[{example_id}] GT={gt} | Difficulty={difficulty:.2f}")
            print(f"Q: {example.get('question', '')[:80]}...")
            print(f"{'=' * 70}")

            for model_name in self.models:
                for method in methods:
                    current += 1
                    display = BUDGET_MODELS[model_name]["display_name"]
                    print(
                        f"  [{current}/{total}] {display} - {method}...",
                        end=" ",
                        flush=True,
                    )

                    result = await self.evaluate_single(example, model_name, method)
                    results.append(result)

                    status = "[OK]" if result.is_correct else f"[X:{result.error_type}]"
                    print(
                        f"{status} Ans={result.final_answer} (${result.cost_usd:.4f}, {result.latency_seconds:.1f}s)"
                    )

        return results


def save_results(
    results: List[ExperimentResult], output_dir: Path, prefix: str = "failure_case"
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{prefix}_{timestamp}.json"

    # Organize data
    data = {
        "timestamp": timestamp,
        "methodology": "FinanceReasoning Failure Case Analysis",
        "description": "Testing on hardest problems and known failure cases",
        "results": [asdict(r) for r in results],
        "summary": {},
        "error_analysis": {},
    }

    # Calculate summary
    summary = {}
    error_counts = {}

    for r in results:
        key = f"{r.model_name}_{r.method}"
        if key not in summary:
            summary[key] = {
                "model": r.model_name,
                "method": r.method,
                "total": 0,
                "correct": 0,
                "total_cost": 0.0,
                "total_latency": 0.0,
            }
        summary[key]["total"] += 1
        summary[key]["correct"] += 1 if r.is_correct else 0
        summary[key]["total_cost"] += r.cost_usd
        summary[key]["total_latency"] += r.latency_seconds

        # Error analysis
        if not r.is_correct and r.error_type:
            if r.error_type not in error_counts:
                error_counts[r.error_type] = 0
            error_counts[r.error_type] += 1

    for key in summary:
        total = summary[key]["total"]
        if total > 0:
            summary[key]["accuracy"] = round(summary[key]["correct"] / total * 100, 1)
            summary[key]["avg_cost"] = round(summary[key]["total_cost"] / total, 6)
            summary[key]["avg_latency"] = round(
                summary[key]["total_latency"] / total, 2
            )

    data["summary"] = summary
    data["error_analysis"] = error_counts

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def print_summary(results: List[ExperimentResult]):
    # Build summary
    summary = {}
    error_counts = {}

    for r in results:
        key = f"{r.model_name}_{r.method}"
        if key not in summary:
            summary[key] = {
                "model": r.model_name,
                "method": r.method,
                "total": 0,
                "correct": 0,
                "cost": 0.0,
                "latency": 0.0,
            }
        summary[key]["total"] += 1
        summary[key]["correct"] += 1 if r.is_correct else 0
        summary[key]["cost"] += r.cost_usd
        summary[key]["latency"] += r.latency_seconds

        if not r.is_correct and r.error_type:
            if r.error_type not in error_counts:
                error_counts[r.error_type] = 0
            error_counts[r.error_type] += 1

    print("\n" + "=" * 90)
    print("FAILURE CASE EXPERIMENT SUMMARY")
    print("=" * 90)

    for method in ["COT", "POT"]:
        print(f"\n{method} Results:")
        print("-" * 80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Avg Cost':<15} {'Avg Latency':<12}")
        print("-" * 80)

        for key, stats in sorted(summary.items()):
            if stats["method"] == method:
                acc = (
                    stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                )
                avg_cost = stats["cost"] / stats["total"] if stats["total"] > 0 else 0
                avg_lat = stats["latency"] / stats["total"] if stats["total"] > 0 else 0
                print(
                    f"{stats['model']:<25} {acc:>6.1f}%      ${avg_cost:<13.6f} {avg_lat:>6.1f}s"
                )

    # Error type analysis
    print("\n" + "=" * 90)
    print("ERROR TYPE ANALYSIS")
    print("=" * 90)
    total_errors = sum(error_counts.values())
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = count / total_errors * 100 if total_errors > 0 else 0
        print(f"  {error_type:<30}: {count:>4} ({pct:>5.1f}%)")

    # Per-example breakdown
    print("\n" + "=" * 90)
    print("PER-EXAMPLE RESULTS")
    print("=" * 90)

    by_example = {}
    for r in results:
        if r.example_id not in by_example:
            by_example[r.example_id] = {
                "gt": r.ground_truth,
                "difficulty": r.difficulty,
                "results": [],
            }
        by_example[r.example_id]["results"].append(r)

    for ex_id, data in by_example.items():
        correct_count = sum(1 for r in data["results"] if r.is_correct)
        total_count = len(data["results"])
        print(
            f"\n[{ex_id}] GT={data['gt']} | Difficulty={data['difficulty']:.2f} | Correct={correct_count}/{total_count}"
        )
        for r in data["results"]:
            status = "[OK]" if r.is_correct else f"[X:{r.error_type}]"
            print(f"  {r.model_name:<22} {r.method}: {status} {r.final_answer}")


async def async_main():
    import argparse

    parser = argparse.ArgumentParser(description="Failure Case Experiment")
    parser.add_argument(
        "--num", type=int, default=10, help="Number of failure cases to test"
    )
    parser.add_argument(
        "--models", type=str, default=None, help="Comma-separated models"
    )
    parser.add_argument("--methods", type=str, default="COT,POT", help="Methods")
    args = parser.parse_args()

    # Load dataset
    data_path = (
        Path(__file__).parent.parent
        / "data/financereasoning/raw/FinanceReasoning/hard.json"
    )
    print(f"[1] Loading dataset from {data_path}...")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Select failure cases
    print(f"[2] Selecting {args.num} hardest/failure cases...")
    examples = select_failure_cases(data, args.num)

    print(f"    Selected {len(examples)} examples:")
    for i, ex in enumerate(examples):
        print(
            f"    {i + 1}. {ex.get('question_id', 'unknown')} - GT: {ex.get('ground_truth')} | Difficulty: {ex.get('difficulty', 0):.2f}"
        )

    # Parse args
    models = args.models.split(",") if args.models else None
    methods = args.methods.split(",")

    # Run experiment
    print(f"\n[3] Initializing experiment...")
    experiment = FailureCaseExperiment(models=models)
    print(f"    Models: {experiment.models}")
    print(f"    Methods: {methods}")

    print(f"\n[4] Running experiment...")
    results = await experiment.run_experiment(examples, methods=methods)

    # Save
    output_dir = Path(__file__).parent / "quick_results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n[5] Saving results...")
    filepath = save_results(results, output_dir)
    print(f"    Saved to: {filepath}")

    # Print summary
    print_summary(results)

    print(f"\n[OK] Experiment complete!")
    print(f"    Results: {filepath}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
