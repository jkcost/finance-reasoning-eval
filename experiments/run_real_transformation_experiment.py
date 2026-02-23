"""
Run REAL transformation experiment with actual API calls.

Based on paper_error_cases_experiment.py pattern.
"""

import asyncio
import json
import sys
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
# PROMPTS
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
# MODEL CONFIGURATIONS
# ============================================================================

BUDGET_MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.0,
        "cost_per_million_input": 0.15,
        "cost_per_million_output": 0.6,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "display_name": "Claude 3 Haiku",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
        "cost_per_million_input": 0.25,
        "cost_per_million_output": 1.25,
    },
}


# ============================================================================
# EVALUATOR CLASS
# ============================================================================


class TransformationEvaluator:
    """Evaluator for transformation experiments."""

    def __init__(self):
        load_dotenv()
        self.config_manager = ConfigManager()
        self.providers = {}

        # Initialize providers
        for model_name, config in BUDGET_MODELS.items():
            model_config = ModelConfig(
                id=model_name,
                name=config["display_name"],
                provider=config["provider"],
                model_id=config["model_id"],
                api_key_env_var=config["api_key_env"],
                max_tokens=4096,
                temperature=config["temperature"],
                cost_per_million_tokens=config["cost_per_million_output"],
            )

            if config["provider"] == "openai":
                self.providers[model_name] = OpenAIProvider(model_config)
            elif config["provider"] == "anthropic":
                self.providers[model_name] = AnthropicProvider(model_config)

    def _build_prompt(self, example: Dict, method: str) -> tuple:
        """Build prompt for evaluation."""
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

    def _extract_cot_answer(self, response: str) -> Optional[float]:
        """Extract answer from COT response."""
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

        # Fallback: extract last number
        numbers = re.findall(r"[+-]?\d+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _execute_pot_code(self, response: str) -> tuple:
        """Execute POT code and return result."""
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
        """Check if prediction matches ground truth."""
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

    async def evaluate_single(
        self, example: Dict, model_name: str, method: str
    ) -> Dict:
        """Evaluate a single example."""

        provider = self.providers[model_name]
        system_prompt, user_prompt = self._build_prompt(example, method)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Create minimal example object
        class MinimalExample:
            def __init__(self, ex):
                self.id = ex.get("question_id", "unknown")
                self.question = ex.get("question", "")
                self.context = ex.get("context", "")
                self.ground_truth = ex.get("ground_truth", "")
                self.python_solution = ex.get("python_solution", "")

        mini_ex = MinimalExample(example)

        try:
            response = await provider.call_model(mini_ex, full_prompt)
            raw_response = response.raw_response

            # Extract answer based on method
            if method == "COT":
                extracted_answer = self._extract_cot_answer(raw_response)
                executed_code = None
                execution_error = None
            else:  # POT
                extracted_answer, executed_code, execution_error = (
                    self._execute_pot_code(raw_response)
                )

            # Check correctness
            is_correct = self._check_answer(extracted_answer, example["ground_truth"])

            return {
                "model": model_name,
                "method": method,
                "prompt_template": system_prompt,
                "full_prompt": full_prompt,
                "raw_response": raw_response,
                "extracted_answer": extracted_answer,
                "ground_truth": example["ground_truth"],
                "is_correct": is_correct,
                "executed_code": executed_code,
                "execution_error": execution_error,
                "error": None,
            }

        except Exception as e:
            return {
                "model": model_name,
                "method": method,
                "error": str(e),
                "raw_response": "",
                "extracted_answer": None,
                "is_correct": False,
            }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


async def run_experiment():
    """Run transformation experiment."""

    # Load transformation samples
    samples_file = Path(
        "experiments/transformation_samples/transformation_samples_v2.json"
    )
    with open(samples_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Take only FIRST problem from EASY
    samples = {"easy": [all_samples["easy"][0]]}

    print("=" * 70)
    print("TRANSFORMATION EXPERIMENT - REAL DATA")
    print("=" * 70)
    print(
        f"Testing: 1 problem with {len(samples['easy'][0]['transformations'])} transformations"
    )
    print(f"Models: {', '.join(BUDGET_MODELS.keys())}")
    print(f"Methods: COT, POT")
    print(
        f"Total API calls: {len(samples['easy'][0]['transformations']) * len(BUDGET_MODELS) * 2}"
    )
    print("=" * 70)
    print()

    # Initialize evaluator
    evaluator = TransformationEvaluator()

    # Results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": list(BUDGET_MODELS.keys()),
            "methods": ["COT", "POT"],
            "sample_size": "small (1 problem)",
        },
        "problems": [],
    }

    # Process the problem
    for problem_data in samples["easy"]:
        original = problem_data["original"]
        transformations = problem_data["transformations"]

        print(f"\n{'=' * 70}")
        print(f"Problem: {original['question_id']}")
        print(f"Question: {original['question'][:80]}...")
        print(f"Transformations: {len(transformations)}")
        print(f"{'=' * 70}\n")

        problem_results = {"original": original, "transformations": []}

        # Evaluate each transformation
        for trans_idx, transformation in enumerate(transformations):
            print(
                f"\nTransformation {trans_idx + 1}/{len(transformations)}: {transformation['transformation_type']}"
            )
            print(f"Description: {transformation['transformation_description']}")

            trans_results = {
                "transformation_info": {
                    "type": transformation["transformation_type"],
                    "description": transformation["transformation_description"],
                    "expected_behavior": transformation["expected_behavior"],
                },
                "transformed_problem": transformation,
                "model_results": [],
            }

            # Test each model with each method
            for model_name in BUDGET_MODELS.keys():
                for method in ["COT", "POT"]:
                    print(f"  {model_name} ({method})...", end=" ", flush=True)

                    result = await evaluator.evaluate_single(
                        transformation, model_name, method
                    )
                    trans_results["model_results"].append(result)

                    if result.get("error"):
                        print(f"[ERROR]: {result['error']}")
                    else:
                        status = "[OK]" if result.get("is_correct") else "[FAIL]"
                        print(f"{status}")

            problem_results["transformations"].append(trans_results)

        results["problems"].append(problem_results)

    # Save results
    output_dir = Path("experiments/transformation_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"real_experiment_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[SUCCESS] Results saved to: {output_file}")
    print(f"{'=' * 70}")

    # Print summary
    total_evals = (
        sum(len(p["transformations"]) for p in results["problems"])
        * len(BUDGET_MODELS)
        * 2
    )
    print(f"\nTotal evaluations: {total_evals}")

    # Count correct refusals vs hallucinations
    correct_refusals = 0
    hallucinations = 0

    for problem in results["problems"]:
        for trans in problem["transformations"]:
            for result in trans["model_results"]:
                if result.get("is_correct"):
                    correct_refusals += 1
                elif not result.get("error"):
                    hallucinations += 1

    print(f"\nResults:")
    print(f"  Correct Refusals: {correct_refusals}")
    print(f"  Hallucinations: {hallucinations}")
    print(f"  Errors: {total_evals - correct_refusals - hallucinations}")

    print("\nNext step: Generate visualization from these results")
    print(f"  python experiments/generate_transformation_visualization.py")
    print(f"  (Update input file to: {output_file})")

    return output_file


if __name__ == "__main__":
    output_file = asyncio.run(run_experiment())
