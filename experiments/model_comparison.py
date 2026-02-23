"""
Multi-Model Comparison Experiment

Compares COT and POT performance across multiple LLM models,
capturing reasoning steps, raw responses, and prompts used.
"""

import asyncio
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, EvaluationConfig, ModelConfig
from prompt_builder import PromptBuilder, PROMPT_TEMPLATES
from response_parser import ResponseParser
from pot_executor import CodeExecutionSandbox, evaluate_pot_response
from metrics_evaluator import MetricsEvaluator
from model_runner import OpenAIProvider, AnthropicProvider, GoogleProvider


@dataclass
class ModelResult:
    """Result from a single model evaluation"""

    model_name: str
    provider: str
    method: str  # COT or POT
    question: str
    ground_truth: Any

    # Model response
    final_answer: Any
    is_correct: bool
    reasoning_steps: List[Dict[str, Any]]
    raw_response: str

    # Prompts used
    system_prompt: str
    user_prompt: str

    # Metrics
    cost_usd: float
    latency_seconds: float

    # For POT
    executed_code: Optional[str] = None
    execution_output: Optional[str] = None


@dataclass
class ComparisonResult:
    """Complete comparison result for one question"""

    example_id: str
    question: str
    context: str
    ground_truth: Any
    python_solution: str
    model_results: List[ModelResult] = field(default_factory=list)


class MultiModelEvaluator:
    """Evaluates multiple models on same questions"""

    # Model configurations
    MODEL_CONFIGS = {
        "claude-sonnet-4": {
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-20250514",
            "display_name": "Claude Sonnet 4",
            "api_key_env": "ANTHROPIC_API_KEY",
            "cost_per_million": 15.0,
        },
        "gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "display_name": "GPT-4o",
            "api_key_env": "OPENAI_API_KEY",
            "cost_per_million": 10.0,
        },
        "gemini-2.0-flash": {
            "provider": "google",
            "model_id": "gemini-2.0-flash-exp",
            "display_name": "Gemini 2.0 Flash",
            "api_key_env": "GOOGLE_API_KEY",
            "cost_per_million": 0.30,
        },
    }

    def __init__(self, models: List[str] = None):
        """Initialize with specific models or all available"""
        # Load environment
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # Filter to models with available API keys
        available_models = []
        for model_name, config in self.MODEL_CONFIGS.items():
            api_key = os.environ.get(config["api_key_env"])
            if api_key:
                available_models.append(model_name)
                print(f"[OK] {model_name}: API key found")
            else:
                print(f"[SKIP] {model_name}: No API key ({config['api_key_env']})")

        # Use specified models or all available
        if models:
            self.models = [m for m in models if m in available_models]
        else:
            self.models = available_models

        if not self.models:
            raise ValueError("No models available. Check API keys in .env file.")

        self.parser = ResponseParser()
        self.pot_executor = CodeExecutionSandbox(timeout_seconds=10)
        self.metrics = MetricsEvaluator()

        # Initialize providers
        self.providers = {}
        for model_name in self.models:
            config = self.MODEL_CONFIGS[model_name]
            model_config = ModelConfig(
                id=model_name,
                name=config["display_name"],
                provider=config["provider"],
                model_id=config["model_id"],
                api_key_env_var=config["api_key_env"],
                max_tokens=4096,
                temperature=0.7,
                cost_per_million_tokens=config["cost_per_million"],
            )

            if config["provider"] == "openai":
                self.providers[model_name] = OpenAIProvider(model_config)
            elif config["provider"] == "anthropic":
                self.providers[model_name] = AnthropicProvider(model_config)
            elif config["provider"] == "google":
                self.providers[model_name] = GoogleProvider(model_config)

    def _build_prompts(self, example: Dict, method: str) -> tuple:
        """Build system and user prompts for given method"""
        question = example.get("question", "")
        context = example.get("context", "")
        python_solution = example.get("python_solution", "")

        if method == "COT":
            template = PROMPT_TEMPLATES["finance_reasoning_compliant"]
        else:
            template = PROMPT_TEMPLATES["pot"]

        # Build user message
        user_prompt = template.user_message_template.format(
            question=question,
            context=context,
            python_solution=python_solution if python_solution else "Not provided",
        )

        return template.system_message, user_prompt

    def _check_answer(
        self, llm_answer: Any, ground_truth: Any, tolerance: float = 0.01
    ) -> bool:
        """Check if answer is correct"""
        if llm_answer is None:
            return False

        try:
            # Try numeric comparison
            gt_str = str(ground_truth).replace(",", "").replace("%", "").strip()
            gt_num = float(gt_str)

            if isinstance(llm_answer, (int, float)):
                llm_num = float(llm_answer)
            else:
                llm_str = str(llm_answer).replace(",", "").replace("%", "").strip()
                llm_num = float(llm_str)

            # Check with tolerance
            if gt_num == 0:
                return abs(llm_num) < tolerance
            return abs(llm_num - gt_num) / abs(gt_num) < tolerance

        except (ValueError, TypeError):
            # String comparison
            return str(llm_answer).strip().lower() == str(ground_truth).strip().lower()

    async def evaluate_single(
        self, example: Dict, model_name: str, method: str
    ) -> ModelResult:
        """Evaluate single example with specific model and method"""

        config = self.MODEL_CONFIGS[model_name]
        provider = self.providers[model_name]

        question = example.get("question", "")
        ground_truth = example.get("ground_truth", "")

        # Build prompts
        system_prompt, user_prompt = self._build_prompts(example, method)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Query model
        start_time = time.time()
        try:
            # Create a minimal Example-like object for the provider
            class MinimalExample:
                def __init__(self, ex):
                    self.id = ex.get("id", "unknown")
                    self.question = ex.get("question", "")
                    self.context = ex.get("context", "")
                    self.ground_truth = ex.get("ground_truth", "")
                    self.python_solution = ex.get("python_solution", "")

            mini_ex = MinimalExample(example)
            response = await provider.call_model(mini_ex, full_prompt)
            latency = response.response_time_seconds

            # LLMResponse already has raw_response extracted
            raw_response = response.raw_response

            # Parse response based on method
            if method == "POT":
                # Execute code
                exec_result = evaluate_pot_response(raw_response)
                final_answer = exec_result.output if exec_result.success else None
                reasoning_steps = []
                executed_code = exec_result.code
                execution_output = (
                    str(exec_result.output)
                    if exec_result.success
                    else exec_result.error
                )
            else:
                # Parse COT response - directly parse raw_response JSON
                import json as json_module

                final_answer = None
                reasoning_steps = []

                # Try to parse JSON directly from raw_response
                try:
                    # Clean up potential markdown code blocks
                    text = raw_response.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                    parsed_json = json_module.loads(text)
                    final_answer = parsed_json.get("final_answer")

                    # Parse reasoning steps
                    steps_data = parsed_json.get("reasoning_steps", [])
                    for step_data in steps_data:
                        if isinstance(step_data, dict):
                            reasoning_steps.append(
                                {
                                    "step_number": step_data.get("step_number", 0),
                                    "step_type": step_data.get("step_type", ""),
                                    "description": step_data.get("description", ""),
                                    "code_snippet": step_data.get("code_snippet", ""),
                                }
                            )
                except json_module.JSONDecodeError:
                    # Fallback: try regex extraction
                    import re

                    match = re.search(r'"final_answer"\s*:\s*(\d+\.?\d*)', raw_response)
                    if match:
                        try:
                            final_answer = float(match.group(1))
                            if final_answer == int(final_answer):
                                final_answer = int(final_answer)
                        except ValueError:
                            pass

                executed_code = None
                execution_output = None

            # Check correctness
            is_correct = self._check_answer(final_answer, ground_truth)

            # Use cost from LLMResponse
            cost = response.cost_usd

            return ModelResult(
                model_name=model_name,
                provider=config["provider"],
                method=method,
                question=question,
                ground_truth=ground_truth,
                final_answer=final_answer,
                is_correct=is_correct,
                reasoning_steps=reasoning_steps,
                raw_response=raw_response,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                cost_usd=round(cost, 6),
                latency_seconds=round(latency, 2),
                executed_code=executed_code,
                execution_output=execution_output,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            latency = time.time() - start_time
            return ModelResult(
                model_name=model_name,
                provider=config["provider"],
                method=method,
                question=question,
                ground_truth=ground_truth,
                final_answer=None,
                is_correct=False,
                reasoning_steps=[],
                raw_response=f"ERROR: {str(e)}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                cost_usd=0.0,
                latency_seconds=round(latency, 2),
                executed_code=None,
                execution_output=str(e),
            )

    async def run_comparison(
        self, examples: List[Dict], methods: List[str] = None
    ) -> List[ComparisonResult]:
        """Run comparison across all models and methods"""

        methods = methods or ["COT", "POT"]
        results = []

        total = len(examples) * len(self.models) * len(methods)
        current = 0

        for example in examples:
            example_id = example.get("id", "unknown")
            model_results = []

            print(f"\n{'=' * 70}")
            print(f"Example: {example_id}")
            print(f"Question: {example.get('question', '')[:80]}...")
            print(f"Ground Truth: {example.get('ground_truth', '')}")
            print(f"{'=' * 70}")

            for model_name in self.models:
                for method in methods:
                    current += 1
                    display_name = self.MODEL_CONFIGS[model_name]["display_name"]
                    print(
                        f"\n  [{current}/{total}] {display_name} - {method}...",
                        end=" ",
                        flush=True,
                    )

                    result = await self.evaluate_single(example, model_name, method)
                    model_results.append(result)

                    status = "[OK]" if result.is_correct else "[X]"
                    print(
                        f"{status} Answer={result.final_answer} (${result.cost_usd:.4f}, {result.latency_seconds:.1f}s)"
                    )

                    # Show reasoning steps for COT
                    if method == "COT" and result.reasoning_steps:
                        print(f"      Reasoning Steps: {len(result.reasoning_steps)}")
                        for step in result.reasoning_steps[:3]:  # Show first 3 steps
                            desc = step.get("description", "")[:50]
                            print(
                                f"        - Step {step.get('step_number', '?')}: {desc}..."
                            )

            results.append(
                ComparisonResult(
                    example_id=example_id,
                    question=example.get("question", ""),
                    context=example.get("context", "")[:500]
                    + "...",  # Truncate context
                    ground_truth=example.get("ground_truth", ""),
                    python_solution=example.get("python_solution", ""),
                    model_results=model_results,
                )
            )

        return results


def save_results(results: List[ComparisonResult], output_dir: Path) -> str:
    """Save comparison results to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{timestamp}.json"
    filepath = output_dir / filename

    # Convert to serializable format
    data = {
        "timestamp": timestamp,
        "models": list(MultiModelEvaluator.MODEL_CONFIGS.keys()),
        "prompts": {
            "COT": {
                "template_name": "finance_reasoning_compliant",
                "system_prompt": PROMPT_TEMPLATES[
                    "finance_reasoning_compliant"
                ].system_message,
                "user_template": PROMPT_TEMPLATES[
                    "finance_reasoning_compliant"
                ].user_message_template,
            },
            "POT": {
                "template_name": "pot",
                "system_prompt": PROMPT_TEMPLATES["pot"].system_message,
                "user_template": PROMPT_TEMPLATES["pot"].user_message_template,
            },
        },
        "results": [],
    }

    for result in results:
        result_dict = {
            "example_id": result.example_id,
            "question": result.question,
            "context": result.context,
            "ground_truth": result.ground_truth,
            "python_solution": result.python_solution,
            "model_results": [asdict(mr) for mr in result.model_results],
        }
        data["results"].append(result_dict)

    # Calculate summary statistics
    summary = calculate_summary(results)
    data["summary"] = summary

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def calculate_summary(results: List[ComparisonResult]) -> Dict:
    """Calculate summary statistics"""
    summary = {}

    for result in results:
        for mr in result.model_results:
            key = f"{mr.model_name}_{mr.method}"
            if key not in summary:
                summary[key] = {
                    "model": mr.model_name,
                    "method": mr.method,
                    "total": 0,
                    "correct": 0,
                    "total_cost": 0.0,
                    "total_latency": 0.0,
                    "avg_reasoning_steps": 0,
                }

            summary[key]["total"] += 1
            summary[key]["correct"] += 1 if mr.is_correct else 0
            summary[key]["total_cost"] += mr.cost_usd
            summary[key]["total_latency"] += mr.latency_seconds
            summary[key]["avg_reasoning_steps"] += len(mr.reasoning_steps)

    # Calculate averages
    for key in summary:
        total = summary[key]["total"]
        if total > 0:
            summary[key]["accuracy"] = round(summary[key]["correct"] / total * 100, 1)
            summary[key]["avg_cost"] = round(summary[key]["total_cost"] / total, 6)
            summary[key]["avg_latency"] = round(
                summary[key]["total_latency"] / total, 2
            )
            summary[key]["avg_reasoning_steps"] = round(
                summary[key]["avg_reasoning_steps"] / total, 1
            )

    return summary


def print_summary(results: List[ComparisonResult]):
    """Print comparison summary table"""
    summary = calculate_summary(results)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Group by method
    for method in ["COT", "POT"]:
        print(f"\n{method} Results:")
        print("-" * 70)
        print(
            f"{'Model':<20} {'Accuracy':<12} {'Avg Cost':<12} {'Avg Latency':<12} {'Avg Steps':<10}"
        )
        print("-" * 70)

        for key, stats in summary.items():
            if stats["method"] == method:
                print(
                    f"{stats['model']:<20} {stats['accuracy']:>6.1f}%      ${stats['avg_cost']:<10.4f} {stats['avg_latency']:>6.1f}s       {stats['avg_reasoning_steps']:>5.1f}"
                )

    # Per-question comparison
    print("\n" + "=" * 80)
    print("PER-QUESTION RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n[{result.example_id}] GT={result.ground_truth}")
        print(f"  Q: {result.question[:70]}...")

        for mr in result.model_results:
            status = "[OK]" if mr.is_correct else "[X] "
            print(f"    {mr.model_name:<18} {mr.method}: {status} {mr.final_answer}")


async def async_main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Model Comparison Experiment")
    parser.add_argument(
        "--num", type=int, default=2, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all available)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="COT,POT",
        help="Comma-separated methods (default: COT,POT)",
    )
    args = parser.parse_args()

    # Load dataset
    data_path = (
        Path(__file__).parent.parent
        / "data/financereasoning/raw/FinanceReasoning/hard.json"
    )
    print(f"[1] Loading dataset from {data_path}...")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Select examples
    examples = data[: args.num]
    print(f"    Selected {len(examples)} examples")

    # Parse models
    models = args.models.split(",") if args.models else None
    methods = args.methods.split(",")

    # Initialize evaluator
    print(f"\n[2] Initializing evaluator...")
    evaluator = MultiModelEvaluator(models=models)
    print(f"    Active Models: {evaluator.models}")
    print(f"    Methods: {methods}")

    # Run comparison
    print(f"\n[3] Running comparison...")
    results = await evaluator.run_comparison(examples, methods=methods)

    # Save results
    output_dir = Path(__file__).parent / "quick_results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n[4] Saving results...")
    filepath = save_results(results, output_dir)
    print(f"    Saved to: {filepath}")

    # Print summary
    print_summary(results)

    print(f"\n[OK] Comparison complete!")
    print(f"    Results: {filepath}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
