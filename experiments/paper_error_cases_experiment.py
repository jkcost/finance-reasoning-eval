"""
Paper Error Cases Experiment - Test specific error cases from Appendix B

This script tests the EXACT error cases mentioned in the paper's appendix (Table 6-14):

Table 6: test-2164 - Misunderstanding of Problem Example
Table 7: test-2162 - Formula Application Error Example
Table 8: test-2017 - Numerical Extraction Error Example
Table 9: test-2208 - Numerical Calculation Error Example
Table 10: test-114 - Unsolvable Problem in CodeFinQA
Table 11: test-142 - Ambiguous Statement in CodeTAT-QA
Table 12: validation-24 - Oversimplified Process in FinanceMath
Table 13: test-8 - Incorrect Answer in FinCode
Table 14: validation-68 - Relaxed Evaluation in FinanceMath

Usage:
    python paper_error_cases_experiment.py
    python paper_error_cases_experiment.py --models gpt-4o,claude-sonnet-4
"""

import asyncio
import json
import sys
import os
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, ModelConfig
from model_runner import OpenAIProvider, AnthropicProvider, GoogleProvider

# Import error analysis module
try:
    from error_analysis import (
        ErrorClassifier,
        ErrorClassification,
        ErrorCategory,
    )
    from error_analysis.error_classifier import ClassificationResult

    ERROR_ANALYSIS_AVAILABLE = True
except ImportError:
    ERROR_ANALYSIS_AVAILABLE = False
    print("[WARN] error_analysis module not available")


# ============================================================================
# PAPER ERROR CASES (from Appendix B, Table 6-14)
# ============================================================================

PAPER_ERROR_CASES = {
    "test-2164": {
        "table": 6,
        "error_type": "Misunderstanding of Problem",
        "description": "DeepSeek-R1 misunderstands the problem requirements",
        "source": "FinanceReasoning-hard",
    },
    "test-2162": {
        "table": 7,
        "error_type": "Formula Application Error",
        "description": "DeepSeek-R1 applies incorrect formula",
        "source": "FinanceReasoning-hard",
    },
    "test-2017": {
        "table": 8,
        "error_type": "Numerical Extraction Error",
        "description": "DeepSeek-R1 extracts wrong numerical values",
        "source": "FinanceReasoning-hard",
    },
    "test-2208": {
        "table": 9,
        "error_type": "Numerical Calculation Error",
        "description": "DeepSeek-R1 makes calculation mistakes",
        "source": "FinanceReasoning-hard",
    },
    "test-114": {
        "table": 10,
        "error_type": "Unsolvable Problem",
        "description": "Original CodeFinQA problem was unsolvable",
        "source": "CodeFinQA",
    },
    "test-142": {
        "table": 11,
        "error_type": "Ambiguous Statement",
        "description": "CodeTAT-QA problem had ambiguous phrasing",
        "source": "CodeTAT-QA",
    },
    "validation-24": {
        "table": 12,
        "error_type": "Oversimplified Process",
        "description": "FinanceMath had oversimplified solution process",
        "source": "FinanceMath",
    },
    "test-8": {
        "table": 13,
        "error_type": "Incorrect Answer",
        "description": "FinCode had incorrect ground truth answer",
        "source": "FinCode",
    },
    "validation-68": {
        "table": 14,
        "error_type": "Relaxed Evaluation",
        "description": "FinanceMath had relaxed evaluation criteria",
        "source": "FinanceMath",
    },
}


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
# MODEL CONFIGURATIONS
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
    level: Optional[str] = None
    paper_table: Optional[int] = None
    paper_error_type: Optional[str] = None
    paper_source: Optional[str] = None
    # Extended error classification (from error_analysis module)
    auto_error_category: Optional[str] = None
    auto_error_confidence: Optional[float] = None
    auto_error_evidence: Optional[str] = None


def load_all_datasets() -> Dict[str, Dict]:
    """Load all datasets and create ID-to-example mapping"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data/financereasoning/raw/FinanceReasoning"

    all_examples = {}

    # Load easy, medium, hard datasets
    for level in ["easy", "medium", "hard"]:
        filepath = data_dir / f"{level}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for example in data:
                    qid = example.get("question_id", "")
                    if qid:
                        example["level"] = level
                        all_examples[qid] = example

    return all_examples


def extract_paper_error_cases(
    all_examples: Dict[str, Dict],
) -> Tuple[List[Dict], List[str]]:
    """Extract the specific error cases from paper Table 6-14"""
    found_cases = []
    missing_ids = []

    for qid, case_info in PAPER_ERROR_CASES.items():
        if qid in all_examples:
            example = all_examples[qid].copy()
            example["paper_table"] = case_info["table"]
            example["paper_error_type"] = case_info["error_type"]
            example["paper_source"] = case_info["source"]
            example["paper_description"] = case_info["description"]
            found_cases.append(example)
        else:
            missing_ids.append(qid)

    # Sort by table number
    found_cases.sort(key=lambda x: x.get("paper_table", 0))

    return found_cases, missing_ids


class PaperErrorCasesExperiment:
    """Run experiments on paper's specific error cases"""

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

        # Initialize error classifier if available
        self.error_classifier = None
        if ERROR_ANALYSIS_AVAILABLE:
            self.error_classifier = ErrorClassifier()
            print("[OK] Error classifier initialized")

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

            # Auto-classify error if incorrect and classifier available
            auto_error_category = None
            auto_error_confidence = None
            auto_error_evidence = None

            if not is_correct and self.error_classifier:
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
                    parse_status="success" if final_answer is not None else "failed",
                )

                classification = self.error_classifier.classify(
                    classification_result,
                    example.get("ground_truth"),
                    example.get("context", ""),
                )
                auto_error_category = classification.category.value
                auto_error_confidence = classification.confidence
                auto_error_evidence = classification.evidence

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
                level=example.get("level"),
                paper_table=example.get("paper_table"),
                paper_error_type=example.get("paper_error_type"),
                paper_source=example.get("paper_source"),
                auto_error_category=auto_error_category,
                auto_error_confidence=auto_error_confidence,
                auto_error_evidence=auto_error_evidence,
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
                level=example.get("level"),
                paper_table=example.get("paper_table"),
                paper_error_type=example.get("paper_error_type"),
                paper_source=example.get("paper_source"),
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
            table = example.get("paper_table", "?")
            error_type = example.get("paper_error_type", "Unknown")

            print(f"\n{'=' * 80}")
            print(f"[Table {table}] {example_id} - {error_type}")
            print(f"GT={gt} | Level={example.get('level', 'unknown')}")
            print(f"Q: {example.get('question', '')[:100]}...")
            print(f"{'=' * 80}")

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

                    status = "[OK]" if result.is_correct else "[X]"
                    print(
                        f"{status} Ans={result.final_answer} (${result.cost_usd:.4f}, {result.latency_seconds:.1f}s)"
                    )

        return results


def save_results(results: List[ExperimentResult], output_dir: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"paper_error_cases_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "methodology": "FinanceReasoning Paper Error Cases (Table 6-14)",
        "description": "Testing specific error cases from Appendix B",
        "paper_error_cases": PAPER_ERROR_CASES,
        "results": [asdict(r) for r in results],
        "summary": {},
        "by_error_type": {},
        "by_table": {},
    }

    # Calculate summary by model/method
    summary = {}
    for r in results:
        key = f"{r.model_name}_{r.method}"
        if key not in summary:
            summary[key] = {
                "model": r.model_name,
                "method": r.method,
                "total": 0,
                "correct": 0,
                "total_cost": 0.0,
            }
        summary[key]["total"] += 1
        summary[key]["correct"] += 1 if r.is_correct else 0
        summary[key]["total_cost"] += r.cost_usd

    for key in summary:
        total = summary[key]["total"]
        if total > 0:
            summary[key]["accuracy"] = round(summary[key]["correct"] / total * 100, 1)

    data["summary"] = summary

    # Summary by error type
    by_error_type = {}
    for r in results:
        et = r.paper_error_type or "Unknown"
        if et not in by_error_type:
            by_error_type[et] = {"total": 0, "correct": 0}
        by_error_type[et]["total"] += 1
        by_error_type[et]["correct"] += 1 if r.is_correct else 0

    for et in by_error_type:
        total = by_error_type[et]["total"]
        by_error_type[et]["accuracy"] = (
            round(by_error_type[et]["correct"] / total * 100, 1) if total > 0 else 0
        )

    data["by_error_type"] = by_error_type

    # Summary by table
    by_table = {}
    for r in results:
        table = r.paper_table or 0
        if table not in by_table:
            by_table[table] = {
                "total": 0,
                "correct": 0,
                "error_type": r.paper_error_type,
            }
        by_table[table]["total"] += 1
        by_table[table]["correct"] += 1 if r.is_correct else 0

    for table in by_table:
        total = by_table[table]["total"]
        by_table[table]["accuracy"] = (
            round(by_table[table]["correct"] / total * 100, 1) if total > 0 else 0
        )

    data["by_table"] = by_table

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def generate_visualization(results: List[ExperimentResult], output_dir: Path) -> str:
    """Generate HTML visualization of results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"paper_error_cases_{timestamp}.html"

    # Organize data
    by_example = {}
    for r in results:
        if r.example_id not in by_example:
            by_example[r.example_id] = {
                "ground_truth": r.ground_truth,
                "paper_table": r.paper_table,
                "paper_error_type": r.paper_error_type,
                "level": r.level,
                "results": [],
            }
        by_example[r.example_id]["results"].append(r)

    # Calculate accuracy by model/method
    model_stats = {}
    for r in results:
        key = f"{r.model_name}_{r.method}"
        if key not in model_stats:
            model_stats[key] = {
                "model": r.model_name,
                "method": r.method,
                "correct": 0,
                "total": 0,
            }
        model_stats[key]["total"] += 1
        model_stats[key]["correct"] += 1 if r.is_correct else 0

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Error Cases Results - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            margin-bottom: 30px; 
            color: #00d4ff;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }}
        .chart-title {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th, td {{ 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ 
            background: rgba(0,212,255,0.2); 
            color: #00d4ff;
            font-weight: 600;
        }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .correct {{ color: #4ade80; }}
        .incorrect {{ color: #f87171; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .badge-table {{ background: rgba(0,212,255,0.2); color: #00d4ff; }}
        .badge-error {{ background: rgba(248,113,113,0.2); color: #f87171; }}
        .badge-level {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
        .example-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .example-title {{
            font-size: 1.1em;
            color: #00d4ff;
        }}
        .question-text {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 0.95em;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Paper Error Cases Experiment</h1>
        <p class="subtitle">Testing specific error cases from FinanceReasoning paper Appendix B (Table 6-14)</p>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Accuracy by Model (POT)</div>
                <canvas id="accuracyChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Accuracy by Error Type</div>
                <canvas id="errorTypeChart"></canvas>
            </div>
        </div>
        
        <h2 style="color: #00d4ff; margin-bottom: 20px;">Summary by Model</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Method</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
            </thead>
            <tbody>
"""

    for key, stats in sorted(model_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        acc_class = "correct" if acc >= 50 else "incorrect"
        html_content += f"""
                <tr>
                    <td>{stats["model"]}</td>
                    <td>{stats["method"]}</td>
                    <td>{stats["correct"]}</td>
                    <td>{stats["total"]}</td>
                    <td class="{acc_class}">{acc:.1f}%</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
        
        <h2 style="color: #00d4ff; margin-bottom: 20px;">Results by Paper Table</h2>
"""

    for ex_id, data in sorted(
        by_example.items(), key=lambda x: x[1].get("paper_table", 0)
    ):
        correct_count = sum(1 for r in data["results"] if r.is_correct)
        total_count = len(data["results"])

        html_content += f"""
        <div class="example-card">
            <div class="example-header">
                <span class="example-title">{ex_id}</span>
                <div>
                    <span class="badge badge-table">Table {data["paper_table"]}</span>
                    <span class="badge badge-error">{data["paper_error_type"]}</span>
                    <span class="badge badge-level">{data["level"]}</span>
                </div>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>Ground Truth:</strong> {data["ground_truth"]} | 
                <strong>Score:</strong> <span class="{"correct" if correct_count > total_count / 2 else "incorrect"}">{correct_count}/{total_count}</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Method</th>
                        <th>Answer</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>
"""
        for r in data["results"]:
            status = "correct" if r.is_correct else "incorrect"
            html_content += f"""
                    <tr>
                        <td>{r.model_name}</td>
                        <td>{r.method}</td>
                        <td>{r.final_answer}</td>
                        <td class="{status}">{"✓" if r.is_correct else "✗"}</td>
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
        </div>
"""

    # Chart data
    pot_models = [s for k, s in model_stats.items() if s["method"] == "POT"]
    model_labels = [s["model"] for s in pot_models]
    model_accuracies = [
        s["correct"] / s["total"] * 100 if s["total"] > 0 else 0 for s in pot_models
    ]

    # Error type stats
    error_type_stats = {}
    for r in results:
        et = r.paper_error_type or "Unknown"
        if et not in error_type_stats:
            error_type_stats[et] = {"correct": 0, "total": 0}
        error_type_stats[et]["total"] += 1
        error_type_stats[et]["correct"] += 1 if r.is_correct else 0

    error_labels = list(error_type_stats.keys())
    error_accuracies = [
        error_type_stats[et]["correct"] / error_type_stats[et]["total"] * 100
        if error_type_stats[et]["total"] > 0
        else 0
        for et in error_labels
    ]

    html_content += f"""
        <script>
            // Accuracy by Model Chart
            new Chart(document.getElementById('accuracyChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(model_labels)},
                    datasets: [{{
                        label: 'Accuracy (%)',
                        data: {json.dumps(model_accuracies)},
                        backgroundColor: 'rgba(0, 212, 255, 0.6)',
                        borderColor: 'rgba(0, 212, 255, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        x: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#888' }} }}
                    }}
                }}
            }});
            
            // Error Type Chart
            new Chart(document.getElementById('errorTypeChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(error_labels)},
                    datasets: [{{
                        label: 'Accuracy (%)',
                        data: {json.dumps(error_accuracies)},
                        backgroundColor: 'rgba(248, 113, 113, 0.6)',
                        borderColor: 'rgba(248, 113, 113, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    indexAxis: 'y',
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        y: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#888' }} }}
                    }}
                }}
            }});
        </script>
    </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(html_path)


def print_summary(results: List[ExperimentResult]):
    # Build summary
    summary = {}
    error_type_stats = {}

    for r in results:
        key = f"{r.model_name}_{r.method}"
        if key not in summary:
            summary[key] = {
                "model": r.model_name,
                "method": r.method,
                "total": 0,
                "correct": 0,
                "cost": 0.0,
            }
        summary[key]["total"] += 1
        summary[key]["correct"] += 1 if r.is_correct else 0
        summary[key]["cost"] += r.cost_usd

        et = r.paper_error_type or "Unknown"
        if et not in error_type_stats:
            error_type_stats[et] = {"total": 0, "correct": 0}
        error_type_stats[et]["total"] += 1
        error_type_stats[et]["correct"] += 1 if r.is_correct else 0

    print("\n" + "=" * 90)
    print("PAPER ERROR CASES EXPERIMENT SUMMARY")
    print("Testing specific cases from Appendix B (Table 6-14)")
    print("=" * 90)

    for method in ["COT", "POT"]:
        print(f"\n{method} Results:")
        print("-" * 80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Correct/Total':<15} {'Cost':<12}")
        print("-" * 80)

        for key, stats in sorted(summary.items()):
            if stats["method"] == method:
                acc = (
                    stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                )
                print(
                    f"{stats['model']:<25} {acc:>6.1f}%      {stats['correct']}/{stats['total']:<12} ${stats['cost']:.4f}"
                )

    print("\n" + "=" * 90)
    print("ACCURACY BY ERROR TYPE (All Models Combined)")
    print("=" * 90)
    for et, stats in sorted(error_type_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {et:<35}: {acc:>6.1f}% ({stats['correct']}/{stats['total']})")

    print("\n" + "=" * 90)
    print("PER-EXAMPLE RESULTS")
    print("=" * 90)

    by_example = {}
    for r in results:
        if r.example_id not in by_example:
            by_example[r.example_id] = {
                "gt": r.ground_truth,
                "table": r.paper_table,
                "error_type": r.paper_error_type,
                "results": [],
            }
        by_example[r.example_id]["results"].append(r)

    for ex_id, data in sorted(by_example.items(), key=lambda x: x[1].get("table", 0)):
        correct_count = sum(1 for r in data["results"] if r.is_correct)
        total_count = len(data["results"])
        print(f"\n[Table {data['table']}] {ex_id} - {data['error_type']}")
        print(f"  GT={data['gt']} | Score={correct_count}/{total_count}")
        for r in data["results"]:
            status = "[OK]" if r.is_correct else "[X] "
            print(f"    {r.model_name:<20} {r.method}: {status} {r.final_answer}")


async def async_main():
    import argparse

    parser = argparse.ArgumentParser(description="Paper Error Cases Experiment")
    parser.add_argument(
        "--models", type=str, default=None, help="Comma-separated models"
    )
    parser.add_argument("--methods", type=str, default="COT,POT", help="Methods")
    args = parser.parse_args()

    print("=" * 80)
    print("PAPER ERROR CASES EXPERIMENT")
    print("Testing specific error cases from Appendix B (Table 6-14)")
    print("=" * 80)

    # Load all datasets
    print(f"\n[1] Loading datasets...")
    all_examples = load_all_datasets()
    print(f"    Loaded {len(all_examples)} total examples")

    # Extract paper error cases
    print(f"\n[2] Extracting paper error cases...")
    error_cases, missing_ids = extract_paper_error_cases(all_examples)

    print(f"    Found {len(error_cases)} error cases:")
    for ex in error_cases:
        print(
            f"      Table {ex['paper_table']}: {ex['question_id']} - {ex['paper_error_type']}"
        )

    if missing_ids:
        print(f"\n    WARNING: Missing IDs (not in dataset): {missing_ids}")

    # Parse args
    models = args.models.split(",") if args.models else None
    methods = args.methods.split(",")

    # Run experiment
    print(f"\n[3] Initializing experiment...")
    experiment = PaperErrorCasesExperiment(models=models)
    print(f"    Models: {experiment.models}")
    print(f"    Methods: {methods}")

    print(f"\n[4] Running experiment...")
    results = await experiment.run_experiment(error_cases, methods=methods)

    # Save
    output_dir = Path(__file__).parent / "quick_results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n[5] Saving results...")
    json_path = save_results(results, output_dir)
    print(f"    JSON: {json_path}")

    html_path = generate_visualization(results, output_dir)
    print(f"    HTML: {html_path}")

    # Print summary
    print_summary(results)

    print(f"\n[OK] Experiment complete!")
    print(f"    JSON Results: {json_path}")
    print(f"    HTML Visualization: {html_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
