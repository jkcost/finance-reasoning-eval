"""
Model Comparison Experiment

각 난이도(easy, medium, hard)에서 동일한 3개 문제를 선택하여
여러 모델의 성능을 비교 분석합니다.

Usage:
    python run_model_comparison.py
    python run_model_comparison.py --budget balanced
"""

import asyncio
import json
import sys
import os
import re
import time
import argparse
import html as html_module
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
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
    LLMErrorAnalyzer,
)
from error_analysis.error_classifier import ErrorClassifier, ClassificationResult
from error_analysis.error_taxonomy import BUDGET_MODEL_SETS


# ============================================================================
# PROMPTS
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
class ComparisonResult:
    """Result from model comparison"""

    # 기본 필드
    example_id: str
    level: str
    question: str  # 전체 질문 텍스트
    ground_truth: Any
    model_name: str
    model_type: str  # general or reasoning
    method: str
    predicted_answer: Any
    is_correct: bool
    error_category: Optional[str]
    error_confidence: Optional[float]
    cost_usd: float
    latency_seconds: float
    execution_error: Optional[str] = None

    # 상세 분석용 필드 (새로 추가)
    context: str = ""  # 문제 맥락 (테이블, 데이터 등)
    raw_response: str = ""  # 모델의 전체 응답
    executed_code: Optional[str] = None  # POT에서 실행된 코드
    error_evidence: str = ""  # 오답 분석 상세 설명


def load_dataset(level: str) -> List[Dict]:
    """Load FinanceReasoning dataset by level"""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data/financereasoning/raw/FinanceReasoning"
    filepath = data_dir / f"{level}.json"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for example in data:
        example["level"] = level

    return data


def select_examples(easy: List[Dict], medium: List[Dict], hard: List[Dict], n: int = 3) -> List[Dict]:
    """Select n examples from each level"""

    selected = []

    for level_data, level_name in [(easy, "easy"), (medium, "medium"), (hard, "hard")]:
        total = len(level_data)
        # n개를 균등 간격으로 선택
        if n >= total:
            indices = list(range(total))
        else:
            step = total / n
            indices = [int(i * step) for i in range(n)]

        for i, idx in enumerate(indices):
            example = level_data[idx].copy()
            example["level"] = level_name
            example["selection_index"] = i + 1
            selected.append(example)

    return selected


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


class ModelComparisonExperiment:
    """Run model comparison experiment"""

    def __init__(self, models: List[str], enable_llm_analysis: bool = True):
        self.models = models
        self.classifier = ErrorClassifier()
        self.enable_llm_analysis = enable_llm_analysis

        # LLM Error Analyzer for detailed wrong answer analysis
        self.llm_analyzer = LLMErrorAnalyzer(model="gpt-4o-mini") if enable_llm_analysis else None

        # Initialize providers
        self.providers = {}
        for model_name in models:
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

        numbers = re.findall(r"[+-]?\d+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _execute_pot_code(self, response: str) -> Tuple[Any, Optional[str], Optional[str]]:
        """Execute POT code"""

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

        if "solution()" not in code or not re.search(r"^\s*solution\(\)", code, re.MULTILINE):
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

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate API cost"""

        model_info = MODEL_REGISTRY[model_name]
        input_cost = (input_tokens / 1_000_000) * model_info.cost_per_million_input
        output_cost = (output_tokens / 1_000_000) * model_info.cost_per_million_output
        return round(input_cost + output_cost, 6)

    async def evaluate_single(
        self, example: Dict, model_name: str, method: str
    ) -> ComparisonResult:
        """Evaluate single example"""

        provider = self.providers[model_name]
        model_info = MODEL_REGISTRY[model_name]
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
            else:
                final_answer, executed_code, execution_error = self._execute_pot_code(raw_response)

            is_correct = self._check_answer(final_answer, example.get("ground_truth"))
            cost = self._calculate_cost(input_tokens, output_tokens, model_name)

            # Classify error if incorrect
            error_category = None
            error_confidence = None
            error_evidence = ""

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
                    parse_status="success" if final_answer is not None else "failed",
                )

                classification = self.classifier.classify(
                    classification_result,
                    example.get("ground_truth"),
                    example.get("context", ""),
                )
                error_category = classification.category.value
                error_confidence = classification.confidence
                error_evidence = classification.evidence if hasattr(classification, 'evidence') else ""

                # LLM-based deep analysis for ALL incorrect answers
                # 모든 오답에 대해 상세 분석 수행 (execution_error 포함)
                if self.llm_analyzer:
                    try:
                        # execution_error인 경우 실행 오류 정보도 포함
                        analysis_context = example.get("context", "")
                        if execution_error:
                            analysis_context += f"\n\n[실행 오류 정보]: {execution_error}"
                            if executed_code:
                                analysis_context += f"\n[실행된 코드]:\n{executed_code}"

                        llm_analysis = self.llm_analyzer.analyze(
                            question=example.get("question", ""),
                            context=analysis_context,
                            ground_truth=example.get("ground_truth"),
                            predicted_answer=final_answer,
                            raw_response=raw_response,
                        )
                        if llm_analysis:
                            # execution_error는 유지하고, 나머지는 LLM 분석 결과로 대체
                            if error_category != "execution_error":
                                error_category = llm_analysis.error_type.lower()
                            error_confidence = llm_analysis.confidence
                            error_evidence = (
                                f"[{llm_analysis.error_type}] {llm_analysis.summary}\n\n"
                                f"상세 분석: {llm_analysis.detailed_analysis}\n\n"
                                f"원인: {llm_analysis.likely_cause}\n\n"
                                f"올바른 접근: {llm_analysis.correct_approach}"
                            )
                    except Exception as llm_err:
                        print(f"    [LLM Analysis Error] {llm_err}")
                        # LLM 분석 실패 시 기본 메시지 유지
                        if not error_evidence:
                            error_evidence = f"분석 실패: {str(llm_err)}"

            return ComparisonResult(
                example_id=example.get("question_id", "unknown"),
                level=example.get("level", "unknown"),
                question=example.get("question", ""),  # 전체 저장 ([:100] 제거)
                ground_truth=example.get("ground_truth"),
                model_name=model_name,
                model_type=model_info.model_type.value,
                method=method,
                predicted_answer=final_answer,
                is_correct=is_correct,
                error_category=error_category,
                error_confidence=error_confidence,
                cost_usd=cost,
                latency_seconds=latency,
                execution_error=execution_error,
                # 새 필드들
                context=example.get("context", ""),
                raw_response=raw_response,
                executed_code=executed_code,
                error_evidence=error_evidence,
            )

        except Exception as e:
            return ComparisonResult(
                example_id=example.get("question_id", "unknown"),
                level=example.get("level", "unknown"),
                question=example.get("question", ""),  # 전체 저장
                ground_truth=example.get("ground_truth"),
                model_name=model_name,
                model_type=model_info.model_type.value,
                method=method,
                predicted_answer=None,
                is_correct=False,
                error_category="execution_error",
                error_confidence=1.0,
                cost_usd=0.0,
                latency_seconds=time.time() - start_time,
                execution_error=str(e),
                # 새 필드들
                context=example.get("context", ""),
                raw_response="",
                executed_code=None,
                error_evidence=f"API 호출 또는 실행 중 예외 발생: {str(e)}",
            )

    async def run(self, examples: List[Dict], methods: List[str] = None) -> List[ComparisonResult]:
        """Run experiment"""

        methods = methods or ["POT"]
        results = []
        total = len(examples) * len(self.models) * len(methods)
        current = 0

        for example in examples:
            example_id = example.get("question_id", "unknown")
            level = example.get("level", "unknown")
            gt = example.get("ground_truth", "")

            print(f"\n{'=' * 80}")
            print(f"[{level.upper()}] {example_id} | GT={gt}")
            print(f"Q: {example.get('question', '')[:80]}...")
            print(f"{'=' * 80}")

            for model_name in self.models:
                for method in methods:
                    current += 1
                    model_info = MODEL_REGISTRY[model_name]
                    print(f"  [{current}/{total}] {model_info.display_name} ({method})...", end=" ", flush=True)

                    result = await self.evaluate_single(example, model_name, method)
                    results.append(result)

                    status = "OK" if result.is_correct else "X"
                    error_info = f" [{result.error_category}]" if result.error_category else ""
                    print(f"[{status}] {result.predicted_answer}{error_info} (${result.cost_usd:.4f}, {result.latency_seconds:.1f}s)")

        return results


def generate_comparison_report(results: List[ComparisonResult], output_dir: Path) -> Dict[str, Path]:
    """Generate comparison reports"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # JSON Report
    # =========================================================================

    # By level
    by_level = {"easy": [], "medium": [], "hard": []}
    for r in results:
        by_level[r.level].append(r)

    # By model
    by_model = {}
    for r in results:
        if r.model_name not in by_model:
            by_model[r.model_name] = {"total": 0, "correct": 0, "cost": 0.0}
        by_model[r.model_name]["total"] += 1
        by_model[r.model_name]["correct"] += 1 if r.is_correct else 0
        by_model[r.model_name]["cost"] += r.cost_usd

    # Model accuracy by level
    model_level_acc = {}
    for r in results:
        key = f"{r.model_name}_{r.level}"
        if key not in model_level_acc:
            model_level_acc[key] = {"total": 0, "correct": 0}
        model_level_acc[key]["total"] += 1
        model_level_acc[key]["correct"] += 1 if r.is_correct else 0

    # Model type comparison
    general_results = [r for r in results if r.model_type == "general"]
    reasoning_results = [r for r in results if r.model_type == "reasoning"]

    general_acc = sum(1 for r in general_results if r.is_correct) / len(general_results) * 100 if general_results else 0
    reasoning_acc = sum(1 for r in reasoning_results if r.is_correct) / len(reasoning_results) * 100 if reasoning_results else 0

    json_report = {
        "experiment": "model_comparison",
        "timestamp": timestamp,
        "summary": {
            "total_examples": len(set(r.example_id for r in results)),
            "total_evaluations": len(results),
            "models": list(by_model.keys()),
        },
        "model_type_comparison": {
            "general": {
                "accuracy": general_acc,
                "total": len(general_results),
                "correct": sum(1 for r in general_results if r.is_correct),
            },
            "reasoning": {
                "accuracy": reasoning_acc,
                "total": len(reasoning_results),
                "correct": sum(1 for r in reasoning_results if r.is_correct),
            },
        },
        "by_model": {
            model: {
                "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                **stats
            }
            for model, stats in by_model.items()
        },
        "by_level": {
            level: {
                "total": len(data),
                "correct": sum(1 for r in data if r.is_correct),
                "accuracy": sum(1 for r in data if r.is_correct) / len(data) * 100 if data else 0,
            }
            for level, data in by_level.items()
        },
        "detailed_results": [asdict(r) for r in results],
    }

    json_path = output_dir / f"model_comparison_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # HTML Report
    # =========================================================================

    # Prepare data for charts
    models = list(by_model.keys())
    model_accuracies = [by_model[m]["correct"] / by_model[m]["total"] * 100 for m in models]

    levels = ["easy", "medium", "hard"]
    level_accuracies = [
        sum(1 for r in by_level[l] if r.is_correct) / len(by_level[l]) * 100 if by_level[l] else 0
        for l in levels
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>모델 비교 분석 - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #00d4ff;
            font-size: 2.2em;
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .summary-label {{ color: #888; margin-top: 5px; font-size: 0.9em; }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .comparison-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
        }}
        .comparison-title {{
            color: #00d4ff;
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .accuracy-badge {{
            background: rgba(0,212,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
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
        }}
        .chart-title {{ color: #00d4ff; margin-bottom: 15px; font-size: 1.1em; }}

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
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 500;
        }}
        .badge-easy {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
        .badge-medium {{ background: rgba(251,191,36,0.2); color: #fbbf24; }}
        .badge-hard {{ background: rgba(248,113,113,0.2); color: #f87171; }}
        .badge-error {{ background: rgba(248,113,113,0.2); color: #f87171; }}
        .badge-general {{ background: rgba(0,212,255,0.2); color: #00d4ff; }}
        .badge-reasoning {{ background: rgba(167,139,250,0.2); color: #a78bfa; }}

        .section-title {{
            color: #00d4ff;
            font-size: 1.3em;
            margin: 25px 0 15px 0;
        }}

        .result-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .level-section {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 15px;
        }}
        .level-title {{
            font-size: 1.1em;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        /* 문제별 상세 분석 스타일 */
        .problem-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #00d4ff;
        }}
        .problem-card.all-correct {{ border-left-color: #4ade80; }}
        .problem-card.all-wrong {{ border-left-color: #f87171; }}
        .problem-card.mixed {{ border-left-color: #fbbf24; }}

        .problem-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        .problem-id {{
            font-size: 1.1em;
            font-weight: 600;
            color: #00d4ff;
        }}
        .problem-stats {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .question-box {{
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .question-label {{
            color: #888;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        .question-text {{
            line-height: 1.6;
            white-space: pre-wrap;
        }}

        .context-box {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}

        .gt-box {{
            display: inline-block;
            background: rgba(74,222,128,0.2);
            color: #4ade80;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin-bottom: 15px;
        }}

        .model-results-table {{
            width: 100%;
            margin-bottom: 10px;
        }}
        .model-results-table th,
        .model-results-table td {{
            padding: 10px;
            font-size: 0.9em;
        }}

        .detail-toggle {{
            background: rgba(0,212,255,0.2);
            border: none;
            color: #00d4ff;
            padding: 5px 12px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.8em;
            transition: all 0.2s;
        }}
        .detail-toggle:hover {{
            background: rgba(0,212,255,0.4);
        }}

        .detail-panel {{
            display: none;
            margin-top: 15px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }}
        .detail-panel.active {{
            display: block;
        }}

        .detail-section {{
            margin-bottom: 15px;
        }}
        .detail-section-title {{
            color: #fbbf24;
            font-size: 0.9em;
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .code-block {{
            background: #282c34;
            border-radius: 8px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85em;
            line-height: 1.5;
        }}
        .code-block pre {{
            margin: 0;
        }}

        .response-block {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.85em;
            line-height: 1.5;
            white-space: pre-wrap;
        }}

        .error-analysis {{
            background: rgba(248,113,113,0.1);
            border: 1px solid rgba(248,113,113,0.3);
            border-radius: 8px;
            padding: 12px;
        }}
        .error-analysis .error-category {{
            color: #f87171;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .error-analysis .error-evidence {{
            color: #ccc;
            font-size: 0.9em;
        }}

        /* 개선된 오답 분석 스타일 */
        .error-analysis-card {{
            background: linear-gradient(135deg, rgba(30,30,50,0.9) 0%, rgba(40,40,60,0.9) 100%);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 15px;
        }}
        .error-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 15px 20px;
            background: rgba(0,0,0,0.3);
        }}
        .error-icon {{
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3em;
        }}
        .error-icon.formula {{ background: rgba(251,191,36,0.2); }}
        .error-icon.misunderstanding {{ background: rgba(167,139,250,0.2); }}
        .error-icon.calculation {{ background: rgba(248,113,113,0.2); }}
        .error-icon.extraction {{ background: rgba(96,165,250,0.2); }}
        .error-icon.unit {{ background: rgba(74,222,128,0.2); }}
        .error-icon.execution {{ background: rgba(248,113,113,0.2); }}
        .error-icon.unknown {{ background: rgba(156,163,175,0.2); }}

        .error-type-info {{
            flex: 1;
        }}
        .error-type-label {{
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .error-type-name {{
            font-size: 1.1em;
            font-weight: 600;
            color: #fbbf24;
        }}
        .error-confidence {{
            background: rgba(255,255,255,0.1);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            color: #888;
        }}

        .error-body {{
            padding: 20px;
        }}
        .error-summary {{
            font-size: 1.05em;
            color: #fff;
            line-height: 1.5;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .error-details-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        @media (max-width: 800px) {{
            .error-details-grid {{ grid-template-columns: 1fr; }}
        }}

        .error-detail-box {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px 15px;
        }}
        .error-detail-box.full-width {{
            grid-column: 1 / -1;
        }}
        .error-detail-label {{
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .error-detail-label::before {{
            content: '';
            width: 3px;
            height: 12px;
            border-radius: 2px;
        }}
        .error-detail-label.cause::before {{ background: #f87171; }}
        .error-detail-label.solution::before {{ background: #4ade80; }}
        .error-detail-label.detail::before {{ background: #60a5fa; }}

        .error-detail-content {{
            color: #ccc;
            font-size: 0.9em;
            line-height: 1.6;
        }}

        .answer-comparison {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            padding: 12px 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        .answer-box {{
            flex: 1;
            text-align: center;
        }}
        .answer-label {{
            font-size: 0.75em;
            color: #888;
            margin-bottom: 4px;
        }}
        .answer-value {{
            font-size: 1.3em;
            font-weight: 600;
        }}
        .answer-value.wrong {{ color: #f87171; }}
        .answer-value.correct {{ color: #4ade80; }}
        .answer-arrow {{
            display: flex;
            align-items: center;
            color: #666;
            font-size: 1.2em;
        }}

        .nav-tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 10px;
        }}
        .nav-tab {{
            background: transparent;
            border: none;
            color: #888;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
        }}
        .nav-tab:hover {{ background: rgba(255,255,255,0.05); color: #fff; }}
        .nav-tab.active {{ background: rgba(0,212,255,0.2); color: #00d4ff; }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* 필터 컨트롤 */
        .filter-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: #ccc;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{ background: rgba(255,255,255,0.1); }}
        .filter-btn.active {{ background: rgba(0,212,255,0.2); border-color: #00d4ff; color: #00d4ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FinanceReasoning 모델 비교 분석</h1>
        <p class="subtitle">난이도별 동일 문제로 모델 성능 비교 | {timestamp}</p>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value">{len(set(r.example_id for r in results))}</div>
                <div class="summary-label">테스트 문제 수</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{len(models)}</div>
                <div class="summary-label">비교 모델 수</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{sum(1 for r in results if r.is_correct)}/{len(results)}</div>
                <div class="summary-label">정답 수</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">${sum(r.cost_usd for r in results):.4f}</div>
                <div class="summary-label">총 비용</div>
            </div>
        </div>

        <!-- Model Type Comparison -->
        <h2 class="section-title">모델 유형별 비교</h2>
        <div class="comparison-grid">
            <div class="comparison-card">
                <div class="comparison-title">
                    <span><span class="badge badge-general">General</span> 일반 모델</span>
                    <span class="accuracy-badge">{general_acc:.1f}%</span>
                </div>
                <p style="color: #888;">N = {len(general_results)}</p>
                <ul style="list-style: none; margin-top: 10px;">
                    {"".join(f'<li style="margin: 5px 0;">{m}: {by_model[m]["correct"]}/{by_model[m]["total"]}</li>' for m in models if MODEL_REGISTRY[m].model_type.value == "general")}
                </ul>
            </div>
            <div class="comparison-card">
                <div class="comparison-title">
                    <span><span class="badge badge-reasoning">Reasoning</span> 추론 모델</span>
                    <span class="accuracy-badge">{reasoning_acc:.1f}%</span>
                </div>
                <p style="color: #888;">N = {len(reasoning_results)}</p>
                <ul style="list-style: none; margin-top: 10px;">
                    {"".join(f'<li style="margin: 5px 0;">{m}: {by_model[m]["correct"]}/{by_model[m]["total"]}</li>' for m in models if MODEL_REGISTRY[m].model_type.value == "reasoning") if reasoning_results else '<li style="color: #666;">추론 모델 없음</li>'}
                </ul>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">모델별 정확도</div>
                <canvas id="modelChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">난이도별 정확도</div>
                <canvas id="levelChart"></canvas>
            </div>
        </div>

        <!-- Model Results Table -->
        <h2 class="section-title">모델별 성능</h2>
        <table>
            <thead>
                <tr>
                    <th>모델</th>
                    <th>유형</th>
                    <th>정답</th>
                    <th>정확도</th>
                    <th>비용</th>
                    <th>Easy</th>
                    <th>Medium</th>
                    <th>Hard</th>
                </tr>
            </thead>
            <tbody>
"""

    for model in models:
        stats = by_model[model]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        model_type = MODEL_REGISTRY[model].model_type.value

        # Per-level accuracy for this model
        easy_acc = sum(1 for r in results if r.model_name == model and r.level == "easy" and r.is_correct)
        medium_acc = sum(1 for r in results if r.model_name == model and r.level == "medium" and r.is_correct)
        hard_acc = sum(1 for r in results if r.model_name == model and r.level == "hard" and r.is_correct)

        easy_total = sum(1 for r in results if r.model_name == model and r.level == "easy")
        medium_total = sum(1 for r in results if r.model_name == model and r.level == "medium")
        hard_total = sum(1 for r in results if r.model_name == model and r.level == "hard")

        html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td><span class="badge badge-{model_type}">{model_type}</span></td>
                    <td>{stats["correct"]}/{stats["total"]}</td>
                    <td class="{"correct" if acc >= 50 else "incorrect"}">{acc:.1f}%</td>
                    <td>${stats["cost"]:.4f}</td>
                    <td>{easy_acc}/{easy_total}</td>
                    <td>{medium_acc}/{medium_total}</td>
                    <td>{hard_acc}/{hard_total}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <!-- Detailed Results by Level (간략 보기) -->
        <h2 class="section-title">문제별 요약</h2>
        <div class="result-grid">
"""

    for level in ["easy", "medium", "hard"]:
        level_results = by_level[level]
        badge_class = f"badge-{level}"

        html_content += f"""
            <div class="level-section">
                <div class="level-title"><span class="badge {badge_class}">{level.upper()}</span></div>
"""

        # Group by example
        examples_in_level = {}
        for r in level_results:
            if r.example_id not in examples_in_level:
                examples_in_level[r.example_id] = {"gt": r.ground_truth, "results": []}
            examples_in_level[r.example_id]["results"].append(r)

        for ex_id, ex_data in examples_in_level.items():
            correct_count = sum(1 for r in ex_data["results"] if r.is_correct)
            total_count = len(ex_data["results"])

            html_content += f"""
                <div style="margin: 10px 0; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <strong>{ex_id}</strong>
                        <span class="{"correct" if correct_count > total_count/2 else "incorrect"}">{correct_count}/{total_count}</span>
                    </div>
                    <div style="font-size: 0.85em; color: #888; margin-bottom: 8px;">GT: {ex_data["gt"]}</div>
                    <table style="font-size: 0.85em; margin: 0;">
                        <tr>
                            <th style="padding: 5px;">모델</th>
                            <th style="padding: 5px;">답변</th>
                            <th style="padding: 5px;">결과</th>
                        </tr>
"""

            for r in ex_data["results"]:
                status_class = "correct" if r.is_correct else "incorrect"
                status_icon = "✓" if r.is_correct else "✗"
                error_badge = f' <span class="badge badge-error">{r.error_category}</span>' if r.error_category else ""

                html_content += f"""
                        <tr>
                            <td style="padding: 5px;">{r.model_name}</td>
                            <td style="padding: 5px;">{r.predicted_answer}</td>
                            <td style="padding: 5px;" class="{status_class}">{status_icon}{error_badge}</td>
                        </tr>
"""

            html_content += """
                    </table>
                </div>
"""

        html_content += """
            </div>
"""

    html_content += """
        </div>

        <!-- 문제별 상세 분석 섹션 (새로 추가) -->
        <h2 class="section-title">문제별 상세 분석</h2>
        <p style="color: #888; margin-bottom: 15px;">각 문제의 전체 텍스트, 컨텍스트, 모델별 추론 과정을 확인할 수 있습니다.</p>

        <!-- 필터 컨트롤 -->
        <div class="filter-controls">
            <button class="filter-btn active" onclick="filterProblems('all')">전체</button>
            <button class="filter-btn" onclick="filterProblems('wrong')">오답 포함</button>
            <button class="filter-btn" onclick="filterProblems('easy')">Easy</button>
            <button class="filter-btn" onclick="filterProblems('medium')">Medium</button>
            <button class="filter-btn" onclick="filterProblems('hard')">Hard</button>
        </div>

        <div id="problem-details">
"""

    # 문제별 상세 분석 카드 생성
    all_examples = {}
    for r in results:
        if r.example_id not in all_examples:
            all_examples[r.example_id] = {
                "level": r.level,
                "question": r.question,
                "context": r.context,
                "ground_truth": r.ground_truth,
                "results": []
            }
        all_examples[r.example_id]["results"].append(r)

    for ex_id, ex_data in all_examples.items():
        correct_count = sum(1 for r in ex_data["results"] if r.is_correct)
        total_count = len(ex_data["results"])

        # 카드 스타일 결정
        if correct_count == total_count:
            card_class = "all-correct"
        elif correct_count == 0:
            card_class = "all-wrong"
        else:
            card_class = "mixed"

        level = ex_data["level"]
        badge_class = f"badge-{level}"

        # 컨텍스트 이스케이프 처리
        context_escaped = html_module.escape(str(ex_data["context"])[:2000]) if ex_data["context"] else "없음"
        question_escaped = html_module.escape(ex_data["question"])

        html_content += f"""
            <div class="problem-card {card_class}" data-level="{level}" data-has-wrong="{str(correct_count < total_count).lower()}">
                <div class="problem-header">
                    <div>
                        <span class="badge {badge_class}">{level.upper()}</span>
                        <span class="problem-id" style="margin-left: 10px;">{ex_id}</span>
                    </div>
                    <div class="problem-stats">
                        <span class="{"correct" if correct_count > total_count/2 else "incorrect"}" style="font-size: 1.2em; font-weight: bold;">
                            {correct_count}/{total_count}
                        </span>
                    </div>
                </div>

                <div class="question-box">
                    <div class="question-label">Question</div>
                    <div class="question-text">{question_escaped}</div>
                </div>

                <details style="margin-bottom: 15px;">
                    <summary style="cursor: pointer; color: #888; font-size: 0.9em;">Context 보기</summary>
                    <div class="context-box">{context_escaped}</div>
                </details>

                <div class="gt-box">Ground Truth: {ex_data["ground_truth"]}</div>

                <table class="model-results-table">
                    <thead>
                        <tr>
                            <th>모델</th>
                            <th>예측값</th>
                            <th>결과</th>
                            <th>에러 유형</th>
                            <th>상세</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        for idx, r in enumerate(ex_data["results"]):
            status_class = "correct" if r.is_correct else "incorrect"
            status_icon = "✓" if r.is_correct else "✗"
            error_cat = r.error_category if r.error_category else "-"
            panel_id = f"detail-{ex_id.replace(' ', '-')}-{r.model_name.replace(' ', '-')}-{idx}"

            html_content += f"""
                        <tr>
                            <td><strong>{r.model_name}</strong><br><small class="badge badge-{r.model_type}">{r.model_type}</small></td>
                            <td>{r.predicted_answer}</td>
                            <td class="{status_class}" style="font-size: 1.2em;">{status_icon}</td>
                            <td>{'<span class="badge badge-error">' + error_cat + '</span>' if r.error_category else '-'}</td>
                            <td><button class="detail-toggle" onclick="toggleDetail('{panel_id}')">상세 ▼</button></td>
                        </tr>
                        <tr>
                            <td colspan="5" style="padding: 0; border: none;">
                                <div id="{panel_id}" class="detail-panel">
"""

            # 오답 분석 (틀린 경우에만)
            if not r.is_correct:
                # 에러 유형별 아이콘 및 색상
                error_icons = {
                    'formula_error': ('🔢', 'formula', '수식 오류'),
                    'misunderstanding': ('🤔', 'misunderstanding', '문제 이해 오류'),
                    'calculation_error': ('🧮', 'calculation', '계산 오류'),
                    'numerical_calculation_error': ('🧮', 'calculation', '수치 계산 오류'),
                    'extraction_error': ('📊', 'extraction', '값 추출 오류'),
                    'unit_error': ('📏', 'unit', '단위 오류'),
                    'execution_error': ('⚠️', 'execution', '실행 오류'),
                    'logic_error': ('🔗', 'formula', '논리 오류'),
                    'incomplete': ('⏳', 'unknown', '불완전한 풀이'),
                    'unknown': ('❓', 'unknown', '미분류'),
                }
                error_key = error_cat.lower().replace(' ', '_') if error_cat else 'unknown'
                icon, icon_class, type_label = error_icons.get(error_key, ('❓', 'unknown', error_cat or '미분류'))

                # error_evidence 파싱
                evidence = r.error_evidence or ""
                summary = ""
                detailed = ""
                cause = ""
                solution = ""

                if evidence:
                    # LLM 분석 결과 파싱
                    import re as re_parse
                    # [TYPE] Summary 형식 파싱
                    summary_match = re_parse.search(r'\[[\w_]+\]\s*(.+?)(?:\n\n|$)', evidence)
                    if summary_match:
                        summary = summary_match.group(1).strip()

                    # 상세 분석 파싱
                    detail_match = re_parse.search(r'상세 분석:\s*(.+?)(?:\n\n|$)', evidence, re_parse.DOTALL)
                    if detail_match:
                        detailed = detail_match.group(1).strip()

                    # 원인 파싱
                    cause_match = re_parse.search(r'원인:\s*(.+?)(?:\n\n|$)', evidence, re_parse.DOTALL)
                    if cause_match:
                        cause = cause_match.group(1).strip()

                    # 올바른 접근 파싱
                    solution_match = re_parse.search(r'올바른 접근:\s*(.+?)(?:\n\n|$)', evidence, re_parse.DOTALL)
                    if solution_match:
                        solution = solution_match.group(1).strip()

                    # 파싱 실패시 전체 텍스트 사용
                    if not summary and not detailed:
                        summary = evidence[:200] + "..." if len(evidence) > 200 else evidence

                summary_escaped = html_module.escape(summary) if summary else "분석 정보 없음"
                detailed_escaped = html_module.escape(detailed) if detailed else ""
                cause_escaped = html_module.escape(cause) if cause else ""
                solution_escaped = html_module.escape(solution) if solution else ""

                html_content += f"""
                                    <div class="error-analysis-card">
                                        <div class="error-header">
                                            <div class="error-icon {icon_class}">{icon}</div>
                                            <div class="error-type-info">
                                                <div class="error-type-label">에러 유형</div>
                                                <div class="error-type-name">{type_label}</div>
                                            </div>
                                            <div class="error-confidence">신뢰도 {(r.error_confidence or 0)*100:.0f}%</div>
                                        </div>
                                        <div class="error-body">
                                            <div class="answer-comparison">
                                                <div class="answer-box">
                                                    <div class="answer-label">모델 예측</div>
                                                    <div class="answer-value wrong">{r.predicted_answer}</div>
                                                </div>
                                                <div class="answer-arrow">→</div>
                                                <div class="answer-box">
                                                    <div class="answer-label">정답</div>
                                                    <div class="answer-value correct">{r.ground_truth}</div>
                                                </div>
                                            </div>
                                            <div class="error-summary">{summary_escaped}</div>
                                            <div class="error-details-grid">
"""
                if detailed_escaped:
                    html_content += f"""
                                                <div class="error-detail-box full-width">
                                                    <div class="error-detail-label detail">상세 분석</div>
                                                    <div class="error-detail-content">{detailed_escaped}</div>
                                                </div>
"""
                if cause_escaped:
                    html_content += f"""
                                                <div class="error-detail-box">
                                                    <div class="error-detail-label cause">원인 추정</div>
                                                    <div class="error-detail-content">{cause_escaped}</div>
                                                </div>
"""
                if solution_escaped:
                    html_content += f"""
                                                <div class="error-detail-box">
                                                    <div class="error-detail-label solution">올바른 접근</div>
                                                    <div class="error-detail-content">{solution_escaped}</div>
                                                </div>
"""
                html_content += """
                                            </div>
                                        </div>
                                    </div>
"""

            # 실행된 코드 (POT인 경우)
            if r.executed_code:
                code_escaped = html_module.escape(r.executed_code)
                html_content += f"""
                                    <div class="detail-section">
                                        <div class="detail-section-title">실행된 코드</div>
                                        <div class="code-block"><pre><code class="language-python">{code_escaped}</code></pre></div>
                                    </div>
"""

            # 실행 에러 (있는 경우)
            if r.execution_error:
                exec_error_escaped = html_module.escape(r.execution_error)
                html_content += f"""
                                    <div class="detail-section">
                                        <div class="detail-section-title">실행 에러</div>
                                        <div class="error-analysis">
                                            <div class="error-evidence">{exec_error_escaped}</div>
                                        </div>
                                    </div>
"""

            # 모델 전체 응답
            raw_response_escaped = html_module.escape(r.raw_response[:3000] + "..." if len(r.raw_response) > 3000 else r.raw_response) if r.raw_response else "응답 없음"
            html_content += f"""
                                    <div class="detail-section">
                                        <div class="detail-section-title">모델 전체 응답</div>
                                        <div class="response-block">{raw_response_escaped}</div>
                                    </div>

                                    <div style="font-size: 0.85em; color: #666; margin-top: 10px;">
                                        비용: ${r.cost_usd:.4f} | 응답시간: {r.latency_seconds:.1f}s
                                    </div>
                                </div>
                            </td>
                        </tr>
"""

        html_content += """
                    </tbody>
                </table>
            </div>
"""

    html_content += f"""
        </div>

        <script>
            // Model accuracy chart
            new Chart(document.getElementById('modelChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(models)},
                    datasets: [{{
                        label: '정확도 (%)',
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
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            // Level accuracy chart
            new Chart(document.getElementById('levelChart'), {{
                type: 'bar',
                data: {{
                    labels: ['Easy', 'Medium', 'Hard'],
                    datasets: [{{
                        label: '정확도 (%)',
                        data: {json.dumps(level_accuracies)},
                        backgroundColor: ['rgba(74, 222, 128, 0.6)', 'rgba(251, 191, 36, 0.6)', 'rgba(248, 113, 113, 0.6)'],
                        borderColor: ['rgba(74, 222, 128, 1)', 'rgba(251, 191, 36, 1)', 'rgba(248, 113, 113, 1)'],
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
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            // 코드 하이라이팅 초기화
            hljs.highlightAll();

            // 상세 패널 토글
            function toggleDetail(panelId) {{
                const panel = document.getElementById(panelId);
                if (panel) {{
                    panel.classList.toggle('active');
                    // 코드 하이라이팅 재적용
                    if (panel.classList.contains('active')) {{
                        panel.querySelectorAll('pre code').forEach((block) => {{
                            hljs.highlightElement(block);
                        }});
                    }}
                }}
            }}

            // 문제 필터링
            function filterProblems(filter) {{
                // 버튼 활성화 상태 변경
                document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                // 문제 카드 필터링
                document.querySelectorAll('.problem-card').forEach(card => {{
                    const level = card.dataset.level;
                    const hasWrong = card.dataset.hasWrong === 'true';

                    let show = false;
                    if (filter === 'all') {{
                        show = true;
                    }} else if (filter === 'wrong') {{
                        show = hasWrong;
                    }} else {{
                        show = level === filter;
                    }}

                    card.style.display = show ? 'block' : 'none';
                }});
            }}
        </script>
    </div>
</body>
</html>
"""

    html_path = output_dir / f"model_comparison_{timestamp}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {"json": json_path, "html": html_path}


def print_summary(results: List[ComparisonResult]):
    """Print summary to console"""

    print("\n" + "=" * 90)
    print("모델 비교 실험 결과 요약")
    print("=" * 90)

    # By model
    by_model = {}
    for r in results:
        if r.model_name not in by_model:
            by_model[r.model_name] = {"total": 0, "correct": 0, "cost": 0.0}
        by_model[r.model_name]["total"] += 1
        by_model[r.model_name]["correct"] += 1 if r.is_correct else 0
        by_model[r.model_name]["cost"] += r.cost_usd

    print(f"\n{'모델':<20} {'정답':<10} {'정확도':<10} {'비용':<10}")
    print("-" * 50)
    for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["correct"]/x[1]["total"] if x[1]["total"] > 0 else 0):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{model:<20} {stats['correct']}/{stats['total']:<7} {acc:>6.1f}%    ${stats['cost']:.4f}")

    # By level
    print(f"\n난이도별 결과:")
    for level in ["easy", "medium", "hard"]:
        level_results = [r for r in results if r.level == level]
        correct = sum(1 for r in level_results if r.is_correct)
        total = len(level_results)
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {level.upper():<8}: {correct}/{total} ({acc:.1f}%)")

    # Model type comparison
    general = [r for r in results if r.model_type == "general"]
    reasoning = [r for r in results if r.model_type == "reasoning"]

    print(f"\n모델 유형별 비교:")
    if general:
        gen_acc = sum(1 for r in general if r.is_correct) / len(general) * 100
        print(f"  General:   {sum(1 for r in general if r.is_correct)}/{len(general)} ({gen_acc:.1f}%)")
    if reasoning:
        rea_acc = sum(1 for r in reasoning if r.is_correct) / len(reasoning) * 100
        print(f"  Reasoning: {sum(1 for r in reasoning if r.is_correct)}/{len(reasoning)} ({rea_acc:.1f}%)")

    print(f"\n총 비용: ${sum(r.cost_usd for r in results):.4f}")


async def main():
    parser = argparse.ArgumentParser(description="Model Comparison Experiment")
    parser.add_argument("--budget", type=str, choices=["economic", "balanced", "full"], default="economic")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models")
    parser.add_argument("--n", type=int, default=3, help="Examples per level")
    parser.add_argument("--methods", type=str, default="POT", help="Methods (COT,POT)")
    parser.add_argument("--no-llm-analysis", action="store_true", help="Disable LLM-based error analysis")
    args = parser.parse_args()

    print("=" * 80)
    print("모델 비교 실험")
    print("각 난이도(Easy/Medium/Hard)에서 동일 문제로 모델 성능 비교")
    print("=" * 80)

    # Load environment
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    # Determine models
    if args.models:
        requested_models = args.models.split(",")
    else:
        requested_models = BUDGET_MODEL_SETS[args.budget]

    print(f"\n[1] API 키 확인...")
    available_models = get_available_models(requested_models)

    if not available_models:
        print("ERROR: 사용 가능한 모델이 없습니다.")
        return

    # Load datasets
    print(f"\n[2] 데이터셋 로딩...")
    easy = load_dataset("easy")
    medium = load_dataset("medium")
    hard = load_dataset("hard")
    print(f"    Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}")

    # Select examples
    print(f"\n[3] 문제 선택 (난이도별 {args.n}개)...")
    examples = select_examples(easy, medium, hard, n=args.n)
    print(f"    총 {len(examples)}개 문제 선택")
    for ex in examples:
        print(f"      [{ex['level'].upper()}] {ex['question_id']}: GT={ex['ground_truth']}")

    # Run experiment
    print(f"\n[4] 실험 실행...")
    enable_llm_analysis = not args.no_llm_analysis
    if enable_llm_analysis:
        print("    LLM 기반 오답 분석 활성화 (gpt-4o-mini)")
    experiment = ModelComparisonExperiment(available_models, enable_llm_analysis=enable_llm_analysis)
    results = await experiment.run(examples, methods=args.methods.split(","))

    # Generate reports
    print(f"\n[5] 리포트 생성...")
    output_dir = project_root / "experiments" / "results" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_paths = generate_comparison_report(results, output_dir)
    print(f"    JSON: {report_paths['json']}")
    print(f"    HTML: {report_paths['html']}")

    # Print summary
    print_summary(results)

    print(f"\n[완료] 실험 완료!")
    print(f"    리포트: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
