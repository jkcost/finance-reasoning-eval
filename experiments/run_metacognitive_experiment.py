"""
Metacognitive Financial LLM Evaluation Experiment

"Do Financial LLMs Know What They Don't Know?"
Evaluates LLMs' ability to recognize information insufficiency
via controlled data manipulation.

Phases:
  A - Baseline accuracy on original problems
  B - Metacognitive test on transformed (unsolvable) problems
  C - RAG impact on metacognitive ability
  D - Cost-optimal strategy analysis (no new API calls)

Usage:
    python experiments/run_metacognitive_experiment.py --phase A --budget economic --n 5
    python experiments/run_metacognitive_experiment.py --phase B --budget economic --n 5 --prompt-strategy metacognitive
    python experiments/run_metacognitive_experiment.py --phase C --budget economic --n 5
    python experiments/run_metacognitive_experiment.py --phase D --results-dir experiments/results/metacognitive/
"""

import asyncio
import json
import sys
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ModelConfig
from model_runner import OpenAIProvider, AnthropicProvider, GoogleProvider

from error_analysis import MODEL_REGISTRY
from error_analysis.error_classifier import ErrorClassifier
from error_analysis.error_taxonomy import BUDGET_MODEL_SETS

from metacognitive_metrics import (
    MetacognitiveResult,
    ResponseType,
    compute_metrics,
    compute_metrics_by_dimension,
    compute_cross_phase_metrics,
)
from refusal_detector import RefusalDetector

# Import transformation utilities
sys.path.insert(0, str(Path(__file__).parent))
from apply_transformations_full import (  # noqa: E402
    apply_transformations,
    normalize_transformation_label,
    validate_transformation,
)


# ============================================================================
# PROMPT STRATEGIES
# ============================================================================

# ============================================================================
# PROMPT STRATEGY: standard
# Source: FinanceReasoning (ACL 2025), BUPT-Reasoning/FinanceReasoning GitHub
#   - utils/prompts.py의 COT/POT system message 원본 그대로 사용
#   - 대조군(control group) 역할: 거부 지시 없이 항상 답변 생성
#   - 논문은 unsolvable 문제를 의도적으로 배제 (future work로 남김):
#     "modeling LRMs' proactive clarification-seeking behavior under
#      insufficient conditions falls outside this work's scope"
# ============================================================================
COT_SYSTEM_STANDARD = (
    "You are a financial expert, you are supposed to answer the given question. "
    "You need to first think through the problem step by step, identifying the exact "
    "variables and values, and documenting each necessary step. Then you are required "
    "to conclude your response with the final answer in your last sentence as "
    "'Therefore, the answer is {final answer}'. The final answer should be a numeric value."
)

POT_SYSTEM_STANDARD = """You are a financial expert, you are supposed to generate a Python program to answer the given question. The returned value of the program is supposed to be the answer. Here is an example of the Python program:
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

# ============================================================================
# PROMPT STRATEGY: metacognitive
# Design: standard 프롬프트에 명시적 거부(abstention) 지시를 추가한 변형
#
# Literature basis:
#   - Feng+2024 (Abstention F1): "If you cannot answer, say 'I don't know'"
#     → 구조화된 변형: INSUFFICIENT_INFORMATION 토큰으로 파싱 가능하게 설계
#   - Kadavath+2022 ("Language Models (Mostly) Know What They Know"):
#     LLM이 자기 불확실성을 인식하는 능력(calibration)이 있음을 보임
#     → "verify that ALL required data is present" 지시의 이론적 근거
#   - Yin+2023 ("Do LLMs Know What They Don't Know"):
#     모델 거부 행동을 직접 거부/조건부 답변/환각적 자신감 3유형으로 분류
#     → RefusalDetector의 REFUSED/CAVEAT/CONFIDENT 3분류 체계의 근거
#
# Key design decisions:
#   - "Do NOT guess, assume, or fabricate" = R-Tuning (Zhang+2023)의
#     anti-hallucination 지시 패턴 차용
#   - INSUFFICIENT_INFORMATION 토큰 = Feng+2024의 "I don't know" 신호를
#     POT(프로그램 출력)에서도 파싱 가능하도록 구조화
# ============================================================================
COT_SYSTEM_METACOGNITIVE = (
    "You are a financial expert, you are supposed to answer the given question. "
    "You need to first think through the problem step by step, identifying the exact "
    "variables and values, and documenting each necessary step.\n\n"
    "IMPORTANT: Before solving, verify that ALL required data is present in the context. "
    "If any critical information is missing, contradictory, or ambiguous, respond with:\n"
    "'INSUFFICIENT_INFORMATION: [explain what data is missing or inconsistent]'\n"
    "Do NOT guess, assume, or fabricate missing values.\n\n"
    "If the data is sufficient, conclude with 'Therefore, the answer is {final answer}'."
)

POT_SYSTEM_METACOGNITIVE = """You are a financial expert, you are supposed to generate a Python program to answer the given question.

IMPORTANT: Before writing code, verify that ALL required data is present in the context.
If any critical information is missing, contradictory, or ambiguous, generate:
```python
def solution():
    return "INSUFFICIENT_INFORMATION: [explain exactly what data is missing or contradictory]"
```
Replace the bracketed text with a specific description, e.g.:
  return "INSUFFICIENT_INFORMATION: Q4 2023 revenue figure is absent from the context"
  return "INSUFFICIENT_INFORMATION: Operating expenses for 2022 are contradictory — two values given"
Do NOT guess, assume, or fabricate missing values.

If the data is sufficient, generate the solution program. Example:
```python
def solution():
    revenue = 600000
    avg_account_receivable = 50000
    receivables_turnover = revenue / avg_account_receivable
    answer = 365 / receivables_turnover
    return answer
```
"""

# ============================================================================
# PROMPT STRATEGY: self_verification
# Design: Chain-of-Verification (CoVe) 패러다임의 pre-generation 변형
#
# Literature basis:
#   - Dhuliawala+2023 (Chain-of-Verification, CoVe):
#     4단계 파이프라인 (Draft → Plan Verification → Execute → Final)
#     → 우리 변형: 생성 후 검증이 아닌 생성 전 데이터 감사(pre-generation audit)
#     → STEP 1 DATA AUDIT = CoVe의 "Plan Verification Questions" 단계에 대응
#   - Rajpurkar+2018 (SQuAD 2.0):
#     답변 불가능 질문(unanswerable)을 평가 체계에 최초 도입
#     → STATUS: AVAILABLE/MISSING 이진 분류가 SQuAD 2.0의
#       has-answer/no-answer 판정과 구조적으로 동일
#
# Key design decisions:
#   - 2단계 강제 분리: 데이터 목록화를 풀이보다 반드시 먼저 수행하게 함
#   - "STOP" 지시: MISSING 발견 시 STEP 2로 진행하지 못하게 명시적 차단
# ============================================================================
COT_SYSTEM_SELF_VERIFICATION = (
    "You are a financial expert. Follow this 2-step process:\n\n"
    "STEP 1 - DATA AUDIT:\n"
    "List ALL data values required to solve this problem.\n"
    "For each required value, mark STATUS: AVAILABLE or MISSING.\n"
    "If any value is MISSING or data is CONTRADICTORY, state:\n"
    "'INSUFFICIENT_INFORMATION: [details]' and STOP.\n\n"
    "STEP 2 - SOLUTION (only if ALL data is AVAILABLE):\n"
    "Solve the problem step by step.\n"
    "Conclude with 'Therefore, the answer is {final answer}'."
)

POT_SYSTEM_SELF_VERIFICATION = """You are a financial expert. Follow this 2-step process:

STEP 1 - DATA AUDIT:
List ALL data values required to solve this problem.
For each required value, mark STATUS: AVAILABLE or MISSING.
If any value is MISSING or data is CONTRADICTORY, generate:
```python
def solution():
    # DATA AUDIT: [list missing/contradictory values here]
    return "INSUFFICIENT_INFORMATION: [specific description of missing or contradictory data]"
```

STEP 2 - SOLUTION (only if ALL data is AVAILABLE):
```python
def solution():
    # Define variables from context
    # Calculate answer
    return answer
```
"""

# ============================================================================
# PROMPT STRATEGY: contradiction_aware
# Design: 모순 탐지를 최우선 단계로 배치한 3단계 파이프라인 (novel contribution)
#
# Motivation:
#   - Phase B 실험에서 Type 5 변환(모순 정보 삽입)에 전 모델이 실패
#   - 기존 3개 전략 모두 모순 탐지 능력 부재 확인 → 명시적 지시 필요
#
# Literature basis:
#   - NLI (Natural Language Inference) 연구에서 contradiction/entailment/neutral
#     3분류 체계를 차용 → STEP 1의 "CONTRADICTIONS or DISCREPANCIES" 탐지
#   - Dhuliawala+2023 (CoVe): self_verification의 2단계에 consistency check를
#     선행 단계로 추가하여 3단계로 확장
#
# Key design decisions:
#   - 모순 탐지를 STEP 1에 배치: 불완전성 검사(STEP 2)보다 선행시켜,
#     모순 데이터로 계산하는 것을 원천 차단
#   - CONTRADICTION_DETECTED 토큰: INSUFFICIENT_INFORMATION과 별도로 두어
#     모순 탐지와 누락 탐지를 구분 가능하게 설계
# ============================================================================
COT_SYSTEM_CONTRADICTION_AWARE = (
    "You are a financial expert. Follow this 3-step process:\n\n"
    "STEP 1 - DATA CONSISTENCY CHECK:\n"
    "Scan the context for any CONTRADICTIONS or DISCREPANCIES.\n"
    "Compare all values for the same metric/item. If two different values exist "
    "for the same data point, flag: 'CONTRADICTION_DETECTED: [details]'\n\n"
    "STEP 2 - DATA COMPLETENESS CHECK:\n"
    "List ALL required data values. For each, mark: AVAILABLE or MISSING.\n"
    "If any value is MISSING, state: 'INSUFFICIENT_INFORMATION: [details]'\n\n"
    "STEP 3 - SOLUTION (only if NO contradictions AND ALL data available):\n"
    "If contradictions were found, respond with:\n"
    "'INSUFFICIENT_INFORMATION: Contradictory data detected — [details]'\n"
    "Otherwise, solve step by step and conclude with "
    "'Therefore, the answer is {final answer}'."
)

POT_SYSTEM_CONTRADICTION_AWARE = """You are a financial expert. Follow this 3-step process:

STEP 1 - DATA CONSISTENCY CHECK:
Scan the context for any CONTRADICTIONS or DISCREPANCIES.
If two different values exist for the same data point, generate:
```python
def solution():
    return "INSUFFICIENT_INFORMATION: CONTRADICTION_DETECTED — [describe conflicting values, e.g. revenue is both $10M and $15M]"
```

STEP 2 - DATA COMPLETENESS CHECK:
List ALL required data values. If any value is MISSING, generate:
```python
def solution():
    return "INSUFFICIENT_INFORMATION: [list the specific missing values required to solve this problem]"
```

STEP 3 - SOLUTION (only if NO contradictions AND ALL data available):
```python
def solution():
    # Define variables from context
    # Calculate answer
    return answer
```
"""

COT_PREFIX = "Let's think step by step to answer the given question.\n"

POT_PREFIX = """Please generate a Python program to answer the given question. The format of the program should be the following:
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
# PROMPT STRATEGY: ic_fewshot (Mitigation — few-shot IC examples)
# Design: Provide concrete examples of information conflicts before the task,
#         teaching the model what contradictions look like in financial data.
# ============================================================================
COT_SYSTEM_IC_FEWSHOT = (
    "You are a financial expert with strong data verification skills.\n\n"
    "CRITICAL: Financial data can contain contradictions from different sources. "
    "Before solving, you MUST cross-check all numeric values in the context.\n\n"
    "EXAMPLE of a data contradiction:\n"
    "  Context says: 'Revenue was $10 million in 2023.'\n"
    "  Later it says: 'Per the audited report, 2023 revenue was $15 million.'\n"
    "  → These two values conflict. The correct response is:\n"
    "  'INSUFFICIENT_INFORMATION: Contradictory data — revenue reported as both "
    "$10M and $15M.'\n\n"
    "Follow these steps:\n"
    "1. List every numeric value and its source in the context.\n"
    "2. Check if any value appears twice with different numbers.\n"
    "3. If contradictions exist, respond with 'INSUFFICIENT_INFORMATION: "
    "Contradictory data detected — [details]'.\n"
    "4. Only if all data is consistent, solve step by step and conclude with "
    "'Therefore, the answer is {final answer}'."
)

POT_SYSTEM_IC_FEWSHOT = """You are a financial expert with strong data verification skills.

CRITICAL: Financial data can contain contradictions from different sources.
Before writing code, you MUST cross-check all numeric values.

EXAMPLE of a data contradiction:
  Context says: 'Revenue was $10 million in 2023.'
  Later it says: 'Per the audited report, 2023 revenue was $15 million.'
  → These conflict. Generate:
```python
def solution():
    return "INSUFFICIENT_INFORMATION: CONTRADICTION_DETECTED — 2023 revenue is reported as both $10M and $15M"
```

Steps:
1. List every numeric value and its source.
2. Check for any value appearing twice with different numbers.
3. If contradictions exist, return "INSUFFICIENT_INFORMATION: CONTRADICTION_DETECTED — [details]".
4. Only if consistent, solve with code.
"""

# ============================================================================
# PROMPT STRATEGY: ic_crosscheck (Mitigation — explicit cross-check instruction)
# Design: System instruction that mandates cross-checking every extracted value
#         against all other mentions in the context before using it.
# ============================================================================
COT_SYSTEM_IC_CROSSCHECK = (
    "You are a financial expert. You MUST follow this verification protocol:\n\n"
    "MANDATORY VERIFICATION PROTOCOL:\n"
    "For EVERY numeric value you extract from the context:\n"
    "  1. Search the ENTIRE context for other mentions of the same metric.\n"
    "  2. If you find different values for the same metric, STOP immediately.\n"
    "  3. Report: 'INSUFFICIENT_INFORMATION: Value conflict for [metric] — "
    "[value1] vs [value2].'\n\n"
    "DO NOT proceed with calculation if ANY value conflict is found.\n"
    "DO NOT assume either value is correct — both are unreliable.\n\n"
    "Only after verifying ALL values are internally consistent, solve step by "
    "step and conclude with 'Therefore, the answer is {final answer}'."
)

POT_SYSTEM_IC_CROSSCHECK = """You are a financial expert. MANDATORY VERIFICATION PROTOCOL:

For EVERY numeric value you extract:
  1. Search the ENTIRE context for other mentions of the same metric.
  2. If different values exist for the same metric, generate:
```python
def solution():
    return "INSUFFICIENT_INFORMATION: VALUE_CONFLICT — [metric] appears as [v1] and [v2] in context"
```

DO NOT proceed with calculation if ANY value conflict exists.
DO NOT assume either value is correct.

Only after verifying ALL values are consistent, write solution code.
"""

PROMPT_SYSTEMS = {
    "standard": {"COT": COT_SYSTEM_STANDARD, "POT": POT_SYSTEM_STANDARD},
    "metacognitive": {"COT": COT_SYSTEM_METACOGNITIVE, "POT": POT_SYSTEM_METACOGNITIVE},
    "self_verification": {
        "COT": COT_SYSTEM_SELF_VERIFICATION,
        "POT": POT_SYSTEM_SELF_VERIFICATION,
    },
    "contradiction_aware": {
        "COT": COT_SYSTEM_CONTRADICTION_AWARE,
        "POT": POT_SYSTEM_CONTRADICTION_AWARE,
    },
    "ic_fewshot": {
        "COT": COT_SYSTEM_IC_FEWSHOT,
        "POT": POT_SYSTEM_IC_FEWSHOT,
    },
    "ic_crosscheck": {
        "COT": COT_SYSTEM_IC_CROSSCHECK,
        "POT": POT_SYSTEM_IC_CROSSCHECK,
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================


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


def select_examples(datasets: Dict[str, List[Dict]], n: int) -> List[Dict]:
    """Select n examples from each level with uniform spacing"""
    selected = []
    for level_name, level_data in datasets.items():
        total = len(level_data)
        if n >= total:
            indices = list(range(total))
        else:
            step = total / n
            indices = [int(i * step) for i in range(n)]

        for idx in indices:
            example = level_data[idx].copy()
            example["level"] = level_name
            selected.append(example)

    return selected


def get_available_models(requested_models: List[str]) -> List[str]:
    """Filter models by available API keys"""
    load_dotenv()
    available = []
    for model_name in requested_models:
        if model_name not in MODEL_REGISTRY:
            print(f"[WARN] Unknown model: {model_name}")
            continue
        model_info = MODEL_REGISTRY[model_name]
        if model_info.provider == "ollama":
            available.append(model_name)
            print(f"[OK] {model_name}: local Ollama model")
        elif os.environ.get(model_info.api_key_env):
            available.append(model_name)
            print(f"[OK] {model_name}: API key found")
        else:
            print(f"[SKIP] {model_name}: No API key ({model_info.api_key_env})")
    return available


# ============================================================================
# METACOGNITIVE EXPERIMENT
# ============================================================================


class MetacognitiveExperiment:
    """Metacognitive evaluation experiment orchestrator"""

    def __init__(
        self,
        models: List[str],
        method: str = "POT",
        enable_rag: bool = False,
    ):
        self.models = models
        self.method = method
        self.enable_rag = enable_rag
        self.classifier = ErrorClassifier()
        self.refusal_detector = RefusalDetector()

        # Initialize RAG if enabled
        self.rag_enhancer = None
        if enable_rag:
            try:
                from rag_enhancer import RAGEnhancer

                eval_dir = Path(__file__).parent.parent / "evaluation"
                self.rag_enhancer = RAGEnhancer(eval_dir / "function_retriever.py")
                print("[OK] RAG enhancer loaded")
            except Exception as e:
                print(f"[WARN] RAG enhancer failed to load: {e}")

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
            elif model_info.provider == "ollama":
                from model_runner import OllamaProvider
                self.providers[model_name] = OllamaProvider(model_config)

    def _build_prompt(
        self,
        example: Dict,
        prompt_strategy: str,
    ) -> Tuple[str, str]:
        """Build prompt with the specified strategy"""
        context = example.get("context", "")
        question = example.get("question", "")

        if context and context != "[]":
            question_input = (
                f"The following question context is provided for your reference.\n"
                f"{context}\n\nQuestion: {question}\n"
            )
        else:
            question_input = f"Question: {question}\n"

        system_prompt = PROMPT_SYSTEMS[prompt_strategy][self.method]

        if self.method == "COT":
            user_prompt = question_input + "\n" + COT_PREFIX
        else:
            user_prompt = question_input + "\n" + POT_PREFIX

        # Apply RAG enhancement if enabled
        if self.rag_enhancer:
            try:
                template = "cot_rag" if self.method == "COT" else "pot_rag"
                enhanced = self.rag_enhancer.enhance_prompt(
                    user_prompt, question, context, template_type=template
                )
                user_prompt = enhanced
            except Exception as e:
                print(f"    [RAG Warning] {e}")

        return system_prompt, user_prompt

    def _extract_answer(
        self, response: str
    ) -> Tuple[Any, Optional[str], Optional[str]]:
        """Extract answer from model response (COT or POT)"""
        if self.method == "COT":
            return self._extract_cot_answer(response), None, None
        return self._execute_pot_code(response)

    def _extract_cot_answer(self, response: str) -> Any:
        """Extract numeric answer from COT response"""
        # Check for INSUFFICIENT_INFORMATION or CONTRADICTION_DETECTED first
        if re.search(r"(?i)INSUFFICIENT[_ ]INFORMATION", response):
            return "INSUFFICIENT_INFORMATION"
        if re.search(r"(?i)CONTRADICTION[_ ]DETECTED", response):
            return "INSUFFICIENT_INFORMATION"

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

    def _execute_pot_code(
        self, response: str
    ) -> Tuple[Any, Optional[str], Optional[str]]:
        """Execute POT code and return (answer, code, error)"""
        code_patterns = [r"```python\s*(.*?)```", r"```\s*(.*?)```"]
        code = None
        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                break
        if not code:
            code = response.strip()

        # Check for INSUFFICIENT_INFORMATION or CONTRADICTION_DETECTED in code
        if "INSUFFICIENT_INFORMATION" in code or "CONTRADICTION_DETECTED" in code:
            return "INSUFFICIENT_INFORMATION", code, None

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
                result = local_vars["result"]
                if result == "INSUFFICIENT_INFORMATION":
                    return "INSUFFICIENT_INFORMATION", code, None
                return result, code, None
            elif "answer" in local_vars:
                return local_vars["answer"], code, None
            return None, code, "No 'result' or 'answer' variable found"
        except Exception as e:
            return None, code, str(e)

    def _check_answer(self, pred: Any, truth: Any, tolerance: float = 0.002) -> bool:
        """Check if prediction matches ground truth"""
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

    def _calculate_cost(
        self, input_tokens: int, output_tokens: int, model_name: str
    ) -> float:
        """Calculate API cost"""
        model_info = MODEL_REGISTRY[model_name]
        input_cost = (input_tokens / 1_000_000) * model_info.cost_per_million_input
        output_cost = (output_tokens / 1_000_000) * model_info.cost_per_million_output
        return round(input_cost + output_cost, 6)

    @staticmethod
    def _truncate(text: str, max_len: int = 50000) -> str:
        """Truncate text for storage"""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    async def evaluate_single(
        self,
        example: Dict,
        model_name: str,
        prompt_strategy: str,
        transformation_type: str = "original",
        original_context: str = "",
        transformation_description: str = "",
    ) -> MetacognitiveResult:
        """Evaluate a single example with metacognitive analysis"""
        provider = self.providers[model_name]
        system_prompt, user_prompt = self._build_prompt(example, prompt_strategy)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        class MinimalExample:
            def __init__(self, ex):
                self.id = ex.get("question_id", "unknown")
                self.question = ex.get("question", "")
                self.context = ex.get("context", "")
                self.ground_truth = ex.get("ground_truth", "")
                self.python_solution = ex.get("python_solution", "")

        mini_ex = MinimalExample(example)

        # Prepare context fields for result
        question_text = example.get("question", "")
        current_context = example.get("context", "")
        ground_truth = example.get("ground_truth")
        ctx_original = self._truncate(original_context or current_context)
        ctx_transformed = (
            self._truncate(current_context) if transformation_type != "original" else ""
        )

        try:
            response = await provider.call_model(mini_ex, full_prompt)
            raw_response = response.raw_response
            input_tokens = response.prompt_tokens
            output_tokens = response.completion_tokens
            latency = response.response_time_seconds
            cost = self._calculate_cost(input_tokens, output_tokens, model_name)

            answer, executed_code, execution_error = self._extract_answer(raw_response)

            # Detect refusal / caveat / confidence
            detection = self.refusal_detector.detect(
                raw_response,
                context=current_context,
                execution_error=execution_error,
                executed_code=executed_code,
            )

            # Override detection if answer is explicitly INSUFFICIENT_INFORMATION
            if answer == "INSUFFICIENT_INFORMATION":
                response_type = ResponseType.REFUSED.value
            else:
                response_type = detection.response_type.value

            # Check baseline correctness (for original problems)
            is_original_correct = None
            if transformation_type == "original":
                is_original_correct = self._check_answer(answer, ground_truth)

            return MetacognitiveResult(
                example_id=example.get("question_id", "unknown"),
                model_name=model_name,
                method=self.method,
                prompt_strategy=prompt_strategy,
                transformation_type=transformation_type,
                response_type=response_type,
                raw_response=raw_response,
                predicted_answer=answer,
                is_original_correct=is_original_correct,
                cost_usd=cost,
                latency_seconds=latency,
                hallucinated_values=detection.hallucinated_values,
                refusal_patterns_matched=detection.refusal_patterns_matched,
                confidence_indicators=detection.confidence_patterns_matched,
                rag_enabled=self.enable_rag,
                question=question_text,
                context_original=ctx_original,
                context_transformed=ctx_transformed,
                transformation_description=transformation_description,
                ground_truth=ground_truth,
            )

        except Exception:
            return MetacognitiveResult(
                example_id=example.get("question_id", "unknown"),
                model_name=model_name,
                method=self.method,
                prompt_strategy=prompt_strategy,
                transformation_type=transformation_type,
                response_type=ResponseType.ERROR.value,
                raw_response="",
                predicted_answer=None,
                cost_usd=0.0,
                latency_seconds=0.0,
                rag_enabled=self.enable_rag,
                question=question_text,
                context_original=ctx_original,
                context_transformed=ctx_transformed,
                transformation_description=transformation_description,
                ground_truth=ground_truth,
            )

    async def run_phase_a(
        self, examples: List[Dict], prompt_strategy: str = "standard"
    ) -> List[MetacognitiveResult]:
        """Phase A: Baseline accuracy on original problems"""
        print(f"\n{'=' * 80}")
        print(
            f"Phase A: Baseline Accuracy (n={len(examples)}, strategy={prompt_strategy})"
        )
        print(f"{'=' * 80}")

        results = []
        total = len(examples) * len(self.models)
        current = 0

        for example in examples:
            eid = example.get("question_id", "unknown")
            level = example.get("level", "unknown")
            print(f"\n[{level.upper()}] {eid} | GT={example.get('ground_truth', '')}")

            for model_name in self.models:
                current += 1
                model_info = MODEL_REGISTRY[model_name]
                print(
                    f"  [{current}/{total}] {model_info.display_name}...",
                    end=" ",
                    flush=True,
                )

                result = await self.evaluate_single(
                    example, model_name, prompt_strategy, "original"
                )
                results.append(result)

                status = "OK" if result.is_original_correct else "X"
                print(f"[{status}] {result.predicted_answer} (${result.cost_usd:.4f})")

        return results

    async def run_phase_b(
        self,
        examples: List[Dict],
        prompt_strategy: str = "metacognitive",
        transformation_types: Optional[List[str]] = None,
    ) -> List[MetacognitiveResult]:
        """Phase B: Metacognitive test on transformed problems"""
        print(f"\n{'=' * 80}")
        print(
            f"Phase B: Metacognitive Test (n={len(examples)}, strategy={prompt_strategy})"
        )
        print(f"{'=' * 80}")

        results = []
        transform_count = 0
        skip_count = 0

        for example in examples:
            eid = example.get("question_id", "unknown")
            level = example.get("level", "unknown")
            transformed_list = apply_transformations(example)

            if not transformed_list:
                print(f"  [SKIP] {eid}: no transformations applicable")
                skip_count += 1
                continue

            # Filter by transformation type if specified
            if transformation_types:
                transformed_list = [
                    t
                    for t in transformed_list
                    if any(
                        tt in t.get("transformation_type", "")
                        for tt in transformation_types
                    )
                ]

            original_context = example.get("context", "")

            for trans in transformed_list:
                trans_type = trans.get("transformation_type", "unknown")
                trans_desc = trans.get("transformation_description", "")

                # Validate transformation
                validation = validate_transformation(trans, example)
                if not validation["valid"]:
                    if validation["reason"] == "hardcoded_solution":
                        print(f"  [SKIP-HC] {eid} {trans_type}: hardcoded solution")
                    else:
                        print(f"  [INVALID] {eid} {trans_type}: {validation['reason']}")
                    continue

                transform_count += 1
                print(f"\n  [{level.upper()}] {eid} | {trans_type}")
                print(f"    {trans_desc}")

                for model_name in self.models:
                    model_info = MODEL_REGISTRY[model_name]
                    print(f"    {model_info.display_name}...", end=" ", flush=True)

                    result = await self.evaluate_single(
                        trans,
                        model_name,
                        prompt_strategy,
                        trans_type,
                        original_context=original_context,
                        transformation_description=trans_desc,
                    )
                    results.append(result)

                    icon = {
                        "refused": "REFUSE",
                        "caveat": "CAVEAT",
                        "confident": "CONF",
                        "error": "ERR",
                    }.get(result.response_type, "?")
                    print(f"[{icon}] (${result.cost_usd:.4f})")

        print(
            f"\nPhase B Summary: {transform_count} transformations, {skip_count} skipped"
        )
        return results

    async def run_phase_c(
        self,
        examples: List[Dict],
        prompt_strategy: str = "metacognitive",
    ) -> List[MetacognitiveResult]:
        """Phase C: RAG impact test (same as Phase B but with RAG enabled)"""
        if not self.rag_enhancer:
            print("[ERROR] RAG enhancer not available. Skipping Phase C.")
            return []

        print(f"\n{'=' * 80}")
        print(
            f"Phase C: RAG Impact Test (n={len(examples)}, strategy={prompt_strategy})"
        )
        print(f"{'=' * 80}")

        # Phase C reuses Phase B logic with RAG already enabled
        return await self.run_phase_b(examples, prompt_strategy)


def analyze_phase_d(results_dir: Path) -> Dict[str, Any]:
    """Phase D: Cost-optimal strategy analysis from saved results"""
    print(f"\n{'=' * 80}")
    print("Phase D: Cost-Optimal Strategy Analysis")
    print(f"{'=' * 80}")

    all_results = []
    result_files = sorted(results_dir.glob("phase_*.json"))

    for fp in result_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        phase_results = []
        for r in data.get("results", []):
            result = MetacognitiveResult.from_dict(r)
            result.transformation_type = normalize_transformation_label(
                result.transformation_type
            )
            phase_results.append(result)
        all_results.extend(phase_results)
        print(f"  Loaded {len(phase_results)} results from {fp.name}")

    if not all_results:
        print("[ERROR] No results found")
        return {}

    # Split Phase A (solvable) and Phase B (unsolvable) results
    phase_a_results = [r for r in all_results if r.transformation_type == "original"]
    phase_b_results = [r for r in all_results if r.transformation_type != "original"]

    # Compute cross-phase metrics (Phase A + B merged) when both available
    if phase_a_results and phase_b_results:
        model_metrics = compute_cross_phase_metrics(phase_a_results, phase_b_results)
        print(
            f"  Cross-phase metrics: {len(phase_a_results)} solvable + {len(phase_b_results)} unsolvable"
        )
    else:
        model_metrics = compute_metrics(phase_b_results)
        print(f"  Phase B only metrics: {len(phase_b_results)} unsolvable")

    # Compute metrics by prompt strategy
    strategy_metrics = compute_metrics_by_dimension(phase_b_results, "prompt_strategy")

    # Compute metrics by transformation type
    transform_metrics = compute_metrics_by_dimension(
        phase_b_results, "transformation_type"
    )

    # Cost-performance analysis
    cost_perf = {}
    for model_name, m in model_metrics.items():
        cost_perf[model_name] = {
            "mc_score": m.mc_score,
            "refusal_f1": m.refusal_f1,
            "refusal_recall": m.refusal_recall,
            "refusal_precision": m.refusal_precision,
            "over_conservatism_rate": m.over_conservatism_rate,
            "total_cost": m.total_cost_usd,
            "mc_per_dollar": (
                m.mc_score / m.total_cost_usd if m.total_cost_usd > 0 else float("inf")
            ),
        }

    # Find optimal
    if cost_perf:
        best_score = max(cost_perf.items(), key=lambda x: x[1]["mc_score"])
        best_efficiency = max(cost_perf.items(), key=lambda x: x[1]["mc_per_dollar"])
        print(f"\n  Best MC Score: {best_score[0]} ({best_score[1]['mc_score']:.3f})")
        print(
            f"  Best Efficiency: {best_efficiency[0]} ({best_efficiency[1]['mc_per_dollar']:.1f} MC/$)"
        )

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(all_results),
        "model_metrics": {k: v.to_dict() for k, v in model_metrics.items()},
        "strategy_metrics": {
            strat: {k: v.to_dict() for k, v in models.items()}
            for strat, models in strategy_metrics.items()
        },
        "transform_metrics": {
            tt: {k: v.to_dict() for k, v in models.items()}
            for tt, models in transform_metrics.items()
        },
        "cost_performance": cost_perf,
    }

    return analysis


# ============================================================================
# REPORT GENERATION
# ============================================================================


def save_results(
    results: List[MetacognitiveResult],
    phase: str,
    output_dir: Path,
    extra_meta: Optional[Dict] = None,
) -> Path:
    """Save phase results to JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compute metrics
    unsolvable_results = [r for r in results if r.transformation_type != "original"]
    metrics = compute_metrics(unsolvable_results) if unsolvable_results else {}

    report = {
        "experiment": "metacognitive_evaluation",
        "phase": phase,
        "timestamp": timestamp,
        "summary": {
            "total_evaluations": len(results),
            "models": list({r.model_name for r in results}),
            "prompt_strategies": list({r.prompt_strategy for r in results}),
            "transformation_types": list({r.transformation_type for r in results}),
            "total_cost_usd": sum(r.cost_usd for r in results),
        },
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "results": [r.to_dict() for r in results],
    }

    # Build prompt catalog: actual prompt texts keyed by (strategy, method)
    strategies_used = {r.prompt_strategy for r in results}
    methods_used = {r.method for r in results}
    prompt_catalog = {}
    for strategy in strategies_used:
        if strategy in PROMPT_SYSTEMS:
            prompt_catalog[strategy] = {
                method: PROMPT_SYSTEMS[strategy][method]
                for method in methods_used
                if method in PROMPT_SYSTEMS[strategy]
            }
    report["prompt_catalog"] = prompt_catalog

    if extra_meta:
        report["extra"] = extra_meta

    filepath = output_dir / f"phase_{phase}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {filepath}")

    # Print summary
    if metrics:
        print(f"\n{'=' * 80}")
        print(f"Phase {phase} Metrics Summary")
        print(f"{'=' * 80}")
        for model_name, m in sorted(metrics.items()):
            print(
                f"  {model_name:20s} | "
                f"RefRecall={m.refusal_recall:.2%} | "
                f"RefPrec={m.refusal_precision:.2%} | "
                f"RefF1={m.refusal_f1:.3f} | "
                f"MC={m.mc_score:.3f} | "
                f"Cost=${m.total_cost_usd:.4f}"
            )

    # Print baseline accuracy for Phase A
    baseline_results = [r for r in results if r.transformation_type == "original"]
    if baseline_results:
        print("\nBaseline Accuracy:")
        by_model: Dict[str, List[MetacognitiveResult]] = {}
        for r in baseline_results:
            by_model.setdefault(r.model_name, []).append(r)
        for model_name, model_results in sorted(by_model.items()):
            correct = sum(1 for r in model_results if r.is_original_correct)
            total = len(model_results)
            print(f"  {model_name:20s} | {correct}/{total} ({correct / total:.1%})")

    return filepath


# ============================================================================
# MAIN
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Metacognitive Financial LLM Evaluation Experiment"
    )
    parser.add_argument(
        "--phase",
        choices=["A", "B", "C", "D", "all"],
        default="A",
        help="Experiment phase to run",
    )
    parser.add_argument(
        "--budget",
        choices=["economic", "balanced", "full"],
        default="economic",
        help="Model budget set",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of examples per difficulty level"
    )
    parser.add_argument(
        "--method",
        choices=["COT", "POT"],
        default="POT",
        help="Reasoning method",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=[
            "standard",
            "metacognitive",
            "self_verification",
            "contradiction_aware",
            "all",
        ],
        default="metacognitive",
        help="Prompt strategy for Phase B/C",
    )
    parser.add_argument(
        "--level",
        choices=["easy", "medium", "hard", "all"],
        default="hard",
        help="Difficulty level",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG enhancement",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (for Phase D analysis)",
    )
    parser.add_argument(
        "--transformation-types",
        nargs="+",
        default=None,
        help="Filter specific transformation types (e.g. 'Type 4' 'Type 5')",
    )

    args = parser.parse_args()

    # Output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "experiments" / "results" / "metacognitive"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase D: analysis only
    if args.phase == "D":
        results_dir = Path(args.results_dir) if args.results_dir else output_dir
        analysis = analyze_phase_d(results_dir)
        if analysis:
            analysis_path = (
                output_dir
                / f"phase_D_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"\n[SAVED] {analysis_path}")
        return

    # Setup models
    requested_models = BUDGET_MODEL_SETS[args.budget]
    available_models = get_available_models(requested_models)
    if not available_models:
        print("[ERROR] No models available. Check API keys.")
        return

    # Load data
    levels = ["easy", "medium", "hard"] if args.level == "all" else [args.level]
    datasets = {}
    for level in levels:
        datasets[level] = load_dataset(level)
        print(f"[DATA] {level}: {len(datasets[level])} examples")

    examples = select_examples(datasets, args.n)
    print(f"\n[SELECT] {len(examples)} examples total ({args.n} per level)")

    # Initialize experiment
    experiment = MetacognitiveExperiment(
        models=available_models,
        method=args.method,
        enable_rag=args.rag,
    )

    # Determine prompt strategies to run
    if args.prompt_strategy == "all":
        strategies = [
            "standard",
            "metacognitive",
            "self_verification",
            "contradiction_aware",
        ]
    else:
        strategies = [args.prompt_strategy]

    all_results = []

    # Run phases
    if args.phase in ("A", "all"):
        for strategy in strategies:
            results_a = await experiment.run_phase_a(examples, strategy)
            save_results(results_a, f"A_{strategy}", output_dir)
            all_results.extend(results_a)

    if args.phase in ("B", "all"):
        for strategy in strategies:
            results_b = await experiment.run_phase_b(
                examples, strategy, args.transformation_types
            )
            save_results(
                results_b,
                f"B_{strategy}",
                output_dir,
                extra_meta={"prompt_strategy": strategy},
            )
            all_results.extend(results_b)

    if args.phase in ("C", "all"):
        if not args.rag and args.phase == "C":
            print("[INFO] Phase C requires --rag flag. Enabling RAG.")
            experiment.enable_rag = True
            try:
                from rag_enhancer import RAGEnhancer

                eval_dir = Path(__file__).parent.parent / "evaluation"
                experiment.rag_enhancer = RAGEnhancer(
                    eval_dir / "function_retriever.py"
                )
            except Exception as e:
                print(f"[ERROR] Cannot load RAG: {e}")
                return

        for strategy in strategies:
            results_c = await experiment.run_phase_c(examples, strategy)
            save_results(
                results_c,
                f"C_{strategy}",
                output_dir,
                extra_meta={"prompt_strategy": strategy, "rag_enabled": True},
            )
            all_results.extend(results_c)

    # Final summary
    total_cost = sum(r.cost_usd for r in all_results)
    print(f"\n{'=' * 80}")
    print("Experiment Complete")
    print(f"  Total evaluations: {len(all_results)}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'=' * 80}")
    print(
        f"\nNext: python experiments/generate_metacognitive_dashboard.py --results-dir {output_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
