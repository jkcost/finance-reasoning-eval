"""
Reasoning Trace Analyzer for Transformation Validation

Analyzes LLM responses to validate transformation effectiveness by tracing
value provenance and measuring context dependency.

Case Classification:
    Case 1 (거부): Model refused → metacognitive success
    Case 2 (오답): Model answered incorrectly → transformation effective
    Case 3 (정답): Model answered correctly → transformation insufficient

Components:
    - NumberExtractor: Extracts numbers from context/response/code
    - ValueProvenanceClassifier: Classifies value origins
    - AwarenessDetector: Detects partial awareness signals
    - ContextDependencyScorer: Scores context independence (Case 3)
    - LLMJudge: Deep analysis via LLM (optional, Tier 2)
    - ReasoningTraceAnalyzer: Orchestrator
"""

import logging
import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ExtractedNumber:
    """A number extracted from text with its source location."""

    value: float
    source_text: str  # original text snippet
    location: str  # "context", "response", "code"


@dataclass
class ValueProvenance:
    """Classification of where a value came from."""

    value: float
    source: (
        str  # from_context | from_removed_data | fabricated | common_constant | derived
    )
    evidence: str


@dataclass
class ReasoningTraceAnalysis:
    """Complete analysis result for a single evaluation."""

    case_type: int  # 1 (refused), 2 (wrong answer), 3 (correct answer)
    values_used: List[ValueProvenance]
    awareness_signals: List[str]
    context_independence_score: float  # 0.0 ~ 1.0 (높을수록 context 비의존)
    independence_factors: List[str]  # 판정 근거 목록 (한글)
    transformation_effective: bool
    summary: str  # 한글
    llm_judge_verdict: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_type": self.case_type,
            "values_used": [
                {"value": v.value, "source": v.source, "evidence": v.evidence}
                for v in self.values_used
            ],
            "awareness_signals": self.awareness_signals,
            "context_independence_score": round(self.context_independence_score, 3),
            "independence_factors": self.independence_factors,
            "transformation_effective": self.transformation_effective,
            "summary": self.summary,
            "llm_judge_verdict": self.llm_judge_verdict,
        }


# ============================================================================
# COMMON CONSTANTS (skip during provenance classification)
# ============================================================================

COMMON_CONSTANTS: Set[float] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    24,
    30,
    52,
    60,
    100,
    180,
    360,
    365,
    1000,
    10000,
    100000,
    1000000,
}


# ============================================================================
# NUMBER EXTRACTOR
# ============================================================================

_NUMBER_PATTERN = re.compile(r"-?[\d,]+\.?\d*")
_PERCENT_PATTERN = re.compile(r"(-?[\d,]+\.?\d*)\s*%")


class NumberExtractor:
    """Extracts numerical values from various text formats."""

    @staticmethod
    def extract_from_context(context: str) -> Set[float]:
        """Extract all numbers from context (JSON/Markdown/Text).

        Also generates common transformations (×100, /100, etc.)
        to catch percentage ↔ decimal conversions.
        """
        numbers: Set[float] = set()
        for match in _NUMBER_PATTERN.findall(context):
            try:
                num = float(match.replace(",", ""))
                numbers.add(num)
                # Common transformations for percentage/unit conversions
                numbers.add(num * 100)
                numbers.add(num / 100)
                numbers.add(num * 1000)
                numbers.add(num / 1000)
            except ValueError:
                continue
        return numbers

    @staticmethod
    def extract_raw_numbers(text: str) -> Set[float]:
        """Extract raw numbers without transformations."""
        numbers: Set[float] = set()
        for match in _NUMBER_PATTERN.findall(text):
            try:
                numbers.add(float(match.replace(",", "")))
            except ValueError:
                continue
        return numbers

    @staticmethod
    def extract_from_python_code(code: str) -> List[ExtractedNumber]:
        """Extract number literals from Python (POT) code."""
        results: List[ExtractedNumber] = []
        # Parse numeric literals from assignments and expressions
        for match in re.finditer(r"=\s*(-?[\d,]+\.?\d*)\b", code):
            try:
                val = float(match.group(1).replace(",", ""))
                results.append(
                    ExtractedNumber(
                        value=val,
                        source_text=match.group(0).strip(),
                        location="code",
                    )
                )
            except ValueError:
                continue
        # Also catch function arguments like round(x, 2)
        for match in re.finditer(r"[\(\[,]\s*(-?[\d,]+\.?\d*)\s*[,\)\]]", code):
            try:
                val = float(match.group(1).replace(",", ""))
                if val not in COMMON_CONSTANTS:
                    results.append(
                        ExtractedNumber(
                            value=val,
                            source_text=match.group(0).strip(),
                            location="code",
                        )
                    )
            except ValueError:
                continue
        return results

    @staticmethod
    def extract_from_text(text: str) -> List[ExtractedNumber]:
        """Extract numbers from COT natural language response."""
        results: List[ExtractedNumber] = []
        for match in _NUMBER_PATTERN.finditer(text):
            try:
                val = float(match.group().replace(",", ""))
                results.append(
                    ExtractedNumber(
                        value=val,
                        source_text=match.group(),
                        location="response",
                    )
                )
            except ValueError:
                continue
        return results


# ============================================================================
# VALUE PROVENANCE CLASSIFIER
# ============================================================================


class ValueProvenanceClassifier:
    """Classifies where values in a response came from."""

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def classify(
        self,
        response_values: List[ExtractedNumber],
        context_values: Set[float],
        removed_values: Set[float],
    ) -> List[ValueProvenance]:
        """Classify each response value's origin.

        Priority: from_context → common_constant → from_removed_data → derived → fabricated
        """
        results: List[ValueProvenance] = []
        seen: Set[float] = set()

        for rv in response_values:
            if rv.value in seen:
                continue
            if rv.value in COMMON_CONSTANTS:
                continue
            seen.add(rv.value)

            provenance = self._classify_single(rv.value, context_values, removed_values)
            results.append(provenance)

        return results

    def _classify_single(
        self,
        value: float,
        context_values: Set[float],
        removed_values: Set[float],
    ) -> ValueProvenance:
        # 1. from_context
        if self._matches_any(value, context_values):
            return ValueProvenance(
                value=value,
                source="from_context",
                evidence="값이 변환된 context에 존재",
            )

        # 2. common_constant (broader check)
        if abs(value) < 2 or value in COMMON_CONSTANTS:
            return ValueProvenance(
                value=value,
                source="common_constant",
                evidence="일반적인 상수 또는 소수값",
            )

        # 3. from_removed_data
        if self._matches_any(value, removed_values):
            return ValueProvenance(
                value=value,
                source="from_removed_data",
                evidence="제거된 데이터에서 유래 — 변환 불충분 가능성",
            )

        # 4. derived (arithmetic of context values)
        if self._is_derived(value, context_values):
            return ValueProvenance(
                value=value,
                source="derived",
                evidence="context 값들의 산술 연산 결과",
            )

        # 5. fabricated
        return ValueProvenance(
            value=value,
            source="fabricated",
            evidence="출처 불명 — context/removed 어디에도 없음",
        )

    def _matches_any(self, value: float, candidates: Set[float]) -> bool:
        for c in candidates:
            if c == 0:
                continue
            if abs(value - c) / max(abs(c), 1e-9) < self.tolerance:
                return True
        return False

    def _is_derived(self, value: float, context_values: Set[float]) -> bool:
        """Check if value is a simple arithmetic result of context values."""
        # Only check raw context values (no transformations)
        raw_vals = [v for v in context_values if v not in COMMON_CONSTANTS and v != 0]
        if len(raw_vals) < 2:
            return False

        # Check pairwise operations (limit to avoid combinatorial explosion)
        for a, b in combinations(raw_vals[:20], 2):
            for result in (a + b, a - b, b - a, a * b):
                if (
                    result != 0
                    and abs(value - result) / max(abs(result), 1e-9) < self.tolerance
                ):
                    return True
            if b != 0 and abs(value - a / b) / max(abs(a / b), 1e-9) < self.tolerance:
                return True
            if a != 0 and abs(value - b / a) / max(abs(b / a), 1e-9) < self.tolerance:
                return True

        return False


# ============================================================================
# AWARENESS DETECTOR
# ============================================================================

AWARENESS_PATTERNS: Dict[str, List[str]] = {
    "acknowledged_missing": [
        r"(?i)\b(data|information|value)s?\s+(is|are|was|were)\s+(missing|unavailable|removed|absent)",
        r"(?i)\bmissing\b.{0,20}\b(data|value|information)",
        r"(?i)\bN/?A\b.{0,10}\b(found|detected|present)",
        r"(?i)\bDATA\s*MISSING\b",
    ],
    "noticed_contradiction": [
        r"(?i)\bcontradiction\b",
        r"(?i)\bconflicting\b.{0,20}\b(data|values|information)",
        r"(?i)\bdiscrepan(cy|cies)\b",
        r"(?i)\binconsisten(t|cy)\b.{0,20}\b(data|values)",
    ],
    "made_assumption": [
        r"(?i)\b(assuming|assume|assumed)\b",
        r"(?i)\blet'?s?\s+assume\b",
        r"(?i)\bI('ll|\s+will)\s+use\b.{0,20}\b(available|remaining)",
        r"(?i)\bbased on (the |available )?assumption",
    ],
    "expressed_uncertainty": [
        r"(?i)\b(not sure|uncertain|unclear)\b",
        r"(?i)\bapproximat(e|ely|ion)\b",
        r"(?i)\bestimate[ds]?\b",
        r"(?i)\brough(ly)?\b.{0,10}\b(answer|result|calculation)",
    ],
}


class AwarenessDetector:
    """Detects partial awareness signals in non-refusal responses."""

    @staticmethod
    def detect(response: str) -> List[str]:
        """Return list of awareness signal types found in response."""
        signals: List[str] = []
        for signal_type, patterns in AWARENESS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response):
                    signals.append(signal_type)
                    break  # one match per signal type
        return signals


# ============================================================================
# CONTEXT DEPENDENCY SCORER (Case 3 only)
# ============================================================================


class ContextDependencyScorer:
    """Scores context independence for Case 3 (correct answer despite transformation).

    높은 점수 = context 비의존 (변환은 유효하나 모델이 context를 사용하지 않음)
    낮은 점수 = context 의존 (변환이 불충분하여 남은 정보로 정답 도달)
    """

    def __init__(self, hardcoded_problem_ids: FrozenSet[str] = frozenset()):
        self._hardcoded_ids = hardcoded_problem_ids

    def score(
        self,
        example_id: str,
        values_used: List[ValueProvenance],
        response: str,
        context_transformed: str,
    ) -> Tuple[float, List[str]]:
        """Compute context independence score and evidence factors.

        Returns:
            (score, factors) where score is 0.0~1.0 (높을수록 context 비의존)
            and factors is a list of Korean descriptions explaining each triggered factor.
        """
        total = 0.0
        factors: List[str] = []

        # Factor 1: Uses removed data values
        removed_vals = [v for v in values_used if v.source == "from_removed_data"]
        if removed_vals:
            total += 0.3
            val_strs = ", ".join(str(v.value) for v in removed_vals[:3])
            factors.append(
                f"제거된 데이터 값 사용 (+0.3): {val_strs}"
                " — 변환으로 삭제된 숫자를 모델이 그대로 사용"
            )

        # Factor 2: Known hardcoded problem
        if example_id in self._hardcoded_ids:
            total += 0.2
            factors.append(
                "하드코딩 솔루션 문제 (+0.2): 데이터셋의 python_solution이"
                " context를 파싱하지 않고 숫자를 직접 사용하는 문제"
            )

        # Factor 3: Doesn't reference context
        context_keywords = self._extract_context_keywords(context_transformed)
        if context_keywords:
            references = sum(
                1 for kw in context_keywords if kw.lower() in response.lower()
            )
            ref_rate = references / len(context_keywords)
            if ref_rate < 0.2:
                total += 0.2
                factors.append(
                    f"context 미참조 (+0.2): 키워드 참조율 {ref_rate:.0%}"
                    " — 모델이 context를 읽지 않고 답변한 것으로 보임"
                )

        # Factor 4: No derivation steps
        has_derivation = bool(
            re.search(
                r"(?i)(therefore|thus|so|hence|which gives|calculating|resulting in)",
                response,
            )
        )
        if not has_derivation:
            total += 0.15
            factors.append(
                "추론 과정 부재 (+0.15): 'therefore', 'thus' 등 단계적 추론 표현이 없음"
            )

        # Factor 5: All values hardcoded in code
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            has_parsing = bool(
                re.search(
                    r"(json\.loads|\.split\(|context|parse|for\s+\w+\s+in)",
                    code,
                )
            )
            if not has_parsing:
                total += 0.15
                factors.append(
                    "코드 내 값 직접 입력 (+0.15): Python 코드가"
                    " context 파싱 없이 숫자를 하드코딩"
                )

        if not factors:
            factors.append("비의존 근거 없음 — context 기반 추론으로 판단")

        return min(total, 1.0), factors

    @staticmethod
    def _extract_context_keywords(context: str) -> List[str]:
        """Extract meaningful keywords from context for reference checking."""
        # Extract capitalized multi-word terms (company names, financial terms)
        keywords: List[str] = []
        for match in re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", context):
            if len(match) > 5:
                keywords.append(match)
        # Also extract table headers
        for match in re.findall(r"\|\s*([A-Za-z][\w\s]+?)\s*\|", context):
            cleaned = match.strip()
            if len(cleaned) > 3 and not cleaned.isdigit():
                keywords.append(cleaned)
        return keywords[:10]  # limit to avoid noise


# ============================================================================
# LLM JUDGE (Tier 2, optional)
# ============================================================================

LLM_JUDGE_PROMPT = """당신은 금융 LLM의 추론 과정을 분석하는 전문가입니다.

## 원본 Context (변환 전)
{context_original}

## 변환된 Context (모델이 받은 것)
{context_transformed}

## 변환 설명
{transformation_description}

## 모델의 응답
```
{raw_response}
```

## 정답
{ground_truth}

## 모델의 답
{predicted_answer}

## 분석 요청
모델이 정답을 맞췄거나 제거된 데이터를 사용한 것으로 보입니다.
다음을 분석해주세요:

1. **값 출처**: 모델이 사용한 핵심 숫자들이 변환된 context에서 왔는가, 아니면 context에 없는 값을 사용했는가?
2. **context 의존도**: 모델이 실제로 변환된 context를 읽고 추론했는가, 아니면 context와 무관하게 답을 도출했는가?
3. **판정**: CONTEXT_INDEPENDENT (context 비의존) / INSUFFICIENT_TRANSFORM (변환 불충분) / AMBIGUOUS (불분명)

다음 형식으로 답변하세요:
VALUE_SOURCES: [각 핵심 값의 출처 분석]
REASONING_PATH: [추론 과정 분석]
VERDICT: [CONTEXT_INDEPENDENT/INSUFFICIENT_TRANSFORM/AMBIGUOUS]
EXPLANATION: [판정 근거 한줄]
"""


class LLMJudge:
    """Uses an LLM to deeply analyze ambiguous cases."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        if OpenAI is not None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._client = OpenAI(api_key=api_key)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def judge(self, result: Dict[str, Any]) -> Optional[str]:
        """Analyze a single result and return verdict string."""
        if not self.is_available:
            return None

        prompt = LLM_JUDGE_PROMPT.format(
            context_original=_truncate(result.get("context_original", ""), 3000),
            context_transformed=_truncate(result.get("context_transformed", ""), 3000),
            transformation_description=result.get("transformation_description", "N/A"),
            raw_response=_truncate(result.get("raw_response", ""), 2000),
            ground_truth=result.get("ground_truth", "N/A"),
            predicted_answer=result.get("predicted_answer", "N/A"),
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "금융 LLM 추론 분석 전문가입니다. 한글로 답변하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("LLM Judge failed: %s", e)
            return None


# ============================================================================
# ORCHESTRATOR
# ============================================================================


class ReasoningTraceAnalyzer:
    """Orchestrates the full reasoning trace analysis pipeline."""

    def __init__(
        self,
        hardcoded_problem_ids: FrozenSet[str] = frozenset(),
        use_llm_judge: bool = False,
        llm_judge_model: str = "gpt-4o-mini",
    ):
        self._extractor = NumberExtractor()
        self._classifier = ValueProvenanceClassifier(tolerance=0.01)
        self._awareness = AwarenessDetector()
        self._ctx_dependency = ContextDependencyScorer(hardcoded_problem_ids)
        self._llm_judge: Optional[LLMJudge] = None
        if use_llm_judge:
            self._llm_judge = LLMJudge(model=llm_judge_model)

    def analyze(self, result: Dict[str, Any]) -> ReasoningTraceAnalysis:
        """Analyze a single Phase B evaluation result.

        Args:
            result: Dict with keys from MetacognitiveResult.to_dict()

        Returns:
            ReasoningTraceAnalysis with case classification and provenance data
        """
        response_type = result.get("response_type", "")
        raw_response = result.get("raw_response", "")
        context_transformed = result.get("context_transformed", "")
        context_original = result.get("context_original", "")
        example_id = result.get("example_id", "")
        ground_truth = result.get("ground_truth")
        predicted = result.get("predicted_answer")

        # Step 1: Case classification
        case_type = self._classify_case(response_type, predicted, ground_truth)

        # Step 2: Extract numbers
        transformed_values = self._extractor.extract_from_context(context_transformed)
        removed_values = self._compute_removed_values(
            context_original, context_transformed
        )
        response_numbers = self._extract_response_numbers(raw_response)

        # Step 3: Classify provenance
        values_used = self._classifier.classify(
            response_numbers, transformed_values, removed_values
        )

        # Step 4: Awareness detection (non-refused only)
        awareness_signals: List[str] = []
        if case_type != 1:
            awareness_signals = self._awareness.detect(raw_response)

        # Step 5: Context independence score (Case 3 only)
        ctx_independence_score = 0.0
        independence_factors: List[str] = []
        if case_type == 3:
            ctx_independence_score, independence_factors = self._ctx_dependency.score(
                example_id, values_used, raw_response, context_transformed
            )

        # Step 6: Transformation effectiveness
        transformation_effective = self._judge_effectiveness(
            case_type, ctx_independence_score
        )

        # Step 7: LLM Judge (optional, for Case 3 or suspicious Case 2)
        llm_verdict = None
        if self._llm_judge and self._should_invoke_llm_judge(
            case_type, values_used, awareness_signals
        ):
            llm_verdict = self._llm_judge.judge(result)

        # Step 8: Generate summary
        summary = self._generate_summary(
            case_type,
            values_used,
            awareness_signals,
            ctx_independence_score,
            transformation_effective,
        )

        return ReasoningTraceAnalysis(
            case_type=case_type,
            values_used=values_used,
            awareness_signals=awareness_signals,
            context_independence_score=ctx_independence_score,
            independence_factors=independence_factors,
            transformation_effective=transformation_effective,
            summary=summary,
            llm_judge_verdict=llm_verdict,
        )

    def _classify_case(
        self, response_type: str, predicted: Any, ground_truth: Any
    ) -> int:
        """Classify into Case 1 (refused), 2 (wrong), 3 (correct)."""
        if response_type in ("refused", "error"):
            return 1

        if predicted is None or ground_truth is None:
            return 2  # can't verify → treat as wrong

        if _answers_match(predicted, ground_truth):
            return 3
        return 2

    def _compute_removed_values(
        self, context_original: str, context_transformed: str
    ) -> Set[float]:
        """Find values present in original but not in transformed context."""
        orig_nums = self._extractor.extract_raw_numbers(context_original)
        trans_nums = self._extractor.extract_raw_numbers(context_transformed)
        return orig_nums - trans_nums

    def _extract_response_numbers(self, raw_response: str) -> List[ExtractedNumber]:
        """Extract numbers from response, preferring code if present."""
        code_match = re.search(r"```python\s*(.*?)```", raw_response, re.DOTALL)
        if code_match:
            return self._extractor.extract_from_python_code(code_match.group(1))
        return self._extractor.extract_from_text(raw_response)

    @staticmethod
    def _judge_effectiveness(case_type: int, ctx_independence_score: float) -> bool:
        """Judge whether the transformation was effective for this result."""
        if case_type in (1, 2):
            return True
        # Case 3: effective only if model answered independently of context
        return ctx_independence_score >= 0.5

    def _should_invoke_llm_judge(
        self,
        case_type: int,
        values_used: List[ValueProvenance],
        awareness_signals: List[str],
    ) -> bool:
        """Decide whether to invoke LLM Judge for deeper analysis."""
        if case_type == 3:
            return True
        if case_type == 2:
            has_removed = any(v.source == "from_removed_data" for v in values_used)
            if has_removed or len(awareness_signals) >= 2:
                return True
        return False

    @staticmethod
    def _generate_summary(
        case_type: int,
        values_used: List[ValueProvenance],
        awareness_signals: List[str],
        ctx_independence_score: float,
        transformation_effective: bool,
    ) -> str:
        """Generate a Korean summary of the analysis."""
        source_counts: Dict[str, int] = {}
        for v in values_used:
            source_counts[v.source] = source_counts.get(v.source, 0) + 1

        if case_type == 1:
            return "모델이 정보 부족을 인식하고 거부 — 변환 유효"

        if case_type == 2:
            parts = ["모델이 오답 제출 — 변환 유효"]
            if source_counts.get("from_removed_data", 0) > 0:
                parts.append(
                    f"(제거된 데이터 {source_counts['from_removed_data']}개 사용)"
                )
            if source_counts.get("fabricated", 0) > 0:
                parts.append(f"(환각값 {source_counts['fabricated']}개)")
            if awareness_signals:
                parts.append(f"부분 인식: {', '.join(awareness_signals)}")
            return " ".join(parts)

        # Case 3
        if ctx_independence_score >= 0.5:
            return f"정답 맞춤 — context 비의존 (score={ctx_independence_score:.2f}), 변환 자체는 유효하나 모델이 context 미사용"
        return f"정답 맞춤 — context 의존 추론 (score={ctx_independence_score:.2f}), 변환 불충분 가능"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _answers_match(predicted: Any, ground_truth: Any, tolerance: float = 0.002) -> bool:
    """Check if predicted answer matches ground truth within tolerance."""
    pred_str = str(predicted).strip().replace(",", "").replace("%", "").replace("$", "")
    gt_str = (
        str(ground_truth).strip().replace(",", "").replace("%", "").replace("$", "")
    )

    try:
        pred_num = float(pred_str)
        gt_num = float(gt_str)
        if gt_num == 0:
            return abs(pred_num) < 1e-9
        return abs(pred_num - gt_num) / abs(gt_num) <= tolerance
    except (ValueError, TypeError):
        return pred_str.lower() == gt_str.lower()


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
