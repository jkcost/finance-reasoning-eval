"""
LLM-based Error Analyzer for FinanceReasoning

Uses a low-cost LLM to analyze why a model's answer is wrong.
Provides detailed reasoning about the error type and cause.
"""

import os
import re
from typing import Any, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class ErrorAnalysisResult:
    """Result from LLM error analysis"""

    error_type: str  # misunderstanding, formula_error, extraction_error, calculation_error, etc.
    summary: str  # One-line summary of what went wrong
    detailed_analysis: str  # Detailed explanation
    likely_cause: str  # What the model likely did wrong
    correct_approach: str  # How to solve correctly
    confidence: float  # 0.0 to 1.0


ERROR_ANALYSIS_PROMPT = """당신은 금융 계산 오류를 분석하는 전문가입니다.
주어진 문제, 정답, 그리고 모델의 오답과 풀이 과정을 보고 왜 틀렸는지 분석해주세요.

반드시 한글로 답변하세요. 쉽고 명확한 표현을 사용하세요.

## 문제
{question}

## 주어진 데이터
{context}

## 정답
{ground_truth}

## 모델의 답
{predicted_answer}

## 모델의 풀이
```
{raw_response}
```

## 분석 요청

1. **오류 유형** (하나 선택):
   - MISUNDERSTANDING: 문제를 잘못 이해함
   - FORMULA_ERROR: 잘못된 공식이나 계산 방법 사용
   - EXTRACTION_ERROR: 데이터에서 잘못된 값을 가져옴
   - CALCULATION_ERROR: 계산 실수
   - UNIT_ERROR: 단위를 혼동함 (예: INR→GBP vs GBP→INR)
   - LOGIC_ERROR: 논리적 오류
   - INCOMPLETE: 풀이가 불완전함

2. **요약**: 무엇이 잘못되었는지 한 문장으로

3. **상세 분석**: 구체적으로 어떤 실수를 했는지

4. **원인**: 모델이 왜 이런 실수를 했을 것 같은지

5. **올바른 풀이**: 어떻게 풀어야 했는지 간단히

다음 형식으로 답변하세요:
ERROR_TYPE: [유형]
SUMMARY: [한 문장 요약]
DETAILED_ANALYSIS: [상세 분석]
LIKELY_CAUSE: [원인]
CORRECT_APPROACH: [올바른 풀이]
CONFIDENCE: [0.0-1.0]
"""


class LLMErrorAnalyzer:
    """Analyzes errors using LLM"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None

        if OpenAI is not None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)

    def analyze(
        self,
        question: str,
        context: str,
        ground_truth: Any,
        predicted_answer: Any,
        raw_response: str,
    ) -> Optional[ErrorAnalysisResult]:
        """Analyze why the model's answer is wrong"""

        if self.client is None:
            return None

        # Truncate long responses
        raw_response_truncated = raw_response[:2000] if len(raw_response) > 2000 else raw_response
        context_truncated = context[:3000] if len(context) > 3000 else context

        prompt = ERROR_ANALYSIS_PROMPT.format(
            question=question,
            context=context_truncated if context else "No context provided",
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
            raw_response=raw_response_truncated,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 금융 계산 오류 분석 전문가입니다. 한글로 답변하고, 쉽고 명확하게 설명하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
            )

            result_text = response.choices[0].message.content
            return self._parse_response(result_text)

        except Exception as e:
            print(f"[LLM Error Analysis] Failed: {e}")
            return None

    def _parse_response(self, text: str) -> ErrorAnalysisResult:
        """Parse LLM response into structured result"""

        def extract_field(field_name: str) -> str:
            pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        error_type = extract_field("ERROR_TYPE")
        summary = extract_field("SUMMARY")
        detailed = extract_field("DETAILED_ANALYSIS")
        cause = extract_field("LIKELY_CAUSE")
        approach = extract_field("CORRECT_APPROACH")

        confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.7

        return ErrorAnalysisResult(
            error_type=error_type or "UNKNOWN",
            summary=summary or "Analysis incomplete",
            detailed_analysis=detailed or "",
            likely_cause=cause or "",
            correct_approach=approach or "",
            confidence=min(max(confidence, 0.0), 1.0),
        )


def analyze_error_with_llm(
    question: str,
    context: str,
    ground_truth: Any,
    predicted_answer: Any,
    raw_response: str,
    model: str = "gpt-4o-mini",
) -> Optional[ErrorAnalysisResult]:
    """Convenience function to analyze a single error"""

    analyzer = LLMErrorAnalyzer(model=model)
    return analyzer.analyze(
        question=question,
        context=context,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer,
        raw_response=raw_response,
    )
