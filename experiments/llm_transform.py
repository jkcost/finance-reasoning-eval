"""
LLM-based Transformation Engine

금융 문제의 context/question을 LLM이 분석하여 변환.
규칙 기반이 아닌 LLM이 금융 맥락을 이해하고 변환을 수행.

Usage:
    from llm_transform import LLMTransformer
    transformer = LLMTransformer(model="gemini-2.5-flash")
    result = await transformer.transform(problem, transform_type="EA-partial")
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Transformation type definitions for LLM prompts
TRANSFORM_SPECS = {
    "EA-partial": {
        "name": "Explicit Absence (Partial)",
        "instruction": """특정 값 하나를 "[DATA MISSING]" 또는 "N/A"로 교체하여 명시적으로 누락을 표시하세요.
- 풀이에 **필수적인** 값만 교체해야 합니다.
- 남은 데이터로 역산(derive)할 수 있는 값은 교체하면 안 됩니다.
- 예: Total = A + B 일 때 Total을 N/A로 바꾸면, A와 B로 역산 가능 → 부적절
- 예: A를 N/A로 바꾸면, Total과 B로 역산 가능 → 부적절
- 역산 불가능한 독립적인 입력값을 골라야 합니다.""",
    },
    "EA-full": {
        "name": "Explicit Absence (Full)",
        "instruction": """전체 데이터 차원(컬럼, 키, 행 그룹)을 삭제하세요.
- 삭제할 대상은 풀이에 필수적인 데이터를 포함해야 합니다.
- 삭제 후 남은 데이터로 역산할 수 없어야 합니다.
- 테이블: 컬럼 전체 삭제 또는 핵심 행 그룹 삭제
- JSON: 전체 키 삭제
- 텍스트: 핵심 데이터를 포함하는 문장들 삭제
- 구조적 변화는 있지만 [DATA MISSING] 같은 마커는 남기지 않습니다.""",
    },
    "SA": {
        "name": "Silent Absence",
        "instruction": """풀이에 필수적인 데이터를 **아무 흔적 없이** 자연스럽게 제거하세요.
- 제거 후 나머지 문맥이 문법적으로 완전하고 자연스러워야 합니다.
- N/A, [DATA MISSING] 같은 마커를 절대 남기지 마세요.
- 누군가 읽었을 때 "뭔가 빠졌다"는 느낌이 들지 않아야 합니다.
- 텍스트: 핵심 데이터를 포함하는 문장을 삭제하되 흐름이 자연스러워야 함
- 테이블: 행을 삭제하거나 값을 빈칸으로 만들되 구조는 유지
- 이것이 가장 어려운 변환입니다 — 자연스러움이 핵심입니다.""",
    },
    "IC": {
        "name": "Information Conflict",
        "instruction": """풀이에 사용되는 핵심 값에 모순되는 데이터를 삽입하세요.
- 원래 값과 다른 값(예: 1.3~1.7배)을 같은 문맥 내에 추가합니다.
- 모순이 도메인적으로 현실적이어야 합니다 (예: "수정된 감사 보고서", "분기별 보고서와 연간 보고서의 차이")
- 어느 값이 맞는지 판단할 수 없는 상황을 만들어야 합니다.
- 단순히 숫자만 바꾸지 말고, 왜 두 값이 존재하는지 맥락을 제공하세요.
- 모순 대상은 반드시 풀이에 사용되는 값이어야 합니다.""",
    },
    "TA": {
        "name": "Temporal Ambiguity",
        "instruction": """시간 참조를 모호하게 만들어 어느 시점의 데이터를 사용해야 하는지 불분명하게 하세요.
- question의 특정 연도를 "해당 기간 말", "직전 회계연도" 등으로 교체
- context에 여러 연도 데이터가 있어야 실제로 모호해집니다
- 한 시점의 데이터만 있으면 모호성이 없으므로 변환 불가로 표시하세요.""",
    },
}

SYSTEM_PROMPT = """당신은 금융 데이터 변환 전문가입니다.
금융 문제의 context 또는 question을 변환하여 "풀 수 없는(unsolvable)" 문제를 만드는 것이 목표입니다.
이 변환은 LLM의 메타인지 능력(정보 부족을 인식하는 능력)을 평가하기 위한 것입니다.

변환 시 반드시 지켜야 할 원칙:
1. 풀이에 **필수적인** 데이터만 제거/변형합니다
2. 남은 데이터로 **역산(derive)할 수 없는** 변환을 합니다
3. 변환 후 문맥이 **자연스러워야** 합니다
4. 응답은 반드시 지정된 JSON 형식으로 출력합니다"""


def _build_transform_prompt(
    problem: Dict,
    transform_type: str,
) -> str:
    """Build the transformation prompt for LLM."""
    spec = TRANSFORM_SPECS[transform_type]
    question = problem.get("question", "")
    context = problem.get("context_original", problem.get("context", ""))
    python_solution = problem.get("python_solution", "")
    ground_truth = problem.get("ground_truth", "")
    ctx_format = problem.get("context_format", problem.get("context_type", ""))
    has_context = bool(
        context and context.strip() and context.strip() not in ("[]", "")
    )

    prompt = f"""## 변환 유형: {spec["name"]} ({transform_type})

{spec["instruction"]}

---

## 원본 문제

**Question:**
{question}

**Context** (format: {ctx_format}):
{context if has_context else "(context 없음 — 모든 데이터가 question에 포함)"}

**Python Solution (풀이 참고):**
```python
{python_solution}
```

**Ground Truth (정답):** {ground_truth}

---

## 지시사항

1. 위 python_solution을 분석하여 **정답을 도출하는 데 필수적인 데이터**를 식별하세요.
2. 해당 데이터를 제거/변형했을 때 **남은 데이터로 역산할 수 있는지** 판단하세요.
3. 역산 불가능한 데이터를 선택하여 **{spec["name"]}** 기준에 맞게 변환하세요.
{"4. context가 없으므로 **question 자체를 변환**하세요." if not has_context else "4. **context를 변환**하세요. question은 변경하지 마세요."}

## 출력 형식 (반드시 이 JSON 형식만 출력)

```json
{{
  "is_transformable": true 또는 false,
  "reason_if_not": "변환 불가 시 이유 (변환 가능하면 null)",
  "critical_data": ["풀이에 필수적인 데이터 목록"],
  "removed_or_modified": "제거/변형한 데이터 설명",
  "derivable_check": "역산 가능성 분석 결과",
  "transformed_{"question" if not has_context else "context"}": "변환된 전체 {"question" if not has_context else "context"}",
  "description": "변환 요약 (한 줄)"
}}
```"""
    return prompt


@dataclass
class TransformResult:
    """Single transformation result."""

    transform_type: str
    success: bool
    transformed_content: Optional[str] = None
    is_question_transform: bool = False
    description: str = ""
    critical_data: Optional[List[str]] = None
    removed_or_modified: str = ""
    derivable_check: str = ""
    reason: str = ""
    raw_response: str = ""
    cost_usd: float = 0.0


class LLMTransformer:
    """LLM-based transformation engine."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        concurrency: int = 5,
    ):
        self.model = model
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency)
        self.total_cost = 0.0
        self.total_calls = 0

        # Detect provider from model name
        if "gemini" in model:
            self.provider = "google"
            self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        elif "gpt" in model:
            self.provider = "openai"
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        elif "claude" in model:
            self.provider = "anthropic"
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            raise ValueError(f"Unknown model: {model}")

        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def _call_llm(self, system: str, user: str) -> Dict[str, Any]:
        """Call LLM API with retry."""
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    if self.provider == "google":
                        return await self._call_google(system, user)
                    elif self.provider == "openai":
                        return await self._call_openai(system, user)
                    elif self.provider == "anthropic":
                        return await self._call_anthropic(system, user)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = (2**attempt) + 1
                    logger.warning(
                        f"API 오류 (시도 {attempt + 1}): {str(e)[:80]}. "
                        f"{wait}초 후 재시도..."
                    )
                    await asyncio.sleep(wait)
                else:
                    return {"text": "", "error": str(e), "cost": 0.0}
        return {"text": "", "error": "max retries exceeded", "cost": 0.0}

    async def _call_google(self, system: str, user: str) -> Dict[str, Any]:
        """Call Google Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        cost = (usage.get("totalTokenCount", 0) / 1_000_000) * 0.15
        return {"text": text, "cost": cost}

    async def _call_openai(self, system: str, user: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("total_tokens", 0) / 1_000_000) * 2.5
        return {"text": text, "cost": cost}

    async def _call_anthropic(self, system: str, user: str) -> Dict[str, Any]:
        """Call Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.1,
        }
        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        cost = (total_tokens / 1_000_000) * 3.0
        return {"text": text, "cost": cost}

    def _parse_response(
        self, raw_text: str, transform_type: str, has_context: bool
    ) -> Dict[str, Any]:
        """Parse LLM JSON response with multiple fallback strategies."""
        text = raw_text.strip()

        # Strategy 1: Extract from markdown code block
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        # Strategy 2: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Find outermost JSON object
        depth = 0
        start_idx = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    try:
                        return json.loads(text[start_idx : i + 1])
                    except json.JSONDecodeError:
                        pass
                    break

        # Strategy 4: Fix common JSON issues and retry
        cleaned = text
        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        # Fix unescaped newlines in strings
        cleaned = cleaned.replace("\n", "\\n")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        logger.debug(f"JSON 파싱 실패 — raw: {text[:200]}")
        return {"is_transformable": False, "reason_if_not": "JSON 파싱 실패"}

    async def transform_single(
        self, problem: Dict, transform_type: str
    ) -> TransformResult:
        """Apply a single transformation type to a problem."""
        context = problem.get("context_original", problem.get("context", ""))
        has_context = bool(
            context and context.strip() and context.strip() not in ("[]", "")
        )

        prompt = _build_transform_prompt(problem, transform_type)
        response = await self._call_llm(SYSTEM_PROMPT, prompt)

        self.total_cost += response.get("cost", 0.0)
        self.total_calls += 1

        if response.get("error"):
            return TransformResult(
                transform_type=transform_type,
                success=False,
                reason=f"API error: {response['error'][:80]}",
                raw_response=response.get("text", ""),
                cost_usd=response.get("cost", 0.0),
            )

        parsed = self._parse_response(response["text"], transform_type, has_context)

        if not parsed.get("is_transformable", False):
            return TransformResult(
                transform_type=transform_type,
                success=False,
                reason=parsed.get("reason_if_not", "not_transformable"),
                raw_response=response["text"],
                cost_usd=response.get("cost", 0.0),
            )

        # Extract transformed content
        content_key = (
            "transformed_question" if not has_context else "transformed_context"
        )
        transformed = parsed.get(content_key, "")

        if not transformed:
            # Try alternate key names
            for key in ["transformed_content", "transformed_text", content_key]:
                if parsed.get(key):
                    transformed = parsed[key]
                    break

        if not transformed:
            return TransformResult(
                transform_type=transform_type,
                success=False,
                reason="no_transformed_content",
                raw_response=response["text"],
                cost_usd=response.get("cost", 0.0),
            )

        return TransformResult(
            transform_type=transform_type,
            success=True,
            transformed_content=transformed,
            is_question_transform=not has_context,
            description=parsed.get("description", ""),
            critical_data=parsed.get("critical_data"),
            removed_or_modified=parsed.get("removed_or_modified", ""),
            derivable_check=parsed.get("derivable_check", ""),
            raw_response=response["text"],
            cost_usd=response.get("cost", 0.0),
        )

    async def transform_problem(
        self,
        problem: Dict,
        types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Apply all transformation types to a single problem.

        Returns dict compatible with run_batch_transformation output format.
        """
        if types is None:
            types = ["EA-partial", "EA-full", "SA", "IC"]

        context = problem.get("context_original", problem.get("context", ""))
        has_context = bool(
            context and context.strip() and context.strip() not in ("[]", "")
        )

        results = {}
        tasks = [self.transform_single(problem, t) for t in types]
        transform_results = await asyncio.gather(*tasks)

        for tr in transform_results:
            if tr.success:
                entry = {
                    "success": True,
                    "description": tr.description,
                    "critical_data": tr.critical_data,
                    "removed_or_modified": tr.removed_or_modified,
                    "derivable_check": tr.derivable_check,
                }
                if tr.is_question_transform:
                    entry["question_transformed"] = tr.transformed_content
                    entry["is_question_transform"] = True
                else:
                    entry["context_transformed"] = tr.transformed_content
                results[tr.transform_type] = entry
            else:
                results[tr.transform_type] = {
                    "success": False,
                    "reason": tr.reason,
                }

        # TA: only if not already in types
        if "TA" not in types:
            results["TA"] = {"success": False, "reason": "skipped"}

        return results
