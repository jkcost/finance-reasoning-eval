"""
LLM Judgment Generator

리뷰어 간 불일치 항목에 대해 LLM이 원본/변환 데이터를 비교 분석하고
최종 판정 + 근거 커멘트를 생성.

Pipeline:
  1. merge_annotations.py → merged.json + disagreements.json
  2. generate_llm_judgment.py → llm_judgments.json  (이 스크립트)
  3. generate_review_summary.py → review_summary.html (judgments 반영)

Usage:
    python experiments/generate_llm_judgment.py
    python experiments/generate_llm_judgment.py --model gemini-2.5-flash
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def build_analysis_prompt(item: Dict[str, Any]) -> str:
    """Build a prompt for LLM to analyze a disagreement."""

    qid = item["qid"]
    ttype = item["ttype"]
    question_orig = item["question"]
    question_transformed = item.get("question_transformed", "")
    context_orig = item["context_original"][:2000]
    context_transformed = item.get("context_transformed", "")[:2000]
    description = item.get("description", "")
    ground_truth = item.get("ground_truth", "")
    critical_data = item.get("critical_data", [])
    removed_or_modified = item.get("removed_or_modified", "")

    reviewer_lines = ""
    for r in item["reviewers"]:
        reviewer_lines += f"  - {r['name']}: {r['judgment']}"
        if r.get("judgment_reason"):
            reviewer_lines += f" | 판정이유: {r['judgment_reason']}"
        if r.get("note"):
            reviewer_lines += f" | 메모: {r['note']}"
        reviewer_lines += "\n"

    llm_response_lines = ""
    for lr in item.get("llm_responses", []):
        model = lr.get("model", "?")
        correct = "정답" if lr.get("is_correct") else "오답"
        predicted = lr.get("predicted_answer", "?")
        resp_type = lr.get("response_type", "?")
        llm_response_lines += f"  - {model}: {resp_type} → {predicted} ({correct})\n"

    question_comparison = ""
    if question_transformed and question_transformed != question_orig:
        question_comparison = f"""
## 원본 Question vs 변환 Question
원본: {question_orig}
변환: {question_transformed}
"""
    else:
        question_comparison = f"""
## Question (변경 없음)
{question_orig}
"""

    return f"""당신은 금융 NLP 연구의 메타인지 평가 전문가입니다.

아래는 LLM의 메타인지 능력을 테스트하기 위한 문제 변환에 대한 리뷰어 간 불일치 항목입니다.
변환의 품질을 평가하고 최종 판정을 내려주세요.

# 변환 정보
- 문제 ID: {qid}
- 변환 타입: {ttype}
- 변환 설명: {description}
- 정답 (Ground Truth): {ground_truth}
- 핵심 데이터: {json.dumps(critical_data, ensure_ascii=False)}
- 제거/변형된 내용: {removed_or_modified}

{question_comparison}

## 원본 Context
{context_orig if context_orig else "(context 없음 — 모든 데이터가 question에 포함)"}

## 변환된 Context
{context_transformed if context_transformed else "(context 없음 — question이 변환됨)"}

## 리뷰어 판정 (불일치)
{reviewer_lines}

## LLM 실험 결과
{llm_response_lines if llm_response_lines else "  (실험 데이터 없음)"}

# 평가 기준

변환 타입별 목적:
- EA-partial: 핵심 값을 [DATA MISSING]으로 표시 → LLM이 "정보 부족"을 인식해야 함
- EA-full: 핵심 칼럼/데이터 전체 삭제 → LLM이 "정보 부족"을 인식해야 함
- SA: 흔적 없이 자연스럽게 제거 → LLM이 부재를 스스로 발견해야 함
- IC-L1: 10배 오타 → LLM이 비정상 수치를 감지해야 함
- IC-L2: 단위 불일치 + 1.5배 오류 삽입 → LLM이 모순을 감지해야 함
- IC-L3: 권위 있는 출처가 모순 값 제시 → LLM이 충돌을 감지해야 함
- IC-L4: 기간 합산 불일치 → LLM이 내부 불일치를 감지해야 함

판정 기준:
1. **승인 (approved)**: 변환이 의도대로 작동하고, LLM의 메타인지를 효과적으로 테스트함
2. **수정필요 (needs_modification)**: 변환 아이디어는 좋지만 구현에 개선이 필요함
3. **부적절 (rejected)**: 변환이 근본적으로 목적을 달성하지 못함 (역산 가능, 정답 동일, 등)

특히 확인할 사항:
- 제거/변형된 데이터가 다른 경로로 역산 가능한가?
- 변환 후에도 정답이 동일하게 나올 수 있는가?
- 영어 context에 한글이 섞여 있지 않은가?
- IC 타입 간 구분이 모호하지 않은가?
- context 없는 문제에서 LLM이 암기로 정답을 맞출 가능성이 있는가?

# 응답 형식 (JSON)

다음 JSON 형식으로만 응답하세요:
{{
  "judgment": "approved" | "needs_modification" | "rejected",
  "confidence": "high" | "medium" | "low",
  "comment": "한글로 2-3문장의 판단 근거",
  "issues": ["발견된 구체적 이슈 목록 (있으면)"],
  "agrees_with": "가장 동의하는 리뷰어 이름",
  "suggestion": "수정 방향 제안 (needs_modification/rejected인 경우)"
}}"""


def prepare_disagreement_items(
    disagreements_path: Path,
    batch_path: Path,
    eval_path: Optional[Path],
) -> List[Dict]:
    """Load and prepare all disagreement items with full context."""

    with open(disagreements_path, "r", encoding="utf-8") as f:
        disagreements = json.load(f)

    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    problem_map = {p["question_id"]: p for p in batch["problems"]}

    eval_map: Dict[tuple, List[Dict]] = {}
    if eval_path and eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            evals = json.load(f)
        for r in evals.get("results", []):
            key = (r["question_id"], r["transformation_type"])
            eval_map.setdefault(key, []).append(r)

    items = []
    for d in disagreements:
        qid = d["qid"]
        ttype = d["ttype"]
        p = problem_map.get(qid, {})
        t = p.get("transformations", {}).get(ttype, {})

        item = {
            "qid": qid,
            "ttype": ttype,
            "question": p.get("question", ""),
            "question_transformed": t.get("question_transformed", ""),
            "ground_truth": p.get("ground_truth", ""),
            "context_original": p.get("context_original", ""),
            "context_transformed": t.get("context_transformed", ""),
            "description": t.get("description", ""),
            "critical_data": t.get("critical_data", []),
            "removed_or_modified": t.get("removed_or_modified", ""),
            "reviewers": [
                {
                    "name": a["assignee"],
                    "judgment": a["judgment"],
                    "judgment_reason": a.get("judgment_reason", ""),
                    "note": a.get("note", ""),
                }
                for a in d["annotations"]
            ],
            "llm_responses": [
                {
                    "model": er.get("model", ""),
                    "response_type": er.get("response_type", ""),
                    "is_correct": er.get("is_correct", False),
                    "predicted_answer": str(er.get("predicted_answer", ""))[:200],
                }
                for er in eval_map.get((qid, ttype), [])
            ],
        }
        items.append(item)

    return items


async def call_gemini(
    client: httpx.AsyncClient,
    prompt: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """Call Google Gemini API directly."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }

    resp = await client.post(url, json=payload, params={"key": api_key}, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return parts[0].get("text", "") if parts else ""


async def call_openai(
    client: httpx.AsyncClient,
    prompt: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Call OpenAI API directly."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    resp = await client.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


MODEL_API_MAP = {
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-2.5-pro": ("gemini", "gemini-2.5-pro"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o": ("openai", "gpt-4o"),
}


async def run_llm_judgments(
    items: List[Dict],
    model: str = "gemini-2.5-flash",
) -> List[Dict]:
    """Run LLM analysis on all disagreement items."""

    results = []

    provider, api_model = MODEL_API_MAP.get(model, ("gemini", model))

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(items):
            qid = item["qid"]
            ttype = item["ttype"]
            logger.info(f"[{i + 1}/{len(items)}] 분석 중: {qid}/{ttype}")

            prompt = build_analysis_prompt(item)

            try:
                if provider == "gemini":
                    raw = await call_gemini(client, prompt, api_model)
                else:
                    raw = await call_openai(client, prompt, api_model)

                # Parse JSON from response
                judgment_data = _parse_json_response(raw)

                if judgment_data:
                    judgment_data["qid"] = qid
                    judgment_data["ttype"] = ttype
                    judgment_data["model_used"] = model
                    judgment_data["raw_response"] = raw[:1000]
                    results.append(judgment_data)
                    logger.info(
                        f"  -> {judgment_data.get('judgment', '?')} "
                        f"(confidence: {judgment_data.get('confidence', '?')})"
                    )
                else:
                    logger.warning("  -> JSON 파싱 실패, 원본 저장")
                    results.append(
                        {
                            "qid": qid,
                            "ttype": ttype,
                            "model_used": model,
                            "judgment": "parse_error",
                            "comment": "LLM 응답에서 JSON을 파싱할 수 없었습니다.",
                            "raw_response": raw[:1000],
                        }
                    )

            except Exception as e:
                logger.error(f"  -> error: {e}")
                results.append(
                    {
                        "qid": qid,
                        "ttype": ttype,
                        "model_used": model,
                        "judgment": "error",
                        "comment": f"API error: {str(e)[:200]}",
                    }
                )

    return results


def _parse_json_response(raw: str) -> Optional[Dict]:
    """Extract JSON from LLM response with multi-stage fallback."""

    # Stage 1: direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Stage 2: find JSON in code blocks
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Stage 3: find outermost { ... } containing "judgment"
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Stage 4: truncated JSON — try to close it
    if '"judgment"' in raw:
        candidate = raw.strip()
        if not candidate.endswith("}"):
            # Try closing arrays and objects
            for suffix in ['"}', '"]"}', '"]}']:
                try:
                    return json.loads(candidate + suffix)
                except json.JSONDecodeError:
                    continue

    return None


def save_judgments(results: List[Dict], output_path: Path) -> None:
    """Save LLM judgments to JSON."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "total": len(results),
        "judgments": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"LLM 판정 저장: {output_path}")

    # Summary
    counts = {}
    for r in results:
        j = r.get("judgment", "unknown")
        counts[j] = counts.get(j, 0) + 1
    logger.info(f"판정 분포: {counts}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM judgments for disagreements"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use for analysis",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="experiments/results/metacognitive/annotations",
    )
    parser.add_argument(
        "--batch-input",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval-input",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/metacognitive/annotations/llm_judgments.json",
    )
    args = parser.parse_args()

    ann_dir = PROJECT_ROOT / args.annotations_dir
    disagreements_path = ann_dir / "disagreements.json"

    if not disagreements_path.exists():
        logger.error(f"불일치 파일 없음: {disagreements_path}")
        logger.info("먼저 merge_annotations.py를 실행하세요.")
        sys.exit(1)

    # Auto-detect batch file
    if args.batch_input:
        batch_path = PROJECT_ROOT / args.batch_input
    else:
        candidates = sorted(
            (PROJECT_ROOT / "experiments/results/metacognitive").glob(
                "batch_transformations_*.json"
            ),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        batch_path = candidates[0] if candidates else None

    # Auto-detect eval file
    if args.eval_input:
        eval_path = PROJECT_ROOT / args.eval_input
    else:
        candidates = sorted(
            (PROJECT_ROOT / "experiments/results/metacognitive").glob(
                "batch_evaluation_*.json"
            ),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        eval_path = candidates[0] if candidates else None

    output_path = PROJECT_ROOT / args.output

    items = prepare_disagreement_items(disagreements_path, batch_path, eval_path)
    logger.info(f"분석 대상: {len(items)}건 불일치")

    results = await run_llm_judgments(items, model=args.model)
    save_judgments(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
