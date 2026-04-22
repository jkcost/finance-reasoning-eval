"""LLM 해설 v2 — 변환 맥락 포함 + 거부/오답 분기.

기존 v1 문제:
  - 변환 내용을 LLM에게 안 줌 → "원래 풀이법"만 반복
  - 거부 케이스도 오답 프롬프트로 분석 → CALCULATION_ERROR 오분류
  - python_solution이 하드코딩이면 오히려 오도 유발

v2 개선:
  - transformation_description + removed_or_modified + context_transformed 입력
  - 거부는 REFUSAL_VALID/OVER_REFUSAL/REFUSAL_WRONG_REASON 판정
  - 오답은 HALLUCINATION(암기) 카테고리 포함
  - CORRECT_APPROACH가 "변환 후에는 정답 도출 불가" 같은 답을 낼 수 있게 허용

Usage:
    python experiments/regenerate_analyses_v2.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from _eval_io import load_eval_records  # noqa: E402
from error_analysis.llm_error_analyzer import ErrorAnalysisResult  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


REFUSAL_PROMPT = """당신은 금융 LLM 평가 전문가입니다. 변환된 문제에 대한 모델의 거부 판정을 분석하세요.
반드시 한글로 답변하세요.

## 원본 문제
{question}

## 정답 계산 코드 (원본 context 기준)
```python
{python_solution}
```

## 정답
{ground_truth}

## 변환 정보
- 변환 유형: {transformation_type}
- 변환 내용: {removed_or_modified}

## 변환된 Context (모델이 실제로 본 것)
{context_transformed}

## 모델 응답 (거부)
{raw_response}

## 태스크
이 모델이 INSUFFICIENT_INFORMATION으로 "거부"했습니다. 거부가 **올바른 판단**인지 분석하세요.

분석 순서:
1. python_solution이 사용하는 **값들**(변수 = 숫자)을 나열
2. 변환된 context에서 **각 값이 남아있는지/사라졌는지** 확인
3. 정답 도출에 필수인 값이 사라졌다면 → **거부 타당** (REFUSAL_VALID)
4. 필요한 값이 다 있는데 거부했다면 → **과도 거부** (OVER_REFUSAL)
5. 거부는 맞지만 이유가 엉뚱하면 → REFUSAL_WRONG_REASON
6. 거부 메시지에 context에 없는 값을 언급하면 → HALLUCINATION

## 응답 형식 (정확히 이 포맷)
ERROR_TYPE: <REFUSAL_VALID|OVER_REFUSAL|REFUSAL_WRONG_REASON|HALLUCINATION>
SUMMARY: <한 줄: 거부가 타당한지 + 왜>
DETAILED_ANALYSIS: <사라진 값 vs 남은 값, 모델 판단 평가 2~3문장>
LIKELY_CAUSE: <한 문장: 이 거부를 유발한 근본 신호/맥락>
CORRECT_APPROACH: <1~2문장: 변환 후 상황에서 모델이 가졌어야 할 판단. 거부가 타당하면 "이 변환 후에는 정답 도출 불가, 거부가 올바름" 식으로 명시>
CONFIDENCE: <0.0~1.0>
"""


WRONG_PROMPT = """당신은 금융 계산 오류 분석 전문가입니다.
반드시 한글로 답변하세요.

## 원본 문제
{question}

## 정답 계산 코드 (원본 기준)
```python
{python_solution}
```

## 정답
{ground_truth}

## 변환 정보
- 변환 유형: {transformation_type}
- 변환 내용: {removed_or_modified}

## 변환된 Context (모델이 실제로 본 것)
{context_transformed}

## 모델 응답
예측값: {predicted_answer}

응답 원문:
{raw_response}

## 태스크
변환 후 모델이 **오답**을 냈습니다. 왜 틀렸는지 분석하세요.

분석 순서:
1. python_solution이 사용하는 값들이 **변환 후 context에 여전히 있는가**?
2. 모델이 응답에서 사용한 값들이 **python_solution과 일치/상이**한가?
3. 모델이 어느 단계에서 틀렸는지 (값 추출 / 공식 / 계산 / 단위)

주의사항:
- 변환으로 값이 제거됐는데 모델이 답을 냈다면: **HALLUCINATION** (암기된 값 사용 가능성)
- 변환 후 unsolvable인데 모델이 엉뚱한 값을 가져다 썼다면: **EXTRACTION_ERROR** + 재검토
- python_solution이 하드코딩 값을 쓰면: context에서 찾을 수 없을 수 있음을 감안하여 분석

## 응답 형식 (정확히 이 포맷)
ERROR_TYPE: <FORMULA_ERROR|EXTRACTION_ERROR|CALCULATION_ERROR|MISUNDERSTANDING|UNIT_ERROR|HALLUCINATION|LOGIC_ERROR|INCOMPLETE>
SUMMARY: <한 줄 요약, 변환 맥락 포함>
DETAILED_ANALYSIS: <2~3문장: 변환 후 어떻게 풀려 했고 어디서 틀렸는지>
LIKELY_CAUSE: <한 문장: 근본 원인>
CORRECT_APPROACH: <1~2문장: 변환 후 상황에서 올바른 접근. 풀 수 없으면 "거부가 올바른 판단" 명시>
CONFIDENCE: <0.0~1.0>
"""


def _parse_response(text: str) -> ErrorAnalysisResult:
    """Parse STATUS fields from v2 response."""

    def extract(label: str) -> str:
        m = re.search(rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    conf_m = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
    conf = float(conf_m.group(1)) if conf_m else 0.85
    return ErrorAnalysisResult(
        error_type=extract("ERROR_TYPE") or "UNKNOWN",
        summary=extract("SUMMARY") or "",
        detailed_analysis=extract("DETAILED_ANALYSIS") or "",
        likely_cause=extract("LIKELY_CAUSE") or "",
        correct_approach=extract("CORRECT_APPROACH") or "",
        confidence=max(0.0, min(1.0, conf)),
    )


async def _call_openai(
    client: OpenAI, model: str, prompt: str, system_msg: str, max_tokens: int = 900
) -> str:
    """Blocking OpenAI call wrapped in executor."""
    loop = asyncio.get_event_loop()

    def _sync() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return await loop.run_in_executor(None, _sync)


def _is_refusal(record: dict[str, Any]) -> bool:
    """Heuristic refusal detection from an eval record."""
    if record.get("response_type") in ("refused", "caveat"):
        return True
    predicted = record.get("predicted_answer", "") or ""
    return str(predicted).startswith("INSUFFICIENT_INFORMATION")


async def generate_one(
    client: OpenAI,
    model: str,
    problem: dict[str, Any],
    record: dict[str, Any],
    transformation_type: str | None,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Generate a single v2 analysis. Picks refusal vs wrong prompt automatically."""
    async with sem:
        question = problem.get("question", "")
        python_solution = problem.get("python_solution", "# not available")
        ground_truth = problem.get("ground_truth", "")

        if transformation_type:
            tx = problem.get("transformations", {}).get(transformation_type, {})
            context_transformed = tx.get("context_transformed", "") or ""
            removed = tx.get("removed_or_modified", "") or tx.get("description", "")
            ttype_label = transformation_type
        else:
            # Original context
            context_transformed = problem.get("context_original", "")
            removed = "(원본 context, 변환 없음)"
            ttype_label = "original"

        predicted = record.get("predicted_answer", "") or ""
        raw = (record.get("raw_response", "") or "")[:1800]
        is_refusal = _is_refusal(record)

        if is_refusal:
            prompt = REFUSAL_PROMPT.format(
                question=question[:1500],
                python_solution=str(python_solution)[:2000],
                ground_truth=ground_truth,
                transformation_type=ttype_label,
                removed_or_modified=str(removed)[:600],
                context_transformed=str(context_transformed)[:2500],
                raw_response=raw,
            )
            system_msg = "당신은 금융 LLM 거부 판정의 타당성을 분석하는 전문가입니다. 한글로 답변하세요."
        else:
            prompt = WRONG_PROMPT.format(
                question=question[:1500],
                python_solution=str(python_solution)[:2000],
                ground_truth=ground_truth,
                transformation_type=ttype_label,
                removed_or_modified=str(removed)[:600],
                context_transformed=str(context_transformed)[:2500],
                predicted_answer=predicted,
                raw_response=raw,
            )
            system_msg = "당신은 금융 계산 오류 분석 전문가입니다. 한글로 답변하세요."

        try:
            text = await _call_openai(client, model, prompt, system_msg)
            result = asdict(_parse_response(text))
            result["is_refusal"] = is_refusal
            result["verification_status"] = "regenerated_v2"
            result["verification_note"] = (
                "v2 프롬프트로 재생성 (변환 맥락 + 거부/오답 분기)"
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(f"v2 generate failed: {exc}")
            return None


def _index_problems(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """qid → problem."""
    return {p["question_id"]: p for p in batch.get("problems", [])}


def _pick_original_pot(
    pot_records: list[dict[str, Any]], qid: str
) -> dict[str, Any] | None:
    """Find any POT record for qid (original-context responses are identical across ttype)."""
    for r in pot_records:
        if r.get("question_id") == qid:
            return r
    return None


def _find_strategy_record(
    strategy_recs: list[dict[str, Any]], qid: str, ttype: str
) -> dict[str, Any] | None:
    """Match by (qid, transformation_type)."""
    for r in strategy_recs:
        if r.get("question_id") == qid and r.get("transformation_type") == ttype:
            return r
    return None


async def run_original(
    client: OpenAI,
    model: str,
    sem: asyncio.Semaphore,
    original_path: Path,
    batch_path: Path,
    pot_original_path: Path,
) -> None:
    """Regenerate all original-context analyses with v2 prompts."""
    if not original_path.exists():
        logger.warning(f"original analysis 없음: {original_path}")
        return

    # Backup
    backup = original_path.with_suffix(".v1.bak.json")
    if not backup.exists():
        shutil.copy2(original_path, backup)
        logger.info(f"[original] backup → {backup.name}")

    data = json.loads(original_path.read_text(encoding="utf-8"))
    analyses = data.get("analyses", {})
    if not analyses:
        return

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    problems_idx = _index_problems(batch)
    pot_records = load_eval_records(pot_original_path)

    targets = []
    for qid in analyses.keys():
        problem = problems_idx.get(qid)
        rec = _pick_original_pot(pot_records, qid)
        if problem and rec:
            targets.append((qid, problem, rec))

    logger.info(f"[original] regenerating {len(targets)} analyses with v2 prompts")
    coros = [generate_one(client, model, p, r, None, sem) for (_, p, r) in targets]
    results = await asyncio.gather(*coros)

    regen_ok = 0
    for (qid, _, _), result in zip(targets, results):
        if result is not None:
            analyses[qid] = result
            regen_ok += 1
    logger.info(f"[original] regenerated: {regen_ok}/{len(targets)}")

    from collections import Counter
    et = Counter(a.get("error_type") for a in analyses.values())
    logger.info(f"[original] error_type dist: {dict(et)}")

    data["analyses"] = analyses
    data.setdefault("metadata", {})["prompt_version"] = "v2"
    original_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def run_transformed(
    client: OpenAI,
    model: str,
    sem: asyncio.Semaphore,
    transformed_path: Path,
    batch_path: Path,
    standard_path: Path,
    metacog_path: Path,
    cot_path: Path,
) -> None:
    """Regenerate all transformed-context analyses (3 strategies) with v2 prompts."""
    if not transformed_path.exists():
        logger.warning(f"transformed analysis 없음: {transformed_path}")
        return

    backup = transformed_path.with_suffix(".v1.bak.json")
    if not backup.exists():
        shutil.copy2(transformed_path, backup)
        logger.info(f"[transformed] backup → {backup.name}")

    data = json.loads(transformed_path.read_text(encoding="utf-8"))
    analyses = data.get("analyses", {})
    if not analyses:
        return

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    problems_idx = _index_problems(batch)

    strategy_recs = {
        "standard": load_eval_records(standard_path) if standard_path.exists() else [],
        "metacognitive": load_eval_records(metacog_path) if metacog_path.exists() else [],
        "cot_trace": load_eval_records(cot_path) if cot_path.exists() else [],
    }

    targets: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = []
    for qid, per_t in analyses.items():
        problem = problems_idx.get(qid)
        if not problem:
            continue
        for ttype, per_s in per_t.items():
            for strategy in per_s.keys():
                rec = _find_strategy_record(strategy_recs.get(strategy, []), qid, ttype)
                if rec:
                    targets.append((qid, ttype, strategy, problem, rec))

    logger.info(f"[transformed] regenerating {len(targets)} cells with v2 prompts")

    coros = [
        generate_one(client, model, problem, rec, ttype, sem)
        for (_, ttype, _, problem, rec) in targets
    ]
    results = await asyncio.gather(*coros)

    regen_ok = 0
    for (qid, ttype, strategy, _, _), result in zip(targets, results):
        if result is not None:
            analyses[qid][ttype][strategy] = result
            regen_ok += 1
    logger.info(f"[transformed] regenerated: {regen_ok}/{len(targets)}")

    from collections import Counter
    combo = Counter((s, a.get("error_type"))
                    for per_t in analyses.values()
                    for per_s in per_t.values()
                    for s, a in per_s.items())
    logger.info("[transformed] strategy × error_type dist:")
    for (s, et), n in sorted(combo.items()):
        logger.info(f"  {s:<14s} {et:<24s} {n}")

    data["analyses"] = analyses
    data.setdefault("metadata", {})["prompt_version"] = "v2"
    transformed_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def main_async(args: argparse.Namespace) -> int:
    """Run both original and transformed with v2 prompts."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY 없음")
        return 1
    client = OpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    await run_original(
        client, args.model, sem,
        args.original, args.batch, args.pot_original,
    )
    await run_transformed(
        client, args.model, sem,
        args.transformed, args.batch,
        args.standard, args.metacognitive, args.cot_trace,
    )
    logger.info("=== v2 재생성 완료 ===")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI."""
    parser = argparse.ArgumentParser(description="v2 해설 재생성 (변환 맥락 + 거부/오답 분기)")
    parser.add_argument(
        "--batch",
        type=Path,
        default=Path("experiments/results/metacognitive/batch_transformations_0_20_subset.json"),
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("experiments/results/metacognitive/original_error_analysis_0_20.json"),
    )
    parser.add_argument(
        "--transformed",
        type=Path,
        default=Path("experiments/results/metacognitive/transformed_error_analysis_0_20.json"),
    )
    parser.add_argument(
        "--pot-original",
        type=Path,
        default=Path("experiments/results/metacognitive/eval_original_context.json"),
    )
    parser.add_argument(
        "--standard",
        type=Path,
        default=Path("experiments/results/metacognitive/batch_evaluation_0_238.json"),
    )
    parser.add_argument(
        "--metacognitive",
        type=Path,
        default=Path("experiments/results/metacognitive/eval_metacognitive.json"),
    )
    parser.add_argument(
        "--cot-trace",
        type=Path,
        default=Path("experiments/results/metacognitive/batch_evaluation_0_20.json"),
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    """Entry."""
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
