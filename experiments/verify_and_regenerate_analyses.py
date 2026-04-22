"""Verify + regenerate LLM 한글 해설 (옵션 B 파이프라인).

단계:
    1. 기존 해설(원본 + 변환 3전략) 전체에 대해 python_solution과 대조 검증
       → verification_status ∈ {consistent, needs_review, contradicts}
    2. needs_review / contradicts 해설만 python_solution을 프롬프트에 포함하여 재생성
    3. 원본 JSON 파일 덮어쓰기 (verification 필드 포함)

Usage:
    python experiments/verify_and_regenerate_analyses.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from _eval_io import load_eval_records  # noqa: E402
from error_analysis.llm_error_analyzer import (  # noqa: E402
    ErrorAnalysisResult,
    LLMErrorAnalyzer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


VERIFY_PROMPT = """당신은 금융 오답 해설의 정확성을 검증하는 전문가입니다.
반드시 한글로 답변하세요.

## 문제
{question}

## 정답 계산 코드 (authoritative, 이 코드가 정답)
```python
{python_solution}
```

## 정답
{ground_truth}

## 모델 응답
{predicted}

## 검증할 해설
- 유형: {error_type}
- 요약: {summary}
- 상세 분석: {detailed_analysis}
- 원인: {likely_cause}
- 정답 접근: {correct_approach}

## 태스크
위 해설의 "정답 접근"과 "상세 분석"이 python_solution 코드의 계산 방법론과
얼마나 일치하는지 판정하세요. 3가지 상태 중 하나:

- CONSISTENT: 해설의 방법론이 python_solution과 본질적으로 같음
- NEEDS_REVIEW: 부분 일치 — 일부 세부(변수명/계수 등)가 어긋나거나 모호
- CONTRADICTS: 해설이 python_solution과 근본적으로 다른 방법/공식 제시

응답 형식 (정확히 이 포맷으로):
STATUS: <CONSISTENT|NEEDS_REVIEW|CONTRADICTS>
NOTE: <한글 한 줄 근거 — 어디가 일치/어긋나는지>
"""


REGEN_PROMPT_WITH_SOLUTION = """당신은 금융 계산 오류를 분석하는 전문가입니다.
주어진 문제, 정답, python_solution(정답 계산 코드), 모델의 오답을 보고 왜 틀렸는지 분석하세요.

반드시 한글로 답변하세요. 쉽고 명확한 표현을 사용하세요.
**"정답 접근"은 반드시 python_solution의 방법론에 기반해야 합니다.**

## 문제
{question}

## 주어진 데이터
{context}

## 정답
{ground_truth}

## 정답 계산 코드 (이 코드가 정답을 도출합니다)
```python
{python_solution}
```

## 모델 응답
예측값: {predicted_answer}

응답 원문:
{raw_response}

## 응답 형식 (정확히 이 포맷으로, 모든 필드 채우세요)
ERROR_TYPE: <FORMULA_ERROR|EXTRACTION_ERROR|CALCULATION_ERROR|MISUNDERSTANDING|UNIT_ERROR|INSUFFICIENT_INFORMATION|OVER_REFUSAL|INCOMPLETE|LOGIC_ERROR>
SUMMARY: <한 줄 요약>
DETAILED_ANALYSIS: <2~3문장, 모델이 어느 단계에서 어떻게 실패했는지>
LIKELY_CAUSE: <한 문장, 근본 원인>
CORRECT_APPROACH: <1~2문장, python_solution에 기반한 올바른 접근법>
CONFIDENCE: <0.0~1.0>
"""


def _parse_verify_response(text: str) -> tuple[str, str]:
    """Parse STATUS / NOTE from verification response."""
    import re
    status_m = re.search(r"STATUS:\s*(\w+)", text, re.IGNORECASE)
    note_m = re.search(r"NOTE:\s*(.+?)(?:\n\n|$)", text, re.DOTALL | re.IGNORECASE)
    status = status_m.group(1).lower() if status_m else "needs_review"
    note = note_m.group(1).strip() if note_m else ""
    # Normalize
    if status.upper() == "CONSISTENT":
        status = "consistent"
    elif status.upper() == "CONTRADICTS":
        status = "contradicts"
    else:
        status = "needs_review"
    return status, note


def _parse_regen_response(text: str) -> ErrorAnalysisResult:
    """Parse regenerated analysis (same format as llm_error_analyzer)."""
    import re

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


async def verify_one(
    client: OpenAI,
    model: str,
    question: str,
    python_solution: str,
    ground_truth: Any,
    predicted: Any,
    analysis: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[str, str]:
    """Return (verification_status, verification_note)."""
    async with sem:
        prompt = VERIFY_PROMPT.format(
            question=question[:1500],
            python_solution=(python_solution or "# no solution code available")[:2000],
            ground_truth=ground_truth,
            predicted=predicted,
            error_type=analysis.get("error_type", ""),
            summary=analysis.get("summary", ""),
            detailed_analysis=analysis.get("detailed_analysis", ""),
            likely_cause=analysis.get("likely_cause", ""),
            correct_approach=analysis.get("correct_approach", ""),
        )
        loop = asyncio.get_event_loop()

        def _call() -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 금융 해설의 정확성 검증 전문가입니다. 한글로 간결히 답변하세요."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=250,
            )
            return resp.choices[0].message.content or ""

        try:
            text = await loop.run_in_executor(None, _call)
            return _parse_verify_response(text)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Verify failed: {exc}")
            return "needs_review", f"verify_api_error: {exc}"


async def regenerate_one(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    ground_truth: Any,
    predicted: Any,
    raw_response: str,
    python_solution: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Regenerate a single analysis with python_solution in the prompt."""
    async with sem:
        prompt = REGEN_PROMPT_WITH_SOLUTION.format(
            question=question[:1500],
            context=(context or "")[:2500],
            ground_truth=ground_truth,
            python_solution=(python_solution or "# not available")[:2000],
            predicted_answer=predicted,
            raw_response=(raw_response or "")[:1500],
        )
        loop = asyncio.get_event_loop()

        def _call() -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 금융 계산 오류 분석 전문가입니다. 한글로 답변하세요."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            return resp.choices[0].message.content or ""

        try:
            text = await loop.run_in_executor(None, _call)
            return asdict(_parse_regen_response(text))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Regenerate failed: {exc}")
            return None


def _index_problems(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """qid → problem (with transformations + python_solution)."""
    return {p["question_id"]: p for p in batch.get("problems", [])}


def _pick_pot_record(
    pot_records: list[dict[str, Any]], qid: str, ttype: str | None = None
) -> dict[str, Any] | None:
    """Find a representative POT record for (qid, ttype).

    ttype=None means original-context (use any POT record for this qid — they share
    the original-context response). For ttype, match transformation_type.
    """
    for r in pot_records:
        if r.get("question_id") != qid:
            continue
        if ttype is None:
            return r
        if r.get("transformation_type") == ttype:
            return r
    return None


async def process_original(
    client: OpenAI,
    model: str,
    concurrency: int,
    original_path: Path,
    batch_path: Path,
    pot_records_path: Path,
) -> int:
    """Verify + (selectively) regenerate original-context analyses."""
    if not original_path.exists():
        logger.warning(f"original analysis file 없음: {original_path}")
        return 0

    original = json.loads(original_path.read_text(encoding="utf-8"))
    analyses = original.get("analyses", {})
    if not analyses:
        return 0

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    problems_idx = _index_problems(batch)
    pot_records = load_eval_records(pot_records_path)

    sem = asyncio.Semaphore(concurrency)
    logger.info(f"[original] verifying {len(analyses)} analyses")

    # Phase A: verify
    verify_tasks = []
    for qid, analysis in analyses.items():
        problem = problems_idx.get(qid)
        pot = _pick_pot_record(pot_records, qid)
        if not problem or not pot:
            continue
        verify_tasks.append((qid, analysis, problem, pot))

    verify_coros = [
        verify_one(
            client, model,
            t[2].get("question", ""),
            t[2].get("python_solution", ""),
            t[2].get("ground_truth", ""),
            t[3].get("predicted_answer", ""),
            t[1], sem,
        )
        for t in verify_tasks
    ]
    verify_results = await asyncio.gather(*verify_coros)

    for (qid, analysis, _, _), (status, note) in zip(verify_tasks, verify_results):
        analysis["verification_status"] = status
        analysis["verification_note"] = note

    from collections import Counter
    vcounter = Counter(a.get("verification_status") for a in analyses.values())
    logger.info(f"[original] verification: {dict(vcounter)}")

    # Phase B: regenerate flagged ones
    to_regen = [(qid, a) for qid, a in analyses.items()
                if a.get("verification_status") in ("needs_review", "contradicts")]
    logger.info(f"[original] regenerating {len(to_regen)} flagged analyses")

    regen_tasks = []
    for qid, analysis in to_regen:
        problem = problems_idx.get(qid)
        pot = _pick_pot_record(pot_records, qid)
        if not problem or not pot:
            continue
        regen_tasks.append((qid, analysis, problem, pot))

    regen_coros = [
        regenerate_one(
            client, model,
            t[2].get("question", ""),
            t[2].get("context_original", ""),
            t[2].get("ground_truth", ""),
            t[3].get("predicted_answer", ""),
            t[3].get("raw_response", ""),
            t[2].get("python_solution", ""),
            sem,
        )
        for t in regen_tasks
    ]
    regen_results = await asyncio.gather(*regen_coros)

    regen_ok = 0
    for (qid, analysis, _, _), new_analysis in zip(regen_tasks, regen_results):
        if new_analysis:
            # Preserve verification metadata, update analysis fields
            new_analysis["verification_status"] = "regenerated"
            new_analysis["verification_note"] = (
                "재생성됨 — python_solution 참조 기반 analysis"
            )
            new_analysis["is_refusal"] = analysis.get("is_refusal", False)
            analyses[qid] = new_analysis
            regen_ok += 1
    logger.info(f"[original] regenerated: {regen_ok}/{len(to_regen)}")

    original.setdefault("metadata", {})["verified"] = True
    original["analyses"] = analyses
    original_path.write_text(
        json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return regen_ok


async def process_transformed(
    client: OpenAI,
    model: str,
    concurrency: int,
    transformed_path: Path,
    batch_path: Path,
    standard_recs_path: Path,
    meta_recs_path: Path,
    cot_recs_path: Path,
) -> int:
    """Verify + regenerate transformed-context analyses (3 strategies)."""
    if not transformed_path.exists():
        logger.warning(f"transformed analysis file 없음: {transformed_path}")
        return 0

    data = json.loads(transformed_path.read_text(encoding="utf-8"))
    analyses = data.get("analyses", {})
    if not analyses:
        return 0

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    problems_idx = _index_problems(batch)

    strategy_recs: dict[str, list[dict[str, Any]]] = {
        "standard": load_eval_records(standard_recs_path) if standard_recs_path.exists() else [],
        "metacognitive": load_eval_records(meta_recs_path) if meta_recs_path.exists() else [],
        "cot_trace": load_eval_records(cot_recs_path) if cot_recs_path.exists() else [],
    }

    def _find_record(strategy: str, qid: str, ttype: str) -> dict[str, Any] | None:
        for r in strategy_recs.get(strategy, []):
            if r.get("question_id") == qid and r.get("transformation_type") == ttype:
                return r
        return None

    sem = asyncio.Semaphore(concurrency)

    # Phase A: verify all 361
    verify_cells: list[tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for qid, per_t in analyses.items():
        for ttype, per_s in per_t.items():
            problem = problems_idx.get(qid)
            if not problem:
                continue
            tx = problem.get("transformations", {}).get(ttype)
            if not tx:
                continue
            for strategy, analysis in per_s.items():
                rec = _find_record(strategy, qid, ttype)
                if rec is None:
                    continue
                verify_cells.append((qid, ttype, strategy, analysis, problem, rec))

    logger.info(f"[transformed] verifying {len(verify_cells)} cells (no streaming)")

    statuses = await asyncio.gather(*[
        verify_one(
            client, model,
            t[4].get("question", ""),
            t[4].get("python_solution", ""),
            t[4].get("ground_truth", ""),
            t[5].get("predicted_answer", ""),
            t[3], sem,
        )
        for t in verify_cells
    ])

    for (qid, ttype, strategy, analysis, _, _), (status, note) in zip(verify_cells, statuses):
        analysis["verification_status"] = status
        analysis["verification_note"] = note

    from collections import Counter
    vcounter = Counter(
        a.get("verification_status")
        for per_t in analyses.values()
        for per_s in per_t.values()
        for a in per_s.values()
    )
    logger.info(f"[transformed] verification: {dict(vcounter)}")

    # Phase B: regenerate flagged
    to_regen: list[tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for qid, per_t in analyses.items():
        for ttype, per_s in per_t.items():
            for strategy, analysis in per_s.items():
                if analysis.get("verification_status") in ("needs_review", "contradicts"):
                    problem = problems_idx.get(qid)
                    rec = _find_record(strategy, qid, ttype)
                    if problem and rec:
                        to_regen.append((qid, ttype, strategy, analysis, problem, rec))

    logger.info(f"[transformed] regenerating {len(to_regen)} flagged cells (no streaming)")

    regen_results_ordered = await asyncio.gather(*[
        regenerate_one(
            client, model,
            t[4].get("question", ""),
            t[4].get("transformations", {}).get(t[1], {}).get("context_transformed", t[4].get("context_original", "")),
            t[4].get("ground_truth", ""),
            t[5].get("predicted_answer", ""),
            t[5].get("raw_response", ""),
            t[4].get("python_solution", ""),
            sem,
        )
        for t in to_regen
    ])

    regen_ok = 0
    for (qid, ttype, strategy, analysis, _, _), new_analysis in zip(to_regen, regen_results_ordered):
        if new_analysis:
            new_analysis["verification_status"] = "regenerated"
            new_analysis["verification_note"] = "재생성됨 — python_solution 기반"
            new_analysis["is_refusal"] = analysis.get("is_refusal", False)
            analyses[qid][ttype][strategy] = new_analysis
            regen_ok += 1

    logger.info(f"[transformed] regenerated: {regen_ok}/{len(to_regen)}")

    data.setdefault("metadata", {})["verified"] = True
    data["analyses"] = analyses
    transformed_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return regen_ok


async def main_async(args: argparse.Namespace) -> int:
    """Run both original and transformed verification + regeneration."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY 없음")
        return 1
    client = OpenAI(api_key=api_key)

    # Force-use shared analyzer for model consistency (not strictly required here)
    _ = LLMErrorAnalyzer(model=args.model)

    regen_orig = await process_original(
        client, args.model, args.concurrency,
        args.original, args.batch, args.pot_original,
    )
    regen_trans = await process_transformed(
        client, args.model, args.concurrency,
        args.transformed, args.batch,
        args.standard, args.metacognitive, args.cot_trace,
    )
    logger.info(f"\n=== 최종: 원본 재생성 {regen_orig} / 변환 재생성 {regen_trans}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI."""
    parser = argparse.ArgumentParser(description="해설 검증 + 재생성 (옵션 B)")
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
