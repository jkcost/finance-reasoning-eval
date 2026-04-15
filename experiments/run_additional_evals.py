"""
추가 평가 실행: 원본 context 응답 + 거부 프롬프트(metacognitive) 응답

기존 batch_evaluation에는 "변환 context + standard prompt"만 있음.
이 스크립트는 두 가지를 추가로 생성:
  1. original: 원본 context + 원본 question → standard prompt
  2. metacognitive: 변환 context + 변환 question → metacognitive prompt

결과는 별도 JSON으로 저장하여 HTML 생성 시 3개 응답을 함께 표시.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from run_batch_evaluation import extract_answer, check_answer, classify_case
from run_metacognitive_experiment import (
    POT_PREFIX,
    POT_SYSTEM_STANDARD,
    POT_SYSTEM_METACOGNITIVE,
)
from evaluation.refusal_detector import RefusalDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"


def build_prompt(question: str, context: str, system_prompt: str) -> str:
    if context and context.strip() and context.strip() not in ("[]", ""):
        question_input = (
            f"The following question context is provided for your reference.\n"
            f"{context}\n\nQuestion: {question}\n"
        )
    else:
        question_input = f"Question: {question}\n"
    return f"{system_prompt}\n\n{question_input}\n{POT_PREFIX}"


async def call_openai(client: httpx.AsyncClient, prompt: str) -> str:
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2048,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def evaluate_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    qid: str,
    ttype: str,
    question: str,
    context: str,
    ground_truth: Any,
    system_prompt: str,
    eval_type: str,
    refusal_detector: RefusalDetector,
) -> Optional[Dict]:
    async with sem:
        try:
            prompt = build_prompt(question, context, system_prompt)
            raw_response = await call_openai(client, prompt)

            answer, executed_code, execution_error = extract_answer(raw_response)

            detection = refusal_detector.detect(
                raw_response,
                context=context,
                execution_error=execution_error,
                executed_code=executed_code,
            )

            if answer == "INSUFFICIENT_INFORMATION":
                response_type = "refused"
            else:
                response_type = detection.response_type.value

            is_correct = check_answer(answer, ground_truth)
            case_type = classify_case(response_type, is_correct, ttype)

            return {
                "question_id": qid,
                "transformation_type": ttype,
                "eval_type": eval_type,
                "model": MODEL,
                "prompt_strategy": eval_type,
                "case_type": case_type,
                "response_type": response_type,
                "predicted_answer": str(answer) if answer is not None else None,
                "ground_truth": str(ground_truth),
                "is_correct": is_correct,
                "raw_response": raw_response,
                "executed_code": executed_code,
                "execution_error": execution_error,
            }
        except Exception as e:
            logger.error(f"  ERROR {qid}/{ttype}/{eval_type}: {e}")
            return None


async def run_evaluations(batch_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    sem = asyncio.Semaphore(5)
    refusal_detector = RefusalDetector()
    original_results = []
    metacognitive_results = []

    problems = batch_data["problems"]
    logger.info(f"문제 수: {len(problems)}")

    async with httpx.AsyncClient() as client:
        # === Experiment 1: Original context + standard prompt ===
        logger.info("=== 실험 1: 원본 context 평가 (standard) ===")
        tasks = []
        for p in problems:
            qid = p["question_id"]
            gt = p.get("ground_truth")
            orig_q = p["question"]
            orig_ctx = p.get("context_original", "")

            for ttype in p.get("transformations", {}):
                t = p["transformations"][ttype]
                if not t.get("success") or ttype == "TA":
                    continue
                tasks.append(
                    evaluate_one(
                        client,
                        sem,
                        qid,
                        ttype,
                        orig_q,
                        orig_ctx,
                        gt,
                        POT_SYSTEM_STANDARD,
                        "original",
                        refusal_detector,
                    )
                )

        for i in range(0, len(tasks), 20):
            batch = tasks[i : i + 20]
            results = await asyncio.gather(*batch)
            for r in results:
                if r:
                    original_results.append(r)
            done = min(i + 20, len(tasks))
            logger.info(f"  원본: {done}/{len(tasks)} (성공: {len(original_results)})")
            await asyncio.sleep(1)

        logger.info(f"원본 평가 완료: {len(original_results)}건")

        # === Experiment 2: Transformed context + metacognitive prompt ===
        logger.info("=== 실험 2: 변환 context + metacognitive 프롬프트 ===")
        tasks = []
        for p in problems:
            qid = p["question_id"]
            gt = p.get("ground_truth")

            for ttype in p.get("transformations", {}):
                t = p["transformations"][ttype]
                if not t.get("success") or ttype == "TA":
                    continue

                ctx = t.get("context_transformed", p.get("context_original", ""))
                q = t.get("question_transformed", p["question"])

                tasks.append(
                    evaluate_one(
                        client,
                        sem,
                        qid,
                        ttype,
                        q,
                        ctx,
                        gt,
                        POT_SYSTEM_METACOGNITIVE,
                        "metacognitive",
                        refusal_detector,
                    )
                )

        for i in range(0, len(tasks), 20):
            batch = tasks[i : i + 20]
            results = await asyncio.gather(*batch)
            for r in results:
                if r:
                    metacognitive_results.append(r)
            done = min(i + 20, len(tasks))
            logger.info(
                f"  메타인지: {done}/{len(tasks)} (성공: {len(metacognitive_results)})"
            )
            await asyncio.sleep(1)

        logger.info(f"메타인지 평가 완료: {len(metacognitive_results)}건")

    return original_results, metacognitive_results


def main():
    input_path = (
        PROJECT_ROOT
        / "experiments/results/metacognitive/batch_transformations_0_238.json"
    )
    output_dir = PROJECT_ROOT / "experiments/results/metacognitive"

    with open(input_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)

    original_results, metacognitive_results = asyncio.run(run_evaluations(batch_data))

    # Save original results
    orig_path = output_dir / "eval_original_context.json"
    with open(orig_path, "w", encoding="utf-8") as f:
        json.dump(
            {"eval_type": "original", "model": MODEL, "results": original_results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"저장: {orig_path} ({len(original_results)}건)")

    # Save metacognitive results
    meta_path = output_dir / "eval_metacognitive.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_type": "metacognitive",
                "model": MODEL,
                "results": metacognitive_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"저장: {meta_path} ({len(metacognitive_results)}건)")

    # Summary stats
    for name, results in [
        ("원본", original_results),
        ("메타인지", metacognitive_results),
    ]:
        correct = sum(1 for r in results if r["is_correct"])
        refused = sum(1 for r in results if r["response_type"] == "refused")
        total = len(results)
        logger.info(
            f"{name}: 정답 {correct}/{total} ({correct / total * 100:.1f}%), "
            f"거부 {refused}/{total} ({refused / total * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
