"""
IC-L2/L3 한글 포함 변환 재생성 스크립트

기존 batch_transformations_0_238.json에서 IC-L2/L3 변환 중
한글이 포함된 항목만 LLM으로 재변환하여 교체.

Usage:
    python experiments/retransform_korean_fix.py
    python experiments/retransform_korean_fix.py --dry-run  # 대상만 확인
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from llm_transform import LLMTransformer, _build_transform_prompt, SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def find_korean_transformations(data: dict, content_only: bool = True) -> list:
    """Find IC-L2/L3 transformations with Korean text.

    Args:
        content_only: If True, only find items where context/question has Korean
                      (description-only Korean is handled separately).
    """
    targets = []
    for p in data["problems"]:
        qid = p["question_id"]
        for ttype in ["IC-L2", "IC-L3"]:
            t = p.get("transformations", {}).get(ttype, {})
            if not t.get("success"):
                continue
            ctx = str(t.get("context_transformed", ""))
            qt = str(t.get("question_transformed", ""))
            desc = str(t.get("description", ""))

            if content_only:
                if has_korean(ctx) or has_korean(qt):
                    targets.append((qid, ttype, p))
            else:
                if has_korean(ctx) or has_korean(qt) or has_korean(desc):
                    targets.append((qid, ttype, p))
    return targets


def fix_description_korean(data: dict, model_unused=None) -> int:
    """Post-process: translate Korean descriptions to English (simple regex)."""
    fixed = 0
    for p in data["problems"]:
        for ttype in ["IC-L2", "IC-L3"]:
            t = p.get("transformations", {}).get(ttype, {})
            if not t.get("success"):
                continue
            desc = t.get("description", "")
            if not has_korean(desc):
                continue
            # Don't touch context/question, only description
            ctx = str(t.get("context_transformed", ""))
            qt = str(t.get("question_transformed", ""))
            if has_korean(ctx) or has_korean(qt):
                continue  # Will be handled by LLM retransform
            # Mark description as needing translation
            t["description"] = "[EN] " + desc
            fixed += 1
    return fixed


async def retransform(targets, model="gemini-2.5-flash"):
    """Re-run LLM transformation for Korean-contaminated items."""
    transformer = LLMTransformer(model=model, concurrency=3)
    results = {}

    try:
        for i, (qid, ttype, problem) in enumerate(targets):
            logger.info(f"[{i + 1}/{len(targets)}] {qid}/{ttype} 재변환 중...")

            prompt = _build_transform_prompt(problem, ttype)
            response = await transformer._call_llm(SYSTEM_PROMPT, prompt)

            raw = response.get("text", "")
            cost = response.get("cost", 0.0)

            # Parse JSON response
            parsed = _parse_response(raw, ttype, problem)
            if parsed:
                # Verify no Korean in result
                ctx_new = str(
                    parsed.get(
                        "context_transformed", parsed.get("question_transformed", "")
                    )
                )
                desc_new = str(parsed.get("description", ""))
                if has_korean(ctx_new) or has_korean(desc_new):
                    logger.warning("  재변환 결과에도 한글 포함! 재시도...")
                    # Retry once with explicit instruction
                    prompt2 = (
                        prompt
                        + "\n\nCRITICAL: Output MUST be entirely in English. No Korean characters allowed."
                    )
                    response2 = await transformer._call_llm(SYSTEM_PROMPT, prompt2)
                    raw2 = response2.get("text", "")
                    parsed2 = _parse_response(raw2, ttype, problem)
                    if parsed2 and not has_korean(str(parsed2)):
                        parsed = parsed2
                        logger.info("  재시도 성공 (한글 제거)")
                    else:
                        logger.warning("  재시도에도 한글 포함 — 원본 유지")
                        continue

                results[(qid, ttype)] = parsed
                logger.info(f"  OK (cost: ${cost:.4f})")
            else:
                logger.warning("  파싱 실패 — 원본 유지")

    finally:
        await transformer.close()

    logger.info(f"재변환 완료: {len(results)}/{len(targets)}개 성공")
    return results


def _parse_response(raw: str, ttype: str, problem: dict) -> dict:
    """Parse LLM JSON response."""
    # Try extracting JSON
    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        text = raw

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try finding { ... }
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : idx + 1])
                        break
                    except json.JSONDecodeError:
                        return None
        else:
            return None

    if not parsed.get("is_transformable", True):
        return None

    return parsed


def apply_results(data: dict, results: dict) -> int:
    """Apply retransformed results back to batch data."""
    applied = 0
    for p in data["problems"]:
        qid = p["question_id"]
        for ttype in ["IC-L2", "IC-L3"]:
            key = (qid, ttype)
            if key not in results:
                continue

            parsed = results[key]
            t = p["transformations"][ttype]

            # Determine if question or context transform
            has_context = bool(
                p.get("context_original", "").strip()
                and p.get("context_original", "").strip() not in ("[]", "")
            )

            if has_context:
                new_ctx = parsed.get("transformed_context", "")
                if new_ctx:
                    t["context_transformed"] = new_ctx
            else:
                new_q = parsed.get("transformed_question", "")
                if new_q:
                    t["question_transformed"] = new_q

            if parsed.get("description"):
                t["description"] = parsed["description"]
            if parsed.get("critical_data"):
                t["critical_data"] = parsed["critical_data"]
            if parsed.get("removed_or_modified"):
                t["removed_or_modified"] = parsed["removed_or_modified"]
            if parsed.get("derivable_check"):
                t["derivable_check"] = parsed["derivable_check"]

            applied += 1

    return applied


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="대상만 확인")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--input",
        default="experiments/results/metacognitive/batch_transformations_0_238.json",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = find_korean_transformations(data)
    logger.info(f"한글 포함 변환: {len(targets)}개")

    by_type = {}
    for qid, ttype, _ in targets:
        by_type[ttype] = by_type.get(ttype, 0) + 1
    for ttype, count in by_type.items():
        logger.info(f"  {ttype}: {count}개")

    if args.dry_run:
        for qid, ttype, _ in targets:
            logger.info(f"  {qid}/{ttype}")
        return

    # Backup
    backup_path = input_path.with_suffix(".json.bak_korean")
    if not backup_path.exists():
        import shutil

        shutil.copy2(input_path, backup_path)
        logger.info(f"백업: {backup_path}")

    results = await retransform(targets, model=args.model)
    applied = apply_results(data, results)

    # Save
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"저장 완료: {applied}개 교체 → {input_path}")

    # Verify
    remaining = find_korean_transformations(data)
    logger.info(f"잔여 한글: {len(remaining)}개")


if __name__ == "__main__":
    asyncio.run(main())
