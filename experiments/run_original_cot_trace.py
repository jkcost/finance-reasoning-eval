"""Run ic_cot_trace prompt on ORIGINAL context (not transformed).

B1 실험의 확장 — 원본 context에도 CoT trace 프롬프트 적용하여 오답 원인 분석.
run_batch_evaluation.py가 변환된 문제 전용이므로 원본은 이 스크립트로 처리.

Usage:
    python experiments/run_original_cot_trace.py \\
        --input experiments/results/metacognitive/batch_transformations_0_20_subset.json \\
        --model gpt-4o-mini \\
        --output experiments/results/metacognitive/eval_original_ic_cot_trace_0_20.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ModelConfig  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from model_runner import OpenAIProvider  # noqa: E402
from refusal_detector import RefusalDetector  # noqa: E402
from run_metacognitive_experiment import PROMPT_SYSTEMS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _extract_answer(raw: str) -> tuple[Any, str | None, str | None]:
    """Extract final answer from a CoT trace response (very permissive)."""
    # Try "Therefore, the answer is X"
    m = re.search(r"[Tt]herefore,\s*the\s+answer\s+is\s+([^\n.]+)", raw)
    if m:
        return m.group(1).strip(), None, None
    # Try INSUFFICIENT_INFORMATION
    if "INSUFFICIENT_INFORMATION" in raw:
        return "INSUFFICIENT_INFORMATION", None, None
    # Fall back to last numeric token in the last 5 lines
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    for line in reversed(lines[-5:]):
        tokens = re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", ""))
        if tokens:
            return tokens[-1], None, None
    return None, None, None


def _check_answer(predicted: Any, ground_truth: Any) -> bool:
    """Loose numeric comparison with 0.2% tolerance."""
    if predicted is None or ground_truth is None:
        return False
    try:
        p = float(str(predicted).replace(",", "").replace("$", "").strip())
        g = float(str(ground_truth).replace(",", "").replace("$", "").strip())
        if g == 0:
            return abs(p) < 1e-6
        return abs(p - g) / abs(g) < 0.002
    except (ValueError, TypeError):
        return str(predicted).strip() == str(ground_truth).strip()


class _MinExample:
    """Minimal stand-in for evaluation.Example used by provider.call_model."""

    def __init__(self, qid: str, question: str, context: str, ground_truth: Any):
        self.id = qid
        self.question = question
        self.context = context
        self.ground_truth = ground_truth
        self.python_solution = ""


async def evaluate_one(
    provider: OpenAIProvider,
    system_prompt: str,
    problem: dict[str, Any],
    sem: asyncio.Semaphore,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a single problem with original context + ic_cot_trace prompt."""
    async with sem:
        qid = problem.get("question_id", "?")
        question = problem.get("question", "")
        context = problem.get("context_original", "")
        ground_truth = problem.get("ground_truth")

        user_prompt = f"{question}\n\nContext:\n{context}"
        full = f"{system_prompt}\n\n{user_prompt}"
        example = _MinExample(qid, question, context, ground_truth)

        try:
            resp = await provider.call_model(example, full)
            raw = resp.raw_response
            predicted, _, _ = _extract_answer(raw)
            is_correct = _check_answer(predicted, ground_truth)

            detector = RefusalDetector()
            detection = detector.detect(raw, context=context)

            # Parse inline INSUFFICIENT_INFORMATION: <reason> — compatible with
            # both the current branch's DetectionResult (no reason field) and
            # the enriched version on PR #3 (has detection.reason).
            inline_reason = None
            if predicted and str(predicted).startswith("INSUFFICIENT_INFORMATION"):
                colon = str(predicted).find(":")
                if colon >= 0:
                    inline_reason = str(predicted)[colon + 1:].strip() or None
            if inline_reason is None and "INSUFFICIENT_INFORMATION:" in raw:
                m = re.search(r"INSUFFICIENT_INFORMATION\s*:\s*(.+?)(?:\n\n|\Z)", raw, re.DOTALL)
                if m:
                    inline_reason = m.group(1).strip().strip("[]\"' ") or None
            detector_reason = getattr(detection, "reason", None)

            return {
                "question_id": qid,
                "transformation_type": "original",
                "eval_type": "original_cot_trace",
                "model": model_name,
                "prompt_strategy": "ic_cot_trace",
                "response_type": detection.response_type.value,
                "predicted_answer": str(predicted) if predicted is not None else None,
                "ground_truth": str(ground_truth),
                "is_correct": is_correct,
                "raw_response": raw,
                "refusal_reason": inline_reason or detector_reason,
                "cost_usd": getattr(resp, "cost_usd", 0.0),
                "tokens_prompt": getattr(resp, "prompt_tokens", 0),
                "tokens_completion": getattr(resp, "completion_tokens", 0),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{qid}] Failed: {exc}")
            return {
                "question_id": qid,
                "transformation_type": "original",
                "eval_type": "original_cot_trace",
                "model": model_name,
                "prompt_strategy": "ic_cot_trace",
                "response_type": "error",
                "raw_response": f"ERROR: {exc}",
                "is_correct": False,
                "ground_truth": str(ground_truth),
            }


async def main() -> int:
    """Run B1 CoT trace on original context for a problem subset."""
    parser = argparse.ArgumentParser(description="Run ic_cot_trace on original contexts")
    parser.add_argument("--input", type=Path, required=True, help="batch_transformations subset JSON")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    load_dotenv()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    problems = data.get("problems", [])
    logger.info(f"Problems: {len(problems)}")

    system_prompt = PROMPT_SYSTEMS["ic_cot_trace"]["POT"]
    logger.info(f"Prompt: ic_cot_trace/POT (length={len(system_prompt)} chars)")

    config = ModelConfig(
        id=args.model,
        name=args.model,
        provider="openai",
        model_id=args.model,
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=4096,
        temperature=0.0,
        cost_per_million_tokens=0.15,
    )
    provider = OpenAIProvider(config)

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [evaluate_one(provider, system_prompt, p, sem, args.model) for p in problems]
    results = await asyncio.gather(*tasks)

    ok = sum(1 for r in results if r.get("is_correct"))
    refused = sum(1 for r in results if r.get("response_type") == "refused")
    errors = sum(1 for r in results if r.get("response_type") == "error")
    logger.info(f"Results: {len(results)} total | correct={ok} refused={refused} errors={errors}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"results": results, "metadata": {
            "input": str(args.input),
            "model": args.model,
            "prompt_strategy": "ic_cot_trace",
            "n": len(results),
        }}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
