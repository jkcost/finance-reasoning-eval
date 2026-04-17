"""
Local Model Evaluation — Ollama 기반 로컬 LLM 실험

API 비용 없이 로컬 모델로 Phase A/B 실험을 실행합니다.

Requirements:
    - Ollama 설치 및 실행: https://ollama.ai
    - 모델 다운로드: ollama pull llama3.1:8b

Usage:
    # 기본 실행 (llama3.1:8b, hard 5문제)
    python experiments/run_local_model_eval.py

    # 모델/문제 수 지정
    python experiments/run_local_model_eval.py --model ollama-qwen2.5-7b --n 10

    # 커스텀 Ollama 엔드포인트
    OLLAMA_BASE_URL=http://remote-host:11434 python experiments/run_local_model_eval.py

    # 변환된 문제 평가 (Phase B)
    python experiments/run_local_model_eval.py \\
        --input experiments/results/metacognitive/batch_transformations_0_238.json \\
        --phase B
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
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_MODELS, LOCAL_MODELS, ModelConfig
from model_runner import OllamaProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# LOCAL MODEL REGISTRY
# ============================================================================

LOCAL_MODEL_REGISTRY = {m.id: m for m in LOCAL_MODELS}

DEFAULT_LOCAL_MODEL = "ollama-llama3.1-8b"


# ============================================================================
# PROMPT
# ============================================================================

POT_SYSTEM = """You are a financial expert solving quantitative finance problems.
Write a Python function `solution()` that returns the final numeric answer.
Use only the data provided in the problem context.
If required data is missing or contradictory, output exactly: INSUFFICIENT_INFORMATION"""

POT_PREFIX = "```python\ndef solution():\n"


def build_prompt(question: str, context: str, strategy: str = "standard") -> str:
    """Build prompt for local model evaluation."""
    if strategy == "metacognitive":
        system = (
            POT_SYSTEM
            + "\nIMPORTANT: If any data needed to solve this problem is missing or "
            "contradictory, you MUST output exactly: INSUFFICIENT_INFORMATION"
        )
    else:
        system = POT_SYSTEM

    if context and context != "[]":
        user = (
            f"The following question context is provided for your reference.\n"
            f"{context}\n\nQuestion: {question}\n"
        )
    else:
        user = f"Question: {question}\n"

    return f"{system}\n\n{user}\n{POT_PREFIX}"


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================


def extract_answer(response: str) -> Tuple[Any, Optional[str]]:
    """Extract numeric answer or INSUFFICIENT_INFORMATION from POT response."""
    if re.search(r"(?i)INSUFFICIENT[_ ]INFORMATION", response):
        return "INSUFFICIENT_INFORMATION", None

    code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if not code_match:
        code_match = re.search(r"(def solution\(\):.*?)(?:\n\n|\Z)", response, re.DOTALL)

    if code_match:
        code = f"def solution():\n{code_match.group(1)}" if not code_match.group(0).startswith("def") else code_match.group(1)
        try:
            namespace: Dict = {}
            exec(code, namespace)  # noqa: S102
            result = namespace.get("solution", lambda: None)()
            if result is not None:
                return result, None
        except Exception as e:
            return None, str(e)

    # Fallback: extract last number from text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response)
    if numbers:
        try:
            return float(numbers[-1].replace(",", "")), None
        except ValueError:
            pass

    return None, "No answer found"


def is_correct(predicted: Any, ground_truth: Any, tolerance: float = 0.002) -> bool:
    """Check if predicted answer matches ground truth within tolerance."""
    if predicted is None:
        return False
    if str(predicted) == str(ground_truth):
        return True
    try:
        p, g = float(predicted), float(ground_truth)
        if g == 0:
            return abs(p - g) < 1e-9
        return abs(p - g) / abs(g) <= tolerance
    except (ValueError, TypeError):
        return False


# ============================================================================
# PHASE A — BASELINE
# ============================================================================


async def run_phase_a(
    provider: OllamaProvider,
    model_id: str,
    dataset_path: Path,
    n: int,
    difficulty: str = "hard",
) -> List[Dict]:
    """Phase A: baseline accuracy on original problems."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    subset = problems[:n]
    logger.info(f"Phase A: {len(subset)}문제 × {model_id}")

    results = []
    for i, prob in enumerate(subset):
        qid = prob.get("id", str(i))
        question = prob.get("question", "")
        context = prob.get("context", "")
        gt = prob.get("answer", prob.get("ground_truth", ""))

        prompt = build_prompt(question, context, "standard")

        from dataset_loader import Example

        example = Example(
            id=qid,
            question=question,
            context=context,
            ground_truth=gt,
        )

        logger.info(f"  [{i + 1}/{len(subset)}] {qid}")
        resp = await provider.call_model(example, prompt)
        raw = resp.raw_response

        predicted, exec_err = extract_answer(raw)
        correct = is_correct(predicted, gt)

        results.append(
            {
                "question_id": qid,
                "model": model_id,
                "phase": "A",
                "ground_truth": gt,
                "predicted": str(predicted),
                "is_correct": correct,
                "execution_error": exec_err,
                "raw_response": raw[:500],
            }
        )
        status = "O" if correct else "X"
        logger.info(f"    {status} predicted={predicted}, gt={gt}")

    return results


# ============================================================================
# PHASE B — METACOGNITIVE (변환된 문제)
# ============================================================================


async def run_phase_b(
    provider: OllamaProvider,
    model_id: str,
    batch_path: Path,
    n: int,
    strategy: str = "metacognitive",
) -> List[Dict]:
    """Phase B: metacognitive evaluation on transformed problems."""
    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)

    problems = batch.get("problems", [])[:n]
    logger.info(f"Phase B: {len(problems)}문제 × {model_id} ({strategy})")

    TRANSFORM_TYPES = ["EA-partial", "EA-full", "SA", "IC-L1", "IC-L2", "IC-L3", "IC-L4"]
    UNSOLVABLE_TYPES = {"EA-partial", "EA-full", "SA"}

    results = []
    from dataset_loader import Example

    for prob in problems:
        qid = prob["question_id"]
        question = prob["question"]
        gt = prob["ground_truth"]
        transforms = prob.get("transformations", {})

        for ttype in TRANSFORM_TYPES:
            tdata = transforms.get(ttype, {})
            if not tdata.get("success"):
                continue

            ctx = tdata.get("context_transformed", prob.get("context_original", ""))
            q = tdata.get("question_transformed", question)
            should_refuse = ttype in UNSOLVABLE_TYPES

            prompt = build_prompt(q, ctx, strategy)
            example = Example(id=f"{qid}/{ttype}", question=q, context=ctx, ground_truth=gt)

            logger.info(f"  {qid}/{ttype}")
            resp = await provider.call_model(example, prompt)
            raw = resp.raw_response

            predicted, exec_err = extract_answer(raw)
            refused = predicted == "INSUFFICIENT_INFORMATION"
            correct_refusal = refused and should_refuse
            false_confidence = not refused and should_refuse

            results.append(
                {
                    "question_id": qid,
                    "transformation_type": ttype,
                    "model": model_id,
                    "phase": "B",
                    "strategy": strategy,
                    "should_refuse": should_refuse,
                    "refused": refused,
                    "correct_refusal": correct_refusal,
                    "false_confidence": false_confidence,
                    "predicted": str(predicted),
                    "ground_truth": gt,
                    "execution_error": exec_err,
                    "raw_response": raw[:500],
                }
            )

    return results


# ============================================================================
# METRICS SUMMARY
# ============================================================================


def summarize_results(results: List[Dict], phase: str) -> Dict:
    """Compute summary metrics from results."""
    if phase == "A":
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        return {"total": total, "correct": correct, "accuracy": correct / total if total else 0}

    # Phase B
    should_refuse = [r for r in results if r["should_refuse"]]
    correctly_refused = [r for r in should_refuse if r["correct_refusal"]]
    false_conf = [r for r in should_refuse if r["false_confidence"]]

    rr = len(correctly_refused) / len(should_refuse) if should_refuse else 0
    return {
        "total": len(results),
        "should_refuse": len(should_refuse),
        "correctly_refused": len(correctly_refused),
        "false_confidence": len(false_conf),
        "refusal_recall": rr,
    }


# ============================================================================
# MAIN
# ============================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(description="Local model (Ollama) evaluation")
    parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_MODEL,
        choices=list(LOCAL_MODEL_REGISTRY.keys()),
        help="Local model to use",
    )
    parser.add_argument(
        "--phase",
        default="A",
        choices=["A", "B"],
        help="Evaluation phase (A=baseline, B=metacognitive)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of problems to evaluate",
    )
    parser.add_argument(
        "--difficulty",
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Dataset difficulty (Phase A only)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Batch transformations JSON path (Phase B only)",
    )
    parser.add_argument(
        "--strategy",
        default="metacognitive",
        choices=["standard", "metacognitive"],
        help="Prompt strategy (Phase B only)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/metacognitive",
        help="Output directory",
    )
    args = parser.parse_args()

    model_config = LOCAL_MODEL_REGISTRY[args.model]
    logger.info(f"모델: {model_config.name}")
    logger.info(f"Ollama URL: {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")

    # Check Ollama is running
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.status_code != 200:
                raise RuntimeError(f"Ollama 응답 오류: {r.status_code}")
        logger.info("Ollama 연결 확인")
    except Exception as e:
        logger.error(f"Ollama에 연결할 수 없습니다: {e}")
        logger.error("Ollama 실행 후 재시도: https://ollama.ai")
        sys.exit(1)

    provider = OllamaProvider(model_config, concurrency_limit=1)

    if args.phase == "A":
        data_path = (
            PROJECT_ROOT
            / "data/financereasoning/raw/FinanceReasoning"
            / f"{args.difficulty}.json"
        )
        if not data_path.exists():
            logger.error(f"데이터 파일 없음: {data_path}")
            sys.exit(1)
        results = await run_phase_a(provider, args.model, data_path, args.n, args.difficulty)
    else:
        if args.input:
            batch_path = PROJECT_ROOT / args.input
        else:
            candidates = sorted(
                (PROJECT_ROOT / "experiments/results/metacognitive").glob(
                    "batch_transformations_*.json"
                ),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if not candidates:
                logger.error("batch_transformations JSON 파일을 찾을 수 없습니다.")
                sys.exit(1)
            batch_path = candidates[0]

        logger.info(f"배치 파일: {batch_path.name}")
        results = await run_phase_b(provider, args.model, batch_path, args.n, args.strategy)

    summary = summarize_results(results, args.phase)
    logger.info(f"\n{'='*50}")
    logger.info(f"결과 요약 ({args.model}, Phase {args.phase})")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.3f}")
        else:
            logger.info(f"  {k}: {v}")

    # Save results
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"local_eval_phase{args.phase}_{safe_model}_{ts}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "model": args.model,
                "phase": args.phase,
                "n": args.n,
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"저장 완료: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
