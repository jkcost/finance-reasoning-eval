"""
test-2009 EA-partial memorization probe.

목적: [DATA MISSING]으로 처리된 borrowed_amount_jpy를 모델이 어떤 값으로 채우는지
     반복 실행하여 분포를 관찰. 암기(항상 1,000,000) vs 임의 예시(분산된 round numbers).

Usage:
    python experiments/probe_test2009_memorization.py
    python experiments/probe_test2009_memorization.py --runs 5 --temperatures 0.0 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig  # noqa: E402
from error_analysis import MODEL_REGISTRY  # noqa: E402
from error_analysis.error_taxonomy import BUDGET_MODEL_SETS  # noqa: E402
from model_runner import (  # noqa: E402
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
from refusal_detector import RefusalDetector  # noqa: E402

from run_batch_evaluation import (  # noqa: E402
    METHOD,
    POT_PREFIX,
    build_prompt,
    calculate_cost,
    check_answer,
    extract_answer,
)
from run_metacognitive_experiment import PROMPT_SYSTEMS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import os  # noqa: E402

QID = "test-2009"
TTYPE = "EA-partial"
GT = "2250"


def load_problem() -> Dict[str, Any]:
    """Load test-2009 EA-partial from batch_transformations."""
    path = (
        Path(__file__).parent.parent
        / "experiments/results/metacognitive/batch_transformations_0_238.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for p in data["problems"]:
        if p["question_id"] == QID:
            t = p["transformations"][TTYPE]
            return {
                "question_id": QID,
                "question": t.get("question_transformed") or p["question"],
                "context": t["context_transformed"],
                "ground_truth": p.get("ground_truth", GT),
            }
    raise RuntimeError(f"{QID} not found")


def build_provider(model_name: str, temperature: float):
    """Initialize a provider with a specified temperature."""
    model_info = MODEL_REGISTRY[model_name]
    api_key = os.environ.get(model_info.api_key_env)
    if not api_key:
        return None
    config = ModelConfig(
        id=model_name,
        name=model_info.display_name,
        provider=model_info.provider,
        model_id=model_info.model_id,
        api_key_env_var=model_info.api_key_env,
        max_tokens=4096,
        temperature=temperature,
        cost_per_million_tokens=model_info.cost_per_million_output,
    )
    if model_info.provider == "openai":
        return OpenAIProvider(config)
    if model_info.provider == "anthropic":
        return AnthropicProvider(config)
    if model_info.provider == "google":
        return GoogleProvider(config)
    return None


def extract_borrowed_amount(code: Optional[str], raw: str) -> Optional[float]:
    """Extract the borrowed amount the model assumed from its code/response."""
    text = code or raw
    if not text:
        return None
    patterns = [
        r"borrowed_amount[_a-z]*\s*=\s*([\d,_.]+)",
        r"borrowed[_a-z]*\s*=\s*([\d,_.]+)",
        r"yen[_a-z]*\s*=\s*([\d,_.]+)",
        r"principal[_a-z]*\s*=\s*([\d,_.]+)",
        r"amount[_a-z]*\s*=\s*([\d,_.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "").replace("_", ""))
            except ValueError:
                continue
    return None


async def one_run(
    provider,
    model_name: str,
    temperature: float,
    run_idx: int,
    problem: Dict[str, Any],
    strategy: str,
    detector: RefusalDetector,
) -> Dict[str, Any]:
    class MinimalExample:
        def __init__(self, qid: str, q: str, c: str, gt: Any):
            self.id = qid
            self.question = q
            self.context = c
            self.ground_truth = gt
            self.python_solution = ""

    mini = MinimalExample(
        problem["question_id"],
        problem["question"],
        problem["context"],
        problem["ground_truth"],
    )
    full_prompt = build_prompt(problem["question"], problem["context"], strategy)

    try:
        resp = await provider.call_model(mini, full_prompt)
        raw = resp.raw_response
        cost = calculate_cost(resp.prompt_tokens, resp.completion_tokens, model_name)
        answer, code, exec_err = extract_answer(raw)
        detection = detector.detect(
            raw, context=problem["context"], execution_error=exec_err, executed_code=code
        )
        if answer == "INSUFFICIENT_INFORMATION":
            rtype = "refused"
        else:
            rtype = detection.response_type.value
        is_correct = check_answer(answer, problem["ground_truth"])
        borrowed = extract_borrowed_amount(code, raw)

        return {
            "model": model_name,
            "temperature": temperature,
            "run": run_idx,
            "strategy": strategy,
            "response_type": rtype,
            "predicted_answer": str(answer) if answer is not None else None,
            "is_correct": is_correct,
            "borrowed_amount_assumed": borrowed,
            "cost_usd": cost,
            "executed_code": code,
            "raw_response": raw,
        }
    except Exception as e:
        logger.error(f"  [ERROR] {model_name} t={temperature} r={run_idx}: {e}")
        return {
            "model": model_name,
            "temperature": temperature,
            "run": run_idx,
            "strategy": strategy,
            "response_type": "error",
            "predicted_answer": None,
            "is_correct": False,
            "borrowed_amount_assumed": None,
            "cost_usd": 0.0,
            "executed_code": None,
            "raw_response": f"ERROR: {str(e)[:200]}",
        }


async def run_probe(
    models: List[str],
    temperatures: List[float],
    runs: int,
    strategy: str,
) -> List[Dict[str, Any]]:
    load_dotenv()
    problem = load_problem()
    logger.info(f"Problem loaded: {QID}/{TTYPE}, GT={problem['ground_truth']}")
    logger.info(f"Question: {problem['question']}")
    logger.info(f"Context snippet: ...{problem['context'][200:350]}...")

    detector = RefusalDetector()
    results: List[Dict[str, Any]] = []

    for model in models:
        for temp in temperatures:
            provider = build_provider(model, temp)
            if provider is None:
                logger.warning(f"Skip {model} (no API key)")
                continue
            logger.info(f"=== {model} @ temp={temp} x {runs} runs ===")
            tasks = [
                one_run(provider, model, temp, i + 1, problem, strategy, detector)
                for i in range(runs)
            ]
            batch = await asyncio.gather(*tasks)
            for r in batch:
                borrowed = r["borrowed_amount_assumed"]
                tag = "REFUSED" if r["response_type"] == "refused" else (
                    "CORRECT" if r["is_correct"] else "WRONG"
                )
                logger.info(
                    f"  run{r['run']}: {tag} "
                    f"pred={r['predicted_answer']} borrowed={borrowed}"
                )
            results.extend(batch)

    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate by (model, temperature)."""
    from collections import Counter, defaultdict

    agg: Dict[tuple, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "refused": 0, "correct": 0, "wrong": 0, "borrowed": Counter()}
    )
    for r in results:
        k = (r["model"], r["temperature"])
        agg[k]["n"] += 1
        if r["response_type"] == "refused":
            agg[k]["refused"] += 1
        elif r["is_correct"]:
            agg[k]["correct"] += 1
        else:
            agg[k]["wrong"] += 1
        if r["borrowed_amount_assumed"] is not None:
            agg[k]["borrowed"][r["borrowed_amount_assumed"]] += 1

    table = []
    for (model, temp), s in sorted(agg.items()):
        borrowed_dist = dict(s["borrowed"].most_common())
        table.append(
            {
                "model": model,
                "temperature": temp,
                "n": s["n"],
                "refused": s["refused"],
                "correct": s["correct"],
                "wrong": s["wrong"],
                "borrowed_distribution": borrowed_dist,
            }
        )
    return {"by_model_temp": table, "total_runs": len(results)}


def main():
    parser = argparse.ArgumentParser(description="Probe test-2009 EA-partial")
    parser.add_argument(
        "--models",
        nargs="+",
        default=BUDGET_MODEL_SETS["economic"],
        help="Models to probe",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.7],
        help="Sampling temperatures",
    )
    parser.add_argument("--runs", type=int, default=5, help="Runs per (model, temp)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="standard",
        choices=list(PROMPT_SYSTEMS.keys()),
    )
    args = parser.parse_args()

    logger.info(
        f"Models={args.models}, temps={args.temperatures}, runs={args.runs}, "
        f"strategy={args.strategy}"
    )

    results = asyncio.run(
        run_probe(args.models, args.temperatures, args.runs, args.strategy)
    )
    summary = summarize(results)

    out_dir = Path(__file__).parent.parent / "experiments/results/metacognitive"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"probe_test2009_EA-partial_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "qid": QID,
                    "ttype": TTYPE,
                    "ground_truth": GT,
                    "strategy": args.strategy,
                    "timestamp": ts,
                },
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — borrowed_amount_jpy 분포")
    logger.info("=" * 70)
    for row in summary["by_model_temp"]:
        logger.info(
            f"{row['model']:24s} t={row['temperature']:.1f}  "
            f"n={row['n']:2d}  refused={row['refused']:2d}  "
            f"correct={row['correct']:2d}  wrong={row['wrong']:2d}"
        )
        for val, cnt in row["borrowed_distribution"].items():
            pct = cnt / row["n"] * 100
            logger.info(f"     borrowed={val:>12,.0f}  : {cnt}/{row['n']} ({pct:.0f}%)")
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
