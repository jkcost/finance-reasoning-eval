"""
Carry-trade principal prior probe.

목적: 모델이 [DATA MISSING]에 항상 1,000,000을 채우는 것이
      (A) test-2009 인스턴스 암기인지
      (B) 금융 carry-trade 장르의 일반적 prior인지 구분.

방법: 동일한 문제 구조에 통화/국가/금리를 바꾼 변형을 제시하고
      각 변형에서 모델이 채우는 값의 분포를 비교.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig  # noqa: E402
from error_analysis import MODEL_REGISTRY  # noqa: E402
from model_runner import (  # noqa: E402
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
from refusal_detector import RefusalDetector  # noqa: E402

from run_batch_evaluation import (  # noqa: E402
    build_prompt,
    calculate_cost,
    extract_answer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
# 각 variant는 EA-partial 스타일로 원금을 [DATA MISSING]으로 비운 carry trade
# 문제. 원본(V0)과 동일한 구조를 유지하되 통화/국가/금리를 변경.

Q = "What is the profit from this carry trade in Brazilian Reals? Answer to the nearest integer."
Q_ALT_CCY = "What is the profit from this carry trade in the target-country currency? Answer to the nearest integer."

VARIANTS: List[Dict[str, str]] = [
    {
        "id": "V0_original",
        "note": "원본 test-2009 EA-partial (Yen → Real, 0.5%/5%, rate 0.05)",
        "question": Q,
        "context": (
            "A global investor is exploring a currency carry trade strategy. "
            "The investor plans to borrow funds in Japan, where the interest rate "
            "is extremely low at 0.5% per annum, and invest these funds in Brazil, "
            "where the interest rate is much higher at 5% per annum. "
            "The investor borrows [DATA MISSING] Japanese Yen and converts this "
            "amount into Brazilian Reals using an exchange rate of 0.05. "
            "The investor holds the investment for one year, assuming that the "
            "exchange rate remains stable during this period."
        ),
    },
    {
        "id": "V1_usd_to_real",
        "note": "통화 변경: Yen→USD (0.05/5%, rate 4.5 BRL/USD)",
        "question": Q,
        "context": (
            "A global investor is exploring a currency carry trade strategy. "
            "The investor plans to borrow funds in the United States, where the "
            "interest rate is low at 1% per annum, and invest these funds in "
            "Brazil, where the interest rate is much higher at 5% per annum. "
            "The investor borrows [DATA MISSING] US Dollars and converts this "
            "amount into Brazilian Reals using an exchange rate of 4.5. "
            "The investor holds the investment for one year, assuming that the "
            "exchange rate remains stable during this period."
        ),
    },
    {
        "id": "V2_chf_to_try",
        "note": "통화 변경: CHF→TRY (0.25%/15%, rate 30)",
        "question": Q_ALT_CCY,
        "context": (
            "A global investor is exploring a currency carry trade strategy. "
            "The investor plans to borrow funds in Switzerland, where the "
            "interest rate is extremely low at 0.25% per annum, and invest these "
            "funds in Turkey, where the interest rate is much higher at 15% per "
            "annum. The investor borrows [DATA MISSING] Swiss Francs and converts "
            "this amount into Turkish Lira using an exchange rate of 30. "
            "The investor holds the investment for one year, assuming that the "
            "exchange rate remains stable during this period."
        ),
    },
    {
        "id": "V3_eur_to_zar",
        "note": "통화 변경: EUR→ZAR (2%/8%, rate 20)",
        "question": Q_ALT_CCY,
        "context": (
            "A global investor is exploring a currency carry trade strategy. "
            "The investor plans to borrow funds in the Eurozone, where the "
            "interest rate is relatively low at 2% per annum, and invest these "
            "funds in South Africa, where the interest rate is higher at 8% per "
            "annum. The investor borrows [DATA MISSING] Euros and converts this "
            "amount into South African Rand using an exchange rate of 20. "
            "The investor holds the investment for one year, assuming that the "
            "exchange rate remains stable during this period."
        ),
    },
    {
        "id": "V4_generic",
        "note": "익명화: country X/Y 로 추상화 (1%/7%, rate 0.5)",
        "question": "What is the profit from this carry trade in the target-country currency? Answer to the nearest integer.",
        "context": (
            "A global investor is exploring a currency carry trade strategy. "
            "The investor plans to borrow funds in country X, where the "
            "interest rate is low at 1% per annum, and invest these funds in "
            "country Y, where the interest rate is higher at 7% per annum. "
            "The investor borrows [DATA MISSING] units of currency X and converts "
            "this amount into currency Y using an exchange rate of 0.5. "
            "The investor holds the investment for one year, assuming that the "
            "exchange rate remains stable during this period."
        ),
    },
]


# ---------------------------------------------------------------------------
# Provider / extraction
# ---------------------------------------------------------------------------
def build_provider(model_name: str, temperature: float):
    info = MODEL_REGISTRY[model_name]
    api_key = os.environ.get(info.api_key_env)
    if not api_key:
        return None
    config = ModelConfig(
        id=model_name,
        name=info.display_name,
        provider=info.provider,
        model_id=info.model_id,
        api_key_env_var=info.api_key_env,
        max_tokens=4096,
        temperature=temperature,
        cost_per_million_tokens=info.cost_per_million_output,
    )
    return {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
    }[info.provider](config)


def extract_borrowed(code: Optional[str], raw: str) -> Optional[float]:
    text = code or raw
    if not text:
        return None
    patterns = [
        r"borrowed_amount[_a-z]*\s*=\s*([\d,_.]+)",
        r"borrowed[_a-z]*\s*=\s*([\d,_.]+)",
        r"principal[_a-z]*\s*=\s*([\d,_.]+)",
        r"loan[_a-z]*\s*=\s*([\d,_.]+)",
        r"amount[_a-z]*\s*=\s*([\d,_.]+)",
        r"borrow\w*\s*=\s*([\d,_.]+)",
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
    temp: float,
    run_idx: int,
    variant: Dict[str, str],
    detector: RefusalDetector,
) -> Dict[str, Any]:
    class Mini:
        def __init__(self, qid, q, c):
            self.id = qid
            self.question = q
            self.context = c
            self.ground_truth = "N/A"
            self.python_solution = ""

    mini = Mini(variant["id"], variant["question"], variant["context"])
    full_prompt = build_prompt(variant["question"], variant["context"], "standard")

    try:
        resp = await provider.call_model(mini, full_prompt)
        raw = resp.raw_response
        cost = calculate_cost(resp.prompt_tokens, resp.completion_tokens, model_name)
        answer, code, exec_err = extract_answer(raw)
        detection = detector.detect(
            raw, context=variant["context"], execution_error=exec_err, executed_code=code
        )
        if answer == "INSUFFICIENT_INFORMATION":
            rtype = "refused"
        else:
            rtype = detection.response_type.value
        borrowed = extract_borrowed(code, raw)
        return {
            "variant": variant["id"],
            "model": model_name,
            "temperature": temp,
            "run": run_idx,
            "response_type": rtype,
            "predicted_answer": str(answer) if answer is not None else None,
            "borrowed_amount_assumed": borrowed,
            "cost_usd": cost,
            "executed_code": code,
            "raw_response": raw,
        }
    except Exception as e:
        logger.error(f"  [ERROR] {model_name}/{variant['id']}/r{run_idx}: {e}")
        return {
            "variant": variant["id"],
            "model": model_name,
            "temperature": temp,
            "run": run_idx,
            "response_type": "error",
            "predicted_answer": None,
            "borrowed_amount_assumed": None,
            "cost_usd": 0.0,
            "executed_code": None,
            "raw_response": f"ERROR: {str(e)[:200]}",
        }


async def run_probe(
    models: List[str], temp: float, runs: int
) -> List[Dict[str, Any]]:
    load_dotenv()
    detector = RefusalDetector()
    results: List[Dict[str, Any]] = []

    for model in models:
        provider = build_provider(model, temp)
        if provider is None:
            logger.warning(f"Skip {model} (no API key)")
            continue
        logger.info(f"=== {model} @ temp={temp} ===")
        for variant in VARIANTS:
            logger.info(f"  variant {variant['id']}: {variant['note']}")
            tasks = [
                one_run(provider, model, temp, i + 1, variant, detector)
                for i in range(runs)
            ]
            batch = await asyncio.gather(*tasks)
            for r in batch:
                tag = "REFUSED" if r["response_type"] == "refused" else "ANSWERED"
                logger.info(
                    f"    run{r['run']}: {tag} borrowed={r['borrowed_amount_assumed']}"
                )
            results.extend(batch)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[tuple, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "refused": 0, "borrowed": Counter()}
    )
    for r in results:
        k = (r["model"], r["variant"])
        agg[k]["n"] += 1
        if r["response_type"] == "refused":
            agg[k]["refused"] += 1
        if r["borrowed_amount_assumed"] is not None:
            agg[k]["borrowed"][r["borrowed_amount_assumed"]] += 1
    return {
        "by_model_variant": [
            {
                "model": m,
                "variant": v,
                "n": s["n"],
                "refused": s["refused"],
                "borrowed_distribution": dict(s["borrowed"].most_common()),
            }
            for (m, v), s in sorted(agg.items())
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Carry-trade principal prior probe")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gemini-2.5-flash", "claude-haiku-4"],
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    results = asyncio.run(run_probe(args.models, args.temperature, args.runs))
    summary = summarize(results)

    out_dir = Path(__file__).parent.parent / "experiments/results/metacognitive"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"probe_carry_trade_prior_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": ts,
                    "temperature": args.temperature,
                    "runs_per_cell": args.runs,
                    "variants": [{"id": v["id"], "note": v["note"]} for v in VARIANTS],
                },
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY — 모델×variant 별 borrowed_amount 분포 (temp={:.1f})".format(args.temperature))
    logger.info("=" * 80)
    for row in summary["by_model_variant"]:
        logger.info(
            f"{row['model']:22s} {row['variant']:18s} n={row['n']:2d} refused={row['refused']:2d}"
        )
        for val, cnt in row["borrowed_distribution"].items():
            pct = cnt / row["n"] * 100
            logger.info(f"    borrowed={val:>15,.0f}  : {cnt}/{row['n']} ({pct:.0f}%)")
    logger.info(f"\n저장: {out}")


if __name__ == "__main__":
    main()
