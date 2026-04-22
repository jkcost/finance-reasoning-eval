"""LLM 기반 한글 해설 — 변환 응답 3 전략 cross-coverage.

변환된 context에 대한 모델 응답 중 거부(refused) 또는 오답(is_correct=False)
케이스에 한글 상세 해설을 생성한다. 3 전략(standard, metacognitive, cot_trace)
전부 대상.

Output:
  experiments/results/metacognitive/transformed_error_analysis_0_20.json
    {
      "test-2001": {
        "EA-partial": {
          "standard": { error_type, summary, detailed_analysis, likely_cause, correct_approach, confidence, is_refusal },
          "metacognitive": {...},
          "cot_trace": {...}
        },
        ...
      },
      ...
    }

Usage:
    python experiments/generate_transformed_error_analysis.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from _eval_io import load_eval_records  # noqa: E402
from error_analysis.llm_error_analyzer import LLMErrorAnalyzer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _index_problems(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """qid → problem (with transformations)."""
    return {p["question_id"]: p for p in batch.get("problems", [])}


def _needs_analysis(record: dict[str, Any]) -> bool:
    """Filter: only analyze records that refused, errored, or got wrong answers."""
    if record.get("is_correct"):
        return False
    rt = record.get("response_type", "")
    if rt == "confident" and not record.get("is_correct"):
        return True  # confident but wrong
    if rt in ("refused", "caveat"):
        return True
    if rt == "error":
        return False  # pipeline-level error, not model behavior
    return not record.get("is_correct", True)


def _target_rows(
    records: list[dict[str, Any]],
    strategy_label: str,
    qid_subset: set[str],
    problems_idx: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str, dict[str, Any], dict[str, Any]]]:
    """Pick (qid, ttype, strategy, problem, record) tuples needing analysis."""
    targets: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = []
    for rec in records:
        qid = rec.get("question_id", "")
        ttype = rec.get("transformation_type", "")
        if qid not in qid_subset or not ttype:
            continue
        if not _needs_analysis(rec):
            continue
        problem = problems_idx.get(qid)
        if not problem:
            continue
        tx = problem.get("transformations", {}).get(ttype)
        if not tx or not tx.get("success"):
            continue
        targets.append((qid, ttype, strategy_label, problem, rec))
    return targets


async def analyze_one(
    analyzer: LLMErrorAnalyzer,
    qid: str,
    ttype: str,
    strategy: str,
    problem: dict[str, Any],
    rec: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[str, str, str, dict[str, Any] | None]:
    """Run a single analyze call for a transformed-context response."""
    async with sem:
        tx = problem["transformations"][ttype]
        context = tx.get("context_transformed") or problem.get("context_original", "")
        question = problem.get("question", "")
        ground_truth = problem.get("ground_truth", "")
        predicted = rec.get("predicted_answer", "")
        raw = rec.get("raw_response", "")
        is_refusal = rec.get("response_type") in ("refused", "caveat") or (
            str(predicted).startswith("INSUFFICIENT_INFORMATION")
        )

        # 거부 케이스면 GT 대신 "변환 의도"를 prompt에 넣어주면 더 나은 분석이 나오지만,
        # 단순화를 위해 기존 분석기를 그대로 재사용. "predicted=INSUFFICIENT_INFORMATION"
        # 입력으로 LLM이 해석 방향을 잡음.

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            analyzer.analyze,
            question,
            context,
            ground_truth,
            predicted,
            raw,
        )
        if result is None:
            return qid, ttype, strategy, None
        payload = asdict(result)
        payload["is_refusal"] = is_refusal
        return qid, ttype, strategy, payload


async def run(
    args: argparse.Namespace,
    qid_subset: set[str],
) -> int:
    """Collect records across 3 strategies and generate analyses."""
    load_dotenv()

    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    problems_idx = _index_problems(batch)

    all_targets: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = []

    if args.standard.exists():
        recs = load_eval_records(args.standard)
        tgts = _target_rows(recs, "standard", qid_subset, problems_idx)
        logger.info(f"standard: {len(tgts)} targets from {args.standard.name}")
        all_targets.extend(tgts)
    if args.metacognitive.exists():
        recs = load_eval_records(args.metacognitive)
        tgts = _target_rows(recs, "metacognitive", qid_subset, problems_idx)
        logger.info(f"metacognitive: {len(tgts)} targets from {args.metacognitive.name}")
        all_targets.extend(tgts)
    if args.cot_trace.exists():
        recs = load_eval_records(args.cot_trace)
        tgts = _target_rows(recs, "cot_trace", qid_subset, problems_idx)
        logger.info(f"cot_trace: {len(tgts)} targets from {args.cot_trace.name}")
        all_targets.extend(tgts)

    logger.info(f"Total targets: {len(all_targets)}")

    analyzer = LLMErrorAnalyzer(model=args.model)
    if analyzer.client is None:
        logger.error("OpenAI client 초기화 실패")
        return 1

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [analyze_one(analyzer, *t, sem) for t in all_targets]

    # Stream progress
    results: list[tuple[str, str, str, dict[str, Any] | None]] = []
    for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
        res = await fut
        results.append(res)
        if i % 20 == 0 or i == len(tasks):
            ok = sum(1 for r in results if r[3] is not None)
            logger.info(f"  진행: {i}/{len(tasks)} (성공 {ok})")

    # Reshape to nested dict
    bundle: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for qid, ttype, strategy, payload in results:
        if payload is None:
            continue
        bundle.setdefault(qid, {}).setdefault(ttype, {})[strategy] = payload

    # Stats
    from collections import Counter
    combos = Counter((s, a.get("error_type", "?"))
                     for per_q in bundle.values()
                     for per_t in per_q.values()
                     for s, a in per_t.items())
    logger.info("\nStrategy × Error Type:")
    for (s, et), n in sorted(combos.items()):
        logger.info(f"  {s:<14s} {et:<22s} {n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "analyses": bundle,
                "metadata": {
                    "model": args.model,
                    "n_targets": len(all_targets),
                    "n_completed": sum(1 for r in results if r[3] is not None),
                    "qid_range": args.qid_range,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote {args.output}")
    return 0


def _resolve_qid_subset(spec: str) -> set[str]:
    """Expand 'test-2000:test-2019' to set."""
    start_s, end_s = spec.split(":")
    start_n = int(start_s.rsplit("-", 1)[-1])
    end_n = int(end_s.rsplit("-", 1)[-1])
    prefix = start_s.rsplit("-", 1)[0]
    return {f"{prefix}-{i}" for i in range(start_n, end_n + 1)}


def parse_args() -> argparse.Namespace:
    """Parse CLI."""
    parser = argparse.ArgumentParser(description="LLM 한글 해설 (변환 응답 3 전략)")
    parser.add_argument(
        "--batch",
        type=Path,
        default=Path("experiments/results/metacognitive/batch_transformations_0_20_subset.json"),
    )
    parser.add_argument(
        "--standard",
        type=Path,
        default=Path("experiments/results/metacognitive/batch_evaluation_0_238.json"),
        help="Standard 전략 평가 결과 (기존 auto-detected)",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/transformed_error_analysis_0_20.json"),
    )
    parser.add_argument("--qid-range", type=str, default="test-2000:test-2019")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    """Entry."""
    args = parse_args()
    qid_subset = _resolve_qid_subset(args.qid_range)
    return asyncio.run(run(args, qid_subset))


if __name__ == "__main__":
    sys.exit(main())
