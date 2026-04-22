"""Run LLM-based Korean error analysis on original-context wrong answers.

기존 evaluation/error_analysis/llm_error_analyzer.py (한글 오답 분석) 를 활용하여
20문제 중 POT 원본 오답 케이스들에 대해 상세 한글 해설 생성.

v4 HTML의 원본 Context 응답 카드 overlay에 삽입.

Usage:
    python experiments/generate_original_error_analysis.py \\
        --batch experiments/results/metacognitive/batch_transformations_0_20_subset.json \\
        --pot experiments/results/metacognitive/eval_original_context.json \\
        --output experiments/results/metacognitive/original_error_analysis_0_20.json
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
    """question_id → problem record."""
    return {p["question_id"]: p for p in batch.get("problems", [])}


async def analyze_one(
    analyzer: LLMErrorAnalyzer,
    problem: dict[str, Any],
    pot_record: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None]:
    """Run one LLM analysis for a single wrong POT answer."""
    async with sem:
        qid = problem["question_id"]
        question = problem.get("question", "")
        context = problem.get("context_original", "") or ""
        ground_truth = problem.get("ground_truth", "")
        predicted = pot_record.get("predicted_answer", "")
        raw = pot_record.get("raw_response", "")

        # analyzer.analyze is sync — run in executor to stay non-blocking
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
            logger.warning(f"[{qid}] LLM analysis returned None")
            return qid, None
        logger.info(f"[{qid}] {result.error_type}: {result.summary[:60]}")
        return qid, asdict(result)


async def run(
    batch_path: Path,
    pot_path: Path,
    output_path: Path,
    qid_subset: set[str],
    model: str,
    concurrency: int,
) -> int:
    """Analyze all POT-wrong original context answers within the subset."""
    load_dotenv()

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    problems_idx = _index_problems(batch)
    pot_records = load_eval_records(pot_path)

    # Pick POT-wrong records in subset — dedupe by qid since eval_original_context
    # has one record per (qid, transformation_type) but the original-context response
    # is identical across transformations.
    targets_by_qid: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for rec in pot_records:
        qid = rec.get("question_id", "")
        if qid not in qid_subset or rec.get("is_correct"):
            continue
        if qid in targets_by_qid:
            continue
        problem = problems_idx.get(qid)
        if problem:
            targets_by_qid[qid] = (problem, rec)
    targets = list(targets_by_qid.values())

    logger.info(f"POT-wrong original cases (dedup): {len(targets)} qids — {sorted(targets_by_qid.keys())}")
    if not targets:
        logger.info("No wrong POT cases — writing empty bundle")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"analyses": {}, "metadata": {"n": 0}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    analyzer = LLMErrorAnalyzer(model=model)
    if analyzer.client is None:
        logger.error("OpenAI client 초기화 실패 — OPENAI_API_KEY 확인")
        return 1

    sem = asyncio.Semaphore(concurrency)
    tasks = [analyze_one(analyzer, p, r, sem) for p, r in targets]
    results = await asyncio.gather(*tasks)

    analyses: dict[str, dict[str, Any]] = {}
    for qid, res in results:
        if res is not None:
            analyses[qid] = res

    logger.info(f"Completed analyses: {len(analyses)}/{len(targets)}")

    from collections import Counter
    err_types = Counter(a["error_type"] for a in analyses.values())
    logger.info(f"Error type distribution: {dict(err_types)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "analyses": analyses,
                "metadata": {
                    "model": model,
                    "n": len(analyses),
                    "batch": str(batch_path),
                    "pot": str(pot_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote {output_path}")
    return 0


def _resolve_qid_subset(spec: str) -> set[str]:
    """Expand 'test-2000:test-2019' inclusive to set."""
    start_s, end_s = spec.split(":")
    start_n = int(start_s.rsplit("-", 1)[-1])
    end_n = int(end_s.rsplit("-", 1)[-1])
    prefix = start_s.rsplit("-", 1)[0]
    return {f"{prefix}-{i}" for i in range(start_n, end_n + 1)}


def parse_args() -> argparse.Namespace:
    """Parse CLI."""
    parser = argparse.ArgumentParser(description="LLM 기반 한글 오답 해설 생성")
    parser.add_argument("--batch", type=Path, required=True, help="batch_transformations subset JSON")
    parser.add_argument("--pot", type=Path, required=True, help="eval_original_context.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/original_error_analysis_0_20.json"),
    )
    parser.add_argument("--qid-range", type=str, default="test-2000:test-2019")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    """Entrypoint."""
    args = parse_args()
    qid_subset = _resolve_qid_subset(args.qid_range)
    return asyncio.run(run(args.batch, args.pot, args.output, qid_subset, args.model, args.concurrency))


if __name__ == "__main__":
    sys.exit(main())
