"""Compute Memorization Score per model × transformation_type (RQ-B).

F1 of plan ``2026-04-21-001-feat-cikm-short-execution-plan.md``.

Memorization Score = fraction of response values classified as ``from_removed_data``.
Higher score ⇒ model recalls original values that were intentionally removed/modified.

R4 (solvable 모집단) 필터:
    If ``--original-context <path>`` is provided, only records whose ``question_id``
    was answered correctly in the original context are kept. Others are reported in
    a separate "noise track" summary but excluded from main Memorization Score.

Usage:
    python experiments/compute_memorization_score.py \\
        --eval-results experiments/results/metacognitive/eval_metacognitive.json \\
        --transformations experiments/results/metacognitive/batch_transformations_0_238.json \\
        --original-context experiments/results/metacognitive/eval_original_context.json \\
        --output experiments/results/metacognitive/memorization_score.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning_trace_analyzer import (  # noqa: E402
    NumberExtractor,
    ValueProvenanceClassifier,
)
from _eval_io import load_eval_records, qid_in_solvable, solvable_qids  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _index_transformations(batch: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    """Build (question_id, transformation_type) → {context_original, context_transformed} index."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    for problem in batch.get("problems", []):
        qid = problem.get("question_id")
        context_original = problem.get("context_original", "") or ""
        for ttype, tx in (problem.get("transformations") or {}).items():
            if not isinstance(tx, dict) or not tx.get("success"):
                continue
            context_transformed = tx.get("transformed_content", "") or ""
            index[(str(qid), ttype)] = {
                "context_original": context_original,
                "context_transformed": context_transformed,
            }
    logger.info(f"Indexed {len(index)} (qid, transformation) pairs")
    return index


def _compute_removed_values(
    context_original: str, context_transformed: str, extractor: NumberExtractor
) -> set[float]:
    """Values present in original but absent (or replaced) in transformed context."""
    original_vals = extractor.extract_from_context(context_original)
    transformed_vals = extractor.extract_from_context(context_transformed)
    return original_vals - transformed_vals


def _extract_response_numbers(raw_response: str, extractor: NumberExtractor) -> list:
    """Extract numbers appearing in a response, using the same extractor."""
    if not raw_response:
        return []
    if hasattr(extractor, "extract_from_response"):
        return extractor.extract_from_response(raw_response)
    # Fallback: treat response as plain text
    values = extractor.extract_from_context(raw_response)
    return [type("N", (), {"value": v})() for v in values]


def score_record(
    record: dict[str, Any],
    tx_index: dict[tuple[str, str], dict[str, str]],
    extractor: NumberExtractor,
    classifier: ValueProvenanceClassifier,
) -> dict[str, Any] | None:
    """Score a single evaluation record's value provenance."""
    qid = record.get("question_id")
    ttype = record.get("transformation_type")
    if not qid or not ttype:
        return None
    contexts = tx_index.get((str(qid), ttype))
    if not contexts:
        return None

    raw_response = record.get("raw_response", "") or ""
    if not raw_response.strip():
        return None

    transformed_vals = extractor.extract_from_context(contexts["context_transformed"])
    removed_vals = _compute_removed_values(
        contexts["context_original"], contexts["context_transformed"], extractor
    )
    response_nums = _extract_response_numbers(raw_response, extractor)
    if not response_nums:
        return None

    provenances = classifier.classify(response_nums, transformed_vals, removed_vals)
    if not provenances:
        return None

    counts: dict[str, int] = defaultdict(int)
    for p in provenances:
        counts[p.source] += 1

    total = sum(counts.values())
    memorization_rate = counts.get("from_removed_data", 0) / total if total else 0.0

    return {
        "question_id": qid,
        "transformation_type": ttype,
        "model": record.get("model"),
        "prompt_strategy": record.get("prompt_strategy"),
        "response_type": record.get("response_type"),
        "is_correct": record.get("is_correct", False),
        "total_values": total,
        "counts": dict(counts),
        "memorization_rate": memorization_rate,
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record scores into per-(model, transformation) Memorization Score."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in scores:
        key = (str(s.get("model")), str(s.get("transformation_type")))
        grouped[key].append(s["memorization_rate"])

    table: list[dict[str, Any]] = []
    for (model, ttype), rates in sorted(grouped.items()):
        table.append({
            "model": model,
            "transformation_type": ttype,
            "n": len(rates),
            "memorization_score": sum(rates) / len(rates) if rates else 0.0,
            "non_zero_rate": sum(1 for r in rates if r > 0) / len(rates) if rates else 0.0,
        })
    return {"by_model_transform": table}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute Memorization Score (RQ-B)")
    parser.add_argument("--eval-results", type=Path, nargs="+", required=True)
    parser.add_argument("--transformations", type=Path, required=True)
    parser.add_argument(
        "--original-context",
        type=Path,
        default=None,
        help="Optional Phase A (original context) results for R4 solvable filter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/memorization_score.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit records for debugging")
    return parser.parse_args()


def main() -> int:
    """Run Memorization Score computation."""
    args = parse_args()
    if not args.transformations.exists():
        logger.error(f"transformations not found: {args.transformations}")
        return 1

    batch = json.loads(args.transformations.read_text(encoding="utf-8"))
    tx_index = _index_transformations(batch)
    solvable = solvable_qids(args.original_context)
    if solvable is not None:
        logger.info(f"Solvable QIDs: {len(solvable)}")

    extractor = NumberExtractor()
    classifier = ValueProvenanceClassifier(tolerance=0.01)

    all_records: list[dict[str, Any]] = []
    for path in args.eval_results:
        if not path.exists():
            logger.warning(f"skipping {path} (missing)")
            continue
        all_records.extend(load_eval_records(path))

    if args.limit:
        all_records = all_records[: args.limit]

    logger.info(f"Scoring {len(all_records)} records")

    scored_main: list[dict[str, Any]] = []
    scored_noise: list[dict[str, Any]] = []
    skipped = 0
    for rec in all_records:
        score = score_record(rec, tx_index, extractor, classifier)
        if score is None:
            skipped += 1
            continue
        if not qid_in_solvable(score["question_id"], solvable):
            scored_noise.append(score)
        else:
            scored_main.append(score)

    logger.info(f"Scored main={len(scored_main)} | noise_track={len(scored_noise)} | skipped={skipped}")

    output = {
        "metadata": {
            "eval_results": [str(p) for p in args.eval_results],
            "transformations": str(args.transformations),
            "original_context": str(args.original_context) if args.original_context else None,
            "solvable_filter_applied": solvable is not None,
            "main_n": len(scored_main),
            "noise_n": len(scored_noise),
            "skipped_n": skipped,
        },
        "main": {
            "aggregated": aggregate(scored_main),
            "per_record": scored_main,
        },
        "noise_track": {
            "aggregated": aggregate(scored_noise),
            "per_record": scored_noise,
        } if solvable is not None else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output}")

    # Console summary
    logger.info("\n=== Memorization Score (main track) ===")
    for row in output["main"]["aggregated"]["by_model_transform"]:
        logger.info(
            f"  {row['model']:<20s} {row['transformation_type']:<12s} "
            f"n={row['n']:4d}  score={row['memorization_score']:.3f}  "
            f"non_zero={row['non_zero_rate']:.2%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
