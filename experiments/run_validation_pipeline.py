"""
Reasoning Trace Validation Pipeline

Analyzes Phase B results to validate transformation effectiveness by tracing
how models use (or don't use) the transformed context data.

Usage:
    python experiments/run_validation_pipeline.py                    # 규칙 기반만
    python experiments/run_validation_pipeline.py --with-llm-judge   # LLM Judge 포함
    python experiments/run_validation_pipeline.py --results-dir DIR  # 경로 지정
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

from hardcoded_solution_detector import build_hardcoded_problem_ids
from apply_transformations_full import normalize_transformation_label
from reasoning_trace_analyzer import ReasoningTraceAnalyzer, ReasoningTraceAnalysis

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "metacognitive"
DATA_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "financereasoning"
    / "raw"
    / "FinanceReasoning"
)
HARD_JSON_PATH = DATA_DIR / "hard.json"


# ============================================================================
# DATA LOADING
# ============================================================================


def _load_example_map() -> Dict[str, Dict[str, Any]]:
    """Load all examples from dataset files, keyed by question_id."""
    examples: Dict[str, Dict[str, Any]] = {}
    for level_file in ("easy.json", "medium.json", "hard.json"):
        fp = DATA_DIR / level_file
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for item in json.load(f):
                qid = item.get("question_id", "")
                if qid:
                    examples[qid] = item
    return examples


def _build_full_context_map(
    example_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Build full context map: (question_id, transformation_type) -> full contexts.

    Re-runs apply_transformations() on original examples to recover
    the full context that was sent to models (Phase B results may
    store truncated contexts).
    """
    from apply_transformations_full import apply_transformations as _apply_trans

    ctx_map: Dict[str, Dict[str, str]] = {}
    for qid, example in example_map.items():
        original_context = example.get("context", "")
        # Store original context
        ctx_map[f"{qid}|original"] = {
            "context_original": original_context,
            "context_transformed": "",
        }
        # Generate transformations
        try:
            transformed_list = _apply_trans(example)
        except Exception:
            continue
        for trans in transformed_list:
            trans_type = normalize_transformation_label(
                trans.get("transformation_type", "")
            )
            ctx_map[f"{qid}|{trans_type}"] = {
                "context_original": original_context,
                "context_transformed": trans.get("context", ""),
            }
    return ctx_map


def load_phase_b_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all Phase B result dicts from JSON files.

    Normalizes transformation labels from legacy format.
    """
    all_results: List[Dict[str, Any]] = []
    for fp in sorted(results_dir.glob("phase_B_*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            r["transformation_type"] = normalize_transformation_label(
                r.get("transformation_type", "")
            )
            all_results.append(r)
    return all_results


def restore_full_contexts(
    results: List[Dict[str, Any]],
    ctx_map: Dict[str, Dict[str, str]],
) -> int:
    """Replace truncated contexts in results with full versions from ctx_map.

    Returns count of restored results.
    """
    restored = 0
    for r in results:
        qid = r.get("example_id", "")
        trans_type = r.get("transformation_type", "")
        key = f"{qid}|{trans_type}"
        if key in ctx_map:
            full = ctx_map[key]
            old_orig = r.get("context_original", "")
            new_orig = full["context_original"]
            new_trans = full["context_transformed"]
            if len(new_orig) > len(old_orig):
                r["context_original"] = new_orig
                restored += 1
            if new_trans and len(new_trans) > len(r.get("context_transformed", "")):
                r["context_transformed"] = new_trans
    return restored


# ============================================================================
# EFFECTIVENESS AGGREGATION
# ============================================================================


def aggregate_effectiveness(
    analyses: List[Tuple[Dict[str, Any], ReasoningTraceAnalysis]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate transformation effectiveness by (example_id, transformation_type).

    Returns dict keyed by "example_id|transformation_type" with:
        - verdict: "effective" / "still_solvable" / "transformation_insufficient"
        - case_distribution: {1: n, 2: n, 3: n}
        - model_results: [{model, case_type, context_independence_score, ...}]
    """
    groups: Dict[str, List[Tuple[Dict, ReasoningTraceAnalysis]]] = defaultdict(list)

    for result, analysis in analyses:
        key = f"{result['example_id']}|{result['transformation_type']}"
        groups[key].append((result, analysis))

    aggregated: Dict[str, Dict[str, Any]] = {}
    for key, items in groups.items():
        case_dist = {1: 0, 2: 0, 3: 0}
        model_results = []
        max_ctx_independence = 0.0

        for result, analysis in items:
            case_dist[analysis.case_type] += 1
            max_ctx_independence = max(
                max_ctx_independence, analysis.context_independence_score
            )
            model_results.append(
                {
                    "model": result.get("model_name", ""),
                    "prompt_strategy": result.get("prompt_strategy", ""),
                    "case_type": analysis.case_type,
                    "context_independence_score": round(
                        analysis.context_independence_score, 3
                    ),
                    "awareness_signals": analysis.awareness_signals,
                    "summary": analysis.summary,
                }
            )

        # Determine verdict
        if case_dist[3] == 0:
            verdict = "effective"
        elif max_ctx_independence >= 0.5:
            verdict = "still_solvable"
        else:
            verdict = "transformation_insufficient"

        example_id, trans_type = key.split("|", 1)
        aggregated[key] = {
            "example_id": example_id,
            "transformation_type": trans_type,
            "verdict": verdict,
            "case_distribution": case_dist,
            "model_results": model_results,
        }

    return aggregated


def compute_summary_stats(
    analyses: List[Tuple[Dict[str, Any], ReasoningTraceAnalysis]],
    aggregated: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute high-level summary statistics."""
    total = len(analyses)
    case_counts = {1: 0, 2: 0, 3: 0}
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {1: 0, 2: 0, 3: 0})
    by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {1: 0, 2: 0, 3: 0})

    source_counts: Dict[str, int] = defaultdict(int)

    for result, analysis in analyses:
        case_counts[analysis.case_type] += 1
        trans_type = result.get("transformation_type", "unknown")
        model = result.get("model_name", "unknown")
        by_type[trans_type][analysis.case_type] += 1
        by_model[model][analysis.case_type] += 1

        for v in analysis.values_used:
            source_counts[v.source] += 1

    verdict_counts = {
        "effective": 0,
        "still_solvable": 0,
        "transformation_insufficient": 0,
    }
    for agg in aggregated.values():
        verdict_counts[agg["verdict"]] += 1

    # Source dataset breakdown
    unique_questions = set()
    unique_models = set()
    unique_strategies = set()
    trans_per_question: Dict[str, set] = defaultdict(set)
    for result, analysis in analyses:
        qid = result.get("example_id", "")
        unique_questions.add(qid)
        unique_models.add(result.get("model_name", "unknown"))
        unique_strategies.add(result.get("prompt_strategy", "unknown"))
        trans_per_question[qid].add(result.get("transformation_type", ""))

    return {
        "total_results": total,
        "source_info": {
            "dataset": "hard.json",
            "dataset_total": 238,
            "unique_questions": len(unique_questions),
            "question_ids": sorted(unique_questions),
            "unique_models": sorted(unique_models),
            "unique_strategies": sorted(unique_strategies),
            "transformations_per_question": {
                qid: len(types) for qid, types in sorted(trans_per_question.items())
            },
        },
        "case_distribution": case_counts,
        "case_by_type": {k: dict(v) for k, v in by_type.items()},
        "case_by_model": {k: dict(v) for k, v in by_model.items()},
        "value_source_counts": dict(source_counts),
        "verdict_counts": verdict_counts,
        "total_transformations": len(aggregated),
    }


# ============================================================================
# OUTPUT
# ============================================================================


def save_results(
    output_path: Path,
    analyses: List[Tuple[Dict[str, Any], ReasoningTraceAnalysis]],
    aggregated: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """Save analysis results to JSON."""
    result_entries = []
    for result, analysis in analyses:
        entry = {
            "example_id": result.get("example_id", ""),
            "model_name": result.get("model_name", ""),
            "prompt_strategy": result.get("prompt_strategy", ""),
            "transformation_type": result.get("transformation_type", ""),
            "response_type": result.get("response_type", ""),
            "predicted_answer": result.get("predicted_answer"),
            "ground_truth": result.get("ground_truth"),
            "raw_response": result.get("raw_response", ""),
            "question": result.get("question", ""),
            "context_original": result.get("context_original", ""),
            "context_transformed": result.get("context_transformed", ""),
            "transformation_description": result.get("transformation_description", ""),
            "analysis": analysis.to_dict(),
        }
        result_entries.append(entry)

    output = {
        "pipeline": "reasoning_trace_validation",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "summary": summary,
        "aggregated_verdicts": list(aggregated.values()),
        "detailed_results": result_entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Results saved to %s", output_path)


def print_console_summary(summary: Dict[str, Any]) -> None:
    """Print a concise summary to console."""
    print("\n" + "=" * 60)
    print("  Reasoning Trace Validation Summary")
    print("=" * 60)

    src = summary.get("source_info", {})
    cd = summary["case_distribution"]
    total = summary["total_results"]
    n_q = src.get("unique_questions", "?")
    ds_total = src.get("dataset_total", "?")
    n_models = len(src.get("unique_models", []))
    n_strats = len(src.get("unique_strategies", []))
    n_trans = summary["total_transformations"]

    print(f"\n  Source: {ds_total}개 문제 중 {n_q}개 사용 (hard.json)")
    print(
        f"  Composition: {n_q} questions × {n_trans} transformations × {n_models} models × {n_strats} strategies"
    )
    print(f"  Total results analyzed: {total}")
    print(f"  Case 1 (거부):  {cd[1]:>4} ({cd[1] / total * 100:.1f}%)")
    print(f"  Case 2 (오답):  {cd[2]:>4} ({cd[2] / total * 100:.1f}%)")
    print(f"  Case 3 (정답):  {cd[3]:>4} ({cd[3] / total * 100:.1f}%)")

    print("\n  --- By Transformation Type ---")
    for trans_type, counts in sorted(summary["case_by_type"].items()):
        t = sum(counts.values())
        print(
            f"  {trans_type[:40]:<42} C1={counts[1]:>2} C2={counts[2]:>2} C3={counts[3]:>2} (n={t})"
        )

    print("\n  --- By Model ---")
    for model, counts in sorted(summary["case_by_model"].items()):
        t = sum(counts.values())
        print(
            f"  {model:<25} C1={counts[1]:>3} C2={counts[2]:>3} C3={counts[3]:>3} (n={t})"
        )

    print("\n  --- Value Source Distribution ---")
    for source, count in sorted(
        summary["value_source_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"  {source:<25} {count:>4}")

    vc = summary["verdict_counts"]
    print("\n  --- Transformation Verdicts ---")
    total_v = summary["total_transformations"]
    print(
        f"  effective:                    {vc['effective']:>3} ({vc['effective'] / max(total_v, 1) * 100:.1f}%)"
    )
    print(
        f"  still_solvable:  {vc['still_solvable']:>3} ({vc['still_solvable'] / max(total_v, 1) * 100:.1f}%)"
    )
    print(
        f"  transformation_insufficient:  {vc['transformation_insufficient']:>3} ({vc['transformation_insufficient'] / max(total_v, 1) * 100:.1f}%)"
    )
    print("=" * 60 + "\n")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Reasoning Trace Validation Pipeline")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing Phase B JSON results",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="Enable LLM Judge for deep analysis of ambiguous cases",
    )
    parser.add_argument(
        "--llm-judge-model",
        default="gpt-4o-mini",
        help="Model to use for LLM Judge (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results-dir/validation_analysis_TIMESTAMP.json)",
    )
    args = parser.parse_args()

    # 1. Load Phase B results
    logger.info("Loading Phase B results from %s", args.results_dir)
    results = load_phase_b_results(args.results_dir)
    if not results:
        logger.error("No Phase B results found in %s", args.results_dir)
        sys.exit(1)
    logger.info("Loaded %d Phase B results", len(results))

    # 1.5. Restore full contexts (Phase B may store truncated contexts)
    logger.info("Loading original examples for context restoration...")
    example_map = _load_example_map()
    logger.info("Loaded %d examples from dataset", len(example_map))
    ctx_map = _build_full_context_map(example_map)
    logger.info("Built context map with %d entries", len(ctx_map))
    restored = restore_full_contexts(results, ctx_map)
    logger.info("Restored full context for %d / %d results", restored, len(results))

    # 2. Build hardcoded problem ID set
    hardcoded_ids = build_hardcoded_problem_ids(HARD_JSON_PATH)
    logger.info("Detected %d hardcoded problems", len(hardcoded_ids))

    # 3. Initialize analyzer
    analyzer = ReasoningTraceAnalyzer(
        hardcoded_problem_ids=hardcoded_ids,
        use_llm_judge=args.with_llm_judge,
        llm_judge_model=args.llm_judge_model,
    )

    # 4. Analyze each result
    analyses: List[Tuple[Dict[str, Any], ReasoningTraceAnalysis]] = []
    for i, result in enumerate(results):
        analysis = analyzer.analyze(result)
        analyses.append((result, analysis))
        if (i + 1) % 100 == 0:
            logger.info("Analyzed %d / %d results", i + 1, len(results))

    logger.info("Analysis complete: %d results", len(analyses))

    # 5. Aggregate effectiveness
    aggregated = aggregate_effectiveness(analyses)

    # 6. Compute summary
    summary = compute_summary_stats(analyses, aggregated)

    # 7. Print & save
    print_console_summary(summary)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.results_dir / f"validation_analysis_{timestamp}.json"

    save_results(output_path, analyses, aggregated, summary)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
