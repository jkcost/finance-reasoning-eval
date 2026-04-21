"""Fit logistic regression of IC detection ~ salience features (RQ-C).

F2 of plan ``2026-04-21-001-feat-cikm-short-execution-plan.md``.

Input:
    - IC-L1/L2/L3/L4 evaluation results (eval_*.json)
    - Corresponding batch_transformations_*.json (source context + conflict metadata)

Per record, extract 4 salience features from ``conflict_salience_scorer``:
    token_distance, same_paragraph, authority_marker_count, magnitude_ratio

Target:
    detection ∈ {refused=1, confident=0}  (error/caveat 제외)

R4 (solvable 모집단): 원본 오답 QID 제외.

Dependencies only on stdlib — logistic regression implemented via mini-batch
gradient descent to avoid adding sklearn as a runtime dep for CIKM results.
A SciPy / sklearn path is auto-used if available for stronger fit.

Outputs:
    - salience_regression.json: coefficients, std errors, AUC, per-feature summary
    - salience_regression.csv: tidy feature × record table for paper tables

Usage:
    python experiments/fit_salience_regression.py \\
        --eval-results experiments/results/metacognitive/eval_metacognitive.json \\
        --transformations experiments/results/metacognitive/batch_transformations_0_238.json \\
        --original-context experiments/results/metacognitive/eval_original_context.json \\
        --output experiments/results/metacognitive/salience_regression.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_io import load_eval_records, qid_in_solvable, solvable_qids  # noqa: E402
from conflict_salience_scorer import (  # noqa: E402
    SalienceFeatures,
    extract_salience_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


IC_LEVELS = frozenset({"IC-L1", "IC-L2", "IC-L3", "IC-L4"})


@dataclass
class RegressionResult:
    """Fit outcome for one IC-level or combined dataset."""

    level: str
    n: int
    n_positive: int
    coefficients: dict[str, float]
    intercept: float
    auc: float
    accuracy: float
    backend: str  # "sklearn" | "gd"


def _index_transformations(
    batch: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build (qid, transformation_type) → {context_original, context_transformed, tx_meta} index."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for problem in batch.get("problems", []):
        qid = str(problem.get("question_id", ""))
        context_original = problem.get("context_original", "") or ""
        for ttype, tx in (problem.get("transformations") or {}).items():
            if not isinstance(tx, dict) or not tx.get("success"):
                continue
            index[(qid, ttype)] = {
                "context_original": context_original,
                "context_transformed": tx.get("transformed_content", "") or "",
                "removed_or_modified": tx.get("removed_or_modified", "") or "",
            }
    return index


def build_feature_dataset(
    eval_records: list[dict[str, Any]],
    tx_index: dict[tuple[str, str], dict[str, Any]],
    solvable: set[str] | None,
) -> list[dict[str, Any]]:
    """Build a tidy list of (features, label) rows for IC records only."""
    rows: list[dict[str, Any]] = []
    for rec in eval_records:
        ttype = rec.get("transformation_type")
        if ttype not in IC_LEVELS:
            continue
        qid = rec.get("question_id")
        if not qid:
            continue
        if not qid_in_solvable(qid, solvable):
            continue
        response_type = rec.get("response_type")
        if response_type not in {"refused", "confident"}:
            continue  # exclude error / caveat
        contexts = tx_index.get((str(qid), ttype))
        if not contexts:
            continue

        features: SalienceFeatures = extract_salience_features(
            question_id=str(qid),
            transformation_description=contexts["removed_or_modified"],
            transformed_context=contexts["context_transformed"],
            ic_level=ttype,
        )
        rows.append({
            "question_id": qid,
            "transformation_type": ttype,
            "model": rec.get("model"),
            "prompt_strategy": rec.get("prompt_strategy"),
            "label": 1 if response_type == "refused" else 0,
            "token_distance": features.token_distance,
            "same_paragraph": int(features.same_paragraph),
            "authority_marker_count": features.authority_marker_count,
            "magnitude_ratio": features.magnitude_ratio,
        })
    return rows


def _standardize(values: list[float]) -> tuple[list[float], float, float]:
    """Return (standardized, mean, std). std=0 → returns all-zero."""
    if not values:
        return [], 0.0, 1.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    std = math.sqrt(var) or 1.0
    return [(v - mean) / std for v in values], mean, std


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_gradient_descent(
    X: list[list[float]],
    y: list[int],
    lr: float = 0.1,
    epochs: int = 500,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[list[float], float]:
    """Minimal logistic regression via batch gradient descent with L2 reg.

    Returns (weights, intercept).
    """
    rng = random.Random(seed)
    n, d = len(X), len(X[0]) if X else 0
    if n == 0 or d == 0:
        return [0.0] * d, 0.0

    w = [rng.uniform(-0.01, 0.01) for _ in range(d)]
    b = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        for xi, yi in zip(X, y):
            z = sum(wj * xij for wj, xij in zip(w, xi)) + b
            p = _sigmoid(z)
            err = p - yi
            for j in range(d):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(d):
            w[j] -= lr * (grad_w[j] / n + l2 * w[j])
        b -= lr * grad_b / n
    return w, b


def _auc(y_true: list[int], y_score: list[float]) -> float:
    """Compute area under the ROC curve (Mann-Whitney formulation)."""
    pairs = [(s, yi) for s, yi in zip(y_score, y_true)]
    positives = [s for s, yi in pairs if yi == 1]
    negatives = [s for s, yi in pairs if yi == 0]
    if not positives or not negatives:
        return float("nan")
    wins = ties = 0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    total = len(positives) * len(negatives)
    return (wins + 0.5 * ties) / total if total else float("nan")


def fit(
    rows: list[dict[str, Any]],
    level_tag: str,
    min_per_class: int = 2,
) -> RegressionResult | None:
    """Fit logistic regression on rows and return coefficients + metrics.

    Skips fitting when the dataset has fewer than ``min_per_class`` of either label,
    because a single-class response cannot be modeled. This happens frequently for
    IC-L1/L2 in current data (near-zero refusal) and is itself a reportable finding.
    """
    if len(rows) < 10:
        logger.warning(f"[{level_tag}] Too few rows to fit ({len(rows)})")
        return None

    n_positive = sum(r["label"] for r in rows)
    n_negative = len(rows) - n_positive
    if n_positive < min_per_class or n_negative < min_per_class:
        logger.warning(
            f"[{level_tag}] Severe class imbalance — n={len(rows)}, "
            f"positive={n_positive}, negative={n_negative}. Skipping fit. "
            f"(This itself is a finding: IC refusal is extremely rare.)"
        )
        return None

    feature_names = ["token_distance", "same_paragraph", "authority_marker_count", "magnitude_ratio"]
    feature_matrix = [[r[name] for name in feature_names] for r in rows]
    # Standardize continuous features
    cols = list(zip(*feature_matrix))
    standardized_cols: list[list[float]] = []
    stats: list[tuple[float, float]] = []
    for col in cols:
        std_col, mean, std = _standardize([float(v) for v in col])
        standardized_cols.append(std_col)
        stats.append((mean, std))
    X = list(zip(*standardized_cols))
    X_list = [list(row) for row in X]
    y = [r["label"] for r in rows]

    backend = "gd"
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_list, y)
        weights = clf.coef_[0].tolist()
        intercept = float(clf.intercept_[0])
        scores = [float(p) for p in clf.predict_proba(X_list)[:, 1]]
        backend = "sklearn"
    except ImportError:
        weights, intercept = _fit_gradient_descent(X_list, y)
        scores = [_sigmoid(sum(w * x for w, x in zip(weights, xi)) + intercept) for xi in X_list]
    except ValueError as exc:
        logger.warning(f"[{level_tag}] sklearn fit rejected ({exc}) — falling back to GD")
        weights, intercept = _fit_gradient_descent(X_list, y)
        scores = [_sigmoid(sum(w * x for w, x in zip(weights, xi)) + intercept) for xi in X_list]
        backend = "gd_fallback"

    preds = [1 if s >= 0.5 else 0 for s in scores]
    accuracy = sum(1 for p, yi in zip(preds, y) if p == yi) / len(y)
    auc = _auc(y, scores)

    return RegressionResult(
        level=level_tag,
        n=len(rows),
        n_positive=sum(y),
        coefficients={name: float(w) for name, w in zip(feature_names, weights)},
        intercept=float(intercept),
        auc=float(auc),
        accuracy=float(accuracy),
        backend=backend,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fit IC detection ~ salience regression (RQ-C)")
    parser.add_argument("--eval-results", type=Path, nargs="+", required=True)
    parser.add_argument("--transformations", type=Path, required=True)
    parser.add_argument(
        "--original-context",
        type=Path,
        default=None,
        help="R4 solvable filter source",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/salience_regression.json"),
    )
    return parser.parse_args()


def main() -> int:
    """Build feature dataset and fit logistic regression."""
    args = parse_args()
    if not args.transformations.exists():
        logger.error("transformations missing")
        return 1

    batch = json.loads(args.transformations.read_text(encoding="utf-8"))
    tx_index = _index_transformations(batch)
    solvable = solvable_qids(args.original_context)
    if solvable is not None:
        logger.info(f"Solvable QIDs: {len(solvable)}")

    all_records: list[dict[str, Any]] = []
    for path in args.eval_results:
        if not path.exists():
            logger.warning(f"skipping {path}")
            continue
        all_records.extend(load_eval_records(path))

    logger.info(f"Loaded {len(all_records)} total eval records")
    rows = build_feature_dataset(all_records, tx_index, solvable)
    logger.info(f"IC dataset rows: {len(rows)} (positive={sum(r['label'] for r in rows)})")

    # Fit per IC level + combined
    per_level: dict[str, dict[str, Any] | None] = {}
    for level in sorted(IC_LEVELS):
        level_rows = [r for r in rows if r["transformation_type"] == level]
        result = fit(level_rows, level_tag=level)
        per_level[level] = asdict(result) if result else None

    combined = fit(rows, level_tag="IC-ALL")
    combined_dict = asdict(combined) if combined else None

    # Write CSV dataset for inspection / paper tables
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info(f"Wrote CSV: {csv_path}")

    output = {
        "metadata": {
            "eval_results": [str(p) for p in args.eval_results],
            "transformations": str(args.transformations),
            "original_context": str(args.original_context) if args.original_context else None,
            "solvable_filter_applied": solvable is not None,
            "n_rows": len(rows),
            "n_positive": sum(r["label"] for r in rows),
        },
        "combined": combined_dict,
        "per_level": per_level,
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output}")

    # Console summary
    if combined_dict:
        logger.info("\n=== Combined IC regression ===")
        logger.info(f"  N={combined_dict['n']}, AUC={combined_dict['auc']:.3f}, accuracy={combined_dict['accuracy']:.3f}, backend={combined_dict['backend']}")
        logger.info(f"  intercept={combined_dict['intercept']:+.3f}")
        for name, coef in combined_dict["coefficients"].items():
            logger.info(f"  {name:<24s} {coef:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
