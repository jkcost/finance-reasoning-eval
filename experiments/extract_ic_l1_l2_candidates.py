"""Extract IC-L1/L2 Hard Example candidates.

CIKM Short 연구질문 [A2] Hard Example 생성 파이프라인의 첫 단계.
batch_transformations JSON에서 IC-L1(10x typo)/IC-L2(단위 불일치) 성공 케이스를
필터링하고, Python solution 치환 후 새 정답 재계산 가능성을 점수화한다.

Candidate 선별 기준:
    1. transformations.<IC-L1|IC-L2>.success == True
    2. is_hardcoded_solution == False (하드코딩 solution은 치환 불가)
    3. python_solution에서 AST로 추출한 숫자 리터럴과
       removed_or_modified 텍스트의 숫자가 최소 1개 이상 매칭

출력:
    experiments/results/metacognitive/hard_example_candidates.json
        {
          "metadata": {...},
          "candidates": [
            {
              "question_id": "test-2009",
              "ic_level": "IC-L1" | "IC-L2",
              "original_value": "500000000",
              "modified_value": "5000000000",
              "modified_expression": "5 billion (per revised statement)",
              "solution_literals": [500000000, 1.5, ...],
              "match_score": 0.75,  # 0~1 (0=치환 거의 불가, 1=명확)
              "needs_human_review": true
            },
            ...
          ]
        }

Usage:
    python experiments/extract_ic_l1_l2_candidates.py \\
        --input experiments/results/metacognitive/batch_transformations_0_238.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


IC_LEVELS = ("IC-L1", "IC-L2")
# Numeric literal pattern: 123, 12.3, 1.2e3, 1,234.56, with optional %/M/B suffix matching.
# We strip commas and suffixes before parsing.
NUMBER_PATTERN = re.compile(
    r"(?<![a-zA-Z_])"
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class Candidate:
    """A single IC-L1/L2 hard-example candidate for new-answer computation."""

    question_id: str
    ic_level: str
    original_value: str
    modified_snippet: str
    solution_literals: list[float]
    matched_literal: float | None
    match_score: float
    needs_human_review: bool
    reason: str


def extract_solution_literals(python_solution: str) -> list[float]:
    """Extract numeric literals from python_solution via AST.

    Returns sorted unique list. Handles int, float, and UnaryOp(-x).
    Skips 0 and 1 (common sentinels that produce false matches).
    """
    if not python_solution or not python_solution.strip():
        return []

    try:
        tree = ast.parse(python_solution)
    except SyntaxError:
        logger.debug(f"AST parse failed for solution (truncated): {python_solution[:80]!r}")
        return []

    literals: set[float] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            literals.add(float(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = node.operand
            if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                literals.add(-float(operand.value))

    # Filter out trivial literals that match too broadly.
    filtered = {v for v in literals if abs(v) not in (0.0, 1.0)}
    return sorted(filtered)


def extract_text_numbers(text: str) -> list[float]:
    """Extract numeric values from free-form text (e.g., removed_or_modified description).

    Handles thousand separators. Returns unique sorted list.
    """
    if not text:
        return []
    values: set[float] = set()
    for match in NUMBER_PATTERN.findall(text):
        cleaned = match.replace(",", "")
        try:
            values.add(float(cleaned))
        except ValueError:
            continue
    return sorted(v for v in values if abs(v) not in (0.0, 1.0))


def best_literal_match(
    solution_literals: list[float], text_numbers: list[float]
) -> tuple[float | None, float]:
    """Score match between solution literals and text numbers.

    Returns (best_matching_literal, score).

    Score tiers:
        1.0   — exact value match
        0.8   — match within 1% (thousand-separator / float rounding)
        0.6   — match within 10x scale (IC-L1 typo signature: solution_lit * 10 ≈ text_num)
        0.4   — match within 1.5x scale (IC-L2/L3 multiplier signature)
        0.0   — no useful match
    """
    if not solution_literals or not text_numbers:
        return None, 0.0

    best_lit: float | None = None
    best_score = 0.0
    for lit in solution_literals:
        if lit == 0:
            continue
        for num in text_numbers:
            ratio = abs(num / lit) if lit else 0.0
            if abs(num - lit) < 1e-9:
                score = 1.0
            elif abs(ratio - 1.0) < 0.01:
                score = 0.8
            elif 9.0 <= ratio <= 11.0 or 1 / 11.0 <= ratio <= 1 / 9.0:
                score = 0.6
            elif 1.3 <= ratio <= 1.7 or 1 / 1.7 <= ratio <= 1 / 1.3:
                score = 0.4
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_lit = lit
    return best_lit, best_score


def build_candidate(
    problem: dict[str, Any], ic_level: str
) -> Candidate | None:
    """Build a Candidate record for a single problem × IC level.

    Returns None if the transformation is unsuccessful or solution is unusable.
    """
    transforms = problem.get("transformations", {})
    tx = transforms.get(ic_level)
    if not tx or not tx.get("success"):
        return None

    if problem.get("is_hardcoded_solution"):
        return Candidate(
            question_id=problem.get("question_id", "?"),
            ic_level=ic_level,
            original_value="",
            modified_snippet=tx.get("removed_or_modified", ""),
            solution_literals=[],
            matched_literal=None,
            match_score=0.0,
            needs_human_review=False,
            reason="hardcoded_solution — cannot recompute answer",
        )

    python_solution = problem.get("python_solution", "")
    removed_desc = tx.get("removed_or_modified", "") or ""

    solution_literals = extract_solution_literals(python_solution)
    text_numbers = extract_text_numbers(removed_desc)

    matched, score = best_literal_match(solution_literals, text_numbers)

    if score == 0.0:
        reason = "no_literal_match"
        needs_review = False
    elif score >= 0.6:
        reason = "strong_match"
        needs_review = True
    else:
        reason = "weak_match"
        needs_review = True

    return Candidate(
        question_id=problem.get("question_id", "?"),
        ic_level=ic_level,
        original_value=str(matched) if matched is not None else "",
        modified_snippet=removed_desc[:300],
        solution_literals=solution_literals[:20],
        matched_literal=matched,
        match_score=round(score, 2),
        needs_human_review=needs_review,
        reason=reason,
    )


def extract_candidates(batch: dict[str, Any]) -> list[Candidate]:
    """Walk batch_transformations JSON and emit candidates."""
    problems = batch.get("problems", [])
    logger.info(f"Scanning {len(problems)} problems for IC-L1/L2 candidates")

    candidates: list[Candidate] = []
    for problem in problems:
        for ic_level in IC_LEVELS:
            cand = build_candidate(problem, ic_level)
            if cand is not None:
                candidates.append(cand)
    return candidates


def summarize(candidates: list[Candidate]) -> dict[str, Any]:
    """Build summary statistics for candidates."""
    by_level: dict[str, dict[str, int]] = {}
    for cand in candidates:
        bucket = by_level.setdefault(
            cand.ic_level,
            {"total": 0, "strong": 0, "weak": 0, "no_match": 0, "hardcoded": 0},
        )
        bucket["total"] += 1
        if cand.reason == "strong_match":
            bucket["strong"] += 1
        elif cand.reason == "weak_match":
            bucket["weak"] += 1
        elif cand.reason == "hardcoded_solution — cannot recompute answer":
            bucket["hardcoded"] += 1
        else:
            bucket["no_match"] += 1
    return by_level


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract IC-L1/L2 hard example candidates")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to batch_transformations_<start>_<end>.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/hard_example_candidates.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="Minimum match_score to include (default: 0.4, covers 1.5x IC-L2 signature)",
    )
    return parser.parse_args()


def main() -> int:
    """Run candidate extraction."""
    args = parse_args()
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    batch = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = extract_candidates(batch)
    summary = summarize(candidates)

    filtered = [c for c in candidates if c.match_score >= args.min_score]
    logger.info(f"Total candidates: {len(candidates)} | Above threshold ({args.min_score}): {len(filtered)}")
    for level, stats in summary.items():
        logger.info(f"  {level}: {stats}")

    output = {
        "metadata": {
            "input": str(args.input),
            "min_score": args.min_score,
            "total_candidates": len(candidates),
            "filtered_count": len(filtered),
            "by_level": summary,
        },
        "candidates": [asdict(c) for c in filtered],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
