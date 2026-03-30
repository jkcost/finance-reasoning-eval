"""
Conflict Salience Scorer

Extracts features from IC transformation results and computes how
"obvious" the conflict is. Used for Fig 3 regression analysis:
logistic regression of salience features → refusal rate.

Features:
  - token_distance: tokens between conflicting values in context
  - same_paragraph: both values in same paragraph
  - authority_marker: count of authority keywords near conflict
  - magnitude_ratio: ratio of conflicting to original value

No API calls needed — operates on existing experimental data.

Reference: Design Doc (office-hours), CEO Plan 2026-03-30
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Authority keywords that make models more likely to accept conflicting values
AUTHORITY_KEYWORDS = [
    "audited",
    "auditor",
    "report",
    "reported",
    "filing",
    "filed",
    "official",
    "confirmed",
    "certified",
    "verified",
    "restated",
    "revised",
    "annual report",
    "10-K",
    "10-Q",
    "SEC",
]


@dataclass
class SalienceFeatures:
    """Features describing how salient (obvious) an IC conflict is."""

    question_id: str
    ic_level: str
    token_distance: int = 0
    same_paragraph: bool = True
    authority_marker_count: int = 0
    magnitude_ratio: float = 1.5
    has_explicit_discrepancy_note: bool = False
    original_value: Optional[float] = None
    conflicting_value: Optional[float] = None

    @property
    def salience_score(self) -> float:
        """Aggregate salience score (0-1). Higher = more obvious conflict."""
        score = 0.0

        # Token distance: closer values are more salient
        if self.token_distance < 50:
            score += 0.3
        elif self.token_distance < 150:
            score += 0.15

        # Same paragraph: more salient
        if self.same_paragraph:
            score += 0.2

        # Magnitude ratio: larger difference is more obvious
        ratio_diff = abs(self.magnitude_ratio - 1.0)
        if ratio_diff >= 5.0:  # L1: 10x
            score += 0.3
        elif ratio_diff >= 0.3:  # L3: 1.5x
            score += 0.1

        # Authority markers reduce salience (model trusts the source)
        if self.authority_marker_count > 0:
            score -= min(0.2, self.authority_marker_count * 0.05)

        # Explicit discrepancy note increases salience
        if self.has_explicit_discrepancy_note:
            score += 0.2

        return max(0.0, min(1.0, score))


def extract_salience_features(
    question_id: str,
    transformation_description: str,
    transformed_context: str,
    ic_level: str = "",
) -> SalienceFeatures:
    """Extract salience features from a single IC transformation.

    Args:
        question_id: Problem identifier.
        transformation_description: Output from transform_type5_*() description.
        transformed_context: The transformed context string.
        ic_level: IC difficulty level label (e.g., "IC-L1: Obvious Typo").

    Returns:
        SalienceFeatures dataclass with computed features.
    """
    features = SalienceFeatures(question_id=question_id, ic_level=ic_level)

    # Extract original and conflicting values from description
    original, conflicting = _parse_values_from_description(transformation_description)
    if original is not None and conflicting is not None:
        features.original_value = original
        features.conflicting_value = conflicting
        if original != 0:
            features.magnitude_ratio = conflicting / original

    # Token distance between the two values in context
    if original is not None and conflicting is not None:
        features.token_distance = _compute_token_distance(
            transformed_context, original, conflicting
        )

    # Same paragraph check
    if original is not None and conflicting is not None:
        features.same_paragraph = _in_same_paragraph(
            transformed_context, original, conflicting
        )

    # Authority marker count near conflict area
    features.authority_marker_count = _count_authority_markers(
        transformed_context, transformation_description
    )

    # Explicit discrepancy note
    discrepancy_patterns = [
        r"discrepancy",
        r"conflicting",
        r"inconsisten",
        r"contradicts",
        r"differs from",
        r"does not match",
    ]
    for pat in discrepancy_patterns:
        if re.search(pat, transformed_context, re.IGNORECASE):
            features.has_explicit_discrepancy_note = True
            break

    return features


def _parse_values_from_description(
    description: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Extract original and conflicting numeric values from transformation description."""
    if not description:
        return None, None

    # Pattern: "Replaced X with Y" or "X vs Y" or "X → Y"
    patterns = [
        r"([\d,.]+)\s*(?:→|->|vs|with)\s*([\d,.]+)",
        r"original[=:]\s*([\d,.]+).*?(?:conflict|contra|stated)[=:]\s*([\d,.]+)",
    ]

    for pat in patterns:
        m = re.search(pat, description)
        if m:
            try:
                v1 = float(m.group(1).replace(",", ""))
                v2 = float(m.group(2).replace(",", ""))
                return v1, v2
            except ValueError:
                continue

    return None, None


def _compute_token_distance(
    context: str, val1: float, val2: float
) -> int:
    """Compute approximate token distance between two values in context."""
    # Find positions of both values
    val1_strs = [str(val1), f"{val1:,.0f}", f"{val1:.2f}"]
    val2_strs = [str(val2), f"{val2:,.0f}", f"{val2:.2f}"]

    pos1 = _find_value_position(context, val1_strs)
    pos2 = _find_value_position(context, val2_strs)

    if pos1 == -1 or pos2 == -1:
        return 999  # Unknown distance

    # Approximate tokens as words between positions
    between = context[min(pos1, pos2) : max(pos1, pos2)]
    return len(between.split())


def _find_value_position(context: str, val_strs: List[str]) -> int:
    """Find the first position of any value representation in context."""
    for vs in val_strs:
        pos = context.find(vs)
        if pos != -1:
            return pos
    return -1


def _in_same_paragraph(context: str, val1: float, val2: float) -> bool:
    """Check if both values appear in the same paragraph."""
    paragraphs = context.split("\n\n")
    val1_strs = [str(val1), f"{val1:,.0f}", f"{val1:.2f}"]
    val2_strs = [str(val2), f"{val2:,.0f}", f"{val2:.2f}"]

    for para in paragraphs:
        has_v1 = any(vs in para for vs in val1_strs)
        has_v2 = any(vs in para for vs in val2_strs)
        if has_v1 and has_v2:
            return True

    return False


def _count_authority_markers(context: str, description: str) -> int:
    """Count authority keywords near the conflict area."""
    count = 0
    context_lower = context.lower()
    for keyword in AUTHORITY_KEYWORDS:
        count += context_lower.count(keyword.lower())
    return count


def score_batch_results(
    results: List[Dict[str, Any]],
) -> List[SalienceFeatures]:
    """Score salience for a batch of IC evaluation results.

    Args:
        results: List of evaluation result dicts, each containing:
            - question_id
            - transformation_type (must contain "IC")
            - transformation_description
            - context (transformed)

    Returns:
        List of SalienceFeatures for IC results only.
    """
    features_list = []

    for r in results:
        t_type = r.get("transformation_type", "")
        if "IC" not in t_type:
            continue

        features = extract_salience_features(
            question_id=r.get("question_id", "unknown"),
            transformation_description=r.get("transformation_description", ""),
            transformed_context=r.get("context", ""),
            ic_level=t_type,
        )
        features_list.append(features)

    return features_list


def compute_regression_data(
    salience_features: List[SalienceFeatures],
    evaluation_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepare data for logistic regression: salience features → refusal outcome.

    Args:
        salience_features: Output from score_batch_results().
        evaluation_results: Evaluation results with response_type field.

    Returns:
        List of dicts with features + refusal outcome, ready for sklearn.
    """
    # Build lookup: question_id → response_type
    response_map: Dict[str, str] = {}
    for r in evaluation_results:
        qid = r.get("question_id", "")
        response_map[qid] = r.get("response_type", "confident")

    regression_data = []
    for sf in salience_features:
        response = response_map.get(sf.question_id, "confident")
        is_refusal = 1 if response in ("refused", "caveat") else 0

        regression_data.append({
            "question_id": sf.question_id,
            "ic_level": sf.ic_level,
            "token_distance": sf.token_distance,
            "same_paragraph": int(sf.same_paragraph),
            "authority_marker_count": sf.authority_marker_count,
            "magnitude_ratio": sf.magnitude_ratio,
            "has_discrepancy_note": int(sf.has_explicit_discrepancy_note),
            "salience_score": sf.salience_score,
            "refusal": is_refusal,
        })

    return regression_data
