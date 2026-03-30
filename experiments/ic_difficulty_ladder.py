"""
IC Conflict Difficulty Ladder (L1-L5)

Extends the base IC transformation with 5 difficulty levels:
  L1 — Obvious Typo: 10x digit/decimal error (rule-based)
  L2 — Unit Mismatch: different units for same quantity (LLM-assisted)
  L3 — Authority Conflict: authoritative source contradicts data (existing IC)
  L4 — Cross-Period Conflict: quarterly sums don't match annual (LLM-assisted)
  L5 — Implicit Ratio: financial ratios are internally inconsistent (LLM-assisted + manual)

Each transform function follows the standard signature:
    (context, question, ...) -> (new_context, description, expected_behavior)
Returns (None, None, None) when the level is not applicable.

Reference: CEO Plan 2026-03-30, Design Doc (office-hours)
"""

import json
import logging
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

try:
    from experiments.apply_transformations_full import (
        _extract_numbers_from_text,
        _is_descriptor_number,
        detect_context_type,
        normalize_context,
    )
except ModuleNotFoundError:
    from apply_transformations_full import (
        _extract_numbers_from_text,
        _is_descriptor_number,
        detect_context_type,
        normalize_context,
    )

logger = logging.getLogger(__name__)

# ============================================================================
# IC DIFFICULTY LEVEL LABELS
# ============================================================================

LABEL_IC_L1 = "IC-L1: Obvious Typo"
LABEL_IC_L2 = "IC-L2: Unit Mismatch"
LABEL_IC_L3 = "IC-L3: Authority Conflict"
LABEL_IC_L4 = "IC-L4: Cross-Period Conflict"
LABEL_IC_L5 = "IC-L5: Implicit Ratio Inconsistency"

IC_LEVELS = [LABEL_IC_L1, LABEL_IC_L2, LABEL_IC_L3, LABEL_IC_L4, LABEL_IC_L5]


# ============================================================================
# L1: OBVIOUS TYPO (rule-based, 10x digit error)
# ============================================================================


def transform_ic_l1_text(
    context: str, question: str, python_solution: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L1: Insert a 10x digit error into a numeric value.

    The error is obvious enough that basic numerical literacy should catch it.
    This is a baseline for IC detection, not true metacognition.
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    matches = list(re.finditer(number_pattern, context))

    if not matches:
        return None, None, None

    q_nums = _extract_numbers_from_text(question)

    # Find a suitable target: not in question, not a descriptor, reasonably large
    best_match = None
    for m in matches:
        val_str = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            val = float(val_str)
        except ValueError:
            continue
        if val <= 1.0:
            continue
        val_norm = str(int(val)) if val == int(val) else str(val)
        if val_norm in q_nums:
            continue
        if _is_descriptor_number(val, context, m):
            continue
        best_match = m
        break

    if best_match is None:
        return None, None, None

    original_str = best_match.group(0)
    val_str_clean = original_str.replace("$", "").replace(",", "").replace("%", "")
    try:
        num_val = float(val_str_clean)
    except ValueError:
        return None, None, None

    # 10x error: multiply or divide by 10
    typo_val = num_val * 10

    # Format in same style
    has_dollar = "$" in original_str
    has_percent = "%" in original_str
    has_decimal = "." in val_str_clean

    if has_decimal:
        decimal_places = len(val_str_clean.split(".")[-1])
        typo_formatted = f"{typo_val:.{decimal_places}f}"
    else:
        typo_formatted = f"{int(typo_val):,}"

    if has_dollar:
        typo_formatted = f"${typo_formatted}"
    if has_percent:
        typo_formatted = f"{typo_formatted}%"

    # Replace at exact match position (not first occurrence in context)
    start, end = best_match.start(), best_match.end()
    new_context = context[:start] + typo_formatted + context[end:]

    return (
        new_context,
        f"L1 typo: {original_str} → {typo_formatted} (10x error)",
        "Model should detect the obviously wrong numeric value",
    )


def transform_ic_l1_json(
    context_dict: Dict, question: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L1: Insert a 10x digit error into a JSON context value."""
    question_lower = question.lower()
    new_context = deepcopy(context_dict)

    for key in list(context_dict.keys()):
        key_lower = key.lower()
        if key_lower in question_lower and isinstance(context_dict[key], dict):
            years = list(context_dict[key].keys())
            if years:
                target_year = years[-1]
                original_val = context_dict[key][target_year]
                try:
                    num_val = float(str(original_val).replace(",", "").replace("%", ""))
                    typo_val = num_val * 10
                    new_context[key][target_year] = typo_val
                    return (
                        new_context,
                        f"L1 typo: {key}[{target_year}] {original_val} → {typo_val} (10x)",
                        "Model should detect the obviously wrong numeric value",
                    )
                except (ValueError, TypeError):
                    continue

    return None, None, None


# ============================================================================
# L2: UNIT MISMATCH
# ============================================================================

# Common financial unit pairs for mismatch generation
_UNIT_PAIRS = [
    ("million", "billion", 1000),
    ("thousand", "million", 1000),
    ("%", " basis points", 100),
]


def transform_ic_l2_text(
    context: str, question: str, python_solution: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L2: Create a unit mismatch for the same quantity.

    Inserts a sentence that restates a value with a different unit,
    creating an implicit contradiction that requires unit awareness.
    """
    # Look for values with explicit units
    unit_patterns = [
        (r"\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)", "currency_scale"),
        (r"([\d,]+(?:\.\d+)?)%", "percentage"),
    ]

    for pattern, unit_type in unit_patterns:
        matches = list(re.finditer(pattern, context, re.IGNORECASE))
        for m in matches:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue

            if val <= 0:
                continue

            # Skip if value appears in question
            if val_str in question:
                continue

            if unit_type == "currency_scale":
                original_unit = m.group(2).lower()
                for u1, u2, factor in _UNIT_PAIRS:
                    if original_unit == u1:
                        # Apply 1.5x error on top of unit conversion to create actual conflict
                        mismatch_val = val / factor * 1.5
                        mismatch_sentence = (
                            f"According to a separate filing, this figure was "
                            f"reported as {mismatch_val:,.2f} {u2}."
                        )
                        # Insert after the sentence containing the match
                        sentences = re.split(r"(?<=[.!?])\s+", context)
                        for i, sent in enumerate(sentences):
                            if m.group(0) in sent:
                                new_sentences = (
                                    sentences[: i + 1]
                                    + [mismatch_sentence]
                                    + sentences[i + 1 :]
                                )
                                new_context = " ".join(new_sentences)
                                return (
                                    new_context,
                                    f"L2 unit mismatch: {val} {u1} vs {mismatch_val:.2f} {u2}",
                                    "Model should detect conflicting units for the same quantity",
                                )
                        break
                    elif original_unit == u2:
                        mismatch_val = val * factor * 1.5
                        mismatch_sentence = (
                            f"According to a separate filing, this figure was "
                            f"reported as {mismatch_val:,.0f} {u1}."
                        )
                        sentences = re.split(r"(?<=[.!?])\s+", context)
                        for i, sent in enumerate(sentences):
                            if m.group(0) in sent:
                                new_sentences = (
                                    sentences[: i + 1]
                                    + [mismatch_sentence]
                                    + sentences[i + 1 :]
                                )
                                new_context = " ".join(new_sentences)
                                return (
                                    new_context,
                                    f"L2 unit mismatch: {val} {u2} vs {mismatch_val:.0f} {u1}",
                                    "Model should detect conflicting units for the same quantity",
                                )
                        break

            elif unit_type == "percentage":
                # % vs basis points (with 1.5x error to create actual conflict)
                bps_val = val * 100 * 1.5
                mismatch_sentence = (
                    f"Note: An alternative source reports this rate as "
                    f"{bps_val:.0f} basis points."
                )
                sentences = re.split(r"(?<=[.!?])\s+", context)
                for i, sent in enumerate(sentences):
                    if m.group(0) in sent:
                        new_sentences = (
                            sentences[: i + 1]
                            + [mismatch_sentence]
                            + sentences[i + 1 :]
                        )
                        new_context = " ".join(new_sentences)
                        return (
                            new_context,
                            f"L2 unit mismatch: {val}% vs {bps_val:.0f} basis points",
                            "Model should detect conflicting units for the same quantity",
                        )

    return None, None, None


# ============================================================================
# L3: AUTHORITY CONFLICT (delegates to existing IC implementation)
# ============================================================================


def transform_ic_l3_text(
    context: str, question: str, python_solution: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L3: Authority conflict — existing IC implementation with 1.5x multiplier.

    Delegates to the proven transform_type5_text logic, re-labeled as L3.
    """
    try:
        from experiments.apply_transformations_full import transform_type5_text
    except ModuleNotFoundError:
        from apply_transformations_full import transform_type5_text

    result = transform_type5_text(context, question, python_solution)
    if result[0] is None:
        return None, None, None
    return (
        result[0],
        f"L3 authority: {result[1]}",
        result[2],
    )


def transform_ic_l3_json(
    context_dict: Dict, question: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L3: Authority conflict for JSON context."""
    try:
        from experiments.apply_transformations_full import transform_type5_json
    except ModuleNotFoundError:
        from apply_transformations_full import transform_type5_json

    result = transform_type5_json(context_dict, question)
    if result[0] is None:
        return None, None, None
    return (
        result[0],
        f"L3 authority: {result[1]}",
        result[2],
    )


def transform_ic_l3_markdown(
    context: str, question: str, python_solution: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L3: Authority conflict for markdown context."""
    try:
        from experiments.apply_transformations_full import transform_type5_markdown
    except ModuleNotFoundError:
        from apply_transformations_full import transform_type5_markdown

    result = transform_type5_markdown(context, question, python_solution)
    if result[0] is None:
        return None, None, None
    return (
        result[0],
        f"L3 authority: {result[1]}",
        result[2],
    )


# ============================================================================
# L4: CROSS-PERIOD CONFLICT (summation inconsistency)
# ============================================================================


def transform_ic_l4_json(
    context_dict: Dict, question: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L4: Make periodic values inconsistent with their annual total.

    Finds a key with multiple year entries where values could represent
    a time series, then modifies one entry to break summation consistency.
    """
    question_lower = question.lower()
    new_context = deepcopy(context_dict)

    for key in list(context_dict.keys()):
        if not isinstance(context_dict[key], dict):
            continue

        years = list(context_dict[key].keys())
        if len(years) < 3:
            continue

        # Try to find numeric values that could be a series
        numeric_years = {}
        for y in years:
            try:
                v = float(str(context_dict[key][y]).replace(",", "").replace("%", ""))
                numeric_years[y] = v
            except (ValueError, TypeError):
                continue

        if len(numeric_years) < 3:
            continue

        # Modify the second-to-last year by 30% to create summation inconsistency
        sorted_years = sorted(numeric_years.keys())
        target_year = sorted_years[-2]
        original_val = numeric_years[target_year]

        if original_val == 0:
            continue

        modified_val = original_val * 1.3
        new_context[key][target_year] = modified_val

        # Add a "total" key that equals the sum of ORIGINAL values
        original_sum = sum(numeric_years.values())
        new_context[key]["_annual_total"] = original_sum

        return (
            new_context,
            f"L4 cross-period: {key}[{target_year}] {original_val} → {modified_val:.2f}, "
            f"total={original_sum} (now inconsistent)",
            "Model should detect that periodic values do not sum to the annual total",
        )

    return None, None, None


def transform_ic_l4_text(
    context: str, question: str, python_solution: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L4: Insert a summation inconsistency into text context.

    Adds an annual total that contradicts the sum of individual period values.
    """
    # Look for series of numbers that could be periodic values
    number_pattern = r"\$?([\d,]+(?:\.\d+)?)"
    matches = list(re.finditer(number_pattern, context))

    if len(matches) < 3:
        return None, None, None

    # Collect numeric values that are similar in magnitude (likely a series)
    values = []
    for m in matches:
        val_str = m.group(1).replace(",", "")
        try:
            val = float(val_str)
            if val > 0 and not _is_descriptor_number(val, context, m):
                values.append((val, m))
        except ValueError:
            continue

    if len(values) < 3:
        return None, None, None

    # Filter to values within same order of magnitude (likely from same series)
    base_val = values[0][0]
    series_candidates = [
        (v, m)
        for v, m in values
        if 0.01 <= v / base_val <= 100  # within 2 orders of magnitude
    ]
    if len(series_candidates) < 3:
        return None, None, None

    # Take first 3-4 same-magnitude values
    series = series_candidates[:4]
    series_sum = sum(v for v, _ in series)
    fake_total = series_sum * 0.85  # 15% off from actual sum

    total_sentence = (
        f"The total for the reporting period was {fake_total:,.2f}, "
        f"as confirmed in the annual filing."
    )

    new_context = context + " " + total_sentence

    return (
        new_context,
        f"L4 cross-period: added fake total {fake_total:.2f} (actual sum: {series_sum:.2f})",
        "Model should detect that the stated total does not match the sum of individual values",
    )


# ============================================================================
# L5: IMPLICIT RATIO INCONSISTENCY
# ============================================================================


def transform_ic_l5_text(
    context: str, question: str, python_solution: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """L5: Create an implicit ratio inconsistency.

    Inserts values that are individually plausible but mathematically
    inconsistent when checked against common financial relationships.
    E.g., profit margin × revenue ≠ stated profit.

    This is the hardest IC level because no single value is obviously wrong.
    """
    # Look for pairs of related financial metrics
    # Pattern: find revenue and profit/margin mentions
    revenue_pattern = r"revenue[s]?\s+(?:of|was|is|were|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?"
    profit_pattern = r"(?:net\s+)?(?:income|profit|earnings)\s+(?:of|was|is|were|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?"
    margin_pattern = (
        r"(?:profit|net|operating)\s+margin\s+(?:of|was|is|were|:)?\s*([\d.]+)%"
    )

    revenue_match = re.search(revenue_pattern, context, re.IGNORECASE)
    profit_match = re.search(profit_pattern, context, re.IGNORECASE)
    margin_match = re.search(margin_pattern, context, re.IGNORECASE)

    if revenue_match and profit_match:
        try:
            rev = float(revenue_match.group(1).replace(",", ""))
            profit = float(profit_match.group(1).replace(",", ""))

            if rev <= 0:
                return None, None, None

            actual_margin = (profit / rev) * 100
            # Insert a contradictory margin that is 2x the actual
            fake_margin = actual_margin * 2
            if fake_margin > 95:
                fake_margin = actual_margin * 0.3

            margin_sentence = (
                f"The company reported a net profit margin of {fake_margin:.1f}% "
                f"for the period."
            )
            new_context = context + " " + margin_sentence

            return (
                new_context,
                f"L5 ratio: revenue={rev}, profit={profit}, "
                f"actual margin={actual_margin:.1f}%, stated margin={fake_margin:.1f}%",
                "Model should detect that stated margin is inconsistent with revenue and profit figures",
            )
        except (ValueError, ZeroDivisionError):
            return None, None, None

    if revenue_match and margin_match:
        try:
            rev = float(revenue_match.group(1).replace(",", ""))
            margin = float(margin_match.group(1))

            if rev <= 0 or margin <= 0:
                return None, None, None

            implied_profit = rev * margin / 100
            fake_profit = implied_profit * 1.8  # 80% off

            profit_sentence = f"Net income for the period totaled ${fake_profit:,.2f}."
            new_context = context + " " + profit_sentence

            return (
                new_context,
                f"L5 ratio: revenue={rev}, margin={margin}%, "
                f"implied profit={implied_profit:.2f}, stated profit={fake_profit:.2f}",
                "Model should detect that stated profit is inconsistent with revenue and margin",
            )
        except (ValueError, ZeroDivisionError):
            return None, None, None

    return None, None, None


# ============================================================================
# ORCHESTRATION: Apply all IC levels to a problem
# ============================================================================


def apply_ic_ladder(
    example: Dict,
    levels: Optional[List[str]] = None,
) -> List[Dict]:
    """Apply IC difficulty ladder transformations to a single problem.

    Args:
        example: Problem dict with 'question', 'context', 'python_solution', etc.
        levels: List of levels to apply (default: all). E.g., ["L1", "L3", "L5"]

    Returns:
        List of transformed problem dicts, each with 'transformation_type' set
        to the IC level label (e.g., "IC-L1: Obvious Typo").
    """
    if levels is None:
        levels = ["L1", "L2", "L3", "L4", "L5"]

    context_raw = example.get("context", "")
    if not context_raw:
        return []

    context = normalize_context(str(context_raw))
    question = example.get("question", "")
    python_solution = example.get("python_solution", "")
    context_type = detect_context_type(context)

    results = []

    level_transforms = {
        "L1": _apply_l1,
        "L2": _apply_l2,
        "L3": _apply_l3,
        "L4": _apply_l4,
        "L5": _apply_l5,
    }

    for level in levels:
        fn = level_transforms.get(level)
        if fn is None:
            logger.warning(f"Unknown IC level: {level}")
            continue

        transformed = fn(context, context_type, question, python_solution, example)
        if transformed is not None:
            results.append(transformed)

    return results


def _apply_l1(
    context: str,
    context_type: str,
    question: str,
    python_solution: str,
    example: Dict,
) -> Optional[Dict]:
    if context_type == "json":
        try:
            ctx_dict = json.loads(context) if isinstance(context, str) else context
            new_ctx, desc, expected = transform_ic_l1_json(ctx_dict, question)
            if new_ctx is not None:
                return _build_result(
                    example, json.dumps(new_ctx), desc, expected, LABEL_IC_L1
                )
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        new_ctx, desc, expected = transform_ic_l1_text(
            context, question, python_solution
        )
        if new_ctx is not None:
            return _build_result(example, new_ctx, desc, expected, LABEL_IC_L1)
    return None


def _apply_l2(
    context: str,
    context_type: str,
    question: str,
    python_solution: str,
    example: Dict,
) -> Optional[Dict]:
    # L2 only works on text/markdown (needs unit keywords)
    if context_type in ("text", "markdown"):
        new_ctx, desc, expected = transform_ic_l2_text(
            context, question, python_solution
        )
        if new_ctx is not None:
            return _build_result(example, new_ctx, desc, expected, LABEL_IC_L2)
    return None


def _apply_l3(
    context: str,
    context_type: str,
    question: str,
    python_solution: str,
    example: Dict,
) -> Optional[Dict]:
    if context_type == "json":
        try:
            ctx_dict = json.loads(context) if isinstance(context, str) else context
            new_ctx, desc, expected = transform_ic_l3_json(ctx_dict, question)
            if new_ctx is not None:
                ctx_str = json.dumps(new_ctx) if isinstance(new_ctx, dict) else new_ctx
                return _build_result(example, ctx_str, desc, expected, LABEL_IC_L3)
        except (json.JSONDecodeError, TypeError):
            pass
    elif context_type == "markdown":
        new_ctx, desc, expected = transform_ic_l3_markdown(
            context, question, python_solution
        )
        if new_ctx is not None:
            return _build_result(example, new_ctx, desc, expected, LABEL_IC_L3)
    else:
        new_ctx, desc, expected = transform_ic_l3_text(
            context, question, python_solution
        )
        if new_ctx is not None:
            return _build_result(example, new_ctx, desc, expected, LABEL_IC_L3)
    return None


def _apply_l4(
    context: str,
    context_type: str,
    question: str,
    python_solution: str,
    example: Dict,
) -> Optional[Dict]:
    if context_type == "json":
        try:
            ctx_dict = json.loads(context) if isinstance(context, str) else context
            new_ctx, desc, expected = transform_ic_l4_json(ctx_dict, question)
            if new_ctx is not None:
                return _build_result(
                    example, json.dumps(new_ctx), desc, expected, LABEL_IC_L4
                )
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        new_ctx, desc, expected = transform_ic_l4_text(
            context, question, python_solution
        )
        if new_ctx is not None:
            return _build_result(example, new_ctx, desc, expected, LABEL_IC_L4)
    return None


def _apply_l5(
    context: str,
    context_type: str,
    question: str,
    python_solution: str,
    example: Dict,
) -> Optional[Dict]:
    # L5 requires extractable financial relationships — text/markdown only
    if context_type in ("text", "markdown"):
        new_ctx, desc, expected = transform_ic_l5_text(
            context, question, python_solution
        )
        if new_ctx is not None:
            return _build_result(example, new_ctx, desc, expected, LABEL_IC_L5)
    return None


def _build_result(
    example: Dict,
    new_context: str,
    description: str,
    expected_behavior: str,
    label: str,
) -> Dict:
    """Build a transformation result dict in standard format."""
    result = deepcopy(example)
    result["context"] = new_context
    result["transformation_type"] = label
    result["transformation_description"] = description
    result["expected_behavior"] = expected_behavior
    return result
