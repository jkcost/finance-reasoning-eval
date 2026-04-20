"""
Transformation Intensity Metrics

변환의 강도를 수치화하여 C1/C2/C3 결과와의 상관관계를 분석.

Metrics:
    - CVRR (Critical Value Removal Ratio): 풀이 필수 값 중 제거된 비율
    - IRR (Information Removal Ratio): 전체 수치 중 제거된 비율
    - CEDR (Character Edit Distance Ratio): 정규화된 편집 거리
    - QCIO (Question-Context Info Overlap): question 값이 context에 남아있는 비율
"""

import re
from typing import Dict, Set, Tuple


def extract_numbers(text: str) -> Set[str]:
    """Extract all meaningful numbers from text.

    Returns normalized string representations for set comparison.
    Ignores single-digit numbers and common noise (e.g., ordinals).
    """
    if not text:
        return set()

    raw = re.findall(r"[\$]?([\d,]+\.?\d*)", text)
    result = set()
    for n in raw:
        clean = n.replace(",", "")
        # Skip single digits, empty, and likely noise
        if len(clean.replace(".", "")) < 2:
            continue
        # Normalize: remove trailing .0
        try:
            val = float(clean)
            if val == int(val) and "." not in clean:
                result.add(str(int(val)))
            else:
                result.add(str(val))
        except ValueError:
            result.add(clean)
    return result


def extract_solution_values(python_solution: str) -> Set[str]:
    """Extract numerical values used in python_solution code.

    Parses variable assignments, function arguments, and literals.
    """
    if not python_solution:
        return set()

    # Find all number literals in code
    raw = re.findall(r"(?<![a-zA-Z_])(\d+\.?\d*(?:e[+-]?\d+)?)", python_solution)
    result = set()
    for n in raw:
        try:
            val = float(n)
            if val == 0:
                continue
            if val == int(val) and "." not in n and "e" not in n.lower():
                result.add(str(int(val)))
            else:
                result.add(str(val))
        except ValueError:
            continue
    return result


def critical_value_removal_ratio(
    original_context: str,
    transformed_context: str,
    python_solution: str,
    question: str,
) -> Tuple[float, Dict]:
    """CVRR: Fraction of solution-required values removed by transformation.

    Returns:
        (ratio, details) where details includes value sets for debugging.
    """
    # Get solution values
    solution_vals = extract_solution_values(python_solution)
    if not solution_vals:
        # Fallback: use question numbers as proxy
        solution_vals = extract_numbers(question)

    if not solution_vals:
        return 0.0, {"solution_values": set(), "removed": set(), "note": "no_values"}

    # What was removed?
    orig_nums = extract_numbers(original_context)
    trans_nums = extract_numbers(transformed_context)
    removed_nums = orig_nums - trans_nums

    # Which solution-critical values were removed?
    critical_removed = solution_vals & removed_nums

    ratio = len(critical_removed) / len(solution_vals) if solution_vals else 0.0

    return ratio, {
        "solution_values": solution_vals,
        "removed_from_context": removed_nums,
        "critical_removed": critical_removed,
        "ratio": ratio,
    }


def info_removal_ratio(
    original_context: str, transformed_context: str
) -> Tuple[float, Dict]:
    """IRR: Fraction of numerical values removed."""
    orig_nums = extract_numbers(original_context)
    trans_nums = extract_numbers(transformed_context)

    if not orig_nums:
        return 0.0, {"original_count": 0, "removed_count": 0}

    removed = orig_nums - trans_nums
    added = trans_nums - orig_nums  # For IC: values added
    ratio = len(removed) / len(orig_nums)

    return ratio, {
        "original_count": len(orig_nums),
        "transformed_count": len(trans_nums),
        "removed_count": len(removed),
        "added_count": len(added),
        "removed_values": removed,
        "added_values": added,
        "ratio": ratio,
    }


def char_edit_distance_ratio(original: str, transformed: str) -> float:
    """CEDR: Normalized character-level edit distance.

    Uses simple length-based approximation to avoid Levenshtein dependency.
    For exact distance, install python-Levenshtein.
    """
    if not original:
        return 0.0

    # Approximate: use length difference + differing character count
    max_len = max(len(original), len(transformed))
    if max_len == 0:
        return 0.0

    # Simple approach: ratio of length change
    len_diff = abs(len(original) - len(transformed))

    # Count character-level overlap for better accuracy
    min_len = min(len(original), len(transformed))
    matches = sum(1 for i in range(min_len) if original[i] == transformed[i])
    mismatches = min_len - matches + len_diff

    return mismatches / max_len


def question_context_info_overlap(
    question: str, transformed_context: str
) -> Tuple[float, Dict]:
    """QCIO: Fraction of question's numerical values still in transformed context.

    High QCIO = question already has the data = transformation may be ineffective.
    """
    q_nums = extract_numbers(question)
    if not q_nums:
        return 0.0, {"question_nums": set(), "overlap": set()}

    c_nums = extract_numbers(transformed_context)
    overlap = q_nums & c_nums

    ratio = len(overlap) / len(q_nums)
    return ratio, {
        "question_nums": q_nums,
        "context_nums": c_nums,
        "overlap": overlap,
        "ratio": ratio,
    }


def question_leakage_score(
    question: str,
    original_context: str,
    transformed_context: str,
    python_solution: str,
) -> Tuple[float, str]:
    """Detect if removed data is also present in the question text.

    Measures: what fraction of context-removed values appear in the question?
    High score = transformation is ineffective because question leaks the data.

    Returns:
        (score, category) where:
        - score 0.0 = no leakage (removed data is not in question)
        - score 1.0 = complete leakage (all removed data is also in question)
        - category: 'none', 'partial', 'full'
    """
    # What was removed from context?
    orig_nums = extract_numbers(original_context)
    trans_nums = extract_numbers(transformed_context)
    removed_nums = orig_nums - trans_nums

    if not removed_nums:
        return 0.0, "none"

    # How many removed values appear in the question?
    q_nums = extract_numbers(question)
    leaked = removed_nums & q_nums

    score = len(leaked) / len(removed_nums) if removed_nums else 0.0

    if score >= 0.5:
        category = "full"
    elif score > 0:
        category = "partial"
    else:
        category = "none"

    return score, category


def compute_all_metrics(
    original_context: str,
    transformed_context: str,
    question: str,
    python_solution: str,
) -> Dict:
    """Compute all intensity metrics for a single transformation."""
    cvrr, cvrr_detail = critical_value_removal_ratio(
        original_context, transformed_context, python_solution, question
    )
    irr, irr_detail = info_removal_ratio(original_context, transformed_context)
    cedr = char_edit_distance_ratio(original_context, transformed_context)
    qcio, qcio_detail = question_context_info_overlap(question, transformed_context)
    leakage, leakage_cat = question_leakage_score(
        question, original_context, transformed_context, python_solution
    )

    return {
        "cvrr": round(cvrr, 4),
        "irr": round(irr, 4),
        "cedr": round(cedr, 4),
        "qcio": round(qcio, 4),
        "leakage_score": round(leakage, 4),
        "leakage_category": leakage_cat,
        "details": {
            "cvrr": {
                k: list(v) if isinstance(v, set) else v for k, v in cvrr_detail.items()
            },
            "irr": {
                k: list(v) if isinstance(v, set) else v for k, v in irr_detail.items()
            },
            "qcio": {
                k: list(v) if isinstance(v, set) else v for k, v in qcio_detail.items()
            },
        },
    }
