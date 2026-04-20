"""
Reverse Calculation Detector

EA-full 변환에서 삭제된 컬럼의 값이 남은 컬럼들로부터 역산 가능한지 탐지.

역산 가능 = 변환이 문제를 unsolvable로 만들지 못함 = LLM이 정답을 맞출 수 있음.
이런 케이스를 Human Review에서 경고 표시하여 리뷰어의 집중 확인 유도.

역산 탐지 방법:
  1. python_solution에서 변수 할당 관계 추출
  2. 삭제된 컬럼의 값이 할당된 변수를 찾음
  3. 해당 변수가 다른 변수들의 산술식으로 표현 가능한지 확인

예시:
  python_solution:
    issued = 500
    treasury = 100
    outstanding = issued - treasury  # outstanding = 400

  EA-full이 "outstanding" 컬럼을 삭제하면:
    → outstanding = issued - treasury 로 역산 가능
    → 경고: "삭제된 값이 다른 컬럼에서 역산 가능 (outstanding = issued - treasury)"

Usage:
    from evaluation.reverse_calc_detector import detect_reverse_calculable
    result = detect_reverse_calculable(
        removed_key="outstanding",
        python_solution=solution_code,
        remaining_keys=["issued", "treasury"],
    )
    if result["is_reversible"]:
        print(f"경고: {result['explanation']}")
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def detect_reverse_calculable(
    removed_key: str,
    python_solution: str,
    remaining_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect if a removed column's value can be reverse-calculated.

    Args:
        removed_key: The column/key name that was removed by EA-full.
        python_solution: The ground-truth python solution code.
        remaining_keys: List of keys still present in the context (optional).

    Returns:
        {
            "is_reversible": bool,
            "confidence": str,  # "high", "medium", "low"
            "explanation": str,  # human-readable explanation
            "formula": str,     # the derivation formula if found
            "variables_used": List[str],  # variables needed for reverse calc
        }
    """
    if not python_solution or not removed_key:
        return _not_reversible()

    removed_lower = removed_key.lower().replace(" ", "_")

    # Parse variable assignments from solution
    assignments = _parse_assignments(python_solution)
    if not assignments:
        return _not_reversible()

    # Find the variable that corresponds to the removed key
    target_var = _find_matching_variable(removed_lower, assignments)
    if not target_var:
        return _not_reversible()

    target_rhs = assignments[target_var]["rhs"]
    target_deps = assignments[target_var]["deps"]

    # Check if target is a simple literal (not computed from other vars)
    if not target_deps:
        # Value is hardcoded, not derived from other variables
        # Check if ANY other variable's formula uses this target
        reverse_formulas = _find_reverse_formulas(target_var, assignments)
        if reverse_formulas:
            formula, deps = reverse_formulas[0]
            return {
                "is_reversible": True,
                "confidence": "high",
                "explanation": (
                    f"'{removed_key}'의 값은 다른 변수 관계식에서 역산 가능: {formula}"
                ),
                "formula": formula,
                "variables_used": list(deps),
            }
        return _not_reversible()

    # Target IS computed from other variables — it can be forward-calculated
    if target_deps:
        if remaining_keys is not None and remaining_keys:
            remaining_lower = {k.lower().replace(" ", "_") for k in remaining_keys}
            all_available = all(
                _any_key_matches(dep, remaining_lower, assignments)
                for dep in target_deps
            )
            if all_available:
                return {
                    "is_reversible": True,
                    "confidence": "high",
                    "explanation": (
                        f"'{removed_key}' = {target_rhs} (풀이 코드에 명시된 공식). "
                        f"필요한 변수 {target_deps}가 모두 context에 남아있음"
                    ),
                    "formula": f"{target_var} = {target_rhs}",
                    "variables_used": list(target_deps),
                }
        else:
            # No remaining_keys info — still flag if formula exists
            return {
                "is_reversible": True,
                "confidence": "medium",
                "explanation": (
                    f"'{removed_key}' = {target_rhs} (풀이 코드에 공식 존재). "
                    f"의존 변수 {target_deps}가 context에 남아있으면 역산 가능"
                ),
                "formula": f"{target_var} = {target_rhs}",
                "variables_used": list(target_deps),
            }

    # Check common financial relationships
    common_result = _check_common_financial_relationships(removed_lower, assignments)
    if common_result:
        return common_result

    return _not_reversible()


def _not_reversible() -> Dict[str, Any]:
    return {
        "is_reversible": False,
        "confidence": "none",
        "explanation": "",
        "formula": "",
        "variables_used": [],
    }


def _parse_assignments(code: str) -> Dict[str, Dict[str, Any]]:
    """Parse variable assignments from Python solution code.

    Returns:
        {var_name: {"rhs": str, "value": float|None, "deps": set[str]}}
    """
    assignments = {}
    for line in code.strip().split("\n"):
        line = line.strip()
        if line.startswith("#") or line.startswith("def ") or not line:
            continue
        if line.startswith("return"):
            continue

        m = re.match(r"(\w+)\s*=\s*(?!=)(.+)", line)
        if not m:
            continue

        var_name = m.group(1)
        rhs = m.group(2).strip()

        # Extract numeric value if simple literal
        value = None
        try:
            value = float(rhs.replace(",", "").replace("_", ""))
        except ValueError:
            pass

        # Extract dependencies (other variable names used in RHS)
        deps = set()
        # Remove string literals and numbers to find variable references
        cleaned = re.sub(r"['\"][^'\"]*['\"]", "", rhs)  # remove strings
        cleaned = re.sub(r"\d+\.?\d*", "", cleaned)  # remove numbers
        for token in re.findall(r"\b([a-zA-Z_]\w*)\b", cleaned):
            if token not in (
                "abs",
                "round",
                "int",
                "float",
                "sum",
                "len",
                "min",
                "max",
                "True",
                "False",
                "None",
            ):
                deps.add(token)

        assignments[var_name] = {"rhs": rhs, "value": value, "deps": deps}

    return assignments


def _find_matching_variable(removed_key: str, assignments: Dict) -> Optional[str]:
    """Find the variable name that corresponds to a removed column key."""
    # Direct match
    if removed_key in assignments:
        return removed_key

    # Fuzzy match: key words overlap
    removed_words = set(removed_key.split("_"))
    best_match = None
    best_score = 0

    for var_name in assignments:
        var_words = set(var_name.lower().split("_"))
        overlap = len(removed_words & var_words)
        if overlap > best_score and overlap >= 1:
            best_score = overlap
            best_match = var_name

    return best_match


def _find_reverse_formulas(target_var: str, assignments: Dict) -> List[tuple]:
    """Find formulas where target_var can be derived from other variables.

    E.g., if A = B - C, then C = B - A (reverse).
    """
    results = []
    for var_name, info in assignments.items():
        if var_name == target_var:
            continue
        if target_var in info["deps"]:
            # This variable uses the target
            # Check if it's a simple arithmetic relationship
            rhs = info["rhs"]
            # Pattern: var = target_var OP other OR var = other OP target_var
            # If var = a + b, then a = var - b
            # If var = a - b, then b = a - var
            # If var = a * b, then a = var / b
            ops = ["+", "-", "*", "/"]
            for op in ops:
                parts = [p.strip() for p in rhs.split(op)]
                if len(parts) == 2:
                    if parts[0] == target_var:
                        other = parts[1]
                        inverse_op = _inverse_op(op, position="left")
                        formula = f"{target_var} = {var_name} {inverse_op} {other}"
                        results.append((formula, {var_name, other}))
                    elif parts[1] == target_var:
                        other = parts[0]
                        inverse_op = _inverse_op(op, position="right")
                        formula = f"{target_var} = {other} {inverse_op} {var_name}"
                        results.append((formula, {var_name, other}))
    return results


def _inverse_op(op: str, position: str) -> str:
    """Get inverse operation for reverse calculation."""
    if position == "left":
        # var = target + other → target = var - other
        return {"+": "-", "-": "+", "*": "/", "/": "*"}.get(op, op)
    else:
        # var = other - target → target = other - var
        # var = other / target → target = other / var
        return {"+": "-", "-": "-", "*": "/", "/": "/"}.get(op, op)


def _any_key_matches(dep: str, remaining_lower: Set[str], assignments: Dict) -> bool:
    """Check if a dependency variable can be found in remaining keys."""
    if dep in remaining_lower:
        return True
    # Check if dep is a computed variable (not from context)
    if dep in assignments and assignments[dep]["value"] is not None:
        return True  # It's a literal, not from context
    # Fuzzy match
    dep_words = set(dep.split("_"))
    for key in remaining_lower:
        key_words = set(key.split("_"))
        if len(dep_words & key_words) >= 1:
            return True
    return False


def _check_common_financial_relationships(
    removed_key: str, assignments: Dict
) -> Optional[Dict[str, Any]]:
    """Check common financial identity relationships.

    Common identities:
      - net_income = revenue - expenses
      - equity = assets - liabilities
      - outstanding = issued - treasury
      - gross_profit = revenue - cogs
      - operating_income = gross_profit - opex
    """
    identities = [
        (
            {"net_income", "profit"},
            {"revenue", "total_revenue"},
            {"expense", "cost", "cogs"},
        ),
        (
            {"equity", "shareholders_equity"},
            {"asset", "total_asset"},
            {"liabilit", "total_liabilit"},
        ),
        ({"outstanding"}, {"issued", "authorized"}, {"treasury"}),
        ({"gross_profit"}, {"revenue", "sales"}, {"cogs", "cost_of_goods"}),
        (
            {"operating_income", "operating_profit"},
            {"gross_profit"},
            {"operating_expense", "opex"},
        ),
        (
            {"eps", "earnings_per_share"},
            {"net_income", "earnings"},
            {"shares", "shares_outstanding"},
        ),
    ]

    for target_patterns, comp_a_patterns, comp_b_patterns in identities:
        if not any(pat in removed_key for pat in target_patterns):
            continue

        # Check if both components are present in assignments
        has_a = any(
            any(pat in var.lower() for pat in comp_a_patterns) for var in assignments
        )
        has_b = any(
            any(pat in var.lower() for pat in comp_b_patterns) for var in assignments
        )

        if has_a and has_b:
            return {
                "is_reversible": True,
                "confidence": "medium",
                "explanation": (
                    f"'{removed_key}'는 일반적인 금융 항등식으로 역산 가능할 수 있음 "
                    f"(예: {list(target_patterns)[0]} = {list(comp_a_patterns)[0]} - {list(comp_b_patterns)[0]})"
                ),
                "formula": f"financial identity: {list(target_patterns)[0]} = f({list(comp_a_patterns)[0]}, {list(comp_b_patterns)[0]})",
                "variables_used": list(comp_a_patterns | comp_b_patterns),
            }

    return None


def analyze_batch_reversibility(
    problems: List[Dict],
) -> Dict[str, Dict[str, Any]]:
    """Analyze all EA-full transformations in a batch for reverse calculability.

    Args:
        problems: List of problem dicts from batch_transformations JSON.

    Returns:
        {question_id: {
            "is_reversible": bool,
            "details": reverse_calc_result,
            "removed_key": str,
        }}
    """
    results = {}

    for problem in problems:
        qid = problem.get("question_id", "")
        ea_full = problem.get("transformations", {}).get("EA-full", {})

        if not ea_full.get("success"):
            continue

        description = ea_full.get("description", "")
        # Extract removed key from description: "Removed column: KEY_NAME"
        removed_key = ""
        m = re.search(
            r"Removed (?:column|key|row)[s]?:\s*(.+)", description, re.IGNORECASE
        )
        if m:
            removed_key = m.group(1).strip()

        if not removed_key:
            continue

        python_solution = problem.get("python_solution", "")
        remaining_keys = []
        # Parse remaining keys from context_original (JSON)
        try:
            import json

            ctx = problem.get("context_original", "")
            if ctx.startswith("{"):
                ctx_dict = json.loads(ctx)
                remaining_keys = [k for k in ctx_dict.keys() if k != removed_key]
        except (json.JSONDecodeError, TypeError):
            pass

        result = detect_reverse_calculable(
            removed_key=removed_key,
            python_solution=python_solution,
            remaining_keys=remaining_keys,
        )

        if result["is_reversible"]:
            results[qid] = {
                "is_reversible": True,
                "details": result,
                "removed_key": removed_key,
            }
            logger.info(f"  [역산 가능] {qid}: {removed_key} — {result['explanation']}")

    return results
