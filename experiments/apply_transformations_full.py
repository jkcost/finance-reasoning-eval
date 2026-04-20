"""
Apply transformations to the FULL FinanceReasoning dataset.

Transformation Taxonomy (v2)
============================

Theoretical Framework
---------------------
Transformations are organized along two orthogonal axes:

| Metacognitive Ability | Signal Strong (Explicit) | Signal Absent (Silent) |
|-----------------------|--------------------------|------------------------|
| **Absence Detection** | EA-partial / EA-full     | SA                     |
| **Conflict Detection**| —                        | IC                     |
| *(Auxiliary) Ambiguity*| —                       | *TA*                   |

References:
  - AbstentionBench (Feng+2024): Underspecified Context vs Contradictory Data
  - Wen+2024 (EMNLP): context perturbation classified by signal strength
  - CheckList (Ribeiro+2020): INV/DIR test — ability separation by type

Main Types (4)
--------------
  - EA-partial (Explicit Absence — Partial):
      Removes a specific value with an explicit marker ([DATA MISSING], N/A).
      Baseline — easiest detection task; lower bound of metacognitive ability.
  - EA-full (Explicit Absence — Full):
      Deletes an entire key/column, removing a whole data dimension.
      Structural absence without markers but with visible schema change.
  - SA (Silent Absence):
      Silently removes data without any markers.
      JSON: empties dict values; Markdown: removes data rows;
      Text: deletes sentences containing question-relevant data.
      Hardest absence type — context reads naturally but critical info is gone.
  - IC (Information Conflict):
      Inserts contradictory data (1.5× value) for the same data point.
      Tests a fundamentally different cognitive process from absence detection.

Auxiliary Type (1)
------------------
  - TA (Temporal Ambiguity):
      Replaces a specific year in the question with "the end of the period".
      Only applicable to ~11 hard problems. Reported separately as auxiliary.

Objectivity & Reproducibility
-----------------------------
- All transformations are **fully deterministic** (no random elements).
- EA/SA are **question-driven**: targets selected by matching question text.
- IC uses an **absolute rule**: last year/row × 1.5 multiplier.
- `validate_transformation()` runs the ground-truth Python solution against
  the transformed context to objectively verify unsolvability.
- `_solution_uses_hardcoded_values()` detects solutions that bypass context
  parsing, preventing false "still_solvable" validation results.

Legacy Label Mapping
--------------------
Previous experiments used "Type N" labels. The mapping is:
  Type 1: Information Removal        → EA-partial
  Type 2: Table Column Removal       → EA-full
  Type 3: Ambiguous Time Period      → TA
  Type 4: Critical Data Removal      → SA
  Type 5: Contradictory Information  → IC

Usage:
    python experiments/apply_transformations_full.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from hardcoded_solution_detector import _solution_uses_hardcoded_values


# ============================================================================
# QUESTION LEAKAGE PREVENTION HELPERS
# ============================================================================


def _extract_numbers_from_text(text: str) -> Set[str]:
    """Extract normalized number strings from text for comparison.

    Returns set of cleaned number strings (no $, no commas).
    Used to detect if a removed value also appears in the question.
    Normalizes integers consistently (75000.0 -> "75000") to avoid
    format-dependent mismatch between question and context.
    """
    if not text:
        return set()
    raw = re.findall(r"\$?([\d,]+(?:\.\d+)?)", text)
    result = set()
    for n in raw:
        clean = n.replace(",", "")
        if len(clean.replace(".", "")) < 2:
            continue
        try:
            val = float(clean)
            # Always normalize: if value is integer, store as int string
            if val == int(val):
                result.add(str(int(val)))
            else:
                result.add(str(val))
        except ValueError:
            result.add(clean)
    return result


def _extract_critical_solution_values(python_solution: str) -> Set[str]:
    """Extract only values that actually contribute to the final answer.

    Traces variable dependencies from the return/answer statement backwards
    to find which numeric assignments are computation-critical.
    Values assigned but not used in the answer chain are excluded.

    Example:
        current_price = 150    # assigned but NOT used in answer
        strike_price = 145     # used: intrinsic = strike - expiration
        answer = intrinsic * shares - premium * shares  # uses strike, not current
        => returns {'145'} but NOT {'150'}
    """
    if not python_solution:
        return set()

    lines = python_solution.strip().split("\n")

    # Step 1: Parse all variable assignments
    assignments: Dict[str, str] = {}  # var_name -> full RHS expression
    var_values: Dict[str, Set[str]] = {}  # var_name -> numeric values in assignment

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("#") or line_s.startswith("def ") or not line_s:
            continue
        # Match: var = expression (but not ==)
        m = re.match(r"(\w+)\s*=\s*(?!=)(.+)", line_s)
        if m:
            var_name = m.group(1)
            rhs = m.group(2)
            assignments[var_name] = rhs
            var_values[var_name] = _extract_numbers_from_text(rhs)

    # Step 2: Find the answer variable (return, answer =, result =)
    answer_vars: Set[str] = set()
    for line in reversed(lines):
        line_s = line.strip()
        if line_s.startswith("return "):
            expr = line_s[7:]
            # Find all variable references in return expression
            answer_vars.update(re.findall(r"\b([a-zA-Z_]\w*)\b", expr))
            break
        m = re.match(r"(answer|result)\s*=\s*(.+)", line_s)
        if m:
            expr = m.group(2)
            answer_vars.update(re.findall(r"\b([a-zA-Z_]\w*)\b", expr))
            break

    if not answer_vars:
        # Fallback: return all values
        return _extract_numbers_from_text(python_solution)

    # Step 3: Trace dependencies backwards (BFS)
    visited: Set[str] = set()
    queue = list(answer_vars)
    while queue:
        var = queue.pop(0)
        if var in visited:
            continue
        visited.add(var)
        if var in assignments:
            # Find all variables referenced in this assignment's RHS
            rhs = assignments[var]
            refs = re.findall(r"\b([a-zA-Z_]\w*)\b", rhs)
            for ref in refs:
                if ref in assignments and ref not in visited:
                    queue.append(ref)

    # Step 4: Collect numeric values from critical variables only
    critical_nums: Set[str] = set()
    for var in visited:
        if var in var_values:
            critical_nums.update(var_values[var])

    return (
        critical_nums if critical_nums else _extract_numbers_from_text(python_solution)
    )


def _value_in_question(value_str: str, question: str) -> bool:
    """Check if a numeric value from context also appears in the question.

    Normalizes both to handle format differences ($, commas, etc).
    """
    clean_val = value_str.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        val = float(clean_val)
    except ValueError:
        return clean_val in question

    q_nums = _extract_numbers_from_text(question)
    # Check normalized match
    if val == int(val):
        return str(int(val)) in q_nums
    return str(val) in q_nums


# ============================================================================
# TYPE LABELS & LEGACY MAPPING
# ============================================================================

LABEL_EA_PARTIAL = "EA-partial: Explicit Absence (Partial)"
LABEL_EA_FULL = "EA-full: Explicit Absence (Full)"
LABEL_SA = "SA: Silent Absence"
LABEL_IC = "IC: Information Conflict"
LABEL_TA = "TA: Temporal Ambiguity"

LEGACY_LABEL_MAP: Dict[str, str] = {
    "Type 1: Information Removal": LABEL_EA_PARTIAL,
    "Type 2: Table Column Removal": LABEL_EA_FULL,
    "Type 3: Ambiguous Time Period": LABEL_TA,
    "Type 4: Critical Data Removal": LABEL_SA,
    "Type 5: Contradictory Information": LABEL_IC,
}


def normalize_transformation_label(label: str) -> str:
    """Convert legacy 'Type N' label to new taxonomy, or return as-is."""
    return LEGACY_LABEL_MAP.get(label, label)


def normalize_context(context: str) -> str:
    """Normalize context stored as Python list-literal string.

    Some hard.json entries store markdown tables as "['| col1 | col2 |\\n|---|---|']"
    instead of actual multiline strings. This function detects and fixes that format.
    """
    s = context.strip()
    if s.startswith("['") and s.endswith("']"):
        inner = s[2:-2]
        # Replace literal \\n (two chars: backslash + n) with actual newlines
        inner = inner.replace("\\n", "\n")
        return inner
    return context


# ============================================================================
# MARKDOWN TABLE PARSER
# ============================================================================


def _parse_markdown_table(context: str) -> Dict[str, Any]:
    """Parse a markdown table into structured data with correct column alignment.

    Returns:
        {
            "header_idx": int,       # line index of header row
            "headers": List[str],    # column names (including row-label column)
            "separator_idx": int,    # line index of --- separator
            "data_rows": List[Dict], # [{line_idx, cells: List[str], raw: str}]
            "col_values": Dict[int, Set[str]],  # column index -> set of numeric strings
            "lines": List[str],      # all lines of context
        }
        Returns None if no valid table found.
    """
    lines = context.split("\n")
    header_idx = -1

    for i, line in enumerate(lines):
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            header_idx = i
            break

    if header_idx == -1:
        return None

    # Parse header — keep ALL columns including empty row-label column
    raw_headers = lines[header_idx].split("|")
    # Trim leading/trailing empty strings from | delimiters
    if raw_headers and raw_headers[0].strip() == "":
        raw_headers = raw_headers[1:]
    if raw_headers and raw_headers[-1].strip() == "":
        raw_headers = raw_headers[:-1]
    headers = [h.strip() for h in raw_headers]

    # Parse data rows with same column count
    data_rows = []
    col_values: Dict[int, Set[str]] = {i: set() for i in range(len(headers))}

    for line_idx in range(header_idx + 2, len(lines)):
        line = lines[line_idx]
        if "|" not in line:
            continue
        raw_cells = line.split("|")
        if raw_cells and raw_cells[0].strip() == "":
            raw_cells = raw_cells[1:]
        if raw_cells and raw_cells[-1].strip() == "":
            raw_cells = raw_cells[:-1]
        cells = [c.strip() for c in raw_cells]

        # Only include if has digits (data row)
        if not re.search(r"\d", line):
            continue

        data_rows.append(
            {
                "line_idx": line_idx,
                "cells": cells,
                "raw": line,
            }
        )

        for ci, cell in enumerate(cells):
            if ci < len(headers):
                col_values[ci] |= _extract_numbers_from_text(cell)

    return {
        "header_idx": header_idx,
        "headers": headers,
        "separator_idx": header_idx + 1,
        "data_rows": data_rows,
        "col_values": col_values,
        "lines": lines,
    }


def _remove_column_from_line(line: str, col_idx: int) -> str:
    """Remove a specific column from a markdown table line by index.

    Column indexing matches _parse_markdown_table: index 0 is the first
    column after the leading |.
    """
    if "|" not in line:
        return line
    raw_cells = line.split("|")
    has_leading = raw_cells[0].strip() == ""
    has_trailing = raw_cells[-1].strip() == ""

    # Work with inner cells only
    inner = raw_cells[1 if has_leading else 0 : -1 if has_trailing else len(raw_cells)]
    if col_idx < 0 or col_idx >= len(inner):
        return line

    inner = inner[:col_idx] + inner[col_idx + 1 :]

    result_parts = []
    if has_leading:
        result_parts.append("")
    result_parts.extend(inner)
    if has_trailing:
        result_parts.append("")
    return "|".join(result_parts)


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================


def transform_type1_json(context_dict: Dict, question: str) -> tuple:
    """Type 1: Information Removal for JSON context.

    Design rationale:
        Removes a specific year entry from a key mentioned in the question,
        simulating a missing data point in a financial time-series.
    Domain relevance:
        HIGH — assumes fiscal-year-keyed dict structure (e.g. {"Revenue": {"2022": 100}}).
    Objectivity:
        Deterministic; selects the first key×year pair that matches the question text.
    Detection difficulty:
        LOW — the key still exists but with a missing year, which is structurally visible.
    Known limitations:
        If the question references a key not present as a top-level dict key,
        no transformation is produced (returns None).
    """
    # Find key mentioned in question
    question_lower = question.lower()

    for key in context_dict.keys():
        key_lower = key.lower()
        if key_lower in question_lower:
            if isinstance(context_dict[key], dict):
                years = list(context_dict[key].keys())
                if years:
                    # Find year mentioned in question
                    for year_key in years:
                        if str(year_key) in question:
                            value = context_dict[key][year_key]
                            # Skip if the value also appears in question (leakage)
                            if _value_in_question(str(value), question):
                                continue
                            new_context = deepcopy(context_dict)
                            del new_context[key][year_key]
                            return (
                                new_context,
                                f"Removed critical data: {key}[{year_key}]",
                                f"Model should recognize missing {key} and refuse to answer",
                            )
                    # Fallback: remove a year NOT in question (still disrupts calculation)
                    for year_key in years:
                        value = context_dict[key][year_key]
                        if not _value_in_question(str(value), question):
                            new_context = deepcopy(context_dict)
                            del new_context[key][year_key]
                            return (
                                new_context,
                                f"Removed critical data: {key}[{year_key}]",
                                f"Model should recognize missing {key} and refuse to answer",
                            )

    return None, None, None


def transform_type2_json(context_dict: Dict, question: str) -> tuple:
    """Type 2: Table Column Removal for JSON context.

    Design rationale:
        Deletes the entire top-level key referenced by the question, simulating
        a missing column in a balance sheet or income statement.
    Domain relevance:
        HIGH — exploits the column-oriented structure of financial statements
        (BS/IS/CF) where each key represents a line item.
    Objectivity:
        Deterministic; removes the first key whose name appears in the question.
    Detection difficulty:
        MODERATE — no marker is left; the model must notice the key is absent.
    Known limitations:
        If the question references a concept not directly matching any key name,
        no transformation is produced.
    """
    question_lower = question.lower()

    for key in context_dict.keys():
        key_lower = key.lower()
        if key_lower in question_lower:
            # Remove entire column
            new_context = deepcopy(context_dict)
            del new_context[key]
            return (
                new_context,
                f"Removed column: {key}",
                f"Model should recognize missing column {key} and refuse to answer",
            )

    return None, None, None


def transform_type3_question(question: str) -> tuple:
    """Type 3: Ambiguous Time Period.

    Design rationale:
        Replaces the first 4-digit year in the question with "the end of the period",
        making the time reference ambiguous when the context contains multi-year data.
    Domain relevance:
        HIGH — financial calculations almost always require a specific fiscal period;
        ambiguity in period reference is a realistic data quality issue.
    Objectivity:
        Deterministic; replaces the first regex match of a 4-digit year (19xx/20xx).
    Detection difficulty:
        MODERATE — the model must realize "the period" is insufficiently specified
        when context contains data for multiple years.
    Known limitations:
        Questions with no year pattern produce no transformation. Questions with only
        one year of data in context may still be solvable (caught by validate_transformation).
    """
    # Find years in question
    year_pattern = r"\b(19|20)\d{2}\b"
    years = re.findall(year_pattern, question)

    if years:
        # Replace first year with vague term
        new_question = re.sub(year_pattern, "the end of the period", question, count=1)
        return (
            new_question,
            f"Made time reference ambiguous: {years[0]} -> vague term",
            "Model should recognize incomplete time period and refuse to answer",
        )

    return None, None, None


def transform_type3_context(context: str, question: str) -> tuple:
    """TA: Replace year references in context with ambiguous terms.

    Design rationale:
        When the question has no explicit year, fall back to modifying
        the context's temporal references. Replaces ONE year with an ambiguous
        term like "the reporting period", creating temporal ambiguity when
        the context contains data for multiple years/periods.
    Precondition:
        Context must contain at least 2 distinct years; replacing a year
        in single-year context doesn't create genuine ambiguity.
    Detection difficulty:
        MODERATE to HIGH — context reads naturally but the model cannot
        determine which period the data refers to.
    """
    year_pattern = r"\b((?:19|20)\d{2})\b"
    years_in_context = re.findall(year_pattern, context)

    if not years_in_context:
        return None, None, None

    unique_years = sorted(set(years_in_context))
    if len(unique_years) < 2:
        return None, None, None

    # Prefer a year mentioned in the question if available
    years_in_question = re.findall(year_pattern, question)
    target_year = None
    for y in years_in_question:
        if y in unique_years:
            target_year = y
            break

    if not target_year:
        # Pick the most recent year (most likely to be the answer target)
        target_year = unique_years[-1]

    replacement = "the reporting period"
    new_context = re.sub(r"\b" + target_year + r"\b", replacement, context)

    return (
        new_context,
        f"Replaced year {target_year} with '{replacement}' in context",
        "Model should recognize that the time period is ambiguous",
    )


def transform_type1_text(context: str, question: str) -> tuple:
    """Type 1: Information Removal for text context.

    Design rationale:
        Replaces a significant number in the text with "[DATA MISSING]",
        simulating an explicitly flagged data gap.
    Domain relevance:
        MEDIUM — number replacement is generic, but targets financial numeric data.
    Objectivity:
        Deterministic; prefers numbers NOT present in the question text
        to avoid question-leakage (where question already contains the value).
    Detection difficulty:
        LOW — the "[DATA MISSING]" marker is an obvious signal.
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    numbers = re.findall(number_pattern, context)

    if not numbers:
        return None, None, None

    def _num_value(n: str) -> float:
        """Parse numeric magnitude for prioritization."""
        clean = n.replace("$", "").replace(",", "").replace("%", "")
        try:
            return float(clean)
        except ValueError:
            return 0

    # Filter: skip numbers in question AND skip small contextual numbers (<= 31)
    # Small numbers are often descriptive (e.g., "12 days", "10 years")
    non_leaked = [n for n in numbers if not _value_in_question(n, question)]
    data_values = [n for n in non_leaked if _num_value(n) > 31]

    # Fallback chain: data_values > non_leaked > all numbers
    target_num = (
        data_values[0] if data_values else non_leaked[0] if non_leaked else numbers[0]
    )

    new_context = context.replace(target_num, "[DATA MISSING]", 1)
    return (
        new_context,
        f"Removed critical number: {target_num}",
        "Model should recognize missing data and refuse to answer",
    )


def transform_type2_text(context: str, question: str) -> tuple:
    """EA-full: Remove all question-relevant data sentences from text context.

    Design rationale:
        Removes ALL sentences containing both question keywords and numeric data.
        Simulates an entire data section being absent from a financial report.
        More aggressive than SA-text (which removes only one sentence).
    Detection difficulty:
        MODERATE — the remaining text is grammatically complete but lacks
        the full data section needed to solve the problem.
    Difference from SA-text:
        SA removes the single most relevant sentence; EA-full removes ALL
        sentences with question-relevant numeric data.
    """
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    if len(sentences) <= 2:
        return None, None, None

    question_lower = question.lower()
    keywords = [
        w
        for w in re.findall(r"\b\w+\b", question_lower)
        if w not in _SA_STOPWORDS and len(w) > 2 and not w.isdigit()
    ]

    if not keywords:
        return None, None, None

    q_nums = _extract_numbers_from_text(question)

    scored = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        keyword_hits = sum(1 for kw in keywords if kw in sent_lower)
        has_number = bool(re.search(r"\d+", sent))
        if keyword_hits > 0 and has_number:
            # Check if sentence has numbers NOT in the question
            sent_nums = _extract_numbers_from_text(sent)
            has_non_leaked = bool(sent_nums - q_nums)
            scored.append((i, keyword_hits, has_non_leaked))

    if not scored:
        return None, None, None

    # Only remove sentences that have non-leaked numbers (effective removal)
    # If all sentences only have leaked numbers, fall back to removing all
    effective = [(i, kw) for i, kw, nl in scored if nl]
    if effective:
        to_remove_set = {i for i, _ in effective}
    else:
        to_remove_set = {i for i, _, _ in scored}

    remaining = [s for i, s in enumerate(sentences) if i not in to_remove_set]

    # All sentences matched — keep the least-relevant one as residual context
    if not remaining:
        least_relevant_idx = min(scored, key=lambda x: x[1])[0]
        to_remove_set.discard(least_relevant_idx)
        remaining = [sentences[least_relevant_idx]]

    new_context = " ".join(remaining)
    return (
        new_context,
        f"Removed {len(to_remove_set)} data-containing sentences entirely",
        "Model should recognize that critical data sections are missing",
    )


def transform_type1_markdown(
    context: str, question: str, python_solution: str = ""
) -> tuple:
    """Type 1: Information Removal for markdown table.

    Design rationale:
        Replaces a numeric cell in a markdown table with "N/A",
        simulating a missing data point in a tabulated financial report.
    Domain relevance:
        HIGH — markdown tables directly mirror financial statement layouts.
    Objectivity:
        Deterministic; prioritizes cells that are (1) used in the solution
        and (2) not present in the question text.
    Detection difficulty:
        LOW — "N/A" is an explicit missing-data marker.
    """
    cell_pattern = r"\|\s*(\$?\d+(?:,\d{3})*(?:\.\d+)?)\s*\|"
    matches = list(re.finditer(cell_pattern, context))

    if not matches:
        return None, None, None

    # Use BFS-traced critical values for better targeting
    critical_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )
    sol_nums = _extract_numbers_from_text(python_solution) if python_solution else set()
    q_nums = _extract_numbers_from_text(question)

    def _cell_score(m: re.Match) -> tuple:
        val = m.group(1)
        val_norm = _extract_numbers_from_text(val)
        is_critical = bool(val_norm & critical_nums) if critical_nums else False
        is_in_solution = bool(val_norm & sol_nums) if sol_nums else False
        is_in_question = bool(val_norm & q_nums)
        # Priority: critical > in_solution > not_in_question
        return (
            is_critical and not is_in_question,
            is_in_solution and not is_in_question,
            not is_in_question,
        )

    matches_sorted = sorted(matches, key=_cell_score, reverse=True)
    match = matches_sorted[0]

    new_context = context[: match.start(1)] + "N/A" + context[match.end(1) :]
    return (
        new_context,
        f"Removed cell value: {match.group(1)}",
        f"Model should recognize missing {match.group(1)} data and refuse to answer",
    )


def transform_type2_markdown(
    context: str, question: str, python_solution: str = ""
) -> tuple:
    """Type 2: Table Column Removal for markdown table.

    Design rationale:
        Removes an entire column from a markdown table, simulating a missing
        data dimension in a financial report.
    Domain relevance:
        HIGH — columns typically represent fiscal periods or line items.
    Objectivity:
        Deterministic; prefers columns containing solution-critical values.
        Falls back to question-matching or last numeric column.
    Detection difficulty:
        MODERATE — the table looks syntactically correct but has fewer columns.
    Known limitations:
        Tables with <=2 data columns are skipped (removing would destroy the table).
    """
    table = _parse_markdown_table(context)
    if table is None:
        return None, None, None

    headers = table["headers"]
    col_values = table["col_values"]
    lines = table["lines"]

    # Need at least 2 data columns (excluding row-label column)
    data_col_start = (
        1 if headers[0] == "" or not any(c.isdigit() for c in headers[0]) else 0
    )
    data_col_count = len(headers) - data_col_start
    if data_col_count <= 2:
        return None, None, None

    # Use critical solution values for scoring (BFS-traced, not all numbers)
    critical_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )
    sol_nums = _extract_numbers_from_text(python_solution) if python_solution else set()
    question_lower = question.lower()

    # Score each data column by solution-criticality
    target_col_idx = -1
    target_col_name = None

    if critical_nums:
        best_score = 0
        for ci in range(data_col_start, len(headers)):
            overlap = len(col_values.get(ci, set()) & critical_nums)
            if overlap > best_score:
                best_score = overlap
                target_col_idx = ci
                target_col_name = headers[ci]

    # Fallback to all solution numbers if critical extraction found nothing
    if target_col_idx == -1 and sol_nums:
        best_score = 0
        for ci in range(data_col_start, len(headers)):
            overlap = len(col_values.get(ci, set()) & sol_nums)
            if overlap > best_score:
                best_score = overlap
                target_col_idx = ci
                target_col_name = headers[ci]

    if target_col_idx == -1:
        # Fallback: question-matching column header
        for idx in range(data_col_start, len(headers)):
            header = headers[idx]
            if header.lower() in question_lower or any(
                word in question_lower
                for word in header.lower().split()
                if len(word) > 2
            ):
                target_col_idx = idx
                target_col_name = header
                break

    if target_col_idx == -1:
        # Last resort: remove the last data column
        target_col_idx = len(headers) - 1
        target_col_name = headers[target_col_idx]

    # Remove column from all lines using aligned helper
    new_lines = []
    for line in lines:
        if "|" in line:
            new_lines.append(_remove_column_from_line(line, target_col_idx))
        else:
            new_lines.append(line)

    new_context = "\n".join(new_lines)
    return (
        new_context,
        f"Removed column: {target_col_name}",
        f"Model should recognize missing column {target_col_name} and refuse to answer",
    )


# ============================================================================
# TYPE 4: CRITICAL DATA REMOVAL (No markers — silent removal)
# ============================================================================


def _extract_numbers_from_python(python_solution: str) -> List[str]:
    """Extract numeric literals from ground truth Python solution"""
    if not python_solution:
        return []
    # Match numeric assignments: variable = 1234.56
    patterns = [
        r"=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"[\(\[,]\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*[\)\],]",
    ]
    numbers = []
    for pattern in patterns:
        for m in re.findall(pattern, python_solution):
            num_str = m.replace(",", "")
            try:
                val = float(num_str)
                if val > 1 and val not in {100, 1000, 365, 12, 52}:
                    numbers.append(m)
            except ValueError:
                continue
    return numbers


def transform_type4_json(
    context_dict: Dict, question: str, python_solution: str = ""
) -> tuple:
    """Type 4: Critical Data Removal — silently remove all key data from JSON.

    Design rationale:
        Empties all year entries for keys mentioned in the question, leaving
        the key structure intact but with no data. Unlike Type 1/2, there is
        no marker or structural deletion — the context looks "normal" at first glance.
    Domain relevance:
        MEDIUM — the question-driven targeting uses financial key names, but
        the silent-emptying mechanism itself is domain-agnostic.
    Objectivity:
        Deterministic; empties all keys whose names appear in the question.
    Detection difficulty:
        HIGH — no explicit marker; the model must notice that data values are missing
        from an otherwise well-formed structure.
    Known limitations:
        If the question does not reference any top-level key by name, no
        transformation is produced.
    """
    question_lower = question.lower()
    removed_keys = []

    new_context = deepcopy(context_dict)

    for key in list(context_dict.keys()):
        key_lower = key.lower()
        if key_lower in question_lower or any(
            word in question_lower for word in key_lower.split()
        ):
            if isinstance(new_context[key], dict):
                # Remove all year entries
                new_context[key] = {}
            else:
                del new_context[key]
            removed_keys.append(key)

    if not removed_keys:
        return None, None, None

    return (
        new_context,
        f"Silently removed all data for: {', '.join(removed_keys)}",
        "Model should detect missing critical data without explicit markers",
    )


def transform_type4_text(
    context: str, question: str, python_solution: str = ""
) -> tuple:
    """Type 4: Critical Data Removal — remove all numbers from text context.

    Design rationale:
        Replaces ALL numeric values in the text with "___", simulating a
        completely redacted financial passage.
    Domain relevance:
        MEDIUM — number stripping is generic but applied to financial text.
    Objectivity:
        Deterministic; replaces every match of the number/currency/percent pattern.
    Detection difficulty:
        HIGH — "___" placeholders are subtle; a model may try to infer values.
    Known limitations:
        Contexts with fewer than 2 numbers are skipped (too little data to strip).
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    numbers_found = re.findall(number_pattern, context)

    if len(numbers_found) < 2:
        return None, None, None

    # Remove all significant numbers (replace with generic text)
    new_context = context
    for num in numbers_found:
        new_context = new_context.replace(num, "___", 1)

    return (
        new_context,
        f"Silently removed {len(numbers_found)} numeric values",
        "Model should detect that critical numeric data is missing",
    )


def transform_type4_markdown(
    context: str, question: str, python_solution: str = ""
) -> tuple:
    """Type 4: Critical Data Removal — remove data rows from markdown table.

    Design rationale:
        Removes all data rows from a markdown table, leaving only the header
        and separator. Simulates a table with column definitions but no values.
    Domain relevance:
        HIGH — financial tables (BS/IS/CF) are the primary data delivery format.
    Objectivity:
        Deterministic; removes every row containing at least one digit below
        the header separator line.
    Detection difficulty:
        HIGH — the table structure is syntactically valid (headers remain);
        a model must notice the absence of data rows.
    Known limitations:
        Tables with non-numeric data rows (text-only) are not removed, which
        may leave partial data.
    """
    lines = context.split("\n")
    header_idx = -1

    for i, line in enumerate(lines):
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            header_idx = i
            break

    if header_idx == -1:
        return None, None, None

    # Keep header and separator, remove all data rows
    new_lines = lines[: header_idx + 2]  # header + separator
    removed_count = 0
    for line in lines[header_idx + 2 :]:
        if "|" in line and re.search(r"\d", line):
            removed_count += 1
        else:
            new_lines.append(line)

    if removed_count == 0:
        return None, None, None

    return (
        "\n".join(new_lines),
        f"Silently removed {removed_count} data rows from table",
        "Model should detect that table has no data rows",
    )


# ============================================================================
# SA TEXT: SILENT ABSENCE — SENTENCE DELETION (NEW)
# ============================================================================

_SA_STOPWORDS = frozenset(
    {
        "what",
        "is",
        "the",
        "of",
        "a",
        "an",
        "in",
        "to",
        "for",
        "and",
        "or",
        "was",
        "were",
        "are",
        "be",
        "been",
        "being",
        "how",
        "much",
        "many",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "that",
        "this",
        "it",
        "its",
        "by",
        "at",
        "on",
        "from",
        "with",
        "as",
        "if",
        "not",
        "no",
        "but",
        "so",
        "than",
        "then",
        "when",
        "which",
        "who",
        "where",
        "why",
        "all",
        "each",
        "per",
        "can",
        "will",
        "would",
        "could",
        "should",
    }
)


def transform_sa_text(context: str, question: str) -> tuple:
    """SA: Silent Absence — delete question-relevant sentences from text.

    Design rationale:
        Identifies sentences containing both question keywords and numeric
        data, then removes them entirely. The remaining text reads naturally
        but lacks information required to solve the problem.
    Detection difficulty:
        HIGH — the remaining context is grammatically complete and coherent;
        no markers hint at missing data.
    Difference from legacy Type 4 text:
        Type 4 replaced numbers with '___' (still a visible marker).
        SA text deletes entire sentences, producing truly marker-free context.
    """
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    if len(sentences) <= 1:
        return None, None, None

    question_lower = question.lower()
    keywords = [
        w
        for w in re.findall(r"\b\w+\b", question_lower)
        if w not in _SA_STOPWORDS and len(w) > 2 and not w.isdigit()
    ]

    if not keywords:
        return None, None, None

    # Score each sentence: keyword hits + must contain a number
    scored: List[tuple] = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        keyword_hits = sum(1 for kw in keywords if kw in sent_lower)
        has_number = bool(re.search(r"\d+", sent))
        if keyword_hits > 0 and has_number:
            scored.append((i, keyword_hits))

    if not scored:
        return None, None, None

    # Prefer sentences whose numbers are NOT leaked in the question
    q_nums = _extract_numbers_from_text(question)

    def _has_non_leaked_number(sent_idx: int) -> bool:
        """Check if sentence has at least one number not in the question."""
        sent = sentences[sent_idx]
        sent_nums = _extract_numbers_from_text(sent)
        return bool(sent_nums - q_nums)

    # Sort by: (1) has non-leaked numbers, (2) keyword relevance
    scored_sorted = sorted(
        scored,
        key=lambda x: (_has_non_leaked_number(x[0]), x[1]),
        reverse=True,
    )

    target_idx = scored_sorted[0][0]
    removed_sent = sentences[target_idx]
    remaining = sentences[:target_idx] + sentences[target_idx + 1 :]

    # Safety: keep at least one sentence
    if not remaining:
        return None, None, None

    new_context = " ".join(remaining)

    return (
        new_context,
        f"Silently removed sentence: {removed_sent[:80]}...",
        "Model should detect missing critical information without markers",
    )


# ============================================================================
# TYPE 5: CONTRADICTORY INFORMATION
# ============================================================================


def transform_type5_json(context_dict: Dict, question: str) -> tuple:
    """Type 5: Add contradictory values to JSON context.

    Design rationale:
        Adds a '_conflicting_report' sub-key with value × 1.5 within the same
        data key, creating an explicit numerical discrepancy that models should
        detect and flag as inconsistent.
    Domain relevance:
        LOW — the 1.5× fixed multiplier is domain-agnostic. However, the
        conflicting-report framing simulates real-world audit discrepancies.
    Objectivity:
        Deterministic; targets the last year of the first key mentioned in the
        question, multiplies by exactly 1.5.
    Detection difficulty:
        VERY HIGH — Phase B experiments (2026-02-19) showed all models (economic
        tier) failed to detect contradictions, even with metacognitive prompts.
        Models tend to adopt the new value without questioning consistency.
    Known limitations:
        - 1.5× multiplier may produce unrealistic values for some financial metrics
        - Only targets the last year entry, so multi-year contradictions are not tested
        - The '_conflicting_report' key name may be too subtle for models to notice
    """
    question_lower = question.lower()
    new_context = deepcopy(context_dict)
    contradictions = []

    for key in list(context_dict.keys()):
        key_lower = key.lower()
        if key_lower in question_lower:
            if isinstance(context_dict[key], dict):
                years = list(context_dict[key].keys())
                if years:
                    target_year = years[-1]
                    original_val = context_dict[key][target_year]
                    try:
                        num_val = float(
                            str(original_val).replace(",", "").replace("%", "")
                        )
                        contra_val = num_val * 1.5
                        # Add conflicting value as sub-key within same entry
                        new_context[key][f"{target_year}_conflicting_report"] = (
                            contra_val
                        )
                        contradictions.append(
                            f"{key}[{target_year}]={original_val} vs {key}[{target_year}_conflicting_report]={contra_val}"
                        )
                    except (ValueError, TypeError):
                        continue

    if not contradictions:
        return None, None, None

    return (
        new_context,
        f"Added contradictions: {'; '.join(contradictions)}",
        "Model should detect contradictory data and refuse or flag inconsistency",
    )


def _is_descriptor_number(val: float, context: str, match: re.Match) -> bool:
    """Check if a number is a descriptor (date, period count) rather than data.

    Descriptor numbers are those that describe structure (e.g., "10 trading days",
    "January 10", "5-year") rather than computation-critical data values.
    These are poor IC targets because models can verify them by counting data.
    """
    pos = match.start()
    # Get surrounding text (50 chars before and after)
    before = context[max(0, pos - 50) : pos].lower()
    after_text = context[pos : min(len(context), pos + 80)].lower()
    match_text = match.group(0).lower()

    # Date patterns: month names near the number
    month_names = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    for month in month_names:
        if month in before[-20:] or month in after_text[:20]:
            return True

    # Year-like numbers (1900-2100)
    if 1900 <= val <= 2100 and val == int(val):
        return True

    # Period descriptors: "N trading days", "N years", "past N", "last N"
    period_words = [
        "trading day",
        "business day",
        "year",
        "month",
        "week",
        "day",
        "quarter",
        "period",
        "session",
    ]
    for pw in period_words:
        if pw in after_text[:30]:
            return True

    # "past N", "last N", "over N", "first N"
    if re.search(r"(past|last|over|first|next)\s*$", before[-15:]):
        return True

    return False


def transform_type5_text(
    context: str, question: str, python_solution: str = ""
) -> tuple:
    """Type 5: Insert contradictory data directly into text context.

    Design rationale (v2):
        Targets a solution-critical numeric value (not descriptors like dates
        or period counts). Skips values that appear in the question (which
        would let the model resolve the contradiction trivially). Duplicates
        the containing sentence with a 1.5× altered value and an explicit
        discrepancy note.
    Improvements over v1:
        - Solution-aware targeting: prefers numbers used in python_solution
        - Question-leak prevention: skips values already in the question
        - Descriptor filtering: skips dates, years, period counts
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    matches = list(re.finditer(number_pattern, context))

    if not matches:
        return None, None, None

    q_nums = _extract_numbers_from_text(question)
    sol_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )

    # Score each number match for IC suitability
    def _ic_score(m: re.Match) -> tuple:
        val_str = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            val = float(val_str)
        except ValueError:
            return (-1, -1, -1, -1)

        if val <= 0.001:
            return (-1, -1, -1, -1)

        # Normalize for comparison
        val_norm = str(int(val)) if val == int(val) else str(val)

        is_in_question = val_norm in q_nums
        is_in_solution = val_norm in sol_nums if sol_nums else False
        is_descriptor = _is_descriptor_number(val, context, m)

        # Priority: (not_in_question, in_solution, not_descriptor, value_size)
        return (
            0 if is_in_question else 1,  # Must not be in question
            1 if is_in_solution else 0,  # Prefer solution-critical
            0 if is_descriptor else 1,  # Avoid descriptors
            val,  # Larger values = more impactful
        )

    scored = [(m, _ic_score(m)) for m in matches]
    scored = [(m, s) for m, s in scored if s[0] >= 0]

    if not scored:
        return None, None, None

    scored.sort(key=lambda x: x[1], reverse=True)

    # Skip if best candidate is in question (all candidates leaked)
    best_match, best_score = scored[0]
    if best_score[0] == 0:
        # All candidates are in question — try anyway with best non-question if exists
        non_q = [(m, s) for m, s in scored if s[0] == 1]
        if non_q:
            best_match, best_score = non_q[0]
        # else: proceed with leaked value (better than nothing)

    target = best_match
    original_str = target.group(0)
    val_str_clean = original_str.replace("$", "").replace(",", "").replace("%", "")

    try:
        num_val = float(val_str_clean)
        contra_val = num_val * 1.5
    except ValueError:
        return None, None, None

    # Format contradictory value in same style as original
    has_dollar = "$" in original_str
    has_percent = "%" in original_str
    has_decimal = "." in val_str_clean

    if has_decimal:
        decimal_places = len(val_str_clean.split(".")[-1])
        contra_formatted = f"{contra_val:.{decimal_places}f}"
    else:
        contra_formatted = f"{int(contra_val):,}"

    if has_dollar:
        contra_formatted = f"${contra_formatted}"
    if has_percent:
        contra_formatted = f"{contra_formatted}%"

    # Find the sentence containing the target number
    sentences = re.split(r"(?<=[.!?])\s+", context)
    target_sent_idx = -1
    for i, sent in enumerate(sentences):
        if original_str in sent:
            target_sent_idx = i
            break

    if target_sent_idx == -1:
        return None, None, None

    # Strategy: REPLACE original value with 1.5x in the original sentence,
    # then ADD a contradictory sentence with the original value after it.
    # This ensures the model cannot simply pick the "first" (original) value.
    original_sentence = sentences[target_sent_idx]
    replaced_sentence = original_sentence.replace(original_str, contra_formatted, 1)

    # The contradiction sentence restates the original value as from "another source"
    contra_sentence = (
        f"Note: There is a discrepancy — one source reports {contra_formatted}, "
        f"while another shows {original_str}."
    )

    # Replace original sentence with the modified one, then add contradiction
    new_sentences = (
        sentences[:target_sent_idx]
        + [replaced_sentence, contra_sentence]
        + sentences[target_sent_idx + 1 :]
    )
    new_context = " ".join(new_sentences)

    return (
        new_context,
        f"Replaced {original_str} with {contra_formatted}, added contradiction",
        "Model should detect two conflicting values in the context and refuse or flag inconsistency",
    )


def _is_date_or_year_cell(cell: str) -> bool:
    """Check if a table cell contains date/year info that should not be multiplied."""
    cell_lower = cell.lower().strip()
    month_names = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "balance",
        "ending",
        "beginning",
        "as of",
        "date",
        "period",
    ]
    for m in month_names:
        if m in cell_lower:
            return True
    # Pure year cell: "2012", "2013"
    nums = re.findall(r"\d+", cell)
    if nums and all(
        1900 <= int(n) <= 2100 for n in nums if n.isdigit() and len(n) == 4
    ):
        if all(len(n) == 4 for n in nums):
            return True
    return False


def transform_type5_markdown(
    context: str, question: str, python_solution: str = ""
) -> tuple:
    """Type 5: Add contradictory row to markdown table.

    Design rationale (v3):
        Selects the data row with the most solution-critical values (not just
        the last row). Replaces its values with 1.5x, then adds the original
        as a conflicting report row. Protects date/year cells.
    Improvements over v2:
        - Solution-aware row selection: picks the row whose values appear
          most in the python_solution's answer chain
        - Avoids summary/balance rows that models may skip
    """
    lines = context.split("\n")
    header_idx = -1

    for i, line in enumerate(lines):
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            header_idx = i
            break

    if header_idx == -1:
        return None, None, None

    # Collect all data rows with numbers
    data_rows = []
    for i in range(header_idx + 2, len(lines)):
        if "|" in lines[i] and re.search(r"\d", lines[i]):
            data_rows.append(i)

    if not data_rows:
        return None, None, None

    # Score each row by solution-criticality
    sol_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )
    q_nums = _extract_numbers_from_text(question)

    def _row_score(row_idx: int) -> tuple:
        row = lines[row_idx]
        row_nums = _extract_numbers_from_text(row)
        # Skip date/summary rows
        if _is_date_or_year_cell(row.split("|")[1] if len(row.split("|")) > 1 else ""):
            return (-1, 0, 0)
        sol_overlap = len(row_nums & sol_nums) if sol_nums else 0
        q_overlap = len(row_nums & q_nums)
        has_data = len(row_nums)
        return (0 if has_data == 0 else 1, sol_overlap, -q_overlap)

    scored_rows = [(idx, _row_score(idx)) for idx in data_rows]
    scored_rows.sort(key=lambda x: x[1], reverse=True)

    # Pick best row (highest solution overlap, avoid date rows)
    target_idx = scored_rows[0][0]

    # Create contradictory row — only multiply data cells, skip date/year cells
    original_row = lines[target_idx]
    cells = original_row.split("|")
    new_cells = []
    modified = False

    for cell in cells:
        # Skip date/year cells entirely
        if _is_date_or_year_cell(cell):
            new_cells.append(cell)
            continue

        nums = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)", cell)
        new_cell = cell
        for num_str in nums:
            try:
                num_val = float(num_str.replace(",", ""))
                # Skip year-like standalone numbers
                if 1900 <= num_val <= 2100 and num_val == int(num_val):
                    continue
                # Skip day-of-month (1-31)
                if 1 <= num_val <= 31 and num_val == int(num_val) and len(num_str) <= 2:
                    continue
                contra_val = num_val * 1.5
                if "." in num_str:
                    decimal_places = len(num_str.split(".")[-1])
                    new_cell = new_cell.replace(
                        num_str, f"{contra_val:.{decimal_places}f}", 1
                    )
                else:
                    new_cell = new_cell.replace(num_str, f"{int(contra_val):,}", 1)
                modified = True
            except ValueError:
                continue
        new_cells.append(new_cell)

    if not modified:
        return None, None, None

    # Strategy: REPLACE original row values with 1.5x, then ADD the original
    # row as "(conflicting report)" after it. This prevents the model from
    # simply using the "first" (original) value.

    # The modified row replaces the original in-place
    modified_row = "|".join(new_cells)

    # The original row gets "(conflicting report)" label
    orig_cells = original_row.split("|")
    for i, cell in enumerate(orig_cells):
        stripped = cell.strip()
        if stripped and not re.match(r"^[\d,.\-\s%$]+$", stripped):
            orig_cells[i] = cell.rstrip() + " (conflicting report) "
            break
    conflict_row = "|".join(orig_cells)

    # Replace original with modified, add original-as-conflict after
    new_lines = (
        lines[:target_idx] + [modified_row, conflict_row] + lines[target_idx + 1 :]
    )

    return (
        "\n".join(new_lines),
        f"Replaced row {target_idx} values with 1.5x, added original as conflict",
        "Model should detect conflicting data rows and flag inconsistency",
    )


# ============================================================================
# QUESTION-ONLY TRANSFORMATIONS
# (For problems where context is empty and all data is in the question)
# ============================================================================

LABEL_Q_EA_PARTIAL = "Q-EA-partial: Question Explicit Absence (Partial)"
LABEL_Q_SA = "Q-SA: Question Silent Absence"
LABEL_Q_IC = "Q-IC: Question Information Conflict"


def transform_question_ea_partial(question: str, python_solution: str = "") -> tuple:
    """EA-partial for question-only problems.

    Replaces a solution-critical number in the question with [DATA MISSING].
    Prioritizes values used in the answer chain (via BFS trace).

    Returns:
        (new_question, description, expected_behavior) or (None, None, None)
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    matches = list(re.finditer(number_pattern, question))
    if not matches:
        return None, None, None

    critical_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )

    def _match_score(m: re.Match) -> tuple:
        val = m.group(0)
        val_norm = _extract_numbers_from_text(val)
        is_critical = bool(val_norm & critical_nums) if critical_nums else False
        # Skip year-like numbers and very small numbers
        clean = val.replace("$", "").replace(",", "").replace("%", "")
        try:
            num_val = float(clean)
        except ValueError:
            return (False, False, 0)
        is_year = 1900 <= num_val <= 2100 and num_val == int(num_val)
        is_small = num_val <= 1 and "%" not in val and "$" not in val
        return (is_critical and not is_year, not is_year and not is_small, num_val)

    scored = sorted(matches, key=_match_score, reverse=True)
    target = scored[0]
    target_val = target.group(0)

    # Verify it's a meaningful target
    score = _match_score(target)
    if not score[1]:  # all candidates are years or trivial
        return None, None, None

    new_question = (
        question[: target.start()] + "[DATA MISSING]" + question[target.end() :]
    )
    return (
        new_question,
        f"Removed from question: {target_val}",
        "Model should recognize missing data in question and refuse to answer",
    )


def transform_question_sa(question: str, python_solution: str = "") -> tuple:
    """SA (Silent Absence) for question-only problems.

    Removes a clause or phrase containing a solution-critical number,
    making the question read naturally but with missing information.

    Strategy: Split question into clauses (by comma, 'and', semicolon),
    remove the clause with the most critical data.

    Returns:
        (new_question, description, expected_behavior) or (None, None, None)
    """
    # Split into clauses by commas, semicolons, "and" connectors
    clause_pattern = r"[,;]\s*|\s+and\s+"
    parts = re.split(clause_pattern, question)

    if len(parts) <= 2:
        # Too few clauses — try sentence-level if multi-sentence
        sentences = re.split(r"(?<=[.!?])\s+", question.strip())
        if len(sentences) <= 1:
            return None, None, None
        parts = sentences
        is_sentence = True
    else:
        is_sentence = False

    critical_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )

    def _clause_score(part: str) -> tuple:
        nums = _extract_numbers_from_text(part)
        has_critical = bool(nums & critical_nums) if critical_nums else False
        has_number = bool(re.search(r"\d", part))
        # Don't remove the actual question part (ends with ?)
        is_question_part = "?" in part
        return (
            has_critical and not is_question_part,
            has_number and not is_question_part,
            len(nums),
        )

    scored = [(i, _clause_score(p)) for i, p in enumerate(parts)]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_idx, best_score = scored[0]
    if not best_score[1]:  # no clause has numbers
        return None, None, None

    removed = parts[best_idx].strip()
    remaining = [p for i, p in enumerate(parts) if i != best_idx]

    if is_sentence:
        new_question = " ".join(remaining)
    else:
        new_question = ", ".join(remaining)
        # Clean up double commas, leading commas
        new_question = re.sub(r",\s*,", ",", new_question)
        new_question = re.sub(r"^\s*,\s*", "", new_question)

    if not new_question.strip():
        return None, None, None

    return (
        new_question.strip(),
        f"Silently removed clause: {removed[:80]}",
        "Model should detect missing critical information in question",
    )


def transform_question_ic(question: str, python_solution: str = "") -> tuple:
    """IC (Information Conflict) for question-only problems.

    Inserts a contradictory value (1.5x) for a solution-critical number
    by adding a parenthetical alternative.

    Example: "$500,000" → "$500,000 (however, a revised estimate suggests $750,000)"

    Returns:
        (new_question, description, expected_behavior) or (None, None, None)
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    matches = list(re.finditer(number_pattern, question))
    if not matches:
        return None, None, None

    critical_nums = (
        _extract_critical_solution_values(python_solution) if python_solution else set()
    )

    def _match_score(m: re.Match) -> tuple:
        val = m.group(0)
        val_norm = _extract_numbers_from_text(val)
        is_critical = bool(val_norm & critical_nums) if critical_nums else False
        clean = val.replace("$", "").replace(",", "").replace("%", "")
        try:
            num_val = float(clean)
        except ValueError:
            return (False, False, 0)
        is_year = 1900 <= num_val <= 2100 and num_val == int(num_val)
        is_descriptor = num_val <= 1 and "%" not in val and "$" not in val
        return (is_critical and not is_year, not is_year and not is_descriptor, num_val)

    scored = sorted(matches, key=_match_score, reverse=True)
    target = scored[0]
    target_val = target.group(0)

    score = _match_score(target)
    if not score[1]:
        return None, None, None

    # Create contradictory value
    clean = target_val.replace("$", "").replace(",", "").replace("%", "")
    try:
        num_val = float(clean)
    except ValueError:
        return None, None, None

    contra_val = num_val * 1.5

    # Format to match original style
    has_dollar = "$" in target_val
    has_percent = "%" in target_val
    has_comma = "," in target_val

    if "." in clean:
        decimal_places = len(clean.split(".")[-1])
        contra_str = f"{contra_val:.{decimal_places}f}"
    else:
        contra_int = int(contra_val)
        contra_str = f"{contra_int:,}" if has_comma else str(contra_int)

    if has_dollar:
        contra_str = "$" + contra_str
    if has_percent:
        contra_str = contra_str + "%"

    insert_text = f"{target_val} (however, a revised estimate suggests {contra_str})"
    new_question = question[: target.start()] + insert_text + question[target.end() :]

    return (
        new_question,
        f"Added conflict in question: {target_val} vs {contra_str}",
        "Model should detect contradictory data in question and flag inconsistency",
    )


# ============================================================================
# TRANSFORMATION VALIDATION
# ============================================================================


# _solution_uses_hardcoded_values is imported from evaluation.hardcoded_solution_detector


def validate_transformation(
    transformed_example: Dict, original_example: Dict
) -> Dict[str, Any]:
    """Validate that a transformation makes the problem unsolvable.

    .. deprecated::
        This function relies on python_solution execution, which fails for ~64%
        of hard.json (hardcoded solutions). Use the reasoning trace validation
        pipeline instead: ``experiments/run_validation_pipeline.py``

        The new pipeline analyzes LLM responses directly:
        - Case 1 (거부) → metacognitive success
        - Case 2 (오답) → transformation effective
        - Case 3 (정답) → memorization vs reasoning analysis

    Runs the ground truth Python solution against the transformed context.
    If execution fails or produces a different answer, the transformation is valid.

    Args:
        transformed_example: The transformed problem
        original_example: The original problem with ground truth

    Returns:
        Dict with 'valid', 'reason', and 'details' keys
    """
    python_solution = original_example.get("python_solution", "")
    ground_truth = original_example.get("ground_truth")

    if not python_solution:
        return {
            "valid": True,
            "reason": "no_python_solution",
            "details": "Cannot validate — no ground truth code available",
        }

    # Check if solution uses only hardcoded values (no context parsing)
    if _solution_uses_hardcoded_values(python_solution):
        return {
            "valid": False,
            "reason": "hardcoded_solution",
            "details": "Solution uses hardcoded values — cannot validate unsolvability",
        }

    # Try executing the ground truth code with transformed context
    try:
        local_vars = {}
        exec(python_solution, {"__builtins__": __builtins__}, local_vars)

        # Check if result matches original ground truth
        result = local_vars.get("result") or local_vars.get("answer")

        if result is None:
            return {
                "valid": True,
                "reason": "execution_no_result",
                "details": "Ground truth code produced no result with transformed data",
            }

        try:
            result_num = float(str(result).replace(",", "").replace("%", ""))
            truth_num = float(str(ground_truth).replace(",", "").replace("%", ""))

            if truth_num != 0 and abs(result_num - truth_num) / abs(truth_num) < 0.01:
                return {
                    "valid": False,
                    "reason": "still_solvable",
                    "details": f"Ground truth code still produces correct answer: {result}",
                }
        except (ValueError, TypeError):
            pass

        return {
            "valid": True,
            "reason": "different_answer",
            "details": f"Ground truth code produced different answer: {result} (expected {ground_truth})",
        }

    except Exception as e:
        return {
            "valid": True,
            "reason": "execution_error",
            "details": f"Ground truth code failed: {str(e)[:200]}",
        }


# ============================================================================
# MAIN TRANSFORMATION LOGIC
# ============================================================================


def detect_context_type(context: str) -> str:
    """Detect context type: json, text, or markdown."""
    if not context or context == "[]":
        return "none"

    # Try JSON
    try:
        json.loads(context)
        return "json"
    except:
        pass

    # Check for markdown table
    if "|" in context and "---" in context:
        return "markdown"

    return "text"


def apply_transformations(example: Dict) -> List[Dict]:
    """Apply all applicable transformations to an example."""
    transformations = []
    context = normalize_context(example.get("context", ""))
    question = example.get("question", "")
    context_type = detect_context_type(context)

    python_solution = example.get("python_solution", "")

    if context_type == "json":
        try:
            context_dict = json.loads(context)

            # EA-partial: Explicit Absence — Partial (was Type 1)
            new_ctx, desc, expected = transform_type1_json(context_dict, question)
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = LABEL_EA_PARTIAL
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)

            # EA-full: Explicit Absence — Full (was Type 2)
            new_ctx, desc, expected = transform_type2_json(context_dict, question)
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = LABEL_EA_FULL
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)

            # SA: Silent Absence (was Type 4)
            new_ctx, desc, expected = transform_type4_json(
                context_dict, question, python_solution
            )
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = LABEL_SA
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)

            # IC: Information Conflict (was Type 5)
            new_ctx, desc, expected = transform_type5_json(context_dict, question)
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = LABEL_IC
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)
        except Exception:
            pass

    elif context_type == "text":
        # EA-partial: Explicit Absence — Partial (was Type 1)
        new_ctx, desc, expected = transform_type1_text(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_EA_PARTIAL
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # EA-full: Explicit Absence — Full (remove all data sentences)
        new_ctx, desc, expected = transform_type2_text(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_EA_FULL
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # SA: Silent Absence — sentence deletion (replaces Type 4 text)
        new_ctx, desc, expected = transform_sa_text(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_SA
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # IC: Information Conflict (was Type 5)
        new_ctx, desc, expected = transform_type5_text(
            context, question, python_solution
        )
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_IC
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

    elif context_type == "markdown":
        # EA-partial: Explicit Absence — Partial (was Type 1)
        new_ctx, desc, expected = transform_type1_markdown(
            context, question, python_solution
        )
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_EA_PARTIAL
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # EA-full: Explicit Absence — Full (was Type 2)
        new_ctx, desc, expected = transform_type2_markdown(
            context, question, python_solution
        )
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_EA_FULL
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # SA: Silent Absence (was Type 4)
        new_ctx, desc, expected = transform_type4_markdown(
            context, question, python_solution
        )
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_SA
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # IC: Information Conflict (was Type 5)
        new_ctx, desc, expected = transform_type5_markdown(
            context, question, python_solution
        )
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_IC
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

    # TA: Temporal Ambiguity (was Type 3, applies to all)
    # Try question first; fall back to context if question has no year
    new_q, desc, expected = transform_type3_question(question)
    if new_q:
        trans = deepcopy(example)
        trans["question"] = new_q
        trans["transformation_type"] = LABEL_TA
        trans["transformation_description"] = desc
        trans["expected_behavior"] = expected
        transformations.append(trans)
    elif context and context_type != "none":
        new_ctx, desc, expected = transform_type3_context(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = LABEL_TA
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

    return transformations


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def main():
    """Apply transformations to full dataset."""

    data_dir = Path("data/financereasoning/raw/FinanceReasoning")
    output_dir = Path("experiments/transformed_datasets")
    output_dir.mkdir(exist_ok=True, parents=True)

    levels = ["easy", "medium", "hard"]

    for level in levels:
        print(f"\n{'=' * 70}")
        print(f"Processing {level.upper()} dataset")
        print(f"{'=' * 70}")

        input_file = data_dir / f"{level}.json"

        if not input_file.exists():
            print(f"[SKIP] {input_file} not found")
            continue

        # Load data
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} problems")

        # Apply transformations
        transformed_data = []
        success_count = 0

        for idx, example in enumerate(data):
            transformations = apply_transformations(example)

            if transformations:
                transformed_data.append(
                    {"original": example, "transformations": transformations}
                )
                success_count += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(data)} problems...")

        print("\nResults:")
        print(f"  Total problems: {len(data)}")
        print(f"  Successfully transformed: {success_count}")
        print(
            f"  Total transformations: {sum(len(p['transformations']) for p in transformed_data)}"
        )

        # Save
        output_file = output_dir / f"{level}_transformed.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] {output_file}")

    print(f"\n{'=' * 70}")
    print("[SUCCESS] All datasets transformed")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. Review transformed datasets")
    print("2. Run evaluation experiment")
    print("3. Generate visualization")


if __name__ == "__main__":
    main()
