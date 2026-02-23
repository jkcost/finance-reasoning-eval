"""
Apply transformations to the FULL FinanceReasoning dataset.

Transformation Taxonomy
=======================

The 5 transformation types are divided into two categories that test
fundamentally different metacognitive abilities:

**Information Absence (Type 1~4)** — Can the model detect MISSING data?
  - Type 1: Selective Removal + Explicit Marker ([DATA MISSING], N/A)
    → Easiest; the marker itself is a strong signal.
  - Type 2: Structural Removal (entire column/key deleted)
    → Moderate; no marker, but the schema change is detectable.
  - Type 3: Temporal Ambiguity (year → "the end of the period")
    → Tests whether the model demands a concrete time reference.
  - Type 4: Silent Removal (numbers stripped without any marker)
    → Hardest absence type; context looks "normal" but values are gone.

**Information Conflict (Type 5)** — Can the model detect CONTRADICTORY data?
  - Type 5: Contradictory Information (1.5× conflicting value inserted)
    → Separate cognitive skill from absence detection.
    → Phase B experiments (2026-02-19) showed all models fail on this type,
      even with metacognitive prompts.

Objectivity & Reproducibility
-----------------------------
- All transformations are **fully deterministic** (no random elements).
- Types 1~4 are **question-driven**: transformation targets are selected
  by matching question text against context keys/values.
- Type 5 uses an **absolute rule**: last year/row × 1.5 multiplier.
- `validate_transformation()` runs the ground-truth Python solution against
  the transformed context to objectively verify unsolvability.
- `_solution_uses_hardcoded_values()` detects solutions that bypass context
  parsing, preventing false "still_solvable" validation results.

Domain Relevance Summary
-------------------------
| Type | Domain Specificity | Rationale                                    |
|------|--------------------|----------------------------------------------|
| 1    | High               | Targets fiscal-year time-series structure     |
| 2    | High               | Exploits tabular financial statement layout   |
| 3    | High               | Relies on fiscal reporting period conventions |
| 4    | Medium             | Numeric removal is generic; targeting is not  |
| 5    | Low                | 1.5× multiplier is domain-agnostic           |

Usage:
    python experiments/apply_transformations_full.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from copy import deepcopy


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
            # Remove a critical year/period from this key
            if isinstance(context_dict[key], dict):
                years = list(context_dict[key].keys())
                if years:
                    # Find year mentioned in question
                    for year_key in years:
                        if str(year_key) in question:
                            # Remove this year
                            new_context = deepcopy(context_dict)
                            del new_context[key][year_key]
                            return (
                                new_context,
                                f"Removed critical data: {key}",
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


def transform_type1_text(context: str, question: str) -> tuple:
    """Type 1: Information Removal for text context.

    Design rationale:
        Replaces the first significant number in the text with "[DATA MISSING]",
        simulating an explicitly flagged data gap.
    Domain relevance:
        MEDIUM — number replacement is generic, but targets financial numeric data.
    Objectivity:
        Deterministic; selects the first number matching the currency/number pattern.
    Detection difficulty:
        LOW — the "[DATA MISSING]" marker is an obvious signal.
    Known limitations:
        May replace a number not critical to the question; validate_transformation
        filters these false positives.
    """
    # Find numbers in context
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?"
    numbers = re.findall(number_pattern, context)

    if numbers and len(numbers) > 0:
        # Remove first significant number
        target_num = numbers[0]
        new_context = context.replace(target_num, "[DATA MISSING]", 1)
        return (
            new_context,
            f"Removed critical number: {target_num}",
            "Model should recognize missing data and refuse to answer",
        )

    return None, None, None


def transform_type1_markdown(context: str, question: str) -> tuple:
    """Type 1: Information Removal for markdown table.

    Design rationale:
        Replaces the first numeric cell in a markdown table with "N/A",
        simulating a missing data point in a tabulated financial report.
    Domain relevance:
        HIGH — markdown tables directly mirror financial statement layouts.
    Objectivity:
        Deterministic; targets the first cell matching the number pattern.
    Detection difficulty:
        LOW — "N/A" is an explicit missing-data marker.
    Known limitations:
        May target a cell irrelevant to the question; validate_transformation filters.
    """
    # Find table cells with numbers
    cell_pattern = r"\|\s*(\$?\d+(?:,\d{3})*(?:\.\d+)?)\s*\|"
    matches = list(re.finditer(cell_pattern, context))

    if matches:
        # Replace first cell with N/A
        match = matches[0]
        new_context = context[: match.start(1)] + "N/A" + context[match.end(1) :]
        return (
            new_context,
            f"Removed cell value in column: {match.group(1)}",
            f"Model should recognize missing {match.group(1)} data and refuse to answer",
        )

    return None, None, None


def transform_type2_markdown(context: str, question: str) -> tuple:
    """Type 2: Table Column Removal for markdown table.

    Design rationale:
        Removes an entire column from a markdown table, simulating a missing
        data dimension in a financial report.
    Domain relevance:
        HIGH — columns typically represent fiscal periods or line items.
    Objectivity:
        Deterministic; prefers the column whose header matches the question text,
        falls back to the second column.
    Detection difficulty:
        MODERATE — the table looks syntactically correct but has fewer columns.
    Known limitations:
        Tables with ≤2 columns are skipped (removing would destroy the table).
    """
    # Find table header
    lines = context.split("\n")
    header_idx = -1

    for i, line in enumerate(lines):
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            header_idx = i
            break

    if header_idx == -1:
        return None, None, None

    # Parse header
    headers = [h.strip() for h in lines[header_idx].split("|") if h.strip()]

    if len(headers) <= 2:
        return None, None, None

    # Find column mentioned in question
    question_lower = question.lower()
    target_col_idx = -1
    target_col_name = None

    for idx, header in enumerate(headers):
        if header.lower() in question_lower or any(
            word in question_lower for word in header.lower().split()
        ):
            target_col_idx = idx
            target_col_name = header
            break

    if target_col_idx == -1:
        # Remove second column as fallback
        target_col_idx = 1
        target_col_name = headers[1]

    # Remove column from all rows
    new_lines = []
    for i, line in enumerate(lines):
        if "|" not in line:
            new_lines.append(line)
            continue

        cells = line.split("|")
        # Keep first and last empty cells, remove target column
        new_cells = (
            [cells[0]]
            + [cells[j] for j in range(1, len(cells) - 1) if j - 1 != target_col_idx]
            + [cells[-1]]
        )
        new_lines.append("|".join(new_cells))

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


def transform_type4_json(context_dict: Dict, question: str, python_solution: str = "") -> tuple:
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


def transform_type4_text(context: str, question: str, python_solution: str = "") -> tuple:
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


def transform_type4_markdown(context: str, question: str, python_solution: str = "") -> tuple:
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
                        num_val = float(str(original_val).replace(",", "").replace("%", ""))
                        contra_val = num_val * 1.5
                        # Add conflicting value as sub-key within same entry
                        new_context[key][f"{target_year}_conflicting_report"] = contra_val
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


def transform_type5_text(context: str, question: str) -> tuple:
    """Type 5: Insert contradictory data directly into text context.

    Design rationale:
        Finds the first significant number, duplicates its containing sentence
        with a 1.5× altered value, and prepends an explicit discrepancy note.
        This creates two conflicting statements about the same quantity.
    Domain relevance:
        LOW — sentence duplication with a numeric change is domain-agnostic,
        though the discrepancy note language references financial reporting.
    Objectivity:
        Deterministic; targets the first number > 0.001, multiplies by 1.5,
        preserves original formatting (currency, percent, decimals).
    Detection difficulty:
        VERY HIGH — despite the explicit "Note: There is a discrepancy" prefix,
        models in Phase B experiments still failed to flag the contradiction.
    Known limitations:
        - Sentence splitting by regex may fail on complex punctuation
        - The explicit discrepancy note makes this easier than real-world cases
          (yet models still fail — indicating fundamental weakness)
    """
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    matches = list(re.finditer(number_pattern, context))

    if not matches:
        return None, None, None

    # Find first significant number (skip trivially small ones)
    target = None
    for m in matches:
        val_str = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            if float(val_str) > 0.001:
                target = m
                break
        except ValueError:
            continue

    if target is None:
        return None, None, None

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
    # Split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", context)
    target_sent_idx = -1
    for i, sent in enumerate(sentences):
        if original_str in sent:
            target_sent_idx = i
            break

    if target_sent_idx == -1:
        return None, None, None

    # Create contradictory sentence by replacing the number
    original_sentence = sentences[target_sent_idx]
    contra_sentence = original_sentence.replace(original_str, contra_formatted, 1)

    # Prepend explicit discrepancy marker
    contra_sentence = (
        f"Note: There is a discrepancy — one source reports {original_str}, "
        f"while another shows {contra_formatted}. "
        + contra_sentence[0].upper() + contra_sentence[1:]
    )

    # Insert contradictory sentence right after the original
    new_sentences = (
        sentences[: target_sent_idx + 1]
        + [contra_sentence]
        + sentences[target_sent_idx + 1 :]
    )
    new_context = " ".join(new_sentences)

    return (
        new_context,
        f"Inserted contradictory sentence: {original_str} vs {contra_formatted}",
        "Model should detect two conflicting values in the context and refuse or flag inconsistency",
    )


def transform_type5_markdown(context: str, question: str) -> tuple:
    """Type 5: Add contradictory row to markdown table.

    Design rationale:
        Duplicates the last data row with all numeric values × 1.5 and labels
        it "(conflicting report)". The model should notice two rows claiming
        different values for the same data point.
    Domain relevance:
        MEDIUM — conflicting rows in financial tables can occur from multiple
        reporting sources or restatements, though the 1.5× rule is artificial.
    Objectivity:
        Deterministic; targets the last numeric row, multiplies all numbers by 1.5.
    Detection difficulty:
        VERY HIGH — models tend to treat the new row as additional data rather
        than a conflicting duplicate.
    Known limitations:
        - "(conflicting report)" label may be too subtle
        - Row duplication with different numbers could be interpreted as a new period
    """
    lines = context.split("\n")
    header_idx = -1

    for i, line in enumerate(lines):
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            header_idx = i
            break

    if header_idx == -1:
        return None, None, None

    # Find last data row with numbers
    last_data_idx = -1
    for i in range(len(lines) - 1, header_idx + 1, -1):
        if "|" in lines[i] and re.search(r"\d", lines[i]):
            last_data_idx = i
            break

    if last_data_idx == -1:
        return None, None, None

    # Create contradictory row by multiplying numbers
    original_row = lines[last_data_idx]
    cells = original_row.split("|")
    new_cells = []
    modified = False

    for cell in cells:
        nums = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)", cell)
        new_cell = cell
        for num_str in nums:
            try:
                num_val = float(num_str.replace(",", ""))
                contra_val = num_val * 1.5
                if "." in num_str:
                    new_cell = new_cell.replace(num_str, f"{contra_val:.2f}", 1)
                else:
                    new_cell = new_cell.replace(num_str, f"{int(contra_val):,}", 1)
                modified = True
            except ValueError:
                continue
        new_cells.append(new_cell)

    if not modified:
        return None, None, None

    # Add "(conflicting report)" label to first non-empty cell
    for i, cell in enumerate(new_cells):
        stripped = cell.strip()
        if stripped and not re.match(r"^[\d,.\-\s%$]+$", stripped):
            new_cells[i] = cell.rstrip() + " (conflicting report) "
            break

    contra_row = "|".join(new_cells)
    new_lines = lines[: last_data_idx + 1] + [contra_row] + lines[last_data_idx + 1 :]

    return (
        "\n".join(new_lines),
        f"Added contradictory row after row {last_data_idx}",
        "Model should detect conflicting data rows and flag inconsistency",
    )


# ============================================================================
# TRANSFORMATION VALIDATION
# ============================================================================


def _solution_uses_hardcoded_values(python_solution: str) -> bool:
    """Check if python_solution uses only hardcoded values (no context parsing).

    If the solution doesn't extract data from context at all, transforming the
    context won't affect the solution output, making validation unreliable.
    """
    context_extraction_patterns = [
        r"json\.loads",
        r"\.split\(",
        r"\bcontext\b",
        r"\bparse\b",
        r"for\s+\w+\s+in\s+",
        r"re\.\w+\(",
        r"\.strip\(",
        r"\.replace\(",
        r"import\s+json",
        r"\beval\(",
        r"float\(\s*['\"]",
        r"int\(\s*['\"]",
    ]
    for pattern in context_extraction_patterns:
        if re.search(pattern, python_solution):
            return False
    return True


def validate_transformation(
    transformed_example: Dict, original_example: Dict
) -> Dict[str, Any]:
    """Validate that a transformation makes the problem unsolvable.

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
            "valid": True,
            "reason": "hardcoded_solution",
            "details": "Solution uses hardcoded values only — context transformation is valid",
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
    context = example.get("context", "")
    question = example.get("question", "")
    context_type = detect_context_type(context)

    python_solution = example.get("python_solution", "")

    if context_type == "json":
        try:
            context_dict = json.loads(context)

            # Type 1: Information Removal
            new_ctx, desc, expected = transform_type1_json(context_dict, question)
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = "Type 1: Information Removal"
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)

            # Type 2: Column Removal
            new_ctx, desc, expected = transform_type2_json(context_dict, question)
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = "Type 2: Table Column Removal"
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)

            # Type 4: Critical Data Removal (silent)
            new_ctx, desc, expected = transform_type4_json(
                context_dict, question, python_solution
            )
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = "Type 4: Critical Data Removal"
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)

            # Type 5: Contradictory Information
            new_ctx, desc, expected = transform_type5_json(context_dict, question)
            if new_ctx:
                trans = deepcopy(example)
                trans["context"] = json.dumps(new_ctx)
                trans["transformation_type"] = "Type 5: Contradictory Information"
                trans["transformation_description"] = desc
                trans["expected_behavior"] = expected
                transformations.append(trans)
        except Exception:
            pass

    elif context_type == "text":
        # Type 1: Information Removal
        new_ctx, desc, expected = transform_type1_text(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 1: Information Removal"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # Type 4: Critical Data Removal
        new_ctx, desc, expected = transform_type4_text(context, question, python_solution)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 4: Critical Data Removal"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # Type 5: Contradictory Information
        new_ctx, desc, expected = transform_type5_text(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 5: Contradictory Information"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

    elif context_type == "markdown":
        # Type 1: Information Removal
        new_ctx, desc, expected = transform_type1_markdown(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 1: Information Removal"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # Type 2: Column Removal
        new_ctx, desc, expected = transform_type2_markdown(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 2: Table Column Removal"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # Type 4: Critical Data Removal
        new_ctx, desc, expected = transform_type4_markdown(
            context, question, python_solution
        )
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 4: Critical Data Removal"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

        # Type 5: Contradictory Information
        new_ctx, desc, expected = transform_type5_markdown(context, question)
        if new_ctx:
            trans = deepcopy(example)
            trans["context"] = new_ctx
            trans["transformation_type"] = "Type 5: Contradictory Information"
            trans["transformation_description"] = desc
            trans["expected_behavior"] = expected
            transformations.append(trans)

    # Type 3: Ambiguous Time (applies to all)
    new_q, desc, expected = transform_type3_question(question)
    if new_q:
        trans = deepcopy(example)
        trans["question"] = new_q
        trans["transformation_type"] = "Type 3: Ambiguous Time Period"
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

        print(f"\nResults:")
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
