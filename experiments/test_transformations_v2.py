"""
Enhanced transformation script with support for all context types:
- JSON (Easy)
- Text paragraphs (Medium)
- Markdown tables (Hard)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import copy

# Paths
DATA_DIR = Path("data/financereasoning/raw/FinanceReasoning")
OUTPUT_DIR = Path("experiments/transformation_samples")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_sample_problems() -> Dict[str, List[Dict]]:
    """Load 2-3 sample problems from each difficulty level."""
    samples = {}

    for level in ["easy", "medium", "hard"]:
        file_path = DATA_DIR / f"{level}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Take first 3 problems as samples
        samples[level] = data[:3]

    return samples


def detect_context_type(context_str: str) -> str:
    """Detect whether context is JSON, markdown table, or text."""
    context_str = context_str.strip()

    # Try JSON first
    if context_str.startswith("{"):
        try:
            json.loads(context_str)
            return "json"
        except json.JSONDecodeError:
            pass

    # Check for markdown table
    if "|" in context_str and "---" in context_str:
        return "markdown_table"

    # Default to text
    return "text"


def parse_markdown_table(table_str: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse markdown table into headers and rows.

    Returns:
        (headers, rows) where rows is list of dicts mapping header -> value
    """
    lines = [line.strip() for line in table_str.split("\n") if line.strip()]

    # Find table lines (contain |)
    table_lines = [line for line in lines if "|" in line]

    if len(table_lines) < 3:  # Need header, separator, and at least 1 data row
        return [], []

    # Parse header - first line with |
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.split("|")]
    # Remove empty strings from start/end
    headers = [h for h in headers if h]

    # Find separator line (contains ---)
    separator_idx = None
    for i, line in enumerate(table_lines):
        if "---" in line or ":---" in line or "---:" in line:
            separator_idx = i
            break

    if separator_idx is None:
        return headers, []

    # Parse data rows (after separator)
    rows = []
    for line in table_lines[separator_idx + 1 :]:
        cells = [c.strip() for c in line.split("|")]
        # Remove empty strings from start/end
        cells = [c for c in cells if c]

        # Match cells to headers (handle mismatched lengths)
        if cells:
            row_dict = {}
            for i in range(min(len(headers), len(cells))):
                row_dict[headers[i]] = cells[i]
            rows.append(row_dict)

    return headers, rows


def extract_numbers_from_text(text: str) -> List[Tuple[str, float]]:
    """
    Extract numbers from text with their context.

    Returns list of (context_phrase, number) tuples.
    """
    # Pattern to match numbers with optional $ and commas
    number_pattern = r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"

    matches = []
    for match in re.finditer(number_pattern, text):
        # Get surrounding context (20 chars before and after)
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        context = text[start:end]

        # Parse number (remove $ and commas)
        num_str = match.group(1).replace(",", "")
        try:
            number = float(num_str)
            matches.append((context, number, match.start(), match.end()))
        except ValueError:
            continue

    return matches


def extract_variables_from_solution(python_solution: str) -> List[str]:
    """Extract variable names from python solution."""
    variables = set()

    # Find assignments: var_name = ...
    assignments = re.findall(r"(\w+)\s*=", python_solution)
    variables.update(assignments)

    # Find dataframe accesses: df["column_name"]
    df_accesses = re.findall(r'df\["([^"]+)"\]', python_solution)
    variables.update(df_accesses)

    # Find dict accesses
    dict_accesses = re.findall(r'(\w+)\["[^"]+"\]', python_solution)
    variables.update(dict_accesses)

    return list(variables)


# ============================================================================
# TRANSFORMATION TYPE 1: Information Removal
# ============================================================================


def transform_type1_json(problem: Dict, context: dict) -> Optional[Dict]:
    """Type 1 for JSON context."""
    variables = extract_variables_from_solution(problem["python_solution"])

    for key in context.keys():
        if key in variables or key.lower().replace(" ", "_") in [
            v.lower() for v in variables
        ]:
            transformed = copy.deepcopy(problem)
            transformed_context = copy.deepcopy(context)

            if isinstance(transformed_context[key], dict):
                sub_keys = list(transformed_context[key].keys())
                if sub_keys:
                    del transformed_context[key][sub_keys[0]]
            else:
                del transformed_context[key]

            transformed["context"] = json.dumps(transformed_context)
            transformed["transformation_type"] = "Type 1: Information Removal"
            transformed["transformation_description"] = f"Removed critical data: {key}"
            transformed["expected_behavior"] = (
                f"Model should recognize missing {key} and refuse to answer"
            )

            return transformed

    return None


def transform_type1_text(problem: Dict, context_str: str) -> Optional[Dict]:
    """Type 1 for text context - remove critical numbers."""
    numbers = extract_numbers_from_text(context_str)

    if not numbers:
        return None

    # Remove the first significant number (> 100)
    for context_phrase, number, start, end in numbers:
        if number > 100:  # Significant number
            transformed = copy.deepcopy(problem)

            # Replace number with [REDACTED]
            transformed_context = (
                context_str[:start] + "[DATA MISSING]" + context_str[end:]
            )

            transformed["context"] = transformed_context
            transformed["transformation_type"] = "Type 1: Information Removal"
            transformed["transformation_description"] = (
                f"Removed critical number: {number}"
            )
            transformed["expected_behavior"] = (
                "Model should recognize missing data and refuse to answer"
            )

            return transformed

    return None


def transform_type1_markdown(problem: Dict, context_str: str) -> Optional[Dict]:
    """Type 1 for markdown table - remove critical cell values."""
    headers, rows = parse_markdown_table(context_str)

    if not rows or not headers:
        return None

    # Remove a value from the first data row, first numeric column
    for i, header in enumerate(headers):
        if i == 0:  # Skip row label column
            continue

        # Check if column has numeric data
        first_value = rows[0].get(header, "")
        if re.search(r"\d", first_value):
            # Found numeric column - remove this value
            transformed = copy.deepcopy(problem)
            transformed_rows = copy.deepcopy(rows)
            transformed_rows[0][header] = "N/A"

            # Reconstruct markdown table
            new_table = reconstruct_markdown_table(headers, transformed_rows)

            # Replace table in context
            transformed_context = re.sub(
                r"\|.*?\|.*?\n\|.*?---.*?\n(?:\|.*?\|\n)+",
                new_table,
                context_str,
                count=1,
            )

            transformed["context"] = transformed_context
            transformed["transformation_type"] = "Type 1: Information Removal"
            transformed["transformation_description"] = (
                f"Removed cell value in column: {header}"
            )
            transformed["expected_behavior"] = (
                f"Model should recognize missing {header} data and refuse to answer"
            )

            return transformed

    return None


def reconstruct_markdown_table(headers: List[str], rows: List[Dict[str, str]]) -> str:
    """Reconstruct markdown table from headers and rows."""
    lines = []

    # Header line
    header_line = "| " + " | ".join(headers) + " |"
    lines.append(header_line)

    # Separator line
    separator = "| " + " | ".join(["---" for _ in headers]) + " |"
    lines.append(separator)

    # Data rows
    for row in rows:
        row_line = "| " + " | ".join([row.get(h, "") for h in headers]) + " |"
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def transform_type1_information_removal(problem: Dict) -> Optional[Dict]:
    """Type 1: Information Removal - unified entry point."""
    context_str = problem["context"]
    context_type = detect_context_type(context_str)

    if context_type == "json":
        context = json.loads(context_str)
        return transform_type1_json(problem, context)
    elif context_type == "text":
        return transform_type1_text(problem, context_str)
    elif context_type == "markdown_table":
        return transform_type1_markdown(problem, context_str)

    return None


# ============================================================================
# TRANSFORMATION TYPE 2: Table Column Removal
# ============================================================================


def transform_type2_json(problem: Dict, context: dict) -> Optional[Dict]:
    """Type 2 for JSON context."""
    if not all(isinstance(v, dict) for v in context.values()):
        return None

    if len(context) < 3:
        return None

    variables = extract_variables_from_solution(problem["python_solution"])

    for key in list(context.keys())[: len(context) // 2]:
        if key in variables or key.lower().replace(" ", "_") in [
            v.lower() for v in variables
        ]:
            transformed = copy.deepcopy(problem)
            transformed_context = copy.deepcopy(context)

            del transformed_context[key]

            transformed["context"] = json.dumps(transformed_context)
            transformed["transformation_type"] = "Type 2: Table Column Removal"
            transformed["transformation_description"] = f"Removed column: {key}"
            transformed["expected_behavior"] = (
                f"Model should recognize missing column {key} and refuse to answer"
            )

            return transformed

    return None


def transform_type2_markdown(problem: Dict, context_str: str) -> Optional[Dict]:
    """Type 2 for markdown table - remove entire column."""
    headers, rows = parse_markdown_table(context_str)

    if not rows or len(headers) < 3:  # Need at least 3 columns
        return None

    # Remove second column (first is usually row labels)
    if len(headers) >= 2:
        column_to_remove = headers[1]
        new_headers = [h for h in headers if h != column_to_remove]
        new_rows = [
            {k: v for k, v in row.items() if k != column_to_remove} for row in rows
        ]

        # Reconstruct table
        new_table = reconstruct_markdown_table(new_headers, new_rows)

        # Replace in context
        transformed = copy.deepcopy(problem)
        transformed_context = re.sub(
            r"\|.*?\|.*?\n\|.*?---.*?\n(?:\|.*?\|\n)+", new_table, context_str, count=1
        )

        transformed["context"] = transformed_context
        transformed["transformation_type"] = "Type 2: Table Column Removal"
        transformed["transformation_description"] = (
            f"Removed column: {column_to_remove}"
        )
        transformed["expected_behavior"] = (
            f"Model should recognize missing column {column_to_remove} and refuse to answer"
        )

        return transformed

    return None


def transform_type2_column_removal(problem: Dict) -> Optional[Dict]:
    """Type 2: Column Removal - unified entry point."""
    context_str = problem["context"]
    context_type = detect_context_type(context_str)

    if context_type == "json":
        context = json.loads(context_str)
        return transform_type2_json(problem, context)
    elif context_type == "markdown_table":
        return transform_type2_markdown(problem, context_str)
    else:
        # Text doesn't have columns
        return None


# ============================================================================
# TRANSFORMATION TYPE 3: Ambiguous Time Period
# ============================================================================


def transform_type3_ambiguous_time(problem: Dict) -> Optional[Dict]:
    """Type 3: Make temporal references ambiguous."""
    question = problem["question"]

    time_patterns = [
        (r"\b(20\d{2})\b", "the end of the period"),
        (r"\b(FY\s*20\d{2})\b", "the fiscal year"),
        (r"\b(Q[1-4]\s*20\d{2})\b", "the quarter"),
        (
            r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b",
            lambda m: f"from {m.group(1)} to the end of the period",
        ),
    ]

    for pattern, replacement in time_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            transformed = copy.deepcopy(problem)

            if callable(replacement):
                transformed_question = re.sub(
                    pattern, replacement, question, flags=re.IGNORECASE
                )
            else:
                transformed_question = re.sub(
                    pattern, replacement, question, flags=re.IGNORECASE
                )

            transformed["question"] = transformed_question
            transformed["transformation_type"] = "Type 3: Ambiguous Time Period"
            transformed["transformation_description"] = (
                f"Made time reference ambiguous: {match.group(0)} -> vague term"
            )
            transformed["expected_behavior"] = (
                "Model should recognize incomplete time period and refuse to answer"
            )

            return transformed

    return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Generate sample transformations for manual review."""

    print("Loading sample problems...")
    samples = load_sample_problems()

    # Only use Type 1, 2, 3 for now (most reliable)
    transformations = [
        transform_type1_information_removal,
        transform_type2_column_removal,
        transform_type3_ambiguous_time,
    ]

    results = {"easy": [], "medium": [], "hard": []}

    print("\nGenerating transformations...\n")

    for level in ["easy", "medium", "hard"]:
        print(f"=== {level.upper()} Problems ===")

        for i, problem in enumerate(samples[level]):
            print(f"\nProblem {i + 1}: {problem['question_id']}")
            print(f"Question: {problem['question'][:80]}...")

            context_type = detect_context_type(problem["context"])
            print(f"Context type: {context_type}")

            problem_results = {"original": problem, "transformations": []}

            # Try each transformation type
            for transform_func in transformations:
                try:
                    transformed = transform_func(problem)
                    if transformed:
                        print(f"  [OK] {transformed['transformation_type']}")
                        problem_results["transformations"].append(transformed)
                    else:
                        print(f"  [SKIP] {transform_func.__name__} - Not applicable")
                except Exception as e:
                    print(f"  [ERROR] {transform_func.__name__} - {str(e)}")

            results[level].append(problem_results)

    # Save results
    output_file = OUTPUT_DIR / "transformation_samples_v2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Results saved to: {output_file}")

    # Generate summary
    print("\n=== SUMMARY ===")
    for level in ["easy", "medium", "hard"]:
        total_problems = len(results[level])
        total_transformations = sum(len(p["transformations"]) for p in results[level])
        print(
            f"{level.upper()}: {total_transformations} transformations across {total_problems} problems"
        )

    print("\nNext step: Review transformation_samples_v2.json manually")


if __name__ == "__main__":
    main()
