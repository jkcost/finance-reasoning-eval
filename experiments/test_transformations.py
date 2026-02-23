"""
Test transformation feasibility on sample FinanceReasoning data.

This script:
1. Loads sample problems from easy/medium/hard datasets
2. Attempts to apply each transformation type
3. Generates 1-3 sample transformations per type
4. Outputs results for manual review
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
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


def parse_context(context_str: str) -> Any:
    """Parse context string (JSON or markdown table)."""
    try:
        # Try JSON first
        return json.loads(context_str)
    except json.JSONDecodeError:
        # It's a markdown table or plain text
        return context_str


def extract_variables_from_solution(python_solution: str) -> List[str]:
    """Extract variable names from python solution."""
    # Simple regex to find variable assignments and usages
    variables = set()

    # Find assignments: var_name = ...
    assignments = re.findall(r"(\w+)\s*=", python_solution)
    variables.update(assignments)

    # Find dataframe accesses: df["column_name"]
    df_accesses = re.findall(r'df\["([^"]+)"\]', python_solution)
    variables.update(df_accesses)

    # Find dict accesses: dict_name["key"]
    dict_accesses = re.findall(r'(\w+)\["[^"]+"\]', python_solution)
    variables.update(dict_accesses)

    return list(variables)


# ============================================================================
# TRANSFORMATION TYPE 1: Information Removal
# ============================================================================


def transform_type1_information_removal(problem: Dict) -> Optional[Dict]:
    """
    Remove critical numerical values from context.

    Returns transformed problem or None if not applicable.
    """
    context = parse_context(problem["context"])

    # Only works with JSON context
    if not isinstance(context, dict):
        return None

    # Extract variables from solution
    variables = extract_variables_from_solution(problem["python_solution"])

    # Find a critical key to remove
    # Strategy: Remove first top-level key that appears in solution
    for key in context.keys():
        if key in variables or key.lower().replace(" ", "_") in [
            v.lower() for v in variables
        ]:
            # Found a critical key - create transformation
            transformed = copy.deepcopy(problem)
            transformed_context = copy.deepcopy(context)

            # Remove the critical key's data (set to null or remove)
            if isinstance(transformed_context[key], dict):
                # If it's a nested dict, remove a critical sub-key
                sub_keys = list(transformed_context[key].keys())
                if sub_keys:
                    # Remove first sub-key
                    del transformed_context[key][sub_keys[0]]
            else:
                # Remove the entire key
                del transformed_context[key]

            transformed["context"] = json.dumps(transformed_context)
            transformed["transformation_type"] = "Type 1: Information Removal"
            transformed["transformation_description"] = f"Removed critical data: {key}"
            transformed["expected_behavior"] = (
                f"Model should recognize missing {key} and refuse to answer"
            )

            return transformed

    return None


# ============================================================================
# TRANSFORMATION TYPE 2: Table Column Removal
# ============================================================================


def transform_type2_column_removal(problem: Dict) -> Optional[Dict]:
    """
    Remove entire columns from tables.

    Returns transformed problem or None if not applicable.
    """
    context = parse_context(problem["context"])

    # Only works with JSON context (dict of dicts)
    if not isinstance(context, dict):
        return None

    # Check if it's a table structure (dict of dicts)
    if not all(isinstance(v, dict) for v in context.values()):
        return None

    # Need at least 3 columns to safely remove 1
    if len(context) < 3:
        return None

    # Extract variables from solution
    variables = extract_variables_from_solution(problem["python_solution"])

    # Find a column (top-level key) to remove
    for key in list(context.keys())[: len(context) // 2]:  # Try first half of columns
        if key in variables or key.lower().replace(" ", "_") in [
            v.lower() for v in variables
        ]:
            transformed = copy.deepcopy(problem)
            transformed_context = copy.deepcopy(context)

            # Remove the entire column
            del transformed_context[key]

            transformed["context"] = json.dumps(transformed_context)
            transformed["transformation_type"] = "Type 2: Table Column Removal"
            transformed["transformation_description"] = f"Removed column: {key}"
            transformed["expected_behavior"] = (
                f"Model should recognize missing column {key} and refuse to answer"
            )

            return transformed

    return None


# ============================================================================
# TRANSFORMATION TYPE 3: Ambiguous Time Period
# ============================================================================


def transform_type3_ambiguous_time(problem: Dict) -> Optional[Dict]:
    """
    Make temporal references ambiguous in the question.

    Returns transformed problem or None if not applicable.
    """
    question = problem["question"]

    # Detect time references
    time_patterns = [
        (r"\b(20\d{2})\b", "the end of the period"),  # Year like 2019
        (r"\b(FY\s*20\d{2})\b", "the fiscal year"),  # FY 2019
        (r"\b(Q[1-4]\s*20\d{2})\b", "the quarter"),  # Q1 2023
        (
            r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b",
            "from START_YEAR to the end",
        ),  # from 2012 to 2017
    ]

    for pattern, replacement in time_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            transformed = copy.deepcopy(problem)

            # Replace the time reference with ambiguous term
            if "from" in pattern:
                # Special handling for ranges
                transformed_question = re.sub(
                    r"from\s+(\d{4})\s+to\s+(\d{4})",
                    lambda m: f"from {m.group(1)} to the end of the period",
                    question,
                    flags=re.IGNORECASE,
                )
            else:
                transformed_question = re.sub(
                    pattern, replacement, question, flags=re.IGNORECASE
                )

            transformed["question"] = transformed_question
            transformed["transformation_type"] = "Type 3: Ambiguous Time Period"
            transformed["transformation_description"] = (
                f"Made time reference ambiguous: {match.group(0)} → {replacement}"
            )
            transformed["expected_behavior"] = (
                "Model should recognize incomplete time period and refuse to answer"
            )

            return transformed

    return None


# ============================================================================
# TRANSFORMATION TYPE 4: Missing Formula Inputs
# ============================================================================


def transform_type4_missing_formula_inputs(problem: Dict) -> Optional[Dict]:
    """
    Remove variables required for financial formulas.

    Returns transformed problem or None if not applicable.
    """
    context = parse_context(problem["context"])
    solution = problem["python_solution"]

    # Only works with JSON context
    if not isinstance(context, dict):
        return None

    # Detect common financial formulas
    formula_patterns = {
        "ratio": r"(\w+)\s*/\s*(\w+)",  # Division suggests ratio
        "yield": r"(\w+)\s*/\s*(\w+)",
        "return": r"\((\w+)\s*-\s*(\w+)\)\s*/\s*(\w+)",  # (end - start) / start
    }

    for formula_type, pattern in formula_patterns.items():
        match = re.search(pattern, solution)
        if match:
            # Found a formula - remove one of the inputs
            variables = match.groups()
            if len(variables) >= 2:
                # Remove the first variable from context
                var_to_remove = variables[0]

                # Find corresponding key in context
                for key in context.keys():
                    if key.lower().replace(" ", "_") == var_to_remove.lower():
                        transformed = copy.deepcopy(problem)
                        transformed_context = copy.deepcopy(context)

                        # Remove the key
                        del transformed_context[key]

                        transformed["context"] = json.dumps(transformed_context)
                        transformed["transformation_type"] = (
                            "Type 4: Missing Formula Inputs"
                        )
                        transformed["transformation_description"] = (
                            f"Removed formula input: {key} (for {formula_type})"
                        )
                        transformed["expected_behavior"] = (
                            f"Model should recognize missing {key} needed for {formula_type} calculation"
                        )

                        return transformed

    return None


# ============================================================================
# TRANSFORMATION TYPE 5: Contradictory Information
# ============================================================================


def transform_type5_contradictory_info(problem: Dict) -> Optional[Dict]:
    """
    Inject conflicting data into the question.

    Returns transformed problem or None if not applicable.
    """
    context = parse_context(problem["context"])
    solution = problem["python_solution"]

    # Only works with JSON context
    if not isinstance(context, dict):
        return None

    # Look for summation in solution
    if "sum(" in solution or "+" in solution:
        # Try to inject contradictory total
        transformed = copy.deepcopy(problem)

        # Get the ground truth
        ground_truth = problem["ground_truth"]

        # Create a contradictory value (20% different)
        contradictory_value = ground_truth * 1.2

        # Add contradiction to question
        transformed["question"] = (
            problem["question"]
            + f" The annual report states the total is {contradictory_value:.2f}."
        )
        transformed["transformation_type"] = "Type 5: Contradictory Information"
        transformed["transformation_description"] = (
            f"Added contradictory total: {contradictory_value:.2f} (actual: {ground_truth})"
        )
        transformed["expected_behavior"] = (
            "Model should detect inconsistency between calculated and stated values"
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

    # Transformation functions
    transformations = [
        transform_type1_information_removal,
        transform_type2_column_removal,
        transform_type3_ambiguous_time,
        transform_type4_missing_formula_inputs,
        transform_type5_contradictory_info,
    ]

    results = {"easy": [], "medium": [], "hard": []}

    print("\nGenerating transformations...\n")

    for level in ["easy", "medium", "hard"]:
        print(f"=== {level.upper()} Problems ===")

        for i, problem in enumerate(samples[level]):
            print(f"\nProblem {i + 1}: {problem['question_id']}")
            print(f"Question: {problem['question'][:80]}...")

            problem_results = {"original": problem, "transformations": []}

            # Try each transformation type
            for transform_func in transformations:
                transformed = transform_func(problem)
                if transformed:
                    print(f"  [OK] {transformed['transformation_type']}")
                    problem_results["transformations"].append(transformed)
                else:
                    print(f"  [SKIP] {transform_func.__name__} - Not applicable")

            results[level].append(problem_results)

    # Save results
    output_file = OUTPUT_DIR / "transformation_samples.json"
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

    print("\nNext step: Review transformation_samples.json manually")


if __name__ == "__main__":
    main()
