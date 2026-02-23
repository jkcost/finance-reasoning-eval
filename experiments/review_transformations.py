"""
Generate a human-readable review document for transformation samples.
"""

import json
from pathlib import Path

INPUT_FILE = Path("experiments/transformation_samples/transformation_samples_v2.json")
OUTPUT_FILE = Path("experiments/transformation_samples/REVIEW.md")


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to max length."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def generate_review_markdown():
    """Generate markdown review document."""

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    lines.append("# Transformation Samples Review\n")
    lines.append("**Generated from:** `transformation_samples_v2.json`\n")
    lines.append(
        "**Purpose:** Manual review of transformation quality before scaling to full dataset\n"
    )
    lines.append("\n---\n")

    for level in ["easy", "medium", "hard"]:
        lines.append(f"\n## {level.upper()} Problems\n")

        for prob_idx, problem_data in enumerate(data[level]):
            original = problem_data["original"]
            transformations = problem_data["transformations"]

            lines.append(f"\n### Problem {prob_idx + 1}: {original['question_id']}\n")
            lines.append(f"**Question:** {original['question']}\n")
            lines.append(f"**Ground Truth:** {original['ground_truth']}\n")
            lines.append(f"**Difficulty Score:** {original.get('difficulty', 'N/A')}\n")

            # Show original context (truncated)
            lines.append(
                f"\n**Original Context (truncated):**\n```\n{truncate(original['context'], 300)}\n```\n"
            )

            # Show python solution (truncated)
            lines.append(
                f"\n**Python Solution (truncated):**\n```python\n{truncate(original['python_solution'], 200)}\n```\n"
            )

            # Show transformations
            lines.append(f"\n**Transformations Generated:** {len(transformations)}\n")

            for trans_idx, trans in enumerate(transformations):
                lines.append(
                    f"\n#### Transformation {trans_idx + 1}: {trans['transformation_type']}\n"
                )
                lines.append(
                    f"- **Description:** {trans['transformation_description']}\n"
                )
                lines.append(f"- **Expected Behavior:** {trans['expected_behavior']}\n")

                # Show what changed
                if trans["question"] != original["question"]:
                    lines.append(f"\n**Question Changed:**\n")
                    lines.append(f"- Original: `{original['question']}`\n")
                    lines.append(f"- Transformed: `{trans['question']}`\n")

                if trans["context"] != original["context"]:
                    lines.append(f"\n**Context Changed (truncated):**\n")
                    lines.append(f"```\n{truncate(trans['context'], 300)}\n```\n")

                # Review questions
                lines.append(f"\n**Review Checklist:**\n")
                lines.append(
                    f"- [ ] Is the removed data truly critical for solving the problem?\n"
                )
                lines.append(
                    f"- [ ] Is this transformation realistic (could occur in real-world data)?\n"
                )
                lines.append(
                    f"- [ ] Is the expected model behavior clear and correct?\n"
                )
                lines.append(f"- [ ] Would a human recognize this as unsolvable?\n")

            lines.append("\n---\n")

    # Summary statistics
    lines.append("\n## Summary Statistics\n")
    for level in ["easy", "medium", "hard"]:
        total_problems = len(data[level])
        total_transformations = sum(len(p["transformations"]) for p in data[level])
        avg_per_problem = (
            total_transformations / total_problems if total_problems > 0 else 0
        )

        lines.append(f"\n### {level.upper()}\n")
        lines.append(f"- Total Problems: {total_problems}\n")
        lines.append(f"- Total Transformations: {total_transformations}\n")
        lines.append(f"- Average per Problem: {avg_per_problem:.1f}\n")

        # Count by type
        type_counts = {}
        for prob in data[level]:
            for trans in prob["transformations"]:
                trans_type = trans["transformation_type"]
                type_counts[trans_type] = type_counts.get(trans_type, 0) + 1

        lines.append(f"\n**By Transformation Type:**\n")
        for trans_type, count in sorted(type_counts.items()):
            lines.append(f"- {trans_type}: {count}\n")

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[SUCCESS] Review document generated: {OUTPUT_FILE}")
    print(f"\nPlease review the document and check:")
    print("1. Are transformations realistic?")
    print("2. Are removed data points truly critical?")
    print("3. Is expected behavior clear?")
    print("\nAfter review, we can proceed to:")
    print("- Create visualization with prompt display")
    print("- Scale to full dataset")


if __name__ == "__main__":
    generate_review_markdown()
