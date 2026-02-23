"""
Run Quick Experiment - One-click script for FinanceReasoning evaluation

This script:
1. Runs quick_evaluation.py with hard.json samples
2. Generates HTML visualization
3. Opens the dashboard in browser

Usage:
    python experiments/run_quick_experiment.py
    python experiments/run_quick_experiment.py --num 3 --model claude-sonnet-4.5
"""

import asyncio
import subprocess
import sys
import webbrowser
from pathlib import Path


async def run_experiment(num_examples: int = 5, model: str = None):
    """Run the complete experiment pipeline"""

    project_root = Path(__file__).parent.parent
    experiments_dir = Path(__file__).parent

    print("=" * 70)
    print("FinanceReasoning Quick Experiment Runner")
    print("=" * 70)
    print(f"Examples: {num_examples}")
    print(f"Model: {model or 'auto-detect'}")
    print("=" * 70)

    # Step 1: Run evaluation
    print("\n[STEP 1/3] Running evaluation...")

    # Import and run directly
    sys.path.insert(0, str(experiments_dir))
    sys.path.insert(0, str(project_root / "evaluation"))

    from quick_evaluation import main as run_evaluation

    # Override sys.argv for argparse
    original_argv = sys.argv
    sys.argv = ["quick_evaluation.py", "--num", str(num_examples)]
    if model:
        sys.argv.extend(["--model", model])

    try:
        result = await run_evaluation()
    finally:
        sys.argv = original_argv

    if not result:
        print("[ERROR] Evaluation failed!")
        return None

    # Step 2: Generate visualization
    print("\n[STEP 2/3] Generating visualization...")

    from visualization import find_latest_results, generate_html_dashboard
    import json

    results_dir = experiments_dir / "quick_results"
    results_file = find_latest_results(results_dir)

    if not results_file:
        print("[ERROR] No results file found!")
        return None

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    output_html = results_file.with_suffix(".html")
    generate_html_dashboard(results, output_html)

    print(f"[OK] Dashboard generated: {output_html}")

    # Step 3: Open in browser
    print("\n[STEP 3/3] Opening dashboard in browser...")

    try:
        webbrowser.open(f"file://{output_html.absolute()}")
        print("[OK] Dashboard opened in browser")
    except Exception as e:
        print(f"[WARN] Could not open browser automatically: {e}")
        print(f"[INFO] Open manually: file://{output_html.absolute()}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults file: {results_file}")
    print(f"Dashboard: {output_html}")
    print(f"\nSummary:")

    metrics = results.get("aggregate_metrics", {})
    print(f"  COT Accuracy: {metrics.get('cot_accuracy', 0) * 100:.1f}%")
    print(f"  POT Accuracy: {metrics.get('pot_accuracy', 0) * 100:.1f}%")
    print(f"  Total Cost: ${metrics.get('total_cost_usd', 0):.4f}")

    return output_html


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run quick FinanceReasoning experiment"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=5, help="Number of examples (default: 5)"
    )
    parser.add_argument("--model", "-m", type=str, help="Model ID to use")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    args = parser.parse_args()

    asyncio.run(run_experiment(num_examples=args.num, model=args.model))


if __name__ == "__main__":
    main()
