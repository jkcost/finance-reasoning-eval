"""
Simple Test Example for FinanceReasoning Benchmark

Quick test to verify model connections and basic functionality.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)

# Add evaluation directory to path (evaluation is now at root)
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from dataset_loader import Example
from config import ConfigManager, EvaluationConfig
from model_runner import ModelRunner
from response_parser import ResponseParser


async def test_single_example():
    """Test a single FinanceReasoning example"""
    config_manager = ConfigManager()
    eval_config = config_manager.get_evaluation_config()

    model_runner = ModelRunner(config_manager, eval_config)

    print("=" * 60)
    print("Single Example Test")
    print("=" * 60)

    # Use a real FinanceReasoning example
    test_example = Example(
        id="test-001",
        question_id="test-2000",
        level="easy",
        source="finance_reasoning",
        question="what would the 2012 shares outstanding in millions have been without the acquisition of smith international? Answer to the nearest integer.",
        context="Schlumberger Limited and subsidiaries reported the following data:",
        python_solution="""
shares_outstanding = 1328
acquisition_cost = 176
answer = shares_outstanding - acquisition_cost
""",
        ground_truth_final=1152,
    )

    print("\nExample:")
    print(f"  Question: {test_example.question[:60]}...")
    print(f"  Ground Truth: {test_example.ground_truth_final}")

    responses = await model_runner.evaluate_example(
        example=test_example,
        template_type="finance_reasoning_compliant",
    )

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)

    for model_id, response in responses.items():
        status = "[OK]" if response.parse_status == "success" else "[FAIL]"
        print(f"\n{status} {model_id}")
        print(f"  Parse Status: {response.parse_status}")
        if response.final_answer is not None:
            print(f"  Final Answer: {response.final_answer}")
        print(f"  Tokens: {response.total_tokens}")
        print(f"  Cost: ${response.cost_usd:.6f}")
        print(f"  Time: {response.response_time_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(test_single_example())
