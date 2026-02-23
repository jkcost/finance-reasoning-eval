"""
Test FinanceReasoning evaluation framework with single example
Verifies all components work before running full evaluation
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add evaluation directory to path (evaluation is now at root)
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager
from dataset_loader import DatasetLoader, Example
from model_runner import ModelRunner
from prompt_builder import PromptBuilder
from response_parser import ResponseParser
from metrics_evaluator import MetricsEvaluator


async def test_single_example():
    """Test evaluation with a single example"""

    # Load environment variables
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded .env from {env_file}")
    else:
        print(f"[WARN] .env file not found at {env_file}")

    print("\n" + "=" * 60)
    print("FinanceReasoning Test Evaluation")
    print("=" * 60)

    # Initialize components
    print("\n[1] Initializing components...")
    config_manager = ConfigManager()
    eval_config = config_manager.get_evaluation_config()

    # Show available models
    print(f"\n[2] Available models:")
    for model in eval_config.models:
        api_key_env_var = model.api_key_env_var
        has_key = api_key_env_var in [k for k in __import__("os").environ.keys()]
        status = "[OK]" if has_key else "[NO KEY]"
        print(f"  - {model.id:20} ({model.provider:10}) {status}")

    # Check if we have at least one model with API key
    import os

    models_with_keys = [
        m
        for m in eval_config.models
        if m.api_key_env_var and os.environ.get(m.api_key_env_var)
    ]

    if not models_with_keys:
        print("\n[ERROR] No models have API keys configured!")
        print("Please add API keys to .env file:")
        print("  - OPENAI_API_KEY=sk-...")
        print("  - ANTHROPIC_API_KEY=sk-ant-...")
        return

    print(f"\n[3] Testing with {len(models_with_keys)} model(s) that have API keys")

    # Load dataset
    print("\n[4] Loading dataset...")
    data_root = Path(__file__).parent.parent / "data" / "financereasoning"
    dataset_loader = DatasetLoader(data_root)

    # Load just 1 hard example for testing
    examples = dataset_loader.load_hard(limit=1, level="hard")

    if not examples:
        print("[ERROR] No examples loaded!")
        return

    example = examples[0]
    print(f"  [OK] Loaded example: {example.id}")
    print(f"  Question: {example.question[:80]}...")
    print(f"  Ground Truth: {example.ground_truth_final}")

    # Build prompt
    print("\n[5] Building prompt...")
    prompt_builder = PromptBuilder("finance_reasoning_compliant")
    prompt = prompt_builder.build_prompt(
        question=example.question,
        context=example.context[:500] + "...",  # Truncate for display
        python_solution=example.python_solution,
    )
    print(f"  [OK] Prompt length: {len(prompt)} characters")

    # Initialize model runner with only models that have API keys
    print("\n[6] Initializing model runner...")
    from config import EvaluationConfig, ModelConfig

    # Test with all available models
    test_config = EvaluationConfig(
        models=models_with_keys,  # Test with all models that have keys
        output_dir="test_results",
        max_cost_usd=1.0,  # Small budget for test
        concurrency_per_provider=1,
    )

    model_runner = ModelRunner(config_manager, test_config)
    print(f"  [OK] Initialized {len(model_runner.providers)} provider(s)")

    # Evaluate example
    print(f"\n[7] Evaluating example with {models_with_keys[0].id}...")
    responses = await model_runner.evaluate_example(
        example=example,
        template_type="finance_reasoning_compliant",
    )

    if not responses:
        print("[ERROR] No responses received!")
        return

    # Parse responses using ResponseParser
    print(f"\n[8] Parsing responses...")
    from response_parser import ResponseParser

    parser = ResponseParser()

    for model_id, response in responses.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_id}")
        print(f"{'=' * 60}")
        print(f"Parse Status: {response.parse_status}")
        print(
            f"Tokens: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})"
        )
        print(f"Cost: ${response.cost_usd:.4f}")
        print(f"Time: {response.response_time_seconds:.2f}s")
        print(f"\nRaw Response:")
        print(
            response.raw_response[:500] + "..."
            if len(response.raw_response) > 500
            else response.raw_response
        )

        # Parse response to extract final_answer
        # Note: We pass the full response_data structure (including nested API response)
        parsed_response = parser.parse_response(
            response.parsed_data,
            provider_type="anthropic",  # We're using Anthropic models
        )
        response.final_answer = parsed_response.final_answer
        print(f"\nParsed Final Answer: {response.final_answer}")

    # Evaluate metrics
    print(f"\n[9] Calculating metrics...")
    metrics_evaluator = MetricsEvaluator()

    # Need to pass example object with all fields
    from model_runner import LLMResponse

    # Create dummy responses dict for evaluation
    metrics = metrics_evaluator.evaluate_example(
        example=example,
        llm_responses=responses,
    )

    # Save results to JSON file
    results_dir = Path(__file__).parent / "test_results"
    results_dir.mkdir(exist_ok=True)

    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "example_id": example.id,
        "question": example.question,
        "ground_truth": example.ground_truth_final,
        "models": [],
    }

    for model_id, metric in metrics.items():
        print(f"\n{'=' * 60}")
        print(f"Metrics for {model_id}")
        print(f"{'=' * 60}")
        print(f"Final Answer Correct: {metric.final_answer_correct}")
        print(f"Step Completeness: {metric.step_completeness:.2%}")
        print(f"Step Order Correct: {metric.step_order_correct}")
        print(f"Reasoning Similarity: {metric.reasoning_similarity:.2%}")
        print(f"Hallucination Rate: {metric.hallucination_rate:.2%}")
        print(f"Overall Score: {metric.overall_reasoning_score:.3f}")

        # Get LLM response
        llm_response = responses[model_id]

        results_summary["models"].append(
            {
                "model_id": model_id,
                "final_answer": metric.ground_truth_final,
                "llm_answer": llm_response.final_answer,
                "final_answer_correct": metric.final_answer_correct,
                "final_answer_matches_type": metric.final_answer_matches_type,
                "step_completeness": metric.step_completeness,
                "step_order_correct": metric.step_order_correct,
                "step_order_score": metric.step_order_score,
                "reasoning_similarity": metric.reasoning_similarity,
                "hallucination_rate": metric.hallucination_rate,
                "has_hallucination": metric.has_hallucination,
                "overall_reasoning_score": metric.overall_reasoning_score,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "cost_usd": llm_response.cost_usd,
                "response_time_seconds": llm_response.response_time_seconds,
            }
        )

    # Save results
    results_file = (
        results_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[10] Results saved to: {results_file}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If test passed, run full evaluation: python main.py")
    print("2. Adjust models in config/local_secrets.yaml")
    print("3. Check .env for API key configuration")


if __name__ == "__main__":
    asyncio.run(test_single_example())
