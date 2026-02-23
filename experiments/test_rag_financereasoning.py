"""
Test FinanceReasoning evaluation with RAG enhancement
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add evaluation directory to path (evaluation is now at root)
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, EvaluationConfig, ModelConfig
from dataset_loader import DatasetLoader, Example
from prompt_builder import PromptBuilder
from model_runner import ModelRunner
from response_parser import ResponseParser
from metrics_evaluator import MetricsEvaluator
from rag_enhancer import RAGEnhancer


async def test_rag():
    """Test RAG-enhanced evaluation"""

    # Load environment variables
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded .env from {env_file}")

    # Initialize components
    print("\n[1] Initializing RAG test...")
    config_manager = ConfigManager()
    eval_config = config_manager.get_evaluation_config()

    # Initialize RAG enhancer
    functions_file = Path(__file__).parent.parent / "data" / "financereasoning" / "raw" / "functions" / "functions-article-all.json"
    print(f"  [OK] RAG enhancer initialized with functions file")

    # Load dataset
    print("\n[2] Loading dataset...")
    data_root = Path(__file__).parent.parent / "data" / "financereasoning"
    dataset_loader = DatasetLoader(data_root)
    examples = dataset_loader.load_hard(limit=1, level="hard")

    if not examples:
        print("[ERROR] No examples loaded!")
        return

    example = examples[0]
    print(f"  [OK] Loaded example: {example.id}")
    print(f"  Question: {example.question[:80]}...")
    print(f"  Ground Truth: {example.ground_truth_final}")

    # Create test config with RAG enabled
    test_config = EvaluationConfig(
        models=config_manager.get_models_for_evaluation()[:1],  # Test with first model
        output_dir="test_results_rag",
        max_cost_usd=1.0,  # Small budget for test
        concurrency_per_provider=1,
    )

    # Initialize RAG-enhanced model runner
    print("\n[3] Initializing RAG model runner...")
    rag_enhancer = RAGEnhancer(functions_file=functions_file)
    model_runner = ModelRunner(config_manager, test_config, rag_enhancer=rag_enhancer)
    print(f"  [OK] Initialized {len(model_runner.providers)} provider(s) with RAG support")

    # Build RAG-enhanced prompt
    print("\n[4] Building RAG-enhanced prompt...")
    question_text = example.question

    # Test RAG enhancement
    rag_enhanced_prompt = rag_enhancer.enhance_prompt(
        original_prompt=f"Question: {question_text}\n\nContext:\n{example.context[:500]}...",
        question=question_text,
        context=example.context[:500],
        top_k=3,
        template_type="cot_rag",
    )

    print(f"  [OK] RAG prompt length: {len(rag_enhanced_prompt)} characters")

    # Create a modified example with RAG prompt
    class RAGExample:
        def __init__(self, id, question, context, python_solution, ground_truth_final, rag_prompt=None):
            from dataset_loader import Example
            self._example = Example(
                id=id,
                question=question,
                context=context,
                ground_truth_final=ground_truth_final,
                python_solution=python_solution,
            )

    rag_example = RAGExample(
        id=example.id,
        question=example.question,
        context=example.context,
        python_solution=example.python_solution,
        ground_truth_final=example.ground_truth_final,
        rag_prompt=rag_enhanced_prompt,
    )

    print(f"\n[5] Running RAG evaluation...")
    print(f"  Question: {rag_example.question[:80]}...")

    responses = await model_runner.evaluate_example(
        example=rag_example,
        template_type="cot_rag",
    )

    if not responses:
        print("[ERROR] No responses received!")
        return

    # Show results
    for model_id, response in responses.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_id}")
        print(f"{'=' * 60}")
        print(f"Parse Status: {response.parse_status}")
        print(f"Tokens: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})")
        print(f"Cost: ${response.cost_usd:.4f}")
        print(f"Time: {response.response_time_seconds:.2f}s")

        if response.raw_response:
            print(f"Response preview: {response.raw_response[:200]}...")
        print(f"{'=' * 60}")

    # Calculate metrics
    print(f"\n[6] Calculating metrics...")

    from metrics_evaluator import MetricsEvaluator
    metrics_evaluator = MetricsEvaluator()

    metrics = metrics_evaluator.evaluate_example(
        example=rag_example,
        llm_responses=responses,
    )

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
        print(f"{'=' * 60}")

    print("\n" + "=" * 60)
    print("RAG Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update config/local_secrets.yaml to enable RAG strategies")
    print("2. Run full evaluation with RAG enabled")
    print("3. Compare COT vs RAG performance")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_rag())
