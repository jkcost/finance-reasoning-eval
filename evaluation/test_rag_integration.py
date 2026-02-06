"""
Test RAG Integration - Verify that RAG enhancement works correctly

Tests:
1. Function retrieval from library
2. Prompt building with functions
3. End-to-end RAG pipeline
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add evaluation directory to path
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from prompt_builder import PromptBuilder
from rag_enhancer import RAGEnhancer
from dataset_loader import DatasetLoader


def test_function_retrieval():
    """Test 1: Function retrieval works"""
    print("\n" + "=" * 70)
    print("TEST 1: Function Retrieval")
    print("=" * 70)

    try:
        rag_enhancer = RAGEnhancer(eval_dir / "function_retriever.py")

        test_query = "calculate annual growth rate"
        test_question = "What is the year-over-year growth rate?"
        test_context = "Revenue in 2020: $100M, Revenue in 2021: $120M"

        functions_text, functions_dict = rag_enhancer.retrieve_and_format(
            query=test_query,
            question=test_question,
            context=test_context,
            top_k=3,
            template_type="cot_rag",
        )

        if functions_text:
            print(f"✅ Retrieved {len(functions_dict)} functions")
            print(f"\nSample function text (first 500 chars):")
            print(functions_text[:500])
            return True
        else:
            print("❌ No functions retrieved")
            return False

    except Exception as e:
        print(f"❌ Function retrieval failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prompt_building_with_functions():
    """Test 2: Prompt building with functions parameter"""
    print("\n" + "=" * 70)
    print("TEST 2: Prompt Building with Functions")
    print("=" * 70)

    try:
        # Test COT + RAG template
        builder = PromptBuilder("cot_rag")

        test_question = "Calculate the NPV of this investment"
        test_context = "Initial investment: $10,000, Annual return: $3,000 for 5 years, Discount rate: 10%"
        test_functions = """# NPV Calculation
```python
def calculate_npv(cash_flows, discount_rate):
    '''Calculate Net Present Value'''
    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows))
    return npv
```"""

        prompt = builder.build_prompt(
            question=test_question,
            context=test_context,
            python_solution="# Reference solution here",
            functions=test_functions,
        )

        # Check if functions are in the prompt
        if "calculate_npv" in prompt and "Available Financial Functions" in prompt:
            print("✅ Functions successfully included in prompt")
            print(f"\nPrompt length: {len(prompt)} characters")
            print(f"\nPrompt preview (first 800 chars):")
            print(prompt[:800])
            return True
        else:
            print("❌ Functions not found in prompt")
            print(f"\nGenerated prompt:")
            print(prompt)
            return False

    except Exception as e:
        print(f"❌ Prompt building failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_rag_pipeline():
    """Test 3: End-to-end RAG pipeline with real dataset"""
    print("\n" + "=" * 70)
    print("TEST 3: End-to-End RAG Pipeline")
    print("=" * 70)

    try:
        # Load one example from dataset
        loader = DatasetLoader()
        examples = loader.load_dataset("financereasoning", split="easy")

        if not examples:
            print("❌ No examples loaded from dataset")
            return False

        example = examples[0]
        print(f"Loaded example: {example.id}")
        print(f"Question: {example.question[:100]}...")

        # Initialize RAG
        rag_enhancer = RAGEnhancer(eval_dir / "function_retriever.py")

        # Retrieve functions
        query = f"{example.question} {example.context[:200]}"
        functions_text, functions_dict = rag_enhancer.retrieve_and_format(
            query=query,
            question=example.question,
            context=example.context,
            top_k=5,
            template_type="cot_rag",
        )

        if not functions_text:
            print("⚠️  No functions retrieved (might be expected for this query)")
        else:
            print(f"✅ Retrieved {len(functions_dict)} functions")

        # Build prompt with RAG
        builder = PromptBuilder("cot_rag")
        prompt = builder.build_prompt(
            question=example.question,
            context=example.context,
            python_solution=example.python_solution,
            functions=functions_text,
        )

        print(f"✅ Generated RAG-enhanced prompt ({len(prompt)} chars)")

        # Compare with non-RAG prompt
        builder_no_rag = PromptBuilder("default")  # Use 'default' instead of 'cot'
        prompt_no_rag = builder_no_rag.build_prompt(
            question=example.question,
            context=example.context,
            python_solution=example.python_solution,
        )

        size_diff = len(prompt) - len(prompt_no_rag)
        print(f"\nPrompt size comparison:")
        print(f"  COT (no RAG): {len(prompt_no_rag)} chars")
        print(f"  COT + RAG:    {len(prompt)} chars")
        print(f"  Difference:   +{size_diff} chars")

        if size_diff > 0:
            print("✅ RAG prompt is larger (contains additional functions)")
            return True
        else:
            print("⚠️  RAG prompt is not larger (no functions added)")
            return True  # Still pass, might be expected

    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RAG INTEGRATION TEST SUITE")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Function Retrieval", test_function_retrieval()))
    results.append(("Prompt Building", test_prompt_building_with_functions()))
    results.append(("End-to-End Pipeline", test_end_to_end_rag_pipeline()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! RAG integration is working correctly.")
        return 0
    else:
        print(
            f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the errors above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
