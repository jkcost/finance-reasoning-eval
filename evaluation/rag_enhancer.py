"""
RAG Enhancer - Retrieval-Augmented Generation for FinanceReasoning

Retrieves relevant financial functions and incorporates them into prompts
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


class RAGEnhancer:
    """Enhances prompts with retrieved financial functions"""

    def __init__(self, function_retriever_path: Path):
        """
        Args:
            function_retriever_path: Path to function_retriever.py module
        """
        self.function_retriever_path = function_retriever_path
        # Correct path: evaluation/../data/financereasoning/raw/functions
        project_root = function_retriever_path.parent.parent
        self.functions_dir = (
            project_root / "data" / "financereasoning" / "raw" / "functions"
        )

    def retrieve_and_format(
        self,
        query: str,
        question: str,
        context: str,
        top_k: int = 5,
        template_type: str = "cot_rag",
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Retrieve relevant functions and format for prompt

        Args:
            query: User query or reasoning
            question: Original question
            context: Problem context
            top_k: Number of functions to retrieve
            template_type: Prompt template type (cot_rag or pot_rag)

        Returns:
            (functions_formatted, functions_dict) tuple
        """

        try:
            # Import function retriever
            import sys
            import os

            eval_dir = self.function_retriever_path.parent
            if str(eval_dir) not in sys.path:
                sys.path.insert(0, str(eval_dir))

            from function_retriever import FunctionRetriever, Function

            # Load function library
            functions_file = self.functions_dir / "functions-article-all.json"

            if not functions_file.exists():
                print(f"[WARN] Functions file not found: {functions_file}")
                return "", None

            retriever = FunctionRetriever(functions_file)

            # Retrieve relevant functions
            functions = retriever.retrieve(query, top_k=top_k)

            # Format for prompt template
            functions_formatted = retriever.format_for_prompt(functions)
            functions_dict = [f.to_dict() for f in functions]

            return functions_formatted, functions_dict

        except Exception as e:
            print(f"[WARN] RAG retrieval failed: {e}")
            return "", None

    def enhance_prompt(
        self,
        original_prompt: str,
        question: str,
        context: str,
        top_k: int = 5,
        template_type: str = "cot_rag",
    ) -> str:
        """
        Enhance prompt with retrieved functions

        Args:
            original_prompt: Original prompt without functions
            question: User query
            context: Problem context
            top_k: Number of functions to retrieve
            template_type: Prompt template type

        Returns:
            Enhanced prompt with function instructions
        """

        functions_formatted, functions_dict = self.retrieve_and_format(
            query=f"{question} {context[:200]}",
            question=question,
            context=context,
            top_k=top_k,
            template_type=template_type,
        )

        if not functions_formatted:
            # Fallback: return original prompt
            return original_prompt

        # Build enhanced prompt
        enhanced_prompt = f"""{original_prompt}

Available Financial Functions:
{functions_formatted}

Instructions:
1. Review the available financial functions above
2. Use relevant functions to help solve the problem
3. Reference functions by name (e.g., "Using calculate_npv function")
4. If no relevant function exists, solve manually
"""

        return enhanced_prompt


def test_rag_enhancement():
    """Test RAG functionality"""

    from pathlib import Path

    print("=" * 60)
    print("RAG Enhancer Tests")
    print("=" * 60)

    # Initialize RAG enhancer
    evaluator_dir = Path(__file__).parent
    rag_enhancer = RAGEnhancer(evaluator_dir / "function_retriever.py")

    # Test 1: Growth calculation query
    test_question = "calculate the annual growth rate"
    test_context = """
Initial investment: $10,000
Ending value: $15,000
Years held: 5
"""

    print(f"\n[Test 1] Growth calculation query")
    print(f"Question: {test_question}")

    enhanced_prompt = rag_enhancer.enhance_prompt(
        original_prompt=f"Question: {test_question}\n\nContext:\n{test_context}",
        question=test_question,
        context=test_context,
        top_k=3,
        template_type="cot_rag",
    )

    print(f"\nEnhanced prompt:")
    print(enhanced_prompt)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag_enhancement()
