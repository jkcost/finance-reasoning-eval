"""
Function Retriever - RAG system for FinanceReasoning

Loads financial functions from functions-article-all.json and retrieves relevant functions
based on query keywords using BM25/Cosine similarity.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Function:
    """Financial function from library"""

    function_id: str
    article_title: str
    function: str
    function_docstring: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_id": self.function_id,
            "article_title": self.article_title,
            "function": self.function,
            "function_docstring": self.function_docstring,
        }


class FunctionRetriever:
    """Retrieves relevant financial functions based on query"""

    def __init__(self, functions_file: Path = None):
        """
        Args:
            functions_file: Path to functions-article-all.json
        """
        self.functions_file = functions_file
        # If functions_file is provided, use its parent directory
        # Otherwise, functions_dir will be None
        self.functions_dir = functions_file.parent if functions_file else None
        self.functions = self._load_functions()
        self._build_index()

    def _load_functions(self) -> List[Function]:
        """Load all functions from JSON file"""

        functions_file = (
            self.functions_dir / "functions-article-all.json"
            if self.functions_dir
            else self.functions_file
        )

        if not functions_file.exists():
            print(f"[WARN] Functions file not found: {functions_file}")
            return []

        with open(functions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        functions = []
        for item in data:
            try:
                func = Function(
                    function_id=item.get("function_id", ""),
                    article_title=item.get("article_title", ""),
                    function=item.get("function", ""),
                    function_docstring=item.get("function_docstring", ""),
                )
                functions.append(func)
            except Exception as e:
                print(f"[WARN] Failed to load function {item.get('function_id')}: {e}")
                continue

        print(f"[OK] Loaded {len(functions)} functions from {functions_file}")
        return functions

    def _build_index(self):
        """Build search index for fast retrieval

        Creates keyword to function mapping for BM25-like retrieval.
        """

        self.keyword_index: Dict[str, List[Function]] = {}

        for func in self.functions:
            # Extract keywords from function name and docstring
            keywords = self._extract_keywords(func)

            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(func)

        print(
            f"[OK] Built keyword index with {len(self.keyword_index)} unique keywords"
        )

    def _extract_keywords(self, func: Function) -> List[str]:
        """Extract keywords from function for indexing

        Extracts:
        - Function name (split by underscore/camelCase)
        - Key terms from docstring
        - Common financial terms
        """

        keywords = []

        # Function name keywords
        func_name = func.function.lower()
        name_parts = re.findall(r"[a-z]+", func_name)
        keywords.extend(name_parts)

        # Extract key terms from docstring
        docstring = func.function_docstring.lower()

        # Financial keywords to prioritize
        financial_terms = [
            "calculate",
            "return",
            "yield",
            "rate",
            "growth",
            "bonus",
            "probability",
            "salary",
            "investment",
            "revenue",
            "income",
            "ratio",
            "percentage",
            "value",
            "cost",
            "profit",
            "margin",
            "depreciation",
            "amortization",
            "npv",
            "irr",
            "cash",
            "flow",
            "earnings",
            "share",
            "dividend",
            "yoy",
            "ytd",
            "annual",
            "yearly",
            "monthly",
            "compounding",
            "discount",
            "interest",
            "principal",
            "payment",
            "annuity",
            "premium",
        ]

        for term in financial_terms:
            if term in docstring:
                keywords.append(term)

        # Extract words from docstring (remove stopwords)
        doc_words = re.findall(r"\b[a-z]{3,}\b", docstring)
        keywords.extend(doc_words)

        # Remove duplicates and filter short words
        keywords = list(set(k for k in keywords if len(k) >= 3))

        return keywords

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Function]:
        """
        Retrieve top-k relevant functions for a given query

        Args:
            query: Question or reasoning context
            top_k: Number of functions to retrieve

        Returns:
            List of relevant Function objects
        """

        # Extract keywords from query
        query_keywords = self._extract_query_keywords(query)
        query_keywords = set(query_keywords)

        # Score functions based on keyword overlap
        scored_functions = []

        for func in self.functions:
            func_keywords = set(self._extract_keywords(func))

            # Calculate overlap score (Jaccard-like)
            intersection = len(query_keywords & func_keywords)
            union = len(query_keywords | func_keywords)
            score = intersection / union if union > 0 else 0.0

            if score > 0:
                scored_functions.append((func, score))

        # Sort by score and return top-k
        scored_functions.sort(key=lambda x: x[1], reverse=True)
        top_functions = [f[0] for f in scored_functions[:top_k]]

        if not top_functions:
            # Fallback: return recent functions
            top_functions = self.functions[:top_k]

        return top_functions

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract keywords from user query"""

        query = query.lower()

        # Extract nouns and numbers
        keywords = re.findall(r"\b[a-z]+\b|\d+", query)

        # Add financial-specific terms if present
        if "return" in query:
            keywords.append("return")
        if "growth" in query:
            keywords.append("growth")
        if "rate" in query:
            keywords.append("rate")

        # Remove common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "of",
            "to",
            "in",
            "for",
            "with",
            "what",
            "how",
            "find",
            "get",
            "calculate",
            "compute",
        }

        keywords = [k for k in keywords if k not in stopwords and len(k) >= 3]

        return list(set(keywords))

    def retrieve_with_llm_optimize(
        self,
        query: str,
        top_k: int = 30,
        llm_optimizer=None,
    ) -> List[Function]:
        """
        Retrieve functions with optional LLM-based query optimization

        Args:
            query: Original query
            top_k: Number of functions to retrieve (before filtering)
            llm_optimizer: Optional LLM to optimize query

        Returns:
            Filtered list of relevant Function objects
        """

        # Initial retrieval
        functions = self.retrieve(query, top_k=top_k)

        # Optional: Use LLM to filter for relevance
        if llm_optimizer is not None and len(functions) > 0:
            functions = self._llm_filter_functions(
                query=query,
                functions=functions,
                llm=llm_optimizer,
            )

        return functions

    def _llm_filter_functions(
        self,
        query: str,
        functions: List[Function],
        llm,
    ) -> List[Function]:
        """
        Use LLM to filter retrieved functions for relevance

        This is a placeholder for future implementation.
        Would call LLM to ask: "Which of these functions are relevant for: {query}?"
        """

        # For now, return as-is (can add LLM filtering later)
        return functions

    def format_for_prompt(
        self,
        functions: List[Function],
    ) -> str:
        """
        Format retrieved functions for inclusion in prompt

        Format:
        ```python
        # Function 1
        def function_name(...):
            ...
        ```

        ```python
        # Function 2
        def function_name(...):
            ...
        ```
        """

        if not functions:
            return ""

        formatted = "Available financial functions:\n\n"

        for func in functions:
            formatted += f"# {func.article_title}\n"
            formatted += f"```python\n{func.function}\n```\n\n"

        return formatted


if __name__ == "__main__":
    import sys

    # Test function retrieval
    functions_file = (
        Path(__file__).parent.parent
        / "raw"
        / "functions"
        / "functions-article-all.json"
    )

    if not functions_file.exists():
        print(f"[ERROR] Functions file not found: {functions_file}")
        print(
            "Expected: data/financereasoning/raw/functions/functions-article-all.json"
        )
        sys.exit(1)

    retriever = FunctionRetriever(functions_file)

    print("=" * 60)
    print("Function Retriever Tests")
    print("=" * 60)

    # Test 1: Retrieve functions for growth-related query
    test_query = "calculate year over year growth rate"
    print(f"\nTest Query: {test_query}")
    functions = retriever.retrieve(test_query, top_k=3)

    print(f"Retrieved {len(functions)} functions:")
    for i, func in enumerate(functions, 1):
        print(f"  {i}. {func.article_title}")
        print(f"     Keywords: {', '.join(retriever._extract_keywords(func)[:5])}")

    # Test 2: Format for prompt
    print("\n" + "=" * 60)
    print("Formatted for Prompt:")
    print("=" * 60)
    print(retriever.format_for_prompt(functions))
