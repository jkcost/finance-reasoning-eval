"""
Error Analysis Framework for FinanceReasoning Evaluation

This package provides tools for classifying and analyzing LLM errors
in financial reasoning tasks, based on the paper's error taxonomy.
"""

from .error_taxonomy import (
    ErrorCategory,
    ErrorClassification,
    ModelType,
    MODEL_REGISTRY,
)
from .error_classifier import ErrorClassifier
from .error_report_generator import ErrorReportGenerator
from .llm_error_analyzer import LLMErrorAnalyzer, ErrorAnalysisResult, analyze_error_with_llm

__all__ = [
    "ErrorCategory",
    "ErrorClassification",
    "ModelType",
    "MODEL_REGISTRY",
    "ErrorClassifier",
    "ErrorReportGenerator",
    "LLMErrorAnalyzer",
    "ErrorAnalysisResult",
    "analyze_error_with_llm",
]
