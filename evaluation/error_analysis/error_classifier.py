"""
Error Classifier for FinanceReasoning Evaluation

Rule-based automatic classification of LLM errors.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .error_taxonomy import ErrorCategory, ErrorClassification


@dataclass
class ClassificationResult:
    """Result from evaluating a single example"""

    example_id: str
    model_name: str
    method: str  # COT or POT
    ground_truth: Any
    predicted_answer: Any
    is_correct: bool
    raw_response: str
    context: str
    execution_error: Optional[str] = None
    executed_code: Optional[str] = None
    parse_status: str = "success"  # success, failed, partial
    deviation: Optional[float] = None  # Relative error


class ErrorClassifier:
    """Rule-based error classifier for financial reasoning tasks"""

    # Tolerance thresholds
    EXACT_MATCH_TOLERANCE = 0.0001  # 0.01%
    ROUNDING_TOLERANCE = 0.002  # 0.2%
    CALCULATION_TOLERANCE = 0.10  # 10%

    def __init__(self):
        pass

    def classify(
        self,
        result: ClassificationResult,
        ground_truth: Any,
        context: str,
    ) -> ErrorClassification:
        """
        Classify an error case.

        Priority order:
        1. Execution Error (POT)
        2. Parsing Error
        3. Rounding Error (< 0.2%)
        4. Calculation Error (< 10%)
        5. Hallucination
        6. Unknown (requires manual classification)
        """

        # 1. Execution Error (POT)
        if result.execution_error:
            return self._classify_execution_error(result)

        # 2. Parsing Error
        if result.parse_status == "failed" or result.predicted_answer is None:
            return ErrorClassification(
                category=ErrorCategory.PARSING_ERROR,
                confidence=0.95,
                evidence="Failed to parse answer from response",
                details={
                    "raw_response_preview": result.raw_response[:200],
                    "parse_status": result.parse_status,
                },
            )

        # 3. Numeric comparison (if answer is available)
        if result.predicted_answer is not None and ground_truth is not None:
            deviation = self._calculate_deviation(result.predicted_answer, ground_truth)
            result.deviation = deviation

            # 3a. Rounding Error (< 0.2%)
            if deviation is not None and deviation < self.ROUNDING_TOLERANCE:
                return ErrorClassification(
                    category=ErrorCategory.ROUNDING_ERROR,
                    confidence=0.90,
                    evidence=f"Answer within {deviation*100:.3f}% of ground truth (< 0.2% tolerance)",
                    details={
                        "predicted": result.predicted_answer,
                        "ground_truth": ground_truth,
                        "deviation_percent": deviation * 100,
                    },
                )

            # 3b. Calculation Error (< 10%)
            if deviation is not None and deviation < self.CALCULATION_TOLERANCE:
                return ErrorClassification(
                    category=ErrorCategory.CALCULATION_ERROR,
                    confidence=0.70,
                    evidence=f"Answer within {deviation*100:.1f}% of ground truth (likely calculation error)",
                    details={
                        "predicted": result.predicted_answer,
                        "ground_truth": ground_truth,
                        "deviation_percent": deviation * 100,
                    },
                )

        # 4. Hallucination detection
        hallucination_result = self._detect_hallucination(result, context)
        if hallucination_result:
            return hallucination_result

        # 5. Unknown (requires manual classification)
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            confidence=0.30,
            evidence="Could not automatically classify error - manual review needed",
            details={
                "predicted": result.predicted_answer,
                "ground_truth": ground_truth,
                "deviation_percent": (
                    (result.deviation * 100) if result.deviation else None
                ),
            },
            auto_classified=False,
        )

    def _classify_execution_error(
        self, result: ClassificationResult
    ) -> ErrorClassification:
        """Classify POT execution errors"""

        error_msg = result.execution_error or ""

        # Common error patterns
        error_patterns = {
            "SyntaxError": "Python syntax error in generated code",
            "NameError": "Undefined variable in generated code",
            "TypeError": "Type mismatch in calculation",
            "ZeroDivisionError": "Division by zero",
            "ValueError": "Invalid value conversion",
            "IndexError": "Index out of range",
            "KeyError": "Missing dictionary key",
        }

        error_type = "Unknown execution error"
        for pattern, description in error_patterns.items():
            if pattern in error_msg:
                error_type = description
                break

        return ErrorClassification(
            category=ErrorCategory.EXECUTION_ERROR,
            confidence=0.95,
            evidence=f"POT code execution failed: {error_type}",
            details={
                "error_message": error_msg,
                "error_type": error_type,
                "executed_code": result.executed_code,
            },
        )

    def _calculate_deviation(
        self, predicted: Any, ground_truth: Any
    ) -> Optional[float]:
        """Calculate relative deviation between predicted and ground truth"""

        try:
            pred_num = float(str(predicted).replace(",", "").replace("%", ""))
            gt_num = float(str(ground_truth).replace(",", "").replace("%", ""))

            if gt_num == 0:
                return abs(pred_num) if pred_num != 0 else 0.0

            return abs(pred_num - gt_num) / abs(gt_num)
        except (ValueError, TypeError):
            return None

    def _detect_hallucination(
        self, result: ClassificationResult, context: str
    ) -> Optional[ErrorClassification]:
        """
        Detect hallucination by checking if used values exist in context.

        Looks for:
        1. Numbers in response that don't appear in context
        2. Entity names not in context
        """

        if not context or context == "[]":
            return None

        # Extract numbers from context
        context_numbers = set()
        for match in re.findall(r"[\d,]+\.?\d*", context):
            try:
                num = float(match.replace(",", ""))
                context_numbers.add(num)
                # Also add common transformations
                context_numbers.add(num * 100)  # Percentage
                context_numbers.add(num / 100)  # Decimal
                context_numbers.add(num * 1000)  # Thousands
                context_numbers.add(num / 1000)  # Millions to thousands
            except ValueError:
                continue

        # Extract numbers from response/code
        response_to_check = result.raw_response
        if result.executed_code:
            response_to_check = result.executed_code

        response_numbers = set()
        for match in re.findall(r"[\d,]+\.?\d*", response_to_check):
            try:
                num = float(match.replace(",", ""))
                # Skip very small numbers (likely indices or constants)
                if num > 0.01 and num not in [1, 2, 100, 1000, 365, 12, 52]:
                    response_numbers.add(num)
            except ValueError:
                continue

        # Find hallucinated numbers (in response but not in context)
        hallucinated = response_numbers - context_numbers

        # Filter out likely non-hallucinated values
        significant_hallucinations = [
            n for n in hallucinated if n > 1 and n < 1e12  # Reasonable range
        ]

        if len(significant_hallucinations) >= 2:
            return ErrorClassification(
                category=ErrorCategory.HALLUCINATION,
                confidence=0.60,
                evidence=f"Found {len(significant_hallucinations)} values in response not present in context",
                details={
                    "hallucinated_values": significant_hallucinations[:5],
                    "context_values_sample": list(context_numbers)[:10],
                },
            )

        return None

    def classify_batch(
        self,
        results: List[ClassificationResult],
        ground_truths: Dict[str, Any],
        contexts: Dict[str, str],
    ) -> Dict[str, ErrorClassification]:
        """Classify multiple results"""

        classifications = {}

        for result in results:
            if result.is_correct:
                continue  # Skip correct answers

            gt = ground_truths.get(result.example_id)
            ctx = contexts.get(result.example_id, "")

            key = f"{result.example_id}_{result.model_name}_{result.method}"
            classifications[key] = self.classify(result, gt, ctx)

        return classifications

    def get_error_distribution(
        self, classifications: Dict[str, ErrorClassification]
    ) -> Dict[ErrorCategory, int]:
        """Get distribution of error categories"""

        distribution = {cat: 0 for cat in ErrorCategory}

        for classification in classifications.values():
            distribution[classification.category] += 1

        return distribution

    def get_model_error_comparison(
        self,
        classifications: Dict[str, ErrorClassification],
    ) -> Dict[str, Dict[ErrorCategory, int]]:
        """Get error distribution by model"""

        model_errors = {}

        for key, classification in classifications.items():
            # Parse key: example_id_model_name_method
            parts = key.rsplit("_", 2)
            if len(parts) >= 2:
                model_name = parts[-2]
            else:
                model_name = "unknown"

            if model_name not in model_errors:
                model_errors[model_name] = {cat: 0 for cat in ErrorCategory}

            model_errors[model_name][classification.category] += 1

        return model_errors


# Singleton instance
_classifier_instance = None


def get_classifier() -> ErrorClassifier:
    """Get singleton classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ErrorClassifier()
    return _classifier_instance
