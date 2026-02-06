"""
MetricsEvaluator - Calculate FinanceReasoning paper metrics

Implements all paper metrics:
1. Final Answer Accuracy
2. Step Completeness
3. Step Order Correctness
4. Reasoning Similarity (Jaccard Index)
5. Hallucination Rate
6. Overall Reasoning Score
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json

from model_runner import LLMResponse

# Optional: Import error analysis for extended classification
try:
    from error_analysis import ErrorClassifier, ErrorClassification, ErrorCategory
    from error_analysis.error_classifier import ClassificationResult

    ERROR_ANALYSIS_AVAILABLE = True
except ImportError:
    ERROR_ANALYSIS_AVAILABLE = False


class StepType(str, Enum):
    """Types of reasoning steps"""

    VARIABLE_DEFINITION = "variable_definition"
    CALCULATION = "calculation"
    CONDITIONAL = "conditional"
    RETURN_RESULT = "return_result"


@dataclass
class GroundTruthStep:
    """Step from ground truth Python solution"""

    step_number: int
    step_type: StepType
    description: str
    code_snippet: str


@dataclass
class ParsedStep:
    """Parsed step from LLM response"""

    step_number: int
    step_type: StepType
    description: str
    code_snippet: str


@dataclass
class EvaluationMetrics:
    """Complete evaluation results for one example"""

    example_id: str
    model_id: str

    # Ground truth
    ground_truth_final: Any

    # Step metrics
    ground_truth_steps: Optional[List[GroundTruthStep]] = None
    llm_steps: Optional[List[ParsedStep]] = None

    # Accuracy metrics
    final_answer_correct: bool = False
    final_answer_matches_type: bool = False

    # Completeness metrics
    step_completeness: float = 0.0  # Percentage of ground truth steps covered

    # Order metrics
    step_order_correct: bool = False
    step_order_score: float = 0.0  # Levenshtein-like similarity

    # Similarity metrics
    reasoning_similarity: float = 0.0  # Jaccard index

    # Hallucination metrics
    has_hallucination: bool = False
    hallucination_rate: float = 0.0

    # Overall score
    overall_reasoning_score: float = 0.0

    # Extended error classification (from error_analysis module)
    error_classification: Optional[Any] = None  # ErrorClassification if available


class MetricsEvaluator:
    """Evaluates LLM responses against FinanceReasoning ground truth"""

    def __init__(self, enable_error_classification: bool = True):
        self.enable_error_classification = (
            enable_error_classification and ERROR_ANALYSIS_AVAILABLE
        )
        self.error_classifier = None

        if self.enable_error_classification:
            self.error_classifier = ErrorClassifier()

    def _parse_ground_truth_steps(self, python_solution: str) -> List[GroundTruthStep]:
        """Parse Python solution into ground truth steps"""

        steps = []
        lines = python_solution.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # Identify step type
            step_type = None
            description = ""

            if "=" in line and len(line.split("=")) == 2:
                # Variable definition: var = value
                step_type = StepType.VARIABLE_DEFINITION
                description = f"Define variable: {line}"
            elif any(op in line for op in ["+", "-", "*", "/", "**"]):
                # Calculation
                step_type = StepType.CALCULATION
                description = f"Calculate: {line}"
            elif "if" in line:
                step_type = StepType.CONDITIONAL
                description = f"Conditional check: {line}"
            elif "return" in line or "answer" in line:
                step_type = StepType.RETURN_RESULT
                description = f"Return result: {line}"

            if step_type:
                step = GroundTruthStep(
                    step_number=line_num,
                    step_type=step_type,
                    description=description,
                    code_snippet=line,
                )
                steps.append(step)

        return steps

    def _parse_llm_steps(
        self, parsed_data: Dict[str, Any]
    ) -> Optional[List[ParsedStep]]:
        """Parse reasoning steps from LLM structured output"""

        if not parsed_data or "reasoning_steps" not in parsed_data:
            return None

        llm_steps = []
        raw_steps = parsed_data["reasoning_steps"]

        for step_num, raw_step in enumerate(raw_steps, 1):
            step_type_str = raw_step.get("step_type", "unknown")
            description = raw_step.get("description", "")
            code_snippet = raw_step.get("code_snippet", "")

            # Map step type
            step_type = None
            try:
                step_type = StepType(step_type_str.lower())
            except ValueError:
                step_type = StepType.VARIABLE_DEFINITION

            step = ParsedStep(
                step_number=step_num,
                step_type=step_type,
                description=description,
                code_snippet=code_snippet,
            )

            llm_steps.append(step)

        return llm_steps

    def _compare_final_answers(
        self, ground_truth: Any, llm_answer: Any
    ) -> tuple[bool, bool]:
        """Compare final answers"""

        # Handle already parsed numeric answers
        try:
            # Convert ground_truth to float (may already be int or float)
            gt_num = float(str(ground_truth).replace(",", ""))

            # If llm_answer is already a number (not string), use it directly
            if isinstance(llm_answer, (int, float)):
                llm_num = float(llm_answer)
                print(
                    f"[DEBUG] llm_answer is already number: {llm_num}, type: {type(llm_answer)}"
                )
            else:
                # Try to extract number from JSON or text
                # First, try to parse as JSON
                try:
                    import json

                    if isinstance(llm_answer, str) and llm_answer.strip().startswith(
                        "{"
                    ):
                        parsed = json.loads(llm_answer)
                        llm_num = float(parsed.get("final_answer", 0))
                        print(f"[DEBUG] Parsed from JSON: {llm_num}")
                    else:
                        # Fallback: try to convert string directly
                        llm_num = float(str(llm_answer).replace(",", ""))
                        print(f"[DEBUG] Converted from string: {llm_num}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[DEBUG] JSON decode error: {e}, trying direct conversion")
                    try:
                        llm_num = float(str(llm_answer).replace(",", ""))
                        print(f"[DEBUG] Converted from string (fallback): {llm_num}")
                    except (ValueError, AttributeError) as e2:
                        print(f"[DEBUG] Direct conversion error: {e2}")
                        return False, False

            is_correct = abs(gt_num - llm_num) < 0.001

            print(
                f"[DEBUG] gt_num: {gt_num}, llm_num: {llm_num}, is_correct: {is_correct}"
            )

            # Extract ANSWER prefix if present
            gt_match_type = "ANSWER: " in str(ground_truth)
            llm_has_prefix = isinstance(
                llm_answer, str
            ) and llm_answer.strip().upper().startswith("ANSWER:")

            # matches_type is True if both have prefix or both don't have prefix
            matches_type = gt_match_type == llm_has_prefix

            return is_correct, matches_type
        except (ValueError, AttributeError) as e:
            print(f"[DEBUG] Exception in _compare_final_answers: {e}")
            return False, False

    def _compare_steps(
        self,
        ground_truth_steps: List[GroundTruthStep],
        llm_steps: Optional[List[ParsedStep]],
    ) -> tuple[float, bool]:
        """Compare reasoning steps"""

        if not llm_steps:
            return 0.0, False

        # Calculate step completeness
        matched_steps = 0
        total_gt_steps = len(ground_truth_steps)

        for gt_step in ground_truth_steps:
            for llm_step in llm_steps:
                if llm_step.step_number == gt_step.step_number:
                    # Check step type match
                    type_match = llm_step.step_type == gt_step.step_type

                    # Check content similarity (fuzzy match for description)
                    desc_similarity = self._similarity_score(
                        gt_step.description.lower(),
                        llm_step.description.lower(),
                    )

                    if type_match and desc_similarity > 0.5:
                        matched_steps += 1
                        break

        if total_gt_steps == 0:
            return 0.0, False

        completeness = matched_steps / total_gt_steps
        return completeness, bool(total_gt_steps == len(llm_steps))

    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings (0-1.0)"""

        s1 = str1.lower()
        s2 = str2.lower()

        # Levenshtein-like distance (simple version)
        if s1 == s2:
            return 1.0
        if s2 in s1:
            return 1.0
        if s1 in s2:
            return 0.5
        if s2 in s1:
            return 0.0

        # Jaccard index
        set1 = set(s1.split())
        set2 = set(s2.split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _calculate_hallucination_rate(
        self,
        ground_truth_steps: List[GroundTruthStep],
        llm_steps: Optional[List[ParsedStep]],
    ) -> float:
        """Calculate hallucination rate"""

        # Hallucination = extra steps not in ground truth
        if not llm_steps:
            return 0.0

        gt_step_numbers = set(s.step_number for s in ground_truth_steps)
        llm_step_numbers = set(s.step_number for s in llm_steps)

        # Extra steps (in LLM but not in ground truth)
        extra_steps = len(llm_step_numbers - gt_step_numbers)

        if len(ground_truth_steps) == 0:
            return 0.0

        rate = extra_steps / len(ground_truth_steps)
        return rate

    def _calculate_step_order_score(
        self,
        ground_truth_steps: List[GroundTruthStep],
        llm_steps: Optional[List[ParsedStep]],
    ) -> float:
        """Calculate step order similarity score"""

        if not llm_steps:
            return 0.0

        # Check if order matches
        is_correct_order = True
        min_len = min(len(ground_truth_steps), len(llm_steps))

        for i in range(min_len):
            gt = ground_truth_steps[i]
            llm = llm_steps[i] if i < len(llm_steps) else None

            if gt.step_type != llm.step_type:
                is_correct_order = False
                break

        # Levenshtein-like score (0-1.0 where 1.0 = perfect match)
        score = 1.0 if is_correct_order else 0.5

        return score

    def _calculate_reasoning_similarity(
        self,
        ground_truth_steps: List[GroundTruthStep],
        llm_steps: Optional[List[ParsedStep]],
    ) -> float:
        """Calculate reasoning similarity (Jaccard index)"""

        if not llm_steps:
            return 0.0

        # Convert to step type sets for comparison
        gt_types = set(s.step_type for s in ground_truth_steps)
        llm_types = set(s.step_type for s in llm_steps)

        if not gt_types or not llm_types:
            return 0.0

        # Jaccard index
        intersection = len(gt_types & llm_types)
        union = len(gt_types | llm_types)

        return intersection / union if union > 0 else 0.0

    def calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """
        Overall Reasoning Score formula from paper:

        Score = (Final Answer Correct × 0.4) +
                (Step Completeness × 0.3) +
                (Step Order Correct × 0.1) +
                ((1 - Hallucination Rate) × 0.2)
        """

        # Ensure all required metrics are present
        required_metrics = [
            "final_answer_correct",
            "step_completeness",
            "step_order_correct",
            "has_hallucination",
        ]

        for metric in required_metrics:
            if getattr(metrics, metric) is None:
                return 0.0

        score = (
            (metrics.final_answer_correct * 0.4)
            + (metrics.step_completeness * 0.3)
            + (metrics.step_order_correct * 0.1)
            + ((1.0 - metrics.hallucination_rate) * 0.2)
        )

        return score

    def evaluate_example(
        self,
        example: Any,
        llm_responses: Dict[str, LLMResponse],
    ) -> Dict[str, EvaluationMetrics]:
        """Evaluate single example across all models"""

        results = {}

        # Parse ground truth steps
        ground_truth_steps = None
        if hasattr(example, "python_solution") and example.python_solution:
            ground_truth_steps = self._parse_ground_truth_steps(example.python_solution)

        for model_id, llm_response in llm_responses.items():
            # Parse LLM steps
            llm_steps = self._parse_llm_steps(llm_response.parsed_data)

            # Compare final answers
            # Use final_answer if available (parsed from ResponseParser), otherwise use raw_response
            answer_to_compare = (
                llm_response.final_answer
                if llm_response.final_answer is not None
                else llm_response.raw_response
            )
            final_correct, matches_type = self._compare_final_answers(
                example.ground_truth_final,
                answer_to_compare,
            )

            # Compare steps
            step_completeness, complete_match = self._compare_steps(
                ground_truth_steps or [],
                llm_steps,
            )

            # Calculate step order score
            step_order_correct = self._calculate_step_order_score(
                ground_truth_steps or [],
                llm_steps,
            )

            # Calculate reasoning similarity
            reasoning_similarity = self._calculate_reasoning_similarity(
                ground_truth_steps or [],
                llm_steps,
            )

            # Calculate hallucination rate
            hallucination_rate = self._calculate_hallucination_rate(
                example.ground_truth_final,
                llm_steps,
            )

            # Extended error classification (if incorrect and enabled)
            error_classification = None
            if (
                not final_correct
                and self.enable_error_classification
                and self.error_classifier
            ):
                classification_result = ClassificationResult(
                    example_id=example.id,
                    model_name=model_id,
                    method="unknown",  # Will be set by caller if needed
                    ground_truth=example.ground_truth_final,
                    predicted_answer=answer_to_compare,
                    is_correct=final_correct,
                    raw_response=llm_response.raw_response,
                    context=getattr(example, "context", ""),
                )
                error_classification = self.error_classifier.classify(
                    classification_result,
                    example.ground_truth_final,
                    getattr(example, "context", ""),
                )

            # Build metrics object
            metrics = EvaluationMetrics(
                example_id=example.id,
                model_id=model_id,
                ground_truth_final=example.ground_truth_final,
                ground_truth_steps=ground_truth_steps,
                llm_steps=llm_steps,
                final_answer_correct=final_correct,
                final_answer_matches_type=matches_type,
                step_completeness=step_completeness,
                step_order_correct=step_order_correct,
                reasoning_similarity=reasoning_similarity,
                has_hallucination=hallucination_rate > 0,
                hallucination_rate=hallucination_rate,
                overall_reasoning_score=self.calculate_overall_score(
                    EvaluationMetrics(
                        example_id=example.id,
                        model_id=model_id,
                        ground_truth_final=example.ground_truth_final,
                        ground_truth_steps=ground_truth_steps,
                        llm_steps=llm_steps,
                        final_answer_correct=final_correct,
                        final_answer_matches_type=matches_type,
                        step_completeness=step_completeness,
                        step_order_correct=step_order_correct,
                        reasoning_similarity=reasoning_similarity,
                        has_hallucination=hallucination_rate > 0,
                        hallucination_rate=hallucination_rate,
                    ),
                ),
                error_classification=error_classification,
            )

            results[model_id] = metrics

        return results


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("MetricsEvaluator - FinanceReasoning Paper Metrics")
    print("=" * 60)

    # Test evaluation
    try:
        from model_runner import LLMResponse

        evaluator = MetricsEvaluator()

        # Test with synthetic data
        test_ground_truth_steps = [
            GroundTruthStep(
                step_number=1,
                step_type=StepType.VARIABLE_DEFINITION,
                description="Define x",
                code_snippet="x = 1",
            ),
            GroundTruthStep(
                step_number=2,
                step_type=StepType.CALCULATION,
                description="Calculate y",
                code_snippet="y = x * 2",
            ),
        ]

        test_llm_response = LLMResponse(
            model_id="test-model",
            example_id="test-001",
            raw_response="ANSWER: 42",
            parsed_data={
                "reasoning_steps": [
                    {
                        "step_number": 1,
                        "step_type": "variable_definition",
                        "description": "Define x",
                        "code_snippet": "x = 1",
                    },
                    {
                        "step_number": 2,
                        "step_type": "calculation",
                        "description": "Calculate y",
                        "code_snippet": "y = x * 2",
                    },
                ],
            },
        )

        results = evaluator.evaluate_example(
            example_id="test-001",
            ground_truth_final=42,
            llm_responses={"test-model": test_llm_response},
        )

        print("✓ Test evaluation successful")
        print(f"  Model: test-model")
        print(f"  Final Answer Correct: {results['test-model'].final_answer_correct}")
        print(f"  Step Completeness: {results['test-model'].step_completeness:.2%}")
        print(f"  Step Order Correct: {results['test-model'].step_order_correct}")
        print(
            f"  Reasoning Similarity: {results['test-model'].reasoning_similarity:.2%}"
        )
        print(f"  Hallucination Rate: {results['test-model'].hallucination_rate:.2%}")
        print(f"  Overall Score: {results['test-model'].overall_reasoning_score:.3f}")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Test error: {e}")
        sys.exit(1)
