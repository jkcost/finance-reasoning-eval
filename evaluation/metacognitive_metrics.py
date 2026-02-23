"""
Metacognitive Metrics for Financial LLM Evaluation

Measures LLMs' ability to recognize information insufficiency:
- Refusal Accuracy: Correct refusals on unsolvable problems
- False Confidence Rate: Confident answers to unsolvable problems
- Hallucination Under Uncertainty: Fabricated values usage
- Metacognitive Score: Composite metric (weighted)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


class ResponseType(str, Enum):
    """How the model responded to an unsolvable problem"""

    REFUSED = "refused"  # Correctly identified as unsolvable
    CAVEAT = "caveat"  # Answered with uncertainty hedging
    CONFIDENT = "confident"  # Answered confidently (wrong)
    ERROR = "error"  # Execution/parsing error


class TransformationType(str, Enum):
    """Types of data transformation applied to create unsolvable problems"""

    INFO_REMOVAL = "Type 1: Information Removal"
    COLUMN_REMOVAL = "Type 2: Table Column Removal"
    AMBIGUOUS_TIME = "Type 3: Ambiguous Time Period"
    CRITICAL_REMOVAL = "Type 4: Critical Data Removal"
    CONTRADICTORY = "Type 5: Contradictory Information"


class PromptStrategy(str, Enum):
    """Prompt strategy for metacognitive evaluation"""

    STANDARD = "standard"
    METACOGNITIVE = "metacognitive"
    SELF_VERIFICATION = "self_verification"
    CONTRADICTION_AWARE = "contradiction_aware"


@dataclass
class MetacognitiveResult:
    """Result from a single metacognitive evaluation"""

    example_id: str
    model_name: str
    method: str  # COT or POT
    prompt_strategy: str
    transformation_type: str
    response_type: str  # ResponseType value
    raw_response: str
    predicted_answer: Any = None
    is_original_correct: Optional[bool] = None  # Baseline correctness
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    hallucinated_values: List[str] = field(default_factory=list)
    refusal_patterns_matched: List[str] = field(default_factory=list)
    confidence_indicators: List[str] = field(default_factory=list)
    rag_enabled: bool = False
    # Problem context fields for detailed case analysis
    question: str = ""
    context_original: str = ""  # Original context (truncated for storage)
    context_transformed: str = ""  # Transformed context (truncated for storage)
    transformation_description: str = ""  # What was changed
    ground_truth: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetacognitiveResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetacognitiveMetrics:
    """Aggregated metacognitive metrics for a model.

    MC Score = Refusal_F1 × (1 - Hallucination_Rate)

    References:
        - Feng+2024: Abstention F1 (Refusal Precision/Recall)
        - Yang+2024: Honesty Score (Over-Conservatism Rate)
        - Cheng+2024: Balanced IDK metrics (IDK 과다 방지 균형)
        - Rajpurkar+2018: SQuAD 2.0 unanswerable evaluation
        - Kadavath+2022: LLM self-knowledge calibration
        - Yin+2023: 3-type refusal behavior classification
    """

    model_name: str
    # Unsolvable (Phase B) counts
    total_unsolvable: int = 0
    correct_refusals: int = 0
    caveat_responses: int = 0
    confident_wrong: int = 0
    error_responses: int = 0
    hallucination_count: int = 0
    total_cost_usd: float = 0.0
    # Solvable (Phase A) counts — for cross-phase metrics
    total_solvable: int = 0
    solvable_correct: int = 0
    solvable_refused: int = 0  # false refusals (over-conservatism)

    # --- Refusal Recall (= legacy refusal_accuracy) ---

    @property
    def refusal_recall(self) -> float:
        """Correctly refused / total unsolvable (Feng+2024, Rajpurkar+2018)"""
        if self.total_unsolvable == 0:
            return 0.0
        return self.correct_refusals / self.total_unsolvable

    @property
    def refusal_accuracy(self) -> float:
        """Alias for refusal_recall (backward compatibility)"""
        return self.refusal_recall

    # --- Refusal Precision ---

    @property
    def refusal_precision(self) -> float:
        """Correctly refused / total refused (Feng+2024).

        When Phase A data is unavailable (solvable_refused unknown),
        assumes no false refusals → precision = 1.0 (graceful degradation).
        """
        total_refused = self.correct_refusals + self.solvable_refused
        if total_refused == 0:
            return 1.0 if self.total_unsolvable == 0 else 0.0
        return self.correct_refusals / total_refused

    # --- Refusal F1 ---

    @property
    def refusal_f1(self) -> float:
        """Harmonic mean of Refusal Precision and Recall (Feng+2024).

        Graceful degradation: if no Phase A data, RP=1.0 → F1 = 2RR/(1+RR).
        """
        rp = self.refusal_precision
        rr = self.refusal_recall
        if rp + rr == 0:
            return 0.0
        return 2 * rp * rr / (rp + rr)

    # --- Over-Conservatism Rate ---

    @property
    def over_conservatism_rate(self) -> float:
        """Falsely refused solvable / total solvable (Yang+2024)"""
        if self.total_solvable == 0:
            return 0.0
        return self.solvable_refused / self.total_solvable

    # --- Legacy metrics ---

    @property
    def false_confidence_rate(self) -> float:
        if self.total_unsolvable == 0:
            return 0.0
        return self.confident_wrong / self.total_unsolvable

    @property
    def hallucination_rate(self) -> float:
        answerable = self.confident_wrong + self.caveat_responses
        if answerable == 0:
            return 0.0
        return self.hallucination_count / answerable

    # --- MC Score ---

    @property
    def mc_score(self) -> float:
        """MC Score = Refusal_F1 × (1 - Hallucination_Rate)

        - No arbitrary weights: F1 balances precision/recall
        - Hallucination penalty as multiplicative factor (independent quality dimension)
        - Graceful degradation: without Phase A data, RP=1.0 → F1=2RR/(1+RR)
        """
        return self.refusal_f1 * (1 - self.hallucination_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_unsolvable": self.total_unsolvable,
            "correct_refusals": self.correct_refusals,
            "caveat_responses": self.caveat_responses,
            "confident_wrong": self.confident_wrong,
            "error_responses": self.error_responses,
            "hallucination_count": self.hallucination_count,
            "total_cost_usd": self.total_cost_usd,
            "total_solvable": self.total_solvable,
            "solvable_correct": self.solvable_correct,
            "solvable_refused": self.solvable_refused,
            "refusal_recall": round(self.refusal_recall, 4),
            "refusal_precision": round(self.refusal_precision, 4),
            "refusal_f1": round(self.refusal_f1, 4),
            "over_conservatism_rate": round(self.over_conservatism_rate, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "false_confidence_rate": round(self.false_confidence_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "mc_score": round(self.mc_score, 4),
        }


def compute_metrics(results: List[MetacognitiveResult]) -> Dict[str, MetacognitiveMetrics]:
    """Compute aggregated metacognitive metrics per model.

    Args:
        results: List of individual metacognitive evaluation results

    Returns:
        Dict mapping model_name to MetacognitiveMetrics
    """
    metrics_by_model: Dict[str, MetacognitiveMetrics] = {}

    for r in results:
        if r.model_name not in metrics_by_model:
            metrics_by_model[r.model_name] = MetacognitiveMetrics(model_name=r.model_name)

        m = metrics_by_model[r.model_name]
        m.total_unsolvable += 1
        m.total_cost_usd += r.cost_usd

        if r.response_type == ResponseType.REFUSED.value:
            m.correct_refusals += 1
        elif r.response_type == ResponseType.CAVEAT.value:
            m.caveat_responses += 1
            if r.hallucinated_values:
                m.hallucination_count += 1
        elif r.response_type == ResponseType.CONFIDENT.value:
            m.confident_wrong += 1
            if r.hallucinated_values:
                m.hallucination_count += 1
        elif r.response_type == ResponseType.ERROR.value:
            m.error_responses += 1

    return metrics_by_model


def compute_cross_phase_metrics(
    phase_a_results: List[MetacognitiveResult],
    phase_b_results: List[MetacognitiveResult],
) -> Dict[str, MetacognitiveMetrics]:
    """Compute metrics combining Phase A (solvable) and Phase B (unsolvable) data.

    Phase B provides unsolvable counts (correct_refusals, confident_wrong, etc.).
    Phase A provides solvable counts (solvable_correct, solvable_refused).
    Together they enable Refusal Precision and F1 computation.

    Args:
        phase_a_results: Baseline results on original (solvable) problems
        phase_b_results: Results on transformed (unsolvable) problems

    Returns:
        Dict mapping model_name to MetacognitiveMetrics with both phases merged
    """
    # Start with Phase B unsolvable metrics
    metrics = compute_metrics(phase_b_results)

    # Merge Phase A solvable data
    for r in phase_a_results:
        if r.model_name not in metrics:
            metrics[r.model_name] = MetacognitiveMetrics(model_name=r.model_name)

        m = metrics[r.model_name]
        m.total_solvable += 1
        m.total_cost_usd += r.cost_usd

        if r.is_original_correct:
            m.solvable_correct += 1

        # A refused response on a solvable problem = false refusal
        if r.response_type == ResponseType.REFUSED.value:
            m.solvable_refused += 1

    return metrics


def compute_metrics_by_dimension(
    results: List[MetacognitiveResult],
    dimension: str,
) -> Dict[str, Dict[str, MetacognitiveMetrics]]:
    """Compute metrics grouped by model and a second dimension.

    Args:
        results: List of evaluation results
        dimension: One of 'transformation_type', 'prompt_strategy', 'rag_enabled'

    Returns:
        Nested dict: {dimension_value: {model_name: MetacognitiveMetrics}}
    """
    grouped: Dict[str, List[MetacognitiveResult]] = {}

    for r in results:
        key = str(getattr(r, dimension, "unknown"))
        grouped.setdefault(key, []).append(r)

    return {dim_val: compute_metrics(dim_results) for dim_val, dim_results in grouped.items()}
