"""
Error Taxonomy for FinanceReasoning Evaluation

Defines the 13 error categories:
- 9 from paper (Table 6-14, Appendix B)
- 4 extended categories for automatic detection
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


class ErrorCategory(str, Enum):
    """Error categories based on paper Table 6-14 + extensions"""

    # =========================================================================
    # Paper Original Categories (Table 6-14, Appendix B)
    # =========================================================================

    # Table 6: Misunderstanding of Problem
    MISUNDERSTANDING = "misunderstanding_of_problem"

    # Table 7: Formula Application Error
    FORMULA_ERROR = "formula_application_error"

    # Table 8: Numerical Extraction Error
    EXTRACTION_ERROR = "numerical_extraction_error"

    # Table 9: Numerical Calculation Error
    CALCULATION_ERROR = "numerical_calculation_error"

    # Table 10: Unsolvable Problem
    UNSOLVABLE = "unsolvable_problem"

    # Table 11: Ambiguous Statement
    AMBIGUOUS = "ambiguous_statement"

    # Table 12: Oversimplified Process
    OVERSIMPLIFIED = "oversimplified_process"

    # Table 13: Incorrect Answer (ground truth error)
    INCORRECT_GT = "incorrect_answer"

    # Table 14: Relaxed Evaluation
    RELAXED_EVAL = "relaxed_evaluation"

    # =========================================================================
    # Extended Categories (Automatic Detection)
    # =========================================================================

    # E10: POT code execution failed
    EXECUTION_ERROR = "execution_error"

    # E11: Response parsing failed
    PARSING_ERROR = "parsing_error"

    # E12: Answer within 0.2% tolerance (rounding)
    ROUNDING_ERROR = "rounding_error"

    # E13: Used values not in context
    HALLUCINATION = "hallucination"

    # Unknown/unclassified
    UNKNOWN = "unknown"


# Error category metadata
ERROR_METADATA = {
    ErrorCategory.MISUNDERSTANDING: {
        "table": 6,
        "description": "Model misunderstands the problem requirements",
        "examples": ["test-2164"],
        "auto_detectable": False,
        "severity": "high",
    },
    ErrorCategory.FORMULA_ERROR: {
        "table": 7,
        "description": "Model applies incorrect formula or equation",
        "examples": ["test-2162"],
        "auto_detectable": False,
        "severity": "high",
    },
    ErrorCategory.EXTRACTION_ERROR: {
        "table": 8,
        "description": "Model extracts wrong numerical values from context",
        "examples": ["test-2017"],
        "auto_detectable": True,
        "severity": "medium",
    },
    ErrorCategory.CALCULATION_ERROR: {
        "table": 9,
        "description": "Model makes arithmetic calculation mistakes",
        "examples": ["test-2208"],
        "auto_detectable": True,
        "severity": "medium",
    },
    ErrorCategory.UNSOLVABLE: {
        "table": 10,
        "description": "Problem is unsolvable with given information",
        "examples": ["test-114"],
        "auto_detectable": False,
        "severity": "low",
    },
    ErrorCategory.AMBIGUOUS: {
        "table": 11,
        "description": "Problem statement is ambiguous",
        "examples": ["test-142"],
        "auto_detectable": False,
        "severity": "low",
    },
    ErrorCategory.OVERSIMPLIFIED: {
        "table": 12,
        "description": "Solution process is oversimplified",
        "examples": ["validation-24"],
        "auto_detectable": False,
        "severity": "medium",
    },
    ErrorCategory.INCORRECT_GT: {
        "table": 13,
        "description": "Ground truth answer is incorrect",
        "examples": ["test-8"],
        "auto_detectable": False,
        "severity": "low",
    },
    ErrorCategory.RELAXED_EVAL: {
        "table": 14,
        "description": "Evaluation criteria too relaxed",
        "examples": ["validation-68"],
        "auto_detectable": True,
        "severity": "low",
    },
    ErrorCategory.EXECUTION_ERROR: {
        "table": None,
        "description": "POT code execution failed",
        "examples": [],
        "auto_detectable": True,
        "severity": "high",
    },
    ErrorCategory.PARSING_ERROR: {
        "table": None,
        "description": "Response parsing failed",
        "examples": [],
        "auto_detectable": True,
        "severity": "high",
    },
    ErrorCategory.ROUNDING_ERROR: {
        "table": None,
        "description": "Answer within 0.2% tolerance",
        "examples": [],
        "auto_detectable": True,
        "severity": "low",
    },
    ErrorCategory.HALLUCINATION: {
        "table": None,
        "description": "Used values not present in context",
        "examples": [],
        "auto_detectable": True,
        "severity": "high",
    },
    ErrorCategory.UNKNOWN: {
        "table": None,
        "description": "Unclassified error",
        "examples": [],
        "auto_detectable": False,
        "severity": "medium",
    },
}


@dataclass
class ErrorClassification:
    """Classification result for a single error case"""

    category: ErrorCategory
    confidence: float  # 0.0 ~ 1.0
    evidence: str  # Classification rationale
    details: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    auto_classified: bool = True
    manual_override: Optional[ErrorCategory] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "details": self.details,
            "auto_classified": self.auto_classified,
            "manual_override": (
                self.manual_override.value if self.manual_override else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorClassification":
        """Create from dictionary"""
        return cls(
            category=ErrorCategory(data["category"]),
            confidence=data["confidence"],
            evidence=data["evidence"],
            details=data.get("details", {}),
            auto_classified=data.get("auto_classified", True),
            manual_override=(
                ErrorCategory(data["manual_override"])
                if data.get("manual_override")
                else None
            ),
        )


class ModelType(str, Enum):
    """Model type classification"""

    GENERAL = "general"  # General-purpose models
    REASONING = "reasoning"  # Reasoning-focused models


@dataclass
class ModelInfo:
    """Model information for comparison"""

    name: str
    display_name: str
    provider: str
    model_type: ModelType
    cost_per_million_input: float
    cost_per_million_output: float
    api_key_env: str
    model_id: str


# Model registry with type classification
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # =========================================================================
    # General-Purpose Models
    # =========================================================================
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        model_type=ModelType.GENERAL,
        cost_per_million_input=2.50,
        cost_per_million_output=10.00,
        api_key_env="OPENAI_API_KEY",
        model_id="gpt-4o",
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        provider="openai",
        model_type=ModelType.GENERAL,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
        api_key_env="OPENAI_API_KEY",
        model_id="gpt-4o-mini",
    ),
    "claude-sonnet-4": ModelInfo(
        name="claude-sonnet-4",
        display_name="Claude Sonnet 4",
        provider="anthropic",
        model_type=ModelType.GENERAL,
        cost_per_million_input=3.00,
        cost_per_million_output=15.00,
        api_key_env="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-20250514",
    ),
    "claude-haiku-4": ModelInfo(
        name="claude-haiku-4",
        display_name="Claude 3 Haiku",
        provider="anthropic",
        model_type=ModelType.GENERAL,
        cost_per_million_input=0.25,
        cost_per_million_output=1.25,
        api_key_env="ANTHROPIC_API_KEY",
        model_id="claude-3-haiku-20240307",
    ),
    "gemini-2.5-flash": ModelInfo(
        name="gemini-2.5-flash",
        display_name="Gemini 2.0 Flash",
        provider="google",
        model_type=ModelType.GENERAL,
        cost_per_million_input=0.075,
        cost_per_million_output=0.30,
        api_key_env="GOOGLE_API_KEY",
        model_id="gemini-2.0-flash",
    ),
    "gemini-2.0-flash": ModelInfo(
        name="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash Exp",
        provider="google",
        model_type=ModelType.GENERAL,
        cost_per_million_input=0.075,
        cost_per_million_output=0.30,
        api_key_env="GOOGLE_API_KEY",
        model_id="gemini-2.0-flash",
    ),
    # =========================================================================
    # Reasoning-Focused Models
    # =========================================================================
    "gpt-o1": ModelInfo(
        name="gpt-o1",
        display_name="GPT o1",
        provider="openai",
        model_type=ModelType.REASONING,
        cost_per_million_input=15.00,
        cost_per_million_output=60.00,
        api_key_env="OPENAI_API_KEY",
        model_id="o1",
    ),
    "gpt-o1-mini": ModelInfo(
        name="gpt-o1-mini",
        display_name="GPT o1 Mini",
        provider="openai",
        model_type=ModelType.REASONING,
        cost_per_million_input=1.10,
        cost_per_million_output=4.40,
        api_key_env="OPENAI_API_KEY",
        model_id="o1-mini",
    ),
    "claude-opus-4": ModelInfo(
        name="claude-opus-4",
        display_name="Claude Opus 4",
        provider="anthropic",
        model_type=ModelType.REASONING,
        cost_per_million_input=15.00,
        cost_per_million_output=75.00,
        api_key_env="ANTHROPIC_API_KEY",
        model_id="claude-opus-4-20250514",
    ),
    "gemini-2.5-pro": ModelInfo(
        name="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        provider="google",
        model_type=ModelType.REASONING,
        cost_per_million_input=1.25,
        cost_per_million_output=10.00,
        api_key_env="GOOGLE_API_KEY",
        model_id="gemini-2.5-pro",
    ),
}


def get_models_by_type(model_type: ModelType) -> List[ModelInfo]:
    """Get all models of a specific type"""
    return [m for m in MODEL_REGISTRY.values() if m.model_type == model_type]


def get_general_models() -> List[ModelInfo]:
    """Get all general-purpose models"""
    return get_models_by_type(ModelType.GENERAL)


def get_reasoning_models() -> List[ModelInfo]:
    """Get all reasoning-focused models"""
    return get_models_by_type(ModelType.REASONING)


# Test model sets for different budgets
BUDGET_MODEL_SETS = {
    "economic": ["gpt-4o-mini", "claude-haiku-4", "gemini-2.5-flash"],
    "balanced": ["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"],
    "full": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-o1-mini",
        "claude-haiku-4",
        "claude-sonnet-4",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
}
