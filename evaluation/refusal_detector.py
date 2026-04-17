"""
Refusal Detector for Metacognitive Evaluation

Classifies LLM responses into:
- REFUSED: Model correctly identified the problem as unsolvable
- CAVEAT: Model answered but expressed uncertainty
- CONFIDENT: Model answered confidently (false confidence)
- ERROR: Model response had execution/parsing errors

Uses pattern matching on response text to detect refusal/uncertainty signals.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from metacognitive_metrics import ResponseType


# ============================================================================
# REFUSAL PATTERNS
# ============================================================================

REFUSAL_PATTERNS = [
    # Explicit refusal keywords
    r"(?i)\b(cannot|can't|unable to)\b.{0,30}\b(solve|answer|determine|calculate|compute)\b",
    r"(?i)\b(insufficient|inadequate|incomplete|missing)\b.{0,20}\b(information|data|details|values)\b",
    r"(?i)\bINSUFFICIENT[_ ]INFORMATION\b",
    r"(?i)\b(not enough|not sufficient)\b.{0,20}\b(information|data)\b",
    r"(?i)\b(impossible|not possible)\b.{0,20}\b(to (solve|answer|determine|calculate))\b",
    r"(?i)\bthe (problem|question) (cannot|can't) be (solved|answered)\b",
    r"(?i)\bmissing.{0,20}(required|necessary|critical|essential|key)\b",
    r"(?i)\b(need|require)s?\b.{0,20}\b(additional|more)\b.{0,20}\b(information|data)\b",
    # Direct refusal patterns
    r"(?i)^I (cannot|can't|am unable to) (provide|give|calculate)",
    r"(?i)this (problem|question) is (unsolvable|unanswerable)",
    r"(?i)there is not enough (information|data) (to|for)",
    # DATA AUDIT patterns (from self-verification prompt)
    r"(?i)DATA AUDIT.*MISSING",
    r"(?i)\bSTATUS:\s*INSUFFICIENT\b",
    # CONTRADICTION patterns (from contradiction_aware prompt strategy)
    r"(?i)\bCONTRADICTION[_ ]DETECTED\b",
    r"(?i)\bcontradictory\b.{0,20}\b(data|values|information|figures)\b",
]

CAVEAT_PATTERNS = [
    # Uncertainty hedging
    r"(?i)\b(assuming|if we assume|let's assume)\b",
    r"(?i)\b(however|but|note that).{0,40}(may not|might not|uncertain|unclear|ambiguous)\b",
    r"(?i)\b(approximate|estimated|rough)\b.{0,20}\b(answer|result|calculation)\b",
    r"(?i)\b(caveat|disclaimer|warning|limitation)\b",
    r"(?i)\b(not (entirely )?sure|uncertain|unclear)\b",
    r"(?i)\b(this assumes|this is based on the assumption)\b",
    r"(?i)\bwithout.{0,30}\b(certainty|knowing|specific|exact)\b",
    # Hedging language
    r"(?i)\b(probably|possibly|likely|perhaps|may be|might be)\b.{0,20}\b(the answer|correct)\b",
    r"(?i)\b(data.{0,10}(missing|unavailable|removed|absent))\b",
]

CONFIDENCE_PATTERNS = [
    # Strong confidence indicators
    r"(?i)\btherefore,?\s*the\s+answer\s+is\b",
    r"(?i)\bthe\s+(final\s+)?answer\s+is\b",
    r"(?i)\bthe\s+result\s+is\b",
    r"(?i)^def solution\(\):",
    r"(?i)\breturn\s+answer\b",
    r"(?i)\banswer\s*=\s*\d",
]


@dataclass
class DetectionResult:
    """Result of refusal detection analysis"""

    response_type: ResponseType
    refusal_patterns_matched: List[str]
    caveat_patterns_matched: List[str]
    confidence_patterns_matched: List[str]
    hallucinated_values: List[str]
    confidence_score: float  # 0.0 (definitely refused) to 1.0 (definitely confident)
    reason: str = ""  # human-readable explanation of the classification


class RefusalDetector:
    """Detects whether a model refused to answer an unsolvable problem"""

    def __init__(self, context_numbers_extractor=None):
        """
        Args:
            context_numbers_extractor: Optional callable that extracts numbers
                from context. If None, uses built-in extraction.
        """
        self._extract_context_numbers = (
            context_numbers_extractor or self._default_extract_numbers
        )

    def detect(
        self,
        response: str,
        context: str = "",
        execution_error: Optional[str] = None,
        executed_code: Optional[str] = None,
    ) -> DetectionResult:
        """Classify a model response as refused/caveat/confident/error.

        Args:
            response: Raw model response text
            context: Original problem context (for hallucination check)
            execution_error: POT execution error if any
            executed_code: POT code that was executed

        Returns:
            DetectionResult with classification and evidence
        """
        if execution_error:
            return DetectionResult(
                response_type=ResponseType.ERROR,
                refusal_patterns_matched=[],
                caveat_patterns_matched=[],
                confidence_patterns_matched=[],
                hallucinated_values=[],
                confidence_score=0.0,
                reason=f"Execution error: {execution_error[:200]}",
            )

        refusal_matches = self._find_matches(response, REFUSAL_PATTERNS)
        caveat_matches = self._find_matches(response, CAVEAT_PATTERNS)
        confidence_matches = self._find_matches(response, CONFIDENCE_PATTERNS)

        hallucinated = self._detect_hallucinated_values(
            response, context, executed_code
        )

        response_type = self._classify(
            refusal_matches, caveat_matches, confidence_matches
        )

        refusal_score = len(refusal_matches) * 2
        caveat_score = len(caveat_matches)
        confidence_score_raw = len(confidence_matches) * 1.5
        total = refusal_score + caveat_score + confidence_score_raw

        if total == 0:
            conf = 0.5
        else:
            conf = confidence_score_raw / total

        reason = self._build_reason(
            response_type, refusal_matches, caveat_matches, hallucinated
        )

        return DetectionResult(
            response_type=response_type,
            refusal_patterns_matched=refusal_matches,
            caveat_patterns_matched=caveat_matches,
            confidence_patterns_matched=confidence_matches,
            hallucinated_values=hallucinated,
            confidence_score=round(conf, 3),
            reason=reason,
        )

    def _build_reason(
        self,
        response_type: "ResponseType",
        refusal_matches: List[str],
        caveat_matches: List[str],
        hallucinated: List[str],
    ) -> str:
        """Build a human-readable explanation for the classification."""
        if response_type == ResponseType.REFUSED:
            n = len(refusal_matches)
            return f"Refused: {n} refusal signal(s) detected (e.g. missing/insufficient data keywords)"
        if response_type == ResponseType.CAVEAT:
            parts = []
            if refusal_matches:
                parts.append(f"{len(refusal_matches)} refusal signal(s)")
            if caveat_matches:
                parts.append(f"{len(caveat_matches)} uncertainty hedge(s)")
            return "Caveat: " + " + ".join(parts) if parts else "Caveat: hedging language detected"
        if response_type == ResponseType.CONFIDENT:
            if hallucinated:
                return f"Confident (hallucination risk): {len(hallucinated)} value(s) not found in context"
            return "Confident: answer provided without expressed uncertainty"
        return ""

    def _find_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Find all matching patterns in text"""
        matched = []
        for pattern in patterns:
            if re.search(pattern, text):
                matched.append(pattern)
        return matched

    def _classify(
        self,
        refusal_matches: List[str],
        caveat_matches: List[str],
        confidence_matches: List[str],
    ) -> ResponseType:
        """Classify response based on matched patterns"""
        has_refusal = len(refusal_matches) > 0
        has_caveat = len(caveat_matches) > 0
        has_confidence = len(confidence_matches) > 0

        if has_refusal and not has_confidence:
            return ResponseType.REFUSED
        if has_refusal and has_confidence:
            # Refusal signal is stronger — model hedged but still answered
            if len(refusal_matches) >= len(confidence_matches):
                return ResponseType.REFUSED
            return ResponseType.CAVEAT
        if has_caveat:
            return ResponseType.CAVEAT
        return ResponseType.CONFIDENT

    def _detect_hallucinated_values(
        self,
        response: str,
        context: str,
        executed_code: Optional[str] = None,
    ) -> List[str]:
        """Detect numerical values in response not present in context"""
        if not context or context == "[]":
            return []

        context_numbers = self._extract_context_numbers(context)
        text_to_check = executed_code if executed_code else response

        response_numbers = set()
        for match in re.findall(r"[\d,]+\.?\d*", text_to_check):
            try:
                num = float(match.replace(",", ""))
                # Skip common constants and small indices
                if num > 1 and num not in {2, 100, 1000, 365, 12, 52, 4, 24, 60}:
                    response_numbers.add(num)
            except ValueError:
                continue

        hallucinated = []
        for num in response_numbers:
            if not self._number_in_context(num, context_numbers):
                hallucinated.append(str(num))

        return hallucinated

    def _number_in_context(self, num: float, context_numbers: set) -> bool:
        """Check if a number (or its common transformations) exists in context"""
        for ctx_num in context_numbers:
            if ctx_num == 0:
                continue
            rel_error = abs(num - ctx_num) / max(abs(ctx_num), 1e-9)
            if rel_error < 0.01:  # 1% tolerance
                return True
        return False

    @staticmethod
    def _default_extract_numbers(text: str) -> set:
        """Extract numerical values from text with common transformations"""
        numbers = set()
        for match in re.findall(r"[\d,]+\.?\d*", text):
            try:
                num = float(match.replace(",", ""))
                numbers.add(num)
                numbers.add(num * 100)
                numbers.add(num / 100)
                numbers.add(num * 1000)
                numbers.add(num / 1000)
            except ValueError:
                continue
        return numbers
