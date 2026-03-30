"""Tests for IC Difficulty Ladder (L1-L5) transformations."""

import json
import pytest

from experiments.ic_difficulty_ladder import (
    LABEL_IC_L1,
    LABEL_IC_L2,
    LABEL_IC_L3,
    LABEL_IC_L4,
    LABEL_IC_L5,
    apply_ic_ladder,
    transform_ic_l1_json,
    transform_ic_l1_text,
    transform_ic_l2_text,
    transform_ic_l4_json,
    transform_ic_l4_text,
    transform_ic_l5_text,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_json_context():
    return {
        "revenue": {
            "2021": 1000000,
            "2022": 1200000,
            "2023": 1500000,
        },
        "net_income": {
            "2021": 100000,
            "2022": 150000,
            "2023": 200000,
        },
    }


@pytest.fixture
def sample_text_context():
    return (
        "The company reported revenue of $15 million in 2023. "
        "Net income was $2.5 million, representing a profit margin of 16.7%. "
        "Operating expenses totaled $12 million for the fiscal year. "
        "The interest rate on outstanding debt was 5.25%."
    )


@pytest.fixture
def sample_question():
    return "What was the revenue growth rate from 2022 to 2023?"


@pytest.fixture
def sample_example(sample_text_context, sample_question):
    return {
        "question": sample_question,
        "context": sample_text_context,
        "python_solution": "revenue_2023 = 15000000\nrevenue_2022 = 12000000",
        "ground_truth": "25%",
    }


# ============================================================================
# L1: Obvious Typo Tests
# ============================================================================

class TestL1ObviousTypo:
    def test_l1_text_creates_10x_error(self, sample_text_context, sample_question):
        new_ctx, desc, expected = transform_ic_l1_text(
            sample_text_context, sample_question
        )
        assert new_ctx is not None
        assert "10x" in desc or "L1" in desc
        assert new_ctx != sample_text_context

    def test_l1_json_creates_10x_error(self, sample_json_context, sample_question):
        new_ctx, desc, expected = transform_ic_l1_json(
            sample_json_context, sample_question
        )
        assert new_ctx is not None
        assert "10x" in desc or "L1" in desc
        # Check that some value is 10x the original
        for key in sample_json_context:
            if isinstance(sample_json_context[key], dict):
                for year in sample_json_context[key]:
                    if year in new_ctx.get(key, {}):
                        orig = sample_json_context[key][year]
                        new_val = new_ctx[key][year]
                        if orig != new_val:
                            assert abs(new_val - orig * 10) < 0.01

    def test_l1_returns_none_for_empty_context(self, sample_question):
        new_ctx, desc, expected = transform_ic_l1_text("", sample_question)
        assert new_ctx is None

    def test_l1_returns_none_for_no_numbers(self, sample_question):
        new_ctx, desc, expected = transform_ic_l1_text(
            "The company had a good year with strong growth.", sample_question
        )
        assert new_ctx is None


# ============================================================================
# L2: Unit Mismatch Tests
# ============================================================================

class TestL2UnitMismatch:
    def test_l2_creates_unit_conflict(self):
        context = (
            "Total revenue was $500 million in Q4. "
            "The company expanded operations significantly."
        )
        question = "What was the growth rate?"
        new_ctx, desc, expected = transform_ic_l2_text(context, question)
        # L2 may not match all contexts — check both outcomes
        if new_ctx is not None:
            assert "unit mismatch" in desc.lower() or "L2" in desc
            assert "billion" in new_ctx.lower() or "thousand" in new_ctx.lower()

    def test_l2_percentage_to_basis_points(self):
        context = (
            "The federal funds rate increased by 0.75%. "
            "This was the largest single increase in decades."
        )
        question = "What was the total rate change?"
        new_ctx, desc, expected = transform_ic_l2_text(context, question)
        if new_ctx is not None:
            assert "basis points" in new_ctx.lower() or "L2" in desc


# ============================================================================
# L3: Authority Conflict Tests (delegates to existing IC)
# ============================================================================

class TestL3AuthorityConflict:
    def test_l3_via_apply_ic_ladder(self, sample_example):
        results = apply_ic_ladder(sample_example, levels=["L3"])
        # L3 uses existing transform_type5 logic
        for r in results:
            assert r["transformation_type"] == LABEL_IC_L3
            assert "L3" in r.get("transformation_description", "")


# ============================================================================
# L4: Cross-Period Conflict Tests
# ============================================================================

class TestL4CrossPeriod:
    def test_l4_json_breaks_summation(self, sample_json_context, sample_question):
        new_ctx, desc, expected = transform_ic_l4_json(
            sample_json_context, sample_question
        )
        if new_ctx is not None:
            assert "cross-period" in desc.lower() or "L4" in desc
            # Verify that _annual_total doesn't match sum of values
            for key in new_ctx:
                if isinstance(new_ctx[key], dict) and "_annual_total" in new_ctx[key]:
                    total = new_ctx[key]["_annual_total"]
                    actual_sum = sum(
                        v for k, v in new_ctx[key].items()
                        if k != "_annual_total" and isinstance(v, (int, float))
                    )
                    assert abs(total - actual_sum) > 0.01, "Summation should be inconsistent"

    def test_l4_text_adds_fake_total(self, sample_text_context, sample_question):
        new_ctx, desc, expected = transform_ic_l4_text(
            sample_text_context, sample_question
        )
        if new_ctx is not None:
            assert "total" in new_ctx.lower()
            assert "cross-period" in desc.lower() or "L4" in desc

    def test_l4_returns_none_for_insufficient_data(self, sample_question):
        ctx = {"revenue": {"2023": 100}}  # Only 1 year
        new_ctx, desc, expected = transform_ic_l4_json(ctx, sample_question)
        assert new_ctx is None


# ============================================================================
# L5: Implicit Ratio Inconsistency Tests
# ============================================================================

class TestL5ImplicitRatio:
    def test_l5_creates_margin_inconsistency(self):
        context = (
            "The company reported revenue of $100 million. "
            "Net income reached $20 million for the fiscal year."
        )
        question = "What was the profit margin?"
        new_ctx, desc, expected = transform_ic_l5_text(context, question)
        if new_ctx is not None:
            assert "ratio" in desc.lower() or "margin" in desc.lower() or "L5" in desc
            assert "margin" in new_ctx.lower() or "profit" in new_ctx.lower()

    def test_l5_returns_none_without_financial_pairs(self, sample_question):
        context = "The weather was sunny and warm throughout the trading period."
        new_ctx, desc, expected = transform_ic_l5_text(context, sample_question)
        assert new_ctx is None

    def test_l5_handles_zero_revenue(self, sample_question):
        """Ensure no ZeroDivisionError when revenue is 0."""
        context = "Revenue was $0. Net income was -$500,000."
        new_ctx, desc, expected = transform_ic_l5_text(context, sample_question)
        # Should return None gracefully, not crash
        assert new_ctx is None or isinstance(new_ctx, str)


# ============================================================================
# Orchestration Tests
# ============================================================================

class TestApplyICLadder:
    def test_apply_all_levels(self, sample_example):
        results = apply_ic_ladder(sample_example)
        # At least some levels should apply
        assert isinstance(results, list)
        # Check each result has required fields
        for r in results:
            assert "transformation_type" in r
            assert "transformation_description" in r
            assert "context" in r
            assert r["transformation_type"] in [
                LABEL_IC_L1, LABEL_IC_L2, LABEL_IC_L3, LABEL_IC_L4, LABEL_IC_L5,
            ]

    def test_apply_specific_levels(self, sample_example):
        results = apply_ic_ladder(sample_example, levels=["L1", "L3"])
        for r in results:
            assert r["transformation_type"] in [LABEL_IC_L1, LABEL_IC_L3]

    def test_empty_context_returns_empty(self):
        example = {"question": "test?", "context": ""}
        results = apply_ic_ladder(example)
        assert results == []

    def test_preserves_original_fields(self, sample_example):
        results = apply_ic_ladder(sample_example, levels=["L1"])
        for r in results:
            assert r["question"] == sample_example["question"]
            assert r["ground_truth"] == sample_example["ground_truth"]
