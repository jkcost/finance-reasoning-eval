"""Tests for Conflict Salience Scorer."""

import pytest

from experiments.conflict_salience_scorer import (
    SalienceFeatures,
    compute_regression_data,
    extract_salience_features,
    score_batch_results,
)


class TestSalienceFeatures:
    def test_salience_score_range(self):
        """Salience score should always be between 0 and 1."""
        sf = SalienceFeatures(
            question_id="test",
            ic_level="IC-L3",
            token_distance=10,
            same_paragraph=True,
            authority_marker_count=5,
            magnitude_ratio=1.5,
        )
        assert 0.0 <= sf.salience_score <= 1.0

    def test_high_salience_for_close_obvious_conflict(self):
        """Close, same-paragraph, large magnitude = high salience."""
        sf = SalienceFeatures(
            question_id="test",
            ic_level="IC-L1",
            token_distance=10,
            same_paragraph=True,
            authority_marker_count=0,
            magnitude_ratio=10.0,  # L1: 10x
            has_explicit_discrepancy_note=True,
        )
        assert sf.salience_score >= 0.7

    def test_low_salience_for_authoritative_hidden_conflict(self):
        """Far, different paragraph, authority markers = low salience."""
        sf = SalienceFeatures(
            question_id="test",
            ic_level="IC-L3",
            token_distance=200,
            same_paragraph=False,
            authority_marker_count=4,
            magnitude_ratio=1.5,
            has_explicit_discrepancy_note=False,
        )
        assert sf.salience_score < 0.3


class TestExtractSalienceFeatures:
    def test_extracts_from_ic_description(self):
        features = extract_salience_features(
            question_id="q1",
            transformation_description="Replaced $10,000 with $15,000, added contradiction",
            transformed_context=(
                "Revenue was $15,000 in 2023. "
                "Note: There is a discrepancy — one source reports $15,000, "
                "while another shows $10,000."
            ),
            ic_level="IC-L3",
        )
        assert features.question_id == "q1"
        assert features.has_explicit_discrepancy_note is True
        assert features.authority_marker_count >= 0

    def test_handles_empty_description(self):
        features = extract_salience_features(
            question_id="q2",
            transformation_description="",
            transformed_context="Some context without numbers.",
            ic_level="IC-L1",
        )
        assert features.original_value is None
        assert features.conflicting_value is None


class TestScoreBatchResults:
    def test_filters_non_ic_results(self):
        results = [
            {"question_id": "q1", "transformation_type": "IC-L3: Authority Conflict",
             "transformation_description": "test", "context": "test"},
            {"question_id": "q2", "transformation_type": "EA-partial: Explicit Absence",
             "transformation_description": "test", "context": "test"},
        ]
        features = score_batch_results(results)
        assert len(features) == 1
        assert features[0].question_id == "q1"


class TestComputeRegressionData:
    def test_regression_data_format(self):
        features = [
            SalienceFeatures(
                question_id="q1", ic_level="IC-L3",
                token_distance=50, same_paragraph=True,
                authority_marker_count=2, magnitude_ratio=1.5,
            )
        ]
        eval_results = [
            {"question_id": "q1", "response_type": "refused"},
        ]
        data = compute_regression_data(features, eval_results)
        assert len(data) == 1
        assert data[0]["refusal"] == 1
        assert "token_distance" in data[0]
        assert "salience_score" in data[0]

    def test_confident_response_maps_to_zero(self):
        features = [
            SalienceFeatures(question_id="q1", ic_level="IC-L3")
        ]
        eval_results = [
            {"question_id": "q1", "response_type": "confident"},
        ]
        data = compute_regression_data(features, eval_results)
        assert data[0]["refusal"] == 0
