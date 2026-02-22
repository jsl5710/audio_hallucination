"""Tests for evaluation metrics."""

import pytest

from audio_hallucination.evaluation.metrics import (
    EvaluationResult,
    compute_hallucination_rate,
)


def test_perfect_predictions():
    predictions = [True, False, True, False]
    labels = [True, False, True, False]
    result = compute_hallucination_rate(predictions, labels)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.hallucination_rate == 0.5


def test_no_predictions():
    result = compute_hallucination_rate([], [])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.hallucination_rate == 0.0


def test_all_false_positives():
    predictions = [True, True, True]
    labels = [False, False, False]
    result = compute_hallucination_rate(predictions, labels)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.hallucination_rate == 0.0


def test_all_false_negatives():
    predictions = [False, False, False]
    labels = [True, True, True]
    result = compute_hallucination_rate(predictions, labels)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.hallucination_rate == 1.0


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_hallucination_rate([True], [True, False])
