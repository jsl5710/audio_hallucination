"""Evaluation metrics for hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics."""

    hallucination_rate: float
    precision: float
    recall: float
    f1: float
    wer: float | None = None


def compute_hallucination_rate(
    predictions: list[bool],
    labels: list[bool],
) -> EvaluationResult:
    """Compute hallucination detection metrics.

    Args:
        predictions: Predicted hallucination flags per segment.
        labels: Ground-truth hallucination flags per segment.

    Returns:
        EvaluationResult with computed metrics.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    if not predictions:
        return EvaluationResult(
            hallucination_rate=0.0, precision=0.0, recall=0.0, f1=0.0
        )

    tp = sum(p and l for p, l in zip(predictions, labels))
    fp = sum(p and not l for p, l in zip(predictions, labels))
    fn = sum(not p and l for p, l in zip(predictions, labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    hallucination_rate = sum(labels) / len(labels)

    return EvaluationResult(
        hallucination_rate=hallucination_rate,
        precision=precision,
        recall=recall,
        f1=f1,
    )
