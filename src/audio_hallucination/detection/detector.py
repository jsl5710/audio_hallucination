"""Core hallucination detection logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HallucinationResult:
    """Result from hallucination detection on an audio segment."""

    segment_start: float
    segment_end: float
    text: str
    confidence: float
    is_hallucination: bool
    method: str


def detect_hallucinations(
    audio_path: str,
    transcript: str | None = None,
    confidence_threshold: float = 0.5,
    methods: list[str] | None = None,
) -> list[HallucinationResult]:
    """Detect hallucinations in audio transcription.

    Args:
        audio_path: Path to the audio file.
        transcript: Optional reference transcript for comparison.
        confidence_threshold: Threshold below which segments are flagged.
        methods: Detection methods to apply. Defaults to all available methods.

    Returns:
        List of HallucinationResult for each detected hallucination.
    """
    if methods is None:
        methods = ["confidence_scoring", "repetition_detection", "silence_alignment"]

    results: list[HallucinationResult] = []

    for method in methods:
        detector = _get_detector(method)
        results.extend(detector(audio_path, transcript, confidence_threshold))

    return results


def _get_detector(method: str):
    """Return the detector function for the given method name."""
    detectors = {
        "confidence_scoring": _detect_by_confidence,
        "repetition_detection": _detect_by_repetition,
        "silence_alignment": _detect_by_silence_alignment,
    }
    if method not in detectors:
        raise ValueError(f"Unknown detection method: {method}")
    return detectors[method]


def _detect_by_confidence(audio_path, transcript, threshold):
    """Detect hallucinations based on model confidence scores."""
    # TODO: Implement confidence-based detection
    return []


def _detect_by_repetition(audio_path, transcript, threshold):
    """Detect hallucinated repetitions in transcription output."""
    # TODO: Implement repetition-based detection
    return []


def _detect_by_silence_alignment(audio_path, transcript, threshold):
    """Detect text generated during silent audio segments."""
    # TODO: Implement silence-alignment detection
    return []
