"""Tests for hallucination detection."""

import pytest

from audio_hallucination.detection.detector import (
    HallucinationResult,
    detect_hallucinations,
    _get_detector,
)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown detection method"):
        _get_detector("nonexistent_method")


def test_detect_hallucinations_returns_list(tmp_path):
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()
    results = detect_hallucinations(str(dummy_audio))
    assert isinstance(results, list)


def test_detect_hallucinations_custom_methods(tmp_path):
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()
    results = detect_hallucinations(
        str(dummy_audio), methods=["confidence_scoring"]
    )
    assert isinstance(results, list)
