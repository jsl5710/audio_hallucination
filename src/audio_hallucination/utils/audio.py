"""Audio processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to the target sample rate.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate.

    Returns:
        Tuple of (audio waveform as numpy array, sample rate).
    """
    import librosa

    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio, sr


def compute_energy(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
    """Compute frame-level energy of an audio signal.

    Args:
        audio: Audio waveform.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.

    Returns:
        Array of energy values per frame.
    """
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_length)[::hop_length]
    return np.sum(frames ** 2, axis=1)


def detect_silence(
    audio: np.ndarray,
    sr: int = 16000,
    threshold_db: float = -40.0,
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect silent regions in audio.

    Args:
        audio: Audio waveform.
        sr: Sample rate.
        threshold_db: Energy threshold in dB below which audio is considered silent.
        min_duration: Minimum duration in seconds for a silence region.

    Returns:
        List of (start_time, end_time) tuples for each silent region.
    """
    import librosa

    intervals = librosa.effects.split(audio, top_db=abs(threshold_db))

    silent_regions = []
    prev_end = 0.0
    for start, end in intervals:
        start_time = prev_end / sr
        end_time = start / sr
        if end_time - start_time >= min_duration:
            silent_regions.append((start_time, end_time))
        prev_end = end

    # Check trailing silence
    total_duration = len(audio) / sr
    last_end = prev_end / sr
    if total_duration - last_end >= min_duration:
        silent_regions.append((last_end, total_duration))

    return silent_regions
