"""Data loading and preprocessing for audio hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioSample:
    """A single audio sample with optional reference transcript."""

    audio_path: Path
    reference_transcript: str | None = None
    sample_rate: int = 16000


def load_audio_samples(
    data_dir: str | Path,
    transcript_file: str | Path | None = None,
) -> list[AudioSample]:
    """Load audio samples from a directory.

    Args:
        data_dir: Directory containing audio files.
        transcript_file: Optional path to a file mapping audio filenames to transcripts.

    Returns:
        List of AudioSample objects.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    transcripts: dict[str, str] = {}
    if transcript_file is not None:
        transcripts = _load_transcripts(Path(transcript_file))

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    samples = []
    for audio_file in sorted(data_dir.iterdir()):
        if audio_file.suffix.lower() in audio_extensions:
            samples.append(
                AudioSample(
                    audio_path=audio_file,
                    reference_transcript=transcripts.get(audio_file.name),
                )
            )

    return samples


def _load_transcripts(transcript_file: Path) -> dict[str, str]:
    """Load transcript mappings from a text file (tab-separated: filename\\ttranscript)."""
    transcripts = {}
    with open(transcript_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts
