# Audio Hallucination

Detection and evaluation of hallucinations in audio-based AI models.

## Project Structure

```
audio_hallucination/
├── src/
│   └── audio_hallucination/       # Main package
│       ├── detection/             # Hallucination detection modules
│       ├── evaluation/            # Evaluation metrics and benchmarks
│       ├── data/                  # Data loading and preprocessing
│       ├── models/                # Model definitions and wrappers
│       └── utils/                 # Shared utilities
├── tests/
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── configs/                       # Configuration files
├── scripts/                       # CLI scripts and entry points
├── notebooks/                     # Jupyter notebooks for exploration
└── docs/                          # Documentation
```

## Setup

```bash
pip install -e .
```

## Usage

```python
from audio_hallucination.detection import detect_hallucinations

results = detect_hallucinations(audio_path="sample.wav", transcript="...")
```

## Testing

```bash
pytest tests/
```

## License

MIT
