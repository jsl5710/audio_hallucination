# Audio Hallucination Detection

Multi-model, multilingual evaluation of hallucination detection in audio-based AI systems. This project runs structured experiments across multiple audio-language models to classify hallucinations by presence, type, and severity.

## Overview

The experiment pipeline sends audio (or text) samples through multiple models, each prompted to independently classify:

1. **Binary detection** - Is there a hallucination? (yes/no)
2. **Type classification** - What kind? (factual_contradiction / factual_fabrication / contextual_inconsistency / none)
3. **Degree assessment** - How severe? (mild / moderate / severe / none)

Each task is evaluated using two prompting strategies: **direct** and **chain-of-thought (CoT)**, resulting in 6 model calls per sample (3 tasks x 2 approaches).

## Models

| Model | Notebook | Notes |
|-------|----------|-------|
| [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | `Hallucination_qwen25omni_experiment.ipynb` | Primary model; uses `qwen_omni_utils` |
| [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | `Hallucination_qwen2audio_experiment.ipynb` | Audio loaded via `librosa` |
| [Gemma 3n E4B](https://huggingface.co/google/gemma-3n-E4B-it) | `Hallucination_gemma3n_experiment.ipynb` | Requires `transformers>=4.53.0`, HF login |
| [LFM2-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B) | `Hallucination_lfm2audio_experiment.ipynb` | Uses `liquid-audio` package; English only |
| [Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) | `Hallucination_stepaudio2_experiment.ipynb` | Requires `transformers==4.49.0` and repo clone |

Each model has its own self-contained Colab notebook to avoid dependency conflicts.

## Languages

- English
- Kazakh
- Russian

## Project Structure

```
audio_hallucination/
├── notebooks/
│   ├── Hallucination_qwen25omni_experiment.ipynb       # Qwen2.5-Omni (primary)
│   ├── Hallucination_qwen2audio_experiment.ipynb       # Qwen2-Audio
│   ├── Hallucination_gemma3n_experiment.ipynb          # Gemma 3n
│   ├── Hallucination_lfm2audio_experiment.ipynb        # LFM2-Audio
│   ├── Hallucination_stepaudio2_experiment.ipynb       # Step-Audio-2
│   └── Hallucination_audio_text_results.ipynb          # Results analysis
├── src/
│   └── audio_hallucination/
├── tests/
├── configs/
├── scripts/
└── docs/
```

## Data

Experiments expect data on Google Drive at:

```
/content/drive/MyDrive/ICASSP_Hallucinaton/Audio_data/
├── English/    # CSV + audio files
├── Kazakh/     # CSV + audio files
└── Russian/    # CSV + audio files
```

Each CSV contains columns: `filename`, `text`, `hallucination`, `hallucination_type`, `hallucination_level`.

## Running Experiments

1. Open any experiment notebook in Google Colab (A100 GPU recommended)
2. Mount Google Drive and run the setup cells
3. Execute the main cell -- experiments auto-resume from checkpoints if interrupted

Results are saved to `hallucination_results/{model_name}/{experiment_type}/` with per-language CSV files and JSON summaries.

### Prompt Design

All models use identical, separated single-task prompts (PROMPT_VERSION=2). Each of the 3 classification tasks (binary, type, degree) gets its own focused prompt rather than a single combined prompt, improving parsing reliability and enabling independent evaluation of each task.

### Checkpointing

Every notebook includes smart checkpoint management that:
- Saves progress every 5 samples
- Resumes from interruptions without re-processing completed samples
- Matches samples by content hash (robust to dataset reordering)
- Detects prompt version changes and restarts automatically

## Results Analysis

`Hallucination_audio_text_results.ipynb` loads results from all models and computes:
- Per-model accuracy, precision, recall, F1
- Confusion matrices for type and degree classification
- Cross-language comparison charts

Output columns used by the results notebook: `pred_{approach}_{task}` (e.g., `pred_direct_binary`, `pred_cot_type`, `pred_direct_degree`).

## License

MIT
