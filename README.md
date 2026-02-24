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

Each model has its own self-contained Colab notebook to avoid dependency conflicts. Equivalent standalone scripts are available in `scripts/` for running on GPU servers.

## Languages

- English
- Kazakh
- Russian

## Project Structure

```
audio_hallucination/
├── notebooks/                                           # Colab notebooks
│   ├── Hallucination_qwen25omni_experiment.ipynb
│   ├── Hallucination_qwen2audio_experiment.ipynb
│   ├── Hallucination_gemma3n_experiment.ipynb
│   ├── Hallucination_lfm2audio_experiment.ipynb
│   ├── Hallucination_stepaudio2_experiment.ipynb
│   └── Hallucination_audio_text_results.ipynb
├── scripts/                                             # Standalone GPU scripts
│   ├── experiment_utils.py                              # Shared utilities
│   ├── run_all_models.sh                                # Run all 5 models + results
│   ├── run_qwen25omni.py                                # Qwen2.5-Omni-3B
│   ├── run_qwen2audio.py                                # Qwen2-Audio-7B
│   ├── run_gemma3n.py                                   # Gemma 3n E4B
│   ├── run_lfm2audio.py                                 # LFM2-Audio-1.5B
│   ├── run_stepaudio2.py                                # Step-Audio-2-mini
│   └── run_results_analysis.py                          # Results & LaTeX tables
├── src/audio_hallucination/                             # Python package
├── tests/
├── configs/
└── docs/
```

## Data & Prior Results

On Google Colab, everything lives in a single Drive folder:

```
ICASSP_Hallucinaton/               (Google Drive)
├── Audio_data/                    # Input: CSVs + audio files per language
│   ├── English/
│   ├── Kazakh/
│   └── Russian/
├── checkpoints/                   # In-progress checkpoint files
├── hallucination_results/         # Completed results from prior runs
└── tables/                        # Generated LaTeX tables & metrics
```

For running on a **GPU server**, you copy these 4 folders into the repo root so the scripts can find them and **resume from prior Colab runs**:

```
audio_hallucination/               (git clone)
├── Audio_data/                    <-- copy from ICASSP_Hallucinaton/
├── checkpoints/                   <-- copy from ICASSP_Hallucinaton/
├── hallucination_results/         <-- copy from ICASSP_Hallucinaton/
├── tables/                        <-- copy from ICASSP_Hallucinaton/
├── scripts/
├── notebooks/
└── ...
```

Each language CSV must contain columns: `filename`, `text`, `hallucination`, `hallucination_type`, `hallucination_level`.

## Running Experiments

### Option A: Google Colab (notebooks)

1. Open any experiment notebook in Google Colab (A100 GPU recommended)
2. Mount Google Drive and run the setup cells
3. Execute the main cell -- experiments auto-resume from checkpoints if interrupted

### Option B: GPU Server (scripts)

All model scripts are in `scripts/` and share a common utilities module (`experiment_utils.py`). Each script is self-contained with its own model setup and CLI arguments.

#### Step 1: Clone and set up data

```bash
# Clone the repo
git clone https://github.com/jsl5710/audio_hallucination.git
cd audio_hallucination

# Copy the ICASSP_Hallucinaton folder contents into the repo root
# This includes Audio_data + prior results/checkpoints for smart resume
cp -r /path/to/ICASSP_Hallucinaton/Audio_data ./
cp -r /path/to/ICASSP_Hallucinaton/checkpoints ./
cp -r /path/to/ICASSP_Hallucinaton/hallucination_results ./
cp -r /path/to/ICASSP_Hallucinaton/tables ./
```

#### Step 2: Run all models at once (recommended)

The script **automatically creates a separate virtual environment per model** (under `venvs/`) so that conflicting dependencies (e.g. Step-Audio-2 needs `transformers==4.49.0`, Gemma needs `>=4.53.0`) don't interfere. On first run each venv is created and packages are installed; on subsequent runs the existing venvs are reused instantly.

```bash
# Set the HuggingFace token (needed for Gemma 3n)
export HF_TOKEN="your_huggingface_token"

# Run all 5 models sequentially, then generate results
chmod +x scripts/run_all_models.sh
./scripts/run_all_models.sh
```

The script will:
- Auto-detect `Audio_data/`, `checkpoints/`, and `hallucination_results/` in the repo root
- Print what prior results it found and resume from where each model left off
- Create `venvs/qwen25omni/`, `venvs/qwen2audio/`, etc. with the right dependencies
- Run each model, then generate LaTeX tables and metrics

To skip specific models, edit the `RUN_*` flags at the top of `run_all_models.sh`.

#### Step 2b: Run individual models manually

If you prefer manual control, create a venv yourself and install from the per-model requirements file:

```bash
# Example: Qwen2.5-Omni-3B
python3 -m venv venvs/qwen25omni
source venvs/qwen25omni/bin/activate
pip install -r requirements/qwen25omni.txt
python scripts/run_qwen25omni.py --data-dir ./Audio_data --batch-size 4
deactivate

# Example: Qwen2-Audio-7B-Instruct
python3 -m venv venvs/qwen2audio
source venvs/qwen2audio/bin/activate
pip install -r requirements/qwen2audio.txt
python scripts/run_qwen2audio.py --data-dir ./Audio_data --batch-size 8 --experiment-types audio text
deactivate

# Example: Gemma 3n E4B
python3 -m venv venvs/gemma3n
source venvs/gemma3n/bin/activate
pip install -r requirements/gemma3n.txt
export HF_TOKEN="your_huggingface_token"
python scripts/run_gemma3n.py --data-dir ./Audio_data --batch-size 4
deactivate

# Example: LFM2-Audio-1.5B (English only)
python3 -m venv venvs/lfm2audio
source venvs/lfm2audio/bin/activate
pip install -r requirements/lfm2audio.txt
python scripts/run_lfm2audio.py --data-dir ./Audio_data --batch-size 4 --languages english
deactivate

# Example: Step-Audio-2-mini
git clone https://github.com/stepfun-ai/Step-Audio2.git ./Step-Audio2
python3 -m venv venvs/stepaudio2
source venvs/stepaudio2/bin/activate
pip install -r requirements/stepaudio2.txt
python scripts/run_stepaudio2.py --data-dir ./Audio_data --step-audio-repo ./Step-Audio2
deactivate

# Results analysis (lightweight, no GPU needed)
python3 -m venv venvs/results
source venvs/results/bin/activate
pip install -r requirements/results.txt
python scripts/run_results_analysis.py --results-dir ./hallucination_results --output-dir ./tables
deactivate
```

#### Requirements files

```
requirements/
├── base.txt          # Shared: torch, torchaudio, accelerate, pandas, tqdm, etc.
├── qwen25omni.txt    # + transformers (Qwen2.5-Omni preview branch), qwen-omni-utils
├── qwen2audio.txt    # + transformers, librosa
├── gemma3n.txt       # + transformers>=4.53.0, librosa
├── lfm2audio.txt     # + liquid-audio
├── stepaudio2.txt    # + transformers==4.49.0, onnxruntime, s3tokenizer, diffusers, hyperpyyaml
└── results.txt       # pandas, numpy, scikit-learn (no GPU deps)
```

#### CLI Arguments (all model scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | **required** | Path to `Audio_data/` directory containing `English/`, `Kazakh/`, `Russian/` |
| `--batch-size` | `1` | Number of samples per batch. Higher values checkpoint less frequently but reduce I/O overhead |
| `--experiment-types` | `audio` | Space-separated list: `audio`, `text`, or `audio text` for both |
| `--languages` | `english kazakh russian` | Which languages to process |
| `--output-dir` | `hallucination_results` | Directory for result CSVs and JSON summaries |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint files |
| `--force-restart` | off | Ignore existing checkpoints and start fresh |
| `--flash-attn` | off | Use Flash Attention 2 (if installed) |
| `--filter-hallucinated-only` | off | Only process samples where `hallucination=yes` |
| `--no-validate` | off | Skip smart resume validation |

#### Model-Specific Arguments

| Script | Extra Argument | Description |
|--------|---------------|-------------|
| `run_gemma3n.py` | `--hf-token` | HuggingFace token (or set `HF_TOKEN` env var; or run `huggingface-cli login`) |
| `run_stepaudio2.py` | `--step-audio-repo` | **Required.** Path to cloned [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) repository |

#### Results Analysis Script

```bash
python scripts/run_results_analysis.py --results-dir ./hallucination_results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | **required** | Path to `hallucination_results/` directory |
| `--output-dir` | `tables` | Where to save LaTeX tables, JSON metrics, and analysis |
| `--models` | auto-detect | Space-separated list of model names to analyze |

Generates:
- LaTeX tables (per model, per experiment type, audio vs text comparison)
- JSON metrics for all models/languages/approaches
- Detailed confusion analysis per model

### Prompt Design

All models use identical, separated single-task prompts (PROMPT_VERSION=2). Each of the 3 classification tasks (binary, type, degree) gets its own focused prompt rather than a single combined prompt, improving parsing reliability and enabling independent evaluation of each task.

### Checkpointing

Both notebooks and scripts include smart checkpoint management that:
- Saves progress after every batch (controlled by `--batch-size` in scripts, every 5 samples in notebooks)
- Resumes from interruptions without re-processing completed samples
- Matches samples by content hash (robust to dataset reordering)
- Detects prompt version changes and restarts automatically

### Output Structure

```
hallucination_results/
├── Qwen2.5-Omni-3B/
│   ├── audio/
│   │   ├── english_results.csv
│   │   ├── kazakh_results.csv
│   │   ├── russian_results.csv
│   │   └── *_summary.json
│   └── text/
│       └── (same structure)
├── Qwen2-Audio-7B-Instruct/
├── gemma-3n-E4B-it/
├── LFM2-Audio-1.5B/
└── Step-Audio-2-mini/
```

## Results Analysis

`Hallucination_audio_text_results.ipynb` (or `scripts/run_results_analysis.py`) loads results from all models and computes:
- Per-model accuracy, precision, recall, F1
- Confusion matrices for type and degree classification
- Cross-language comparison charts
- LaTeX tables for paper inclusion

Output columns: `pred_{approach}_{task}` (e.g., `pred_direct_binary`, `pred_cot_type`, `pred_direct_degree`).

## License

MIT
