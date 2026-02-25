# Audio Hallucination Detection

Multi-model, multilingual evaluation of hallucination detection in audio-based AI systems. This project runs structured experiments across multiple audio-language models to classify hallucinations by presence, type, and severity across three languages (English, Kazakh, Russian).

## Overview

The experiment pipeline sends audio (or text) samples through 5 models. Each model is independently prompted to classify:

1. **Binary detection** - Is there a hallucination? (`yes` / `no`)
2. **Type classification** - What kind? (`factual_contradiction` / `factual_fabrication` / `contextual_inconsistency` / `none`)
3. **Degree assessment** - How severe? (`mild` / `moderate` / `severe` / `none`)

Each task is evaluated using two prompting strategies: **direct** and **chain-of-thought (CoT)**, resulting in **6 model calls per sample** (3 tasks x 2 approaches).

## Models

| Model | Size | Script | Notebook | Key Dependency | Notes |
|-------|------|--------|----------|----------------|-------|
| [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 3B | `run_qwen25omni.py` | `Hallucination_qwen25omni_experiment.ipynb` | transformers (preview branch) + `qwen-omni-utils` | Primary model |
| [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | 7B | `run_qwen2audio.py` | `Hallucination_qwen2audio_experiment.ipynb` | transformers + `librosa` | Standard HF transformers |
| [Gemma 3n E4B](https://huggingface.co/google/gemma-3n-E4B-it) | 4B (quantized) | `run_gemma3n.py` | `Hallucination_gemma3n_experiment.ipynb` | `transformers>=4.53.0` | Requires HuggingFace token |
| [LFM2-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B) | 1.5B | `run_lfm2audio.py` | `Hallucination_lfm2audio_experiment.ipynb` | `liquid-audio` | **English only** |
| [Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) | - | `run_stepaudio2.py` | `Hallucination_stepaudio2_experiment.ipynb` | `transformers==4.49.0` | Requires cloned [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) repo |

> **Why separate environments?** Step-Audio-2 requires `transformers==4.49.0`, Qwen2.5-Omni requires a custom preview branch, and Gemma requires `transformers>=4.53.0`. Each model runs in its own virtual environment to avoid conflicts.

## Languages

- **English**
- **Kazakh**
- **Russian**

## Project Structure

```
audio_hallucination/                  (this repository)
│
├── scripts/                          # Standalone GPU server scripts
│   ├── run_all_models.sh            #   Run all 5 models end-to-end (recommended)
│   ├── experiment_utils.py          #   Shared: prompts, data loading, checkpoint manager, runner
│   ├── run_qwen25omni.py            #   Qwen2.5-Omni-3B experiment
│   ├── run_qwen2audio.py            #   Qwen2-Audio-7B-Instruct experiment
│   ├── run_gemma3n.py               #   Gemma 3n E4B experiment
│   ├── run_lfm2audio.py             #   LFM2-Audio-1.5B experiment
│   ├── run_stepaudio2.py            #   Step-Audio-2-mini experiment
│   ├── run_results_analysis.py      #   Compute metrics & generate LaTeX tables
│   └── run_detection.py             #   CLI for hallucination detection (package)
│
├── requirements/                     # Per-model dependency files
│   ├── base.txt                     #   Shared: torch, torchaudio, accelerate, pandas, tqdm
│   ├── qwen25omni.txt               #   + transformers preview branch, qwen-omni-utils
│   ├── qwen2audio.txt               #   + transformers, librosa
│   ├── gemma3n.txt                  #   + transformers>=4.53.0, librosa
│   ├── lfm2audio.txt                #   + liquid-audio
│   ├── stepaudio2.txt               #   + transformers==4.49.0, onnxruntime, s3tokenizer, etc.
│   └── results.txt                  #   pandas, numpy, scikit-learn (no GPU needed)
│
├── notebooks/                        # Google Colab notebooks (original experiments)
│   ├── Hallucination_qwen25omni_experiment.ipynb
│   ├── Hallucination_qwen2audio_experiment.ipynb
│   ├── Hallucination_gemma3n_experiment.ipynb
│   ├── Hallucination_lfm2audio_experiment.ipynb
│   ├── Hallucination_stepaudio2_experiment.ipynb
│   └── Hallucination_audio_text_results.ipynb
│
├── src/audio_hallucination/          # Python package (detection utilities)
├── tests/
├── configs/
├── docs/
└── pyproject.toml
```

**After setup** (see below), the repo root will also contain these data/output folders:

```
audio_hallucination/
├── Audio_data/                       # Input data (copied from ICASSP_Hallucinaton/)
│   ├── English/                     #   CSV + audio files
│   ├── Kazakh/                      #   CSV + audio files
│   └── Russian/                     #   CSV + audio files
├── checkpoints/                      # In-progress checkpoint files
├── hallucination_results/            # Completed result CSVs + JSON summaries
├── tables/                           # Generated LaTeX tables & JSON metrics
└── venvs/                            # Auto-created per-model virtual environments
```

These folders are gitignored and will not be committed.

---

## Running on a GPU Server

### Step 1: Clone and set up data

```bash
# Clone the repo
git clone https://github.com/jsl5710/audio_hallucination.git
cd audio_hallucination

# Copy the ICASSP_Hallucinaton folder contents into the repo root.
# This brings in the audio data AND prior Colab results/checkpoints so
# experiments resume where they left off.
cp -r /path/to/ICASSP_Hallucinaton/Audio_data ./
cp -r /path/to/ICASSP_Hallucinaton/checkpoints ./
cp -r /path/to/ICASSP_Hallucinaton/hallucination_results ./
cp -r /path/to/ICASSP_Hallucinaton/tables ./
```

Your repo root should now look like:

```
audio_hallucination/
├── Audio_data/
│   ├── English/     (CSV + .wav/.mp3/.flac files)
│   ├── Kazakh/
│   └── Russian/
├── checkpoints/            (prior run checkpoints)
├── hallucination_results/  (prior run results)
├── tables/                 (prior tables)
├── scripts/
├── requirements/
├── notebooks/
└── ...
```

Each language folder must contain a CSV with these columns:

| Column | Description |
|--------|-------------|
| `filename` | Audio filename (e.g. `sample_001.wav`) |
| `text` | Transcript / text content |
| `hallucination` | `yes` or `no` |
| `hallucination_type` | `factual_contradiction`, `factual_fabrication`, `contextual_inconsistency`, or `none` |
| `hallucination_level` | `mild`, `moderate`, `severe`, or `none` |

### Step 2: Run all models (recommended)

```bash
# Set the HuggingFace token (needed for Gemma 3n)
export HF_TOKEN="your_huggingface_token"

# Make the script executable and run
chmod +x scripts/run_all_models.sh
./scripts/run_all_models.sh
```

**What this does:**

1. Validates `Audio_data/` exists and shows what language folders are available
2. Detects prior results in `hallucination_results/` and `checkpoints/` for smart resume
3. For each model, creates an isolated virtual environment (`venvs/<model>/`) and installs its specific dependencies from `requirements/<model>.txt`
4. Runs each model script sequentially using the model's venv Python
5. Auto-clones the [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) repo if needed
6. Generates LaTeX tables and metrics from all results
7. Prints a color-coded pass/fail summary

On the **first run**, creating venvs and installing packages takes time. On subsequent runs the existing venvs are **reused instantly**.

### Step 2b: Run individual models manually

If you prefer manual control, create a venv and install from the per-model requirements file:

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

---

## Configuration Reference

### `run_all_models.sh` variables

These are set at the top of the script. Edit as needed before running.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./Audio_data` | Path to data directory (auto-detected in repo root) |
| `BATCH_SIZE` | `4` | Samples per batch |
| `EXPERIMENT_TYPES` | `audio` | `audio`, `text`, or `audio text` |
| `LANGUAGES` | `english kazakh russian` | Space-separated |
| `OUTPUT_DIR` | `./hallucination_results` | Where result CSVs are saved |
| `CHECKPOINT_DIR` | `./checkpoints` | Where checkpoint files are saved |
| `TABLES_DIR` | `./tables` | Where LaTeX tables and metrics go |
| `VENVS_DIR` | `./venvs` | Where per-model virtual environments are created |
| `HF_TOKEN` | from `$HF_TOKEN` env var | HuggingFace token for Gemma 3n |
| `STEP_AUDIO_REPO` | `./Step-Audio2` | Path to cloned Step-Audio2 repo (auto-cloned if missing) |
| `FORCE_RESTART` | `false` | Set `true` to ignore all prior checkpoints |
| `FLASH_ATTN` | `false` | Set `true` to use Flash Attention 2 |
| `FILTER_HALLUCINATED_ONLY` | `false` | Set `true` to only process `hallucination=yes` samples |
| `RUN_QWEN25OMNI` | `true` | Set `false` to skip Qwen2.5-Omni-3B |
| `RUN_QWEN2AUDIO` | `true` | Set `false` to skip Qwen2-Audio-7B |
| `RUN_GEMMA3N` | `true` | Set `false` to skip Gemma 3n E4B |
| `RUN_LFM2AUDIO` | `true` | Set `false` to skip LFM2-Audio-1.5B |
| `RUN_STEPAUDIO2` | `true` | Set `false` to skip Step-Audio-2-mini |
| `RUN_RESULTS` | `true` | Set `false` to skip results analysis |

### CLI arguments (all model scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | **required** | Path to `Audio_data/` containing `English/`, `Kazakh/`, `Russian/` |
| `--batch-size` | `1` | Samples per batch. Higher = fewer checkpoints, less I/O overhead |
| `--experiment-types` | `audio` | Space-separated: `audio`, `text`, or `audio text` |
| `--languages` | `english kazakh russian` | Which languages to process |
| `--output-dir` | `hallucination_results` | Directory for result CSVs and JSON summaries |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint files |
| `--force-restart` | off | Ignore existing checkpoints and start fresh |
| `--flash-attn` | off | Use Flash Attention 2 (if installed) |
| `--filter-hallucinated-only` | off | Only process samples where `hallucination=yes` |
| `--no-validate` | off | Skip smart resume validation |

### Model-specific arguments

| Script | Extra Argument | Description |
|--------|---------------|-------------|
| `run_gemma3n.py` | `--hf-token` | HuggingFace token (or set `HF_TOKEN` env var, or run `huggingface-cli login`) |
| `run_stepaudio2.py` | `--step-audio-repo` | **Required.** Path to cloned [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) repository |

### Results analysis script

```bash
python scripts/run_results_analysis.py --results-dir ./hallucination_results --output-dir ./tables
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | **required** | Path to `hallucination_results/` directory |
| `--output-dir` | `tables` | Where to save LaTeX tables, JSON metrics, and analysis |
| `--models` | auto-detect | Space-separated model names to analyze (auto-detects all found) |

### Requirements files

```
requirements/
├── base.txt          # Shared: torch>=2.0, torchaudio>=2.0, accelerate, pandas, tqdm, etc.
├── qwen25omni.txt    # + transformers@v4.51.3-Qwen2.5-Omni-preview, qwen-omni-utils[decord]
├── qwen2audio.txt    # + transformers, librosa
├── gemma3n.txt       # + transformers>=4.53.0, librosa
├── lfm2audio.txt     # + liquid-audio
├── stepaudio2.txt    # + transformers==4.49.0, onnxruntime, s3tokenizer, diffusers, hyperpyyaml
└── results.txt       # pandas, numpy, scikit-learn (no GPU dependencies)
```

Each model file includes `-r base.txt` to pull in the shared dependencies automatically.

---

## How It Works

### Experiment Flow

1. **Load data** - Read CSV from each language folder, load audio files (or text)
2. **Smart resume** - Check for existing results/checkpoints, match samples by content hash, skip already-processed samples
3. **Run inference** - For each sample, run 6 model calls (3 tasks x 2 approaches)
4. **Save checkpoints** - After every batch, save results CSV + checkpoint file
5. **Generate summary** - Per-language JSON summary with prediction distributions
6. **Results analysis** - Compute F1, accuracy, precision, recall; generate LaTeX tables

### Prompt Design

All models use identical, separated single-task prompts (**PROMPT_VERSION=2**). Each of the 3 classification tasks gets its own focused prompt rather than a single combined prompt. This improves parsing reliability and enables independent evaluation per task.

- **Audio prompts**: Ask the model to analyze "audio content"
- **Text prompts**: Same structure but ask about "text content" (auto-derived from audio prompts)
- **Direct**: Straightforward question asking for JSON output
- **Chain-of-Thought (CoT)**: "Think step-by-step" before giving the final answer

### Smart Checkpointing

Both notebooks and scripts include a `SmartCheckpointManager` that:

- **Saves progress** after every batch (`--batch-size` controls frequency; default 1)
- **Resumes from interruptions** without re-processing completed samples
- **Matches samples by content hash** (MD5 of filename + text + language), so it is robust to dataset reordering or row additions
- **Detects prompt version changes** and automatically starts fresh if prompts have changed
- **Transfers results between Colab and server** - prior results from `hallucination_results/` are matched to the current dataset

If a run is interrupted (Ctrl+C, crash, OOM), just re-run the same command - it picks up where it left off.

### Output Structure

```
hallucination_results/
├── Qwen2.5-Omni-3B/
│   ├── audio/
│   │   ├── english_results.csv       # Per-sample predictions
│   │   ├── kazakh_results.csv
│   │   ├── russian_results.csv
│   │   ├── english_summary.json      # Prediction distribution summary
│   │   ├── kazakh_summary.json
│   │   └── russian_summary.json
│   └── text/
│       └── (same structure)
├── Qwen2-Audio-7B-Instruct/
│   └── ...
├── gemma-3n-E4B-it/
│   └── ...
├── LFM2-Audio-1.5B/
│   └── ...
└── Step-Audio-2-mini/
    └── ...
```

**Result CSV columns** (added by the scripts):

| Column | Description |
|--------|-------------|
| `pred_direct_binary` | Direct approach binary prediction (`yes`/`no`) |
| `pred_cot_binary` | CoT approach binary prediction |
| `pred_direct_type` | Direct approach type classification |
| `pred_cot_type` | CoT approach type classification |
| `pred_direct_degree` | Direct approach severity assessment |
| `pred_cot_degree` | CoT approach severity assessment |
| `pred_*_raw` | Raw model response for each prediction |

### Results Analysis Output

`run_results_analysis.py` generates:

- **LaTeX tables** - Per model/experiment type, direct vs CoT comparison, audio vs text comparison
- **JSON metrics** - F1, accuracy for all model/language/approach combinations
- **Detailed analysis** - Confusion matrices, type/level distribution comparisons
- **Audio vs text comparison** - Which input modality performs better per task

```
tables/
├── table_Qwen2_5-Omni-3B_audio.tex
├── table_Qwen2_5-Omni-3B_text.tex
├── comparison_Qwen2_5-Omni-3B_direct.tex
├── comparison_Qwen2_5-Omni-3B_cot.tex
├── all_metrics.json
├── detailed_analysis/
│   └── detailed_*.json
└── audio_text_comparison/
    ├── audio_text_comparison.json
    └── comparison_summary.txt
```

---

## Running on Google Colab (notebooks)

The `notebooks/` directory contains the original Colab notebooks. Each model has its own notebook to avoid dependency conflicts.

1. Open any experiment notebook in Google Colab (A100 GPU recommended)
2. Mount Google Drive and run the setup cells
3. Execute the main cell - experiments auto-resume from checkpoints if interrupted

The notebooks read/write to `/content/drive/MyDrive/ICASSP_Hallucinaton/` on Google Drive. The scripts read/write to the same folder structure but in the repo root.

## License

MIT
