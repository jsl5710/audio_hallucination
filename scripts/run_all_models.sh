#!/bin/bash
# =============================================================================
# Run All Hallucination Classification Experiments
# =============================================================================
#
# Each model runs in its own virtual environment to avoid dependency conflicts
# (e.g. Step-Audio-2 needs transformers==4.49.0 while Gemma needs >=4.53.0).
#
# Setup:
#   1. git clone the repo
#   2. Copy the ICASSP_Hallucinaton folder contents into the repo root:
#        cp -r /path/to/ICASSP_Hallucinaton/Audio_data ./
#        cp -r /path/to/ICASSP_Hallucinaton/checkpoints ./
#        cp -r /path/to/ICASSP_Hallucinaton/hallucination_results ./
#        cp -r /path/to/ICASSP_Hallucinaton/tables ./
#   3. Set HF token: export HF_TOKEN="hf_your_token_here"
#   4. Run: ./scripts/run_all_models.sh
#
# The script creates venvs/ with a separate environment per model.
# On first run this takes a while (pip install). On subsequent runs the
# existing venvs are reused instantly.
#
# =============================================================================

set -e  # Exit on error

# ========================== CONFIGURATION ==========================

# Project root (auto-detected: parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Path to Audio_data directory (default: ./Audio_data in the repo root)
DATA_DIR="${PROJECT_ROOT}/Audio_data"

# Batch size for processing (higher = fewer checkpoints, faster I/O)
BATCH_SIZE=4

# Experiment types to run: "audio", "text", or "audio text" for both
EXPERIMENT_TYPES="audio"

# Languages to process: "english kazakh russian" (space-separated)
LANGUAGES="english kazakh russian"

# Output directories (default: inside repo root, matching Colab layout)
OUTPUT_DIR="${PROJECT_ROOT}/hallucination_results"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
TABLES_DIR="${PROJECT_ROOT}/tables"

# Virtual environments directory
VENVS_DIR="${PROJECT_ROOT}/venvs"

# HuggingFace token for Gemma 3n (REQUIRED for Gemma)
# Set via: export HF_TOKEN="hf_your_token_here" before running this script
HF_TOKEN="${HF_TOKEN:-your_huggingface_token_here}"

# Path to cloned Step-Audio2 repo (REQUIRED for Step-Audio-2)
# Will auto-clone into repo root if not found
STEP_AUDIO_REPO="${PROJECT_ROOT}/Step-Audio2"

# Set to "true" to ignore existing checkpoints and start fresh
FORCE_RESTART="false"

# Set to "true" to use Flash Attention 2 (must be installed)
FLASH_ATTN="false"

# Set to "true" to only process hallucinated samples
FILTER_HALLUCINATED_ONLY="false"

# Which models to run (set to false to skip)
RUN_QWEN25OMNI=true
RUN_QWEN2AUDIO=true
RUN_GEMMA3N=true
RUN_LFM2AUDIO=true
RUN_STEPAUDIO2=true
RUN_RESULTS=true

# ========================== END CONFIGURATION ==========================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_DIR="${PROJECT_ROOT}/requirements"

# Build common CLI arguments
COMMON_ARGS="--data-dir ${DATA_DIR} --batch-size ${BATCH_SIZE} --experiment-types ${EXPERIMENT_TYPES} --languages ${LANGUAGES} --output-dir ${OUTPUT_DIR} --checkpoint-dir ${CHECKPOINT_DIR}"

if [ "$FORCE_RESTART" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --force-restart"
fi

if [ "$FLASH_ATTN" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --flash-attn"
fi

if [ "$FILTER_HALLUCINATED_ONLY" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --filter-hallucinated-only"
fi

# Always skip interactive validation for automated runs
COMMON_ARGS="${COMMON_ARGS} --no-validate"

# ========================== VENV HELPERS ==========================

setup_venv() {
    # Creates a virtual environment and installs requirements if needed.
    # Usage: setup_venv <venv_name> <requirements_file>
    local venv_name="$1"
    local req_file="$2"
    local venv_path="${VENVS_DIR}/${venv_name}"

    if [ ! -d "$venv_path" ]; then
        log_info "Creating virtual environment: ${venv_name}"
        python3 -m venv "$venv_path"
        log_info "Installing dependencies from ${req_file}..."
        "${venv_path}/bin/pip" install --upgrade pip -q
        "${venv_path}/bin/pip" install -r "$req_file"
        log_ok "Environment ${venv_name} ready"
    else
        log_ok "Reusing existing environment: ${venv_name}"
    fi
}

# ========================== VALIDATION ==========================

echo ""
echo "============================================================"
echo "  Hallucination Classification - Run All Models"
echo "============================================================"
echo ""

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    log_error "Data directory not found: ${DATA_DIR}"
    log_error ""
    log_error "Expected layout after setup:"
    log_error "  audio_hallucination/           (this repo)"
    log_error "  ├── Audio_data/                (copy from ICASSP_Hallucinaton/)"
    log_error "  │   ├── English/"
    log_error "  │   ├── Kazakh/"
    log_error "  │   └── Russian/"
    log_error "  ├── checkpoints/               (copy from ICASSP_Hallucinaton/ for resume)"
    log_error "  ├── hallucination_results/     (copy from ICASSP_Hallucinaton/ for resume)"
    log_error "  └── scripts/"
    exit 1
fi

log_ok "Data directory: ${DATA_DIR}"

# Check for language folders
for lang_folder in English Kazakh Russian; do
    if [ -d "${DATA_DIR}/${lang_folder}" ]; then
        csv_count=$(find "${DATA_DIR}/${lang_folder}" -name "*.csv" | wc -l)
        audio_count=$(find "${DATA_DIR}/${lang_folder}" -name "*.wav" -o -name "*.mp3" -o -name "*.flac" 2>/dev/null | wc -l)
        log_ok "  ${lang_folder}/: ${csv_count} CSV, ${audio_count} audio files"
    else
        log_warn "  ${lang_folder}/ not found"
    fi
done

# Check for prior results (for smart resume)
if [ -d "$OUTPUT_DIR" ]; then
    model_count=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$model_count" -gt 0 ]; then
        log_ok "Prior results found (${model_count} model(s)) - will resume from checkpoints"
        for model_dir in "$OUTPUT_DIR"/*/; do
            model_name=$(basename "$model_dir")
            for exp_dir in "$model_dir"*/; do
                [ -d "$exp_dir" ] || continue
                exp_type=$(basename "$exp_dir")
                result_count=$(find "$exp_dir" -name "*_results.csv" 2>/dev/null | wc -l)
                if [ "$result_count" -gt 0 ]; then
                    log_info "  ${model_name}/${exp_type}: ${result_count} language result(s)"
                fi
            done
        done
    fi
else
    log_info "No prior results found - starting fresh"
fi

if [ -d "$CHECKPOINT_DIR" ]; then
    ckpt_count=$(find "$CHECKPOINT_DIR" -name "*_progress.json" 2>/dev/null | wc -l)
    if [ "$ckpt_count" -gt 0 ]; then
        log_ok "Checkpoints found (${ckpt_count} in-progress) - will resume incomplete runs"
    fi
fi

echo ""
log_info "Project root:     ${PROJECT_ROOT}"
log_info "Batch size:       ${BATCH_SIZE}"
log_info "Experiment types: ${EXPERIMENT_TYPES}"
log_info "Languages:        ${LANGUAGES}"
log_info "Output dir:       ${OUTPUT_DIR}"
log_info "Checkpoint dir:   ${CHECKPOINT_DIR}"
log_info "Venvs dir:        ${VENVS_DIR}"
log_info "Force restart:    ${FORCE_RESTART}"
log_info "Flash Attention:  ${FLASH_ATTN}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${TABLES_DIR}" "${VENVS_DIR}"

# Track results
SUCCEEDED=()
FAILED=()
SKIPPED=()

run_model() {
    local name="$1"
    local venv_name="$2"
    local req_file="$3"
    local cmd="$4"

    echo ""
    echo "============================================================"
    log_info "Starting: ${name}"
    echo "============================================================"

    # Set up venv and install deps
    setup_venv "$venv_name" "$req_file"

    # Run the script using the venv's Python
    local venv_python="${VENVS_DIR}/${venv_name}/bin/python"
    local full_cmd="${venv_python} ${cmd}"

    echo "Command: ${full_cmd}"
    echo ""

    if eval "$full_cmd"; then
        log_ok "${name} completed successfully"
        SUCCEEDED+=("$name")
    else
        log_error "${name} failed (exit code: $?)"
        FAILED+=("$name")
    fi
}

# ========================== MODEL 1: Qwen2.5-Omni-3B ==========================

if [ "$RUN_QWEN25OMNI" = true ]; then
    run_model "Qwen2.5-Omni-3B" \
        "qwen25omni" \
        "${REQ_DIR}/qwen25omni.txt" \
        "${SCRIPT_DIR}/run_qwen25omni.py ${COMMON_ARGS}"
else
    SKIPPED+=("Qwen2.5-Omni-3B")
fi

# ========================== MODEL 2: Qwen2-Audio-7B ==========================

if [ "$RUN_QWEN2AUDIO" = true ]; then
    run_model "Qwen2-Audio-7B-Instruct" \
        "qwen2audio" \
        "${REQ_DIR}/qwen2audio.txt" \
        "${SCRIPT_DIR}/run_qwen2audio.py ${COMMON_ARGS}"
else
    SKIPPED+=("Qwen2-Audio-7B-Instruct")
fi

# ========================== MODEL 3: Gemma 3n E4B ==========================

if [ "$RUN_GEMMA3N" = true ]; then
    run_model "Gemma-3n-E4B" \
        "gemma3n" \
        "${REQ_DIR}/gemma3n.txt" \
        "${SCRIPT_DIR}/run_gemma3n.py ${COMMON_ARGS} --hf-token ${HF_TOKEN}"
else
    SKIPPED+=("Gemma-3n-E4B")
fi

# ========================== MODEL 4: LFM2-Audio-1.5B ==========================

if [ "$RUN_LFM2AUDIO" = true ]; then
    run_model "LFM2-Audio-1.5B" \
        "lfm2audio" \
        "${REQ_DIR}/lfm2audio.txt" \
        "${SCRIPT_DIR}/run_lfm2audio.py ${COMMON_ARGS}"
else
    SKIPPED+=("LFM2-Audio-1.5B")
fi

# ========================== MODEL 5: Step-Audio-2-mini ==========================

if [ "$RUN_STEPAUDIO2" = true ]; then
    # Auto-clone Step-Audio2 repo if needed
    if [ ! -d "$STEP_AUDIO_REPO" ]; then
        log_info "Cloning Step-Audio2 repository..."
        git clone https://github.com/stepfun-ai/Step-Audio2.git "$STEP_AUDIO_REPO"
    fi

    run_model "Step-Audio-2-mini" \
        "stepaudio2" \
        "${REQ_DIR}/stepaudio2.txt" \
        "${SCRIPT_DIR}/run_stepaudio2.py ${COMMON_ARGS} --step-audio-repo ${STEP_AUDIO_REPO}"
else
    SKIPPED+=("Step-Audio-2-mini")
fi

# ========================== RESULTS ANALYSIS ==========================

if [ "$RUN_RESULTS" = true ]; then
    echo ""
    echo "============================================================"
    log_info "Generating results tables and metrics"
    echo "============================================================"

    # Results analysis has minimal deps - use its own lightweight venv
    setup_venv "results" "${REQ_DIR}/results.txt"
    local results_python="${VENVS_DIR}/results/bin/python"

    if "${VENVS_DIR}/results/bin/python" "${SCRIPT_DIR}/run_results_analysis.py" \
        --results-dir "${OUTPUT_DIR}" \
        --output-dir "${TABLES_DIR}"; then
        log_ok "Results analysis completed"
        SUCCEEDED+=("Results Analysis")
    else
        log_error "Results analysis failed"
        FAILED+=("Results Analysis")
    fi
else
    SKIPPED+=("Results Analysis")
fi

# ========================== SUMMARY ==========================

echo ""
echo "============================================================"
echo "  FINAL SUMMARY"
echo "============================================================"
echo ""

if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    log_ok "Succeeded (${#SUCCEEDED[@]}):"
    for s in "${SUCCEEDED[@]}"; do echo "    - $s"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Failed (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    log_warn "Skipped (${#SKIPPED[@]}):"
    for k in "${SKIPPED[@]}"; do echo "    - $k"; done
fi

echo ""
log_info "Results directory: ${OUTPUT_DIR}/"
log_info "Tables directory:  ${TABLES_DIR}/"
log_info "Venvs directory:   ${VENVS_DIR}/"
echo ""

# Exit with error if any model failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
