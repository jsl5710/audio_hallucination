#!/bin/bash
# =============================================================================
# Run All Hallucination Classification Experiments
# =============================================================================
#
# This script runs all 5 model experiments and then generates results tables.
# Edit the CONFIGURATION section below before running.
#
# Usage:
#   chmod +x scripts/run_all_models.sh
#   ./scripts/run_all_models.sh
#
# =============================================================================

set -e  # Exit on error

# ========================== CONFIGURATION ==========================
# Edit these variables to match your server setup

# Path to the Audio_data directory (REQUIRED - change this)
DATA_DIR="/path/to/Audio_data"

# Batch size for processing (higher = fewer checkpoints, faster I/O)
BATCH_SIZE=4

# Experiment types to run: "audio", "text", or "audio text" for both
EXPERIMENT_TYPES="audio"

# Languages to process: "english kazakh russian" (space-separated)
LANGUAGES="english kazakh russian"

# Output directories
OUTPUT_DIR="hallucination_results"
CHECKPOINT_DIR="checkpoints"
TABLES_DIR="tables"

# HuggingFace token for Gemma 3n (REQUIRED for Gemma)
HF_TOKEN="${HF_TOKEN:-your_huggingface_token_here}"

# Path to cloned Step-Audio2 repo (REQUIRED for Step-Audio-2)
# Will auto-clone if not found
STEP_AUDIO_REPO="./Step-Audio2"

# Set to "true" to ignore existing checkpoints and start fresh
FORCE_RESTART="false"

# Set to "true" to use Flash Attention 2 (must be installed)
FLASH_ATTN="false"

# Set to "true" to only process hallucinated samples
FILTER_HALLUCINATED_ONLY="false"

# Which models to run (comment out any you want to skip)
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

# ========================== VALIDATION ==========================

echo ""
echo "============================================================"
echo "  Hallucination Classification - Run All Models"
echo "============================================================"
echo ""

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    log_error "Data directory not found: ${DATA_DIR}"
    log_error "Edit DATA_DIR in this script to point to your Audio_data/ directory."
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

echo ""
log_info "Batch size: ${BATCH_SIZE}"
log_info "Experiment types: ${EXPERIMENT_TYPES}"
log_info "Languages: ${LANGUAGES}"
log_info "Output dir: ${OUTPUT_DIR}"
log_info "Force restart: ${FORCE_RESTART}"
log_info "Flash Attention: ${FLASH_ATTN}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

# Track results
SUCCEEDED=()
FAILED=()
SKIPPED=()

run_model() {
    local name="$1"
    local cmd="$2"

    echo ""
    echo "============================================================"
    log_info "Starting: ${name}"
    echo "============================================================"
    echo "Command: ${cmd}"
    echo ""

    if eval "$cmd"; then
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
        "python ${SCRIPT_DIR}/run_qwen25omni.py ${COMMON_ARGS}"
else
    SKIPPED+=("Qwen2.5-Omni-3B")
fi

# ========================== MODEL 2: Qwen2-Audio-7B ==========================

if [ "$RUN_QWEN2AUDIO" = true ]; then
    run_model "Qwen2-Audio-7B-Instruct" \
        "python ${SCRIPT_DIR}/run_qwen2audio.py ${COMMON_ARGS}"
else
    SKIPPED+=("Qwen2-Audio-7B-Instruct")
fi

# ========================== MODEL 3: Gemma 3n E4B ==========================

if [ "$RUN_GEMMA3N" = true ]; then
    run_model "Gemma-3n-E4B" \
        "HF_TOKEN=${HF_TOKEN} python ${SCRIPT_DIR}/run_gemma3n.py ${COMMON_ARGS} --hf-token ${HF_TOKEN}"
else
    SKIPPED+=("Gemma-3n-E4B")
fi

# ========================== MODEL 4: LFM2-Audio-1.5B ==========================

if [ "$RUN_LFM2AUDIO" = true ]; then
    run_model "LFM2-Audio-1.5B" \
        "python ${SCRIPT_DIR}/run_lfm2audio.py ${COMMON_ARGS}"
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
        "python ${SCRIPT_DIR}/run_stepaudio2.py ${COMMON_ARGS} --step-audio-repo ${STEP_AUDIO_REPO}"
else
    SKIPPED+=("Step-Audio-2-mini")
fi

# ========================== RESULTS ANALYSIS ==========================

if [ "$RUN_RESULTS" = true ]; then
    echo ""
    echo "============================================================"
    log_info "Generating results tables and metrics"
    echo "============================================================"

    if python "${SCRIPT_DIR}/run_results_analysis.py" \
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
echo ""

# Exit with error if any model failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
