#!/bin/bash
set -euo pipefail

# ============================================================================
# C4 Dataset Download and Preprocessing
# ============================================================================
# Two methods are available:
#
# Method 1: HuggingFace Hub (recommended for simplicity)
#   - Downloads C4 via the HuggingFace datasets API
#   - Requires: `huggingface-cli login` and access approval for allenai/c4
#
# Method 2: Arrow subset download
#   - Downloads specific Arrow files from HuggingFace Hub
#   - More control over download size
# ============================================================================

METHOD=${C4_METHOD:-"hub"}
NUM_FILES=${C4_NUM_FILES:-50}
NUM_PROC=${NUM_PROC:-8}
C4_LOCAL_DIR=${C4_LOCAL_DIR:-"./data/c4_en_subset"}
C4_TOKENIZED_DIR=${C4_TOKENIZED_DIR:-"./data/tokenized/c4"}

echo "=========================================="
echo "C4 Preprocessing (method=${METHOD})"
echo "=========================================="

if [ "${METHOD}" = "hub" ]; then
    # Method 1: Direct HuggingFace download + tokenization
    echo "Downloading and tokenizing C4 via HuggingFace datasets API..."
    python -m src.utils cache_data \
        --dataset_name "allenai/c4" \
        --out_dir "${C4_TOKENIZED_DIR}" \
        --tokenizer_name "EleutherAI/pythia-1b"

elif [ "${METHOD}" = "subset" ]; then
    # Method 2: Arrow file subset download + tokenization
    echo "Step 1: Downloading ${NUM_FILES} C4 Arrow files..."
    python -m src.download_c4 download_subset \
        --out_dir "${C4_LOCAL_DIR}" \
        --num_files ${NUM_FILES}

    echo "Step 2: Tokenizing local C4 files..."
    python -m src.utils cache_c4_local \
        --out_dir "${C4_TOKENIZED_DIR}" \
        --local_dir "${C4_LOCAL_DIR}/en" \
        --tokenizer_name "EleutherAI/pythia-1b" \
        --num_files ${NUM_FILES} \
        --num_proc ${NUM_PROC}
else
    echo "Error: Unknown method '${METHOD}'. Use 'hub' or 'subset'."
    exit 1
fi

echo "C4 preprocessing complete! Tokenized data saved to: ${C4_TOKENIZED_DIR}"
