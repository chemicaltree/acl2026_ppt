#!/bin/bash
set -euo pipefail

# ============================================================================
# Evaluation Pipeline
# ============================================================================
# Evaluates trained models on:
#   - C4 validation loss
#   - BLiMP aggregate accuracy
# ============================================================================

OUTPUT_BASE=${OUTPUT_BASE:-"./output"}

echo "=========================================="
echo "Evaluation Pipeline"
echo "=========================================="

CONDITIONS=(
    "non_ppt"
    "mp_struct"
    "mp_struct_core"
    "dyck_1"
    "shuff_dyck"
    "random"
    "mp_struct_no_merge"
    "mp_struct_no_move"
    "mp_struct_no_agree"
)

for COND in "${CONDITIONS[@]}"; do
    PT_DIR="${OUTPUT_BASE}/pt/${COND}"

    if [ ! -d "${PT_DIR}" ]; then
        echo "Skipping ${COND}: directory ${PT_DIR} not found"
        continue
    fi

    echo ""
    echo "--- Evaluating: ${COND} ---"

    python eval_checkpoint.py main \
        --super_dir "${PT_DIR}" \
        --do_eval True \
        --do_blimp True \
        --compute_id False
done

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
