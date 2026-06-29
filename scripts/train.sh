#!/bin/bash
set -euo pipefail

# ============================================================================
# Training Pipeline: PPT -> PT
# ============================================================================
# Reproduces the main experiments from Mita et al. (2026).
#
# Stage 1 (PPT): Pre-pretraining on synthetic data (500 steps)
# Stage 2 (PT):  Pretraining on C4 natural language (25,000 steps)
#
# Hyperparameters (Table 3 in the paper):
#   - Model:          Pythia-1B
#   - Batch size:     16 (x2 gradient accumulation = effective 32)
#   - Max seq length: 1024
#   - Learning rate:  5e-4 (cosine with min_lr=5e-5)
#   - Warmup:         1000 steps
#   - Weight decay:   0.1
#   - Precision:      bf16
# ============================================================================

MODEL=${MODEL:-"EleutherAI/pythia-1b"}
SEED=${SEED:-3407}
OUTPUT_BASE=${OUTPUT_BASE:-"./output"}
C4_DATA=${C4_DATA:-"./data/tokenized/c4"}

PPT_STEPS=500
PT_STEPS=25000
PPT_SAVE=500
PT_SAVE=2000

# Common training arguments
COMMON_ARGS="--bsz 16 --gradient_accumulation_steps 2 --lr 5e-4 --warmup_steps 1000 \
    --weight_decay 0.1 --max_grad_norm 1.0 --max_seq_length 1024 --seed ${SEED}"

echo "=========================================="
echo "Training Pipeline (seed=${SEED})"
echo "=========================================="

# ============================================================================
# PPT Conditions
# ============================================================================
PPT_CONDITIONS=(
    "mp_struct"
    "mp_struct_core"
    "dyck_1"
    "shuff_dyck"
    "random"
)

# --- Stage 1: Pre-PreTraining (PPT) ---
echo ""
echo "--- Stage 1: Pre-PreTraining (500 steps) ---"

for COND in "${PPT_CONDITIONS[@]}"; do
    echo ""
    echo "[PPT] Condition: ${COND}"
    python train.py \
        --model_name ${MODEL} \
        --data_dir ./data/tokenized/${COND} \
        --output_dir ${OUTPUT_BASE}/ppt/${COND} \
        --save_steps ${PPT_SAVE} \
        --max_steps ${PPT_STEPS} \
        --reinit True \
        ${COMMON_ARGS}
done

# --- Stage 2: PreTraining on C4 (25,000 steps) ---
echo ""
echo "--- Stage 2: PreTraining on C4 (25,000 steps) ---"

# Non-PPT baseline (random init -> C4)
echo ""
echo "[PT] Non-PPT baseline"
python train.py \
    --model_name ${MODEL} \
    --data_dir ${C4_DATA} \
    --output_dir ${OUTPUT_BASE}/pt/non_ppt \
    --save_steps ${PT_SAVE} \
    --max_steps ${PT_STEPS} \
    --reinit True \
    ${COMMON_ARGS}

# PPT -> PT (continue from PPT checkpoint)
for COND in "${PPT_CONDITIONS[@]}"; do
    echo ""
    echo "[PT] Condition: ${COND} (from PPT checkpoint)"
    python train.py \
        --model_name ${OUTPUT_BASE}/ppt/${COND}/checkpoint-${PPT_STEPS} \
        --data_dir ${C4_DATA} \
        --output_dir ${OUTPUT_BASE}/pt/${COND} \
        --save_steps ${PT_SAVE} \
        --max_steps ${PT_STEPS} \
        --reinit False \
        ${COMMON_ARGS}
done

# ============================================================================
# MP-STRUCT Ablations
# ============================================================================
echo ""
echo "--- MP-STRUCT Ablations ---"

ABLATIONS=(
    "mp_struct_no_merge"
    "mp_struct_no_move"
    "mp_struct_no_agree"
)

for ABL in "${ABLATIONS[@]}"; do
    echo ""
    echo "[PPT Ablation] ${ABL}"
    python train.py \
        --model_name ${MODEL} \
        --data_dir ./data/tokenized/${ABL} \
        --output_dir ${OUTPUT_BASE}/ppt/${ABL} \
        --save_steps ${PPT_SAVE} \
        --max_steps ${PPT_STEPS} \
        --reinit True \
        ${COMMON_ARGS}

    echo "[PT Ablation] ${ABL} -> C4"
    python train.py \
        --model_name ${OUTPUT_BASE}/ppt/${ABL}/checkpoint-${PPT_STEPS} \
        --data_dir ${C4_DATA} \
        --output_dir ${OUTPUT_BASE}/pt/${ABL} \
        --save_steps ${PT_SAVE} \
        --max_steps ${PT_STEPS} \
        --reinit False \
        ${COMMON_ARGS}
done

echo ""
echo "=========================================="
echo "Training pipeline complete!"
echo "=========================================="
