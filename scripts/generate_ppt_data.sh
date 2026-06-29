#!/bin/bash
set -euo pipefail

# ============================================================================
# Generate all PPT (Pre-PreTraining) data
# ============================================================================
# This script generates synthetic data for all PPT conditions described in
# Mita et al. (2026) "Language Acquisition Device in Large Language Models."
#
# PPT Conditions:
#   1. MP-STRUCT       - Minimalist Grammar derivations
#   2. MP-STRUCT CORE  - Simplified constrained structures with head diversity
#   3. 1-Dyck          - Pure context-free nesting (baseline)
#   4. kk-Shuffle Dyck - Cross-serial dependencies (k=64, from Hu et al. 2025)
#   5. Random          - i.i.d. uniform tokens (control)
#
# Ablations (MP-STRUCT variants):
#   - w/o Merge
#   - w/o Move
#   - w/o Agree
# ============================================================================

N=${N:-100000}
MAX_LENGTH=${MAX_LENGTH:-1024}

echo "=========================================="
echo "Generating PPT data (N=${N}, max_length=${MAX_LENGTH})"
echo "=========================================="

# --- 1. MP-STRUCT (Full) ---
echo "[1/5] Generating MP-STRUCT..."
python -m src.mp_struct generate \
    --file_dir ./data/mp_struct \
    --n ${N} \
    --max_length ${MAX_LENGTH}

python -m src.utils cache_data \
    --dataset_name data/mp_struct/mp_struct_ids_${N}_${MAX_LENGTH}.txt \
    --out_dir ./data/tokenized/mp_struct

# --- 2. MP-STRUCT CORE ---
echo "[2/5] Generating MP-STRUCT CORE..."
python -m src.mp_struct_core generate \
    --out_dir ./data/mp_struct_core \
    --n ${N} \
    --length ${MAX_LENGTH}

python -m src.utils cache_data \
    --dataset_name data/mp_struct_core/mp_struct_core_ids.txt \
    --out_dir ./data/tokenized/mp_struct_core

# --- 3. 1-Dyck ---
echo "[3/5] Generating 1-Dyck..."
python -m src.grammar generate_dyck \
    --file_dir ./data/dyck_1 \
    --num_symbols 1 \
    --n ${N} \
    --target_length ${MAX_LENGTH}

python -m src.utils cache_data \
    --dataset_name data/dyck_1/dyck_sequences_1_1_16.txt \
    --out_dir ./data/tokenized/dyck_1

# --- 4. kk-Shuffle Dyck (k=64) ---
echo "[4/5] Generating kk-Shuffle Dyck (k=64)..."
python -m src.grammar generate_shuff_dyck \
    --file_dir ./data/shuff_dyck \
    --num_symbols 64 \
    --n ${N} \
    --target_length ${MAX_LENGTH} \
    --p 0.51

python -m src.utils cache_data \
    --dataset_name data/shuff_dyck/dyck_sequences_cross_serial_64_0.51.txt \
    --out_dir ./data/tokenized/shuff_dyck

# --- 5. Random ---
echo "[5/5] Generating Random baseline..."
python -m src.grammar generate_random \
    --file_dir ./data/random \
    --vocab_size 128 \
    --n ${N} \
    --target_length ${MAX_LENGTH}

python -m src.utils cache_data \
    --dataset_name data/random/random_sequences_128.txt \
    --out_dir ./data/tokenized/random

# ============================================================================
# MP-STRUCT Ablations
# ============================================================================
echo "=========================================="
echo "Generating MP-STRUCT ablation data"
echo "=========================================="

# --- w/o Merge ---
echo "[Ablation 1/3] MP-STRUCT w/o Merge..."
python -m src.mp_struct generate \
    --file_dir ./data/mp_struct_no_merge \
    --n ${N} \
    --max_length ${MAX_LENGTH} \
    --enable_merge False

python -m src.utils cache_data \
    --dataset_name data/mp_struct_no_merge/mp_struct_ids_${N}_${MAX_LENGTH}.txt \
    --out_dir ./data/tokenized/mp_struct_no_merge

# --- w/o Move ---
echo "[Ablation 2/3] MP-STRUCT w/o Move..."
python -m src.mp_struct generate \
    --file_dir ./data/mp_struct_no_move \
    --n ${N} \
    --max_length ${MAX_LENGTH} \
    --enable_move False

python -m src.utils cache_data \
    --dataset_name data/mp_struct_no_move/mp_struct_ids_${N}_${MAX_LENGTH}.txt \
    --out_dir ./data/tokenized/mp_struct_no_move

# --- w/o Agree ---
echo "[Ablation 3/3] MP-STRUCT w/o Agree..."
python -m src.mp_struct generate \
    --file_dir ./data/mp_struct_no_agree \
    --n ${N} \
    --max_length ${MAX_LENGTH} \
    --enable_agree False

python -m src.utils cache_data \
    --dataset_name data/mp_struct_no_agree/mp_struct_ids_${N}_${MAX_LENGTH}.txt \
    --out_dir ./data/tokenized/mp_struct_no_agree

echo "=========================================="
echo "All PPT data generated successfully!"
echo "=========================================="
