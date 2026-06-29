#!/bin/bash
set -euo pipefail

# ============================================================================
# Full Reproduction Pipeline
# ============================================================================
# End-to-end reproduction of Mita et al. (2026) experiments.
#
# Steps:
#   1. Generate all PPT synthetic data
#   2. Download and preprocess C4
#   3. Train all conditions (PPT -> PT)
#   4. Evaluate all checkpoints
#
# Usage:
#   bash scripts/run_all.sh
#
# Environment variables:
#   SEED         - Random seed (default: 3407). Paper uses 3 seeds.
#   C4_METHOD    - C4 download method: "hub" or "subset" (default: "hub")
#   OUTPUT_BASE  - Output directory (default: "./output")
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Full Reproduction: Mita et al. (2026)"
echo "  Language Acquisition Device in Large Language Models"
echo "============================================================"
echo ""

# Step 1: Generate PPT data
echo ">>> Step 1: Generating PPT data..."
bash "${SCRIPT_DIR}/generate_ppt_data.sh"

# Step 2: Preprocess C4
echo ""
echo ">>> Step 2: Preprocessing C4..."
bash "${SCRIPT_DIR}/preproc_c4.sh"

# Step 3: Training
echo ""
echo ">>> Step 3: Training..."
bash "${SCRIPT_DIR}/train.sh"

# Step 4: Evaluation
echo ""
echo ">>> Step 4: Evaluation..."
bash "${SCRIPT_DIR}/eval.sh"

echo ""
echo "============================================================"
echo "Reproduction complete!"
echo "============================================================"
