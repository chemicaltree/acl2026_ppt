# LAD-PPT: Language Acquisition Device in Large Language Models

Reproduction code for our ACL 2026 paper:

**"Language Acquisition Device in Large Language Models"**
Masato Mita, Taiga Someya, Ryo Yoshida, Yohei Oseki

[[Paper]](https://aclanthology.org/2026.acl-long.895/)
[[arxiv]](https://arxiv.org/abs/2605.16758)

This codebase builds on the experimental framework from [Hu et al. (2025)](https://github.com/michahu/pre-pretraining).

## Overview

We propose MP-STRUCT and MP-STRUCT CORE, linguistically-structured synthetic data for pre-pretraining (PPT) that encodes hierarchical composition, feature-based dependencies, and long-distance displacement based on the Minimalist Program (MP). Models pre-pretrained on these sequences for just 500 steps achieve up to 31% token efficiency gains when subsequently trained on natural language.

## Setup

```bash
git clone https://github.com/osekilab/LAD-PPT.git
cd LAD-PPT
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.12
- CUDA-compatible GPU (tested on NVIDIA RTX 6000 Ada, 48GB)
- `flash-attn` (requires CUDA toolkit)

**C4 dataset access:** The C4 dataset is gated on HuggingFace. You need to:
1. Run `huggingface-cli login`
2. Accept the terms at https://huggingface.co/datasets/allenai/c4

## Quick Start (Full Reproduction)

```bash
bash scripts/run_all.sh
```

This runs the entire pipeline: data generation, C4 preprocessing, training, and evaluation. See below for step-by-step instructions.

## Step-by-Step Reproduction

### Step 1: Generate PPT Data

```bash
bash scripts/generate_ppt_data.sh
```

This generates synthetic sequences for all five PPT conditions and three ablation variants:

| Condition | Description |
|---|---|
| MP-STRUCT | Minimalist Grammar derivations (Merge + Agree + Move) |
| MP-STRUCT CORE | Simplified constrained structures with head diversity |
| 1-Dyck | Pure context-free nesting baseline |
| kk-Shuffle Dyck | Cross-serial dependencies (k=64, from Hu et al. 2025) |
| Random | i.i.d. uniform token control |
| MP-STRUCT w/o Merge | Ablation: no hierarchical brackets |
| MP-STRUCT w/o Move | Ablation: no displacement/traces |
| MP-STRUCT w/o Agree | Ablation: no feature agreement |

To generate individual conditions manually:
```bash
# MP-STRUCT (defaults match the paper: strip_lexical=True, p_c_wh=0.2, p_wh_goal=0.2)
python -m src.mp_struct generate --file_dir ./data/mp_struct

# MP-STRUCT CORE (defaults match the paper: use_head_diversity=True)
python -m src.mp_struct_core generate --out_dir ./data/mp_struct_core

# kk-Shuffle Dyck
python -m src.grammar generate_shuff_dyck --file_dir ./data/shuff_dyck
```

### Step 2: Download and Preprocess C4

```bash
# Method 1: Via HuggingFace datasets API (recommended)
bash scripts/preproc_c4.sh

# Method 2: Download Arrow file subset
C4_METHOD=subset bash scripts/preproc_c4.sh
```

### Step 3: Training (PPT -> PT)

```bash
bash scripts/train.sh
```

The training pipeline has two stages:
- **PPT (Stage 1):** 500 steps on synthetic data with random weight initialization
- **PT (Stage 2):** 25,000 steps on C4, continuing from the PPT checkpoint

To run with a specific seed:
```bash
SEED=42 bash scripts/train.sh
```

The paper reports results averaged over 3 random seeds.

### Step 4: Evaluation

```bash
bash scripts/eval.sh
```

Evaluates all trained checkpoints on:
- **C4 validation loss** (cross-entropy)
- **BLiMP** (aggregate linguistic acceptability accuracy)

## Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Model | EleutherAI/pythia-1b |
| Batch size | 16 |
| Gradient accumulation | 2 (effective batch size: 32) |
| Max sequence length | 1024 |
| Learning rate | 5e-4 |
| Min learning rate | 5e-5 |
| LR schedule | Cosine with warmup |
| Warmup steps | 1000 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW (beta1=0.9, beta2=0.999, eps=1e-6) |
| Precision | bf16 |
| PPT steps | 500 |
| PT steps | 25,000 |

## Repository Structure

```
LAD-PPT/
├── train.py                      # Training script (SFTTrainer-based)
├── eval_checkpoint.py            # Evaluation (C4 loss, BLiMP)
├── src/
│   ├── mp_struct.py              # MP-STRUCT data generation
│   ├── mp_struct_core.py         # MP-STRUCT CORE data generation
│   ├── grammar.py                # Formal language generation (Dyck, Shuffle-Dyck, Random)
│   ├── utils.py                  # Data tokenization and caching utilities
│   └── download_c4.py            # C4 dataset download utilities
├── scripts/
│   ├── run_all.sh                # End-to-end reproduction
│   ├── generate_ppt_data.sh      # Generate all PPT synthetic data
│   ├── preproc_c4.sh             # C4 download and preprocessing
│   ├── train.sh                  # Full training pipeline
│   └── eval.sh                   # Evaluation pipeline
├── requirements.txt
└── pyproject.toml
```

## Key Metrics

- **MRS (Marginal Rate of Substitution):** How many PT steps each PPT step saves
- **Efficiency Gain:** Fraction of PT steps saved by PPT

## Acknowledgements

This work builds on the pre-pretraining framework by [Hu et al. (2025)](https://aclanthology.org/2025.acl-long.478/):

> Michael Y. Hu, Jackson Petty, Chuan Shi, William Merrill, Tal Linzen. "Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases." ACL 2025.

Original code: https://github.com/michahu/pre-pretraining

## Citation

```bibtex
@inproceedings{mita-etal-2026-language,
    title = "Language Acquisition Device in Large Language Models",
    author = "Mita, Masato  and
      Someya, Taiga  and
      Yoshida, Ryo  and
      Oseki, Yohei",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.895/",
    pages = "19564--19577",
    ISBN = "979-8-89176-390-6",
    abstract = "Large Language Models (LLMs) remain substantially less data-efficient than humans. Pre-pretraining (PPT) on synthetic languages has been proposed to close this gap, with prior work emphasizing highly expressive formal languages such as $k$-Shuffle Dyck. Inspired by the Language Acquisition Device (LAD) hypothesis, which posits that innate constraints preemptively restrict the learner{'}s hypothesis space to natural-language-like structure, we propose LAD-inspired PPT: pre-pretraining on MP-STRUCT, a formal language whose strings encode hierarchical composition, feature-based dependencies, and long-distance displacement via MERGE, AGREE, and MOVE. A brief 500-step PPT with MP-STRUCT matches strong formal-language baselines in token efficiency while additionally imparting a human-like resistance to structurally implausible languages. Analyzing simplified variants, we find that MP-STRUCT CORE outperforms $k$-Shuffle Dyck despite not being definable in C-RASP (a formal bound on transformer expressivity), challenging the prior hypothesis that effective PPT languages must be both hierarchically expressive and circuit-theoretically learnable. We show that functional landmarks, which reduce dependency resolution ambiguity, are a key driver, suggesting that effective PPT design depends not only on expressivity but also on the accessibility of dependency resolution."
}
```
