# Few-Shot Personalized Emotion Recognition from Physiological Signals with Multitask Meta-Learning

Personalized binary valence/arousal recognition from physiological signals (ECG, EDA, BVP), treating each user as a task. Code for the paper of the same title.

## Overview

Inter-individual variability limits subject-independent affect models, while per-user models need too much labeled data. This repo tackles both with:

- **Multitask learning (MTL)** for *known* users — a shared CNN–LSTM backbone with per-user heads (hard parameter sharing), plus uncertainty-weighting (UW) and PCGrad gradient-surgery variants. MTL is compared against population single-task (P-STL, one global model) and per-user single-task (STL) baselines.
- **Negative-transfer diagnostics** — the per-user Negative Transfer Gap (NTG, MTL vs. STL accuracy) and inter-task affinity scores (TAG, Fifty et al. 2021) to show sharing helps some users and harms others.
- **Multitask meta-learning (MTML)** for *unseen* users — Reptile learns a backbone initialization that adapts from a 10-shot support set. Episodes are sampled single-task (ST), multi-task (MT), or guided by mutual information (MI) between users.

Evaluated on three VR datasets — **VREED**, **SpaceVR-EQ**, **SpaceVR-EM** (the SpaceVR sets are referred to as `dssn_eq` / `dssn_em` in code).

## Repository structure

```
src/                  config, data loaders, models, training loops, TAG probe, utils
datasets/
  dataset_configs/    per-dataset loaders + unified loader.py
  preprocessing/      raw-signal preprocessing notebooks (VREED, DSSN_EQ, DSSN_EM)
experiments/
  MTL_baselines/      pstl, stl, mtl_hps, mtl_uw, mtl_pcgrad
  TAG_analysis/       tag_train.py  (MTL-HPS + per-batch inter-task affinity)
  MTML_baselines/     reptile_st, reptile_mt, reptile_mi  (+ MMD/dynamic ablations)
  collect_results.py  aggregate PKL results -> CSV
  run_all_experiments.py
analysis/             class_balance, mtl_roc_auc, mtl_vs_stl_gains (NTG),
                      significance_tests (Wilcoxon/McNemar), tag_analysis
results/              outputs (not tracked)   z_archive/  retired scripts
```

## Datasets

| Dataset | Sensor | Modalities | Participants | Window / stride |
|---|---|---|---|---|
| VREED | Biopac MP150 (ECG100C + EDA100C) | ECG, EDA (2000 Hz → 256 Hz) | 26 (8 of 34 excluded) | 2560 / 1280 |
| SpaceVR-EQ | Equivital | ECG, EDA (16 → 256 Hz) | 34 (5 excluded) | 2560 / 1280 |
| SpaceVR-EM | Empatica EmbracePlus | BVP (64 Hz), EDA (4 → 64 Hz), HR | 28 | 640 / 320 |

All signals are band-pass filtered, z-score normalised per participant, and segmented into 10-second sliding windows with 50% overlap. SAM ratings are binarised: ≥ 5 → High (1), < 5 → Low (0). Valence and arousal are trained as two independent binary tasks. VREED is public; SpaceVR is available from the authors on request. Preprocessing lives in `datasets/preprocessing/`.

## Models

The shared backbone is two 1D conv layers (128, 64 filters; BN + ReLU + max-pool) → LSTM(64) → mean-pool. Each user head is Linear(128) → Linear(64) → Linear(1).

| Script | Method |
|---|---|
| `MTL_baselines/pstl.py` | P-STL — one global model on pooled data |
| `MTL_baselines/stl.py` | STL — independent model per user |
| `MTL_baselines/mtl_hps.py` | MTL-HPS — shared backbone + per-user heads |
| `MTL_baselines/mtl_uw.py` | MTL + homoscedastic uncertainty weighting |
| `MTL_baselines/mtl_pcgrad.py` | MTL + PCGrad gradient projection |
| `TAG_analysis/tag_train.py` | MTL-HPS while probing per-batch inter-task affinity |
| `MTML_baselines/reptile_st.py` | MTML-ST — 1 user per episode |
| `MTML_baselines/reptile_mt.py` | MTML-MT — K users adapted sequentially per episode |
| `MTML_baselines/reptile_mi.py` | MTML-MI — anchor + high-/low-MI users (static MI matrix) |

`reptile_mi_dynamic.py` (MI recomputed from learned representations) and `reptile_mmd_signal/window.py` (MMD-based similarity) are additional ablation samplers; `mmd_utils.py` supports them.

## Installation

```bash
pip install -r requirements.txt   # Python 3.10+, PyTorch 2.0+ (CUDA optional)
```

## Usage

All experiment scripts take `--dataset {vreed, dssn_eq, dssn_em}` (default `vreed`).

```bash
python experiments/MTL_baselines/mtl_hps.py --dataset vreed
python experiments/TAG_analysis/tag_train.py --dataset vreed
python experiments/MTML_baselines/reptile_mi.py --dataset dssn_em
python experiments/run_all_experiments.py --dataset vreed   # full suite

python experiments/collect_results.py        # -> results/results_summary.csv
python analysis/mtl_vs_stl_gains.py --dataset vreed   # per-user NTG
python analysis/significance_tests.py                 # MTL-HPS vs STL
python analysis/tag_analysis.py --dataset vreed       # affinity vs gains
```

Hyperparameters, dataset registry, and participant-level train/test splits are centralised in `src/config.py`. Meta-learning uses a balanced support set of 10 samples per class (`K_PER_CLASS = 10`). Determinism is enforced via `set_all_seeds()` (`SEED = 42`).

## Citation

If you use this code or the SpaceVR dataset, please cite the paper. The SpaceVR dataset is available from the authors on request.
```
https://github.com/roysup/Personalised-Emotion-Recognition-Multitask-Meta
```
