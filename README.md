<!-- # Few-Shot Personalized Emotion Recognition from Physiological Signals with Multitask Meta-Learning

> Binary arousal/valence classification from ECG, EDA, and BVP using MTL, STL, and Reptile-based MTML across three VR-based physiological datasets.

---

## Overview

This repository implements a personalized framework for dimensional emotion recognition from physiological signals. Inter-individual variability in physiological responses poses a core challenge for affect recognition: population-level models generalise poorly to individuals, while per-user models require prohibitive amounts of labelled data.

The framework addresses this through two complementary strategies:

- **Multitask Learning (MTL)**: Each user is modelled as a distinct task within a hard parameter sharing (HPS) CNN–LSTM architecture. A shared backbone learns population-level physiological representations; user-specific heads capture individual affective patterns.
- **Multitask Meta-Learning (MTML)**: A Reptile-based meta-learning extension learns an initialisation that enables rapid few-shot adaptation to previously unseen users with minimal calibration data.

Experiments are conducted on three VR-based physiological datasets — **VREED**, **DSSN-EQ**, and **DSSN-EM** — across ECG, EDA, and BVP modalities.

---

## Repository Structure

```
.
├── src/
│   ├── config.py          # All hyperparameters, dataset registry, train/test splits
│   ├── data.py            # Sliding window extraction, BalancedSampler, support/query loaders
│   ├── models.py          # SingleTaskModel, MTLModel, MTLModelUW, BaseFeatureExtractor, TaskHead
│   ├── training.py        # Training loops, evaluation, PCGrad, Reptile outer update
│   └── utils.py           # Seeds, metrics, k-fold splits, aggregation, plots
│
├── datasets/
│   └── dataset_configs/
│       ├── vreed.py       # VREED loader
│       ├── dssn_eq.py     # DSSN-EQ loader
│       ├── dssn_em.py     # DSSN-EM loader
│       └── loader.py      # Unified load_dataset() entry point
│
├── experiments/
│   ├── MTL_baselines/
│   │   ├── pstl.py        # Population Single-Task Learning
│   │   ├── stl.py         # Per-participant Single-Task Learning
│   │   ├── mtl_hps.py     # MTL Hard Parameter Sharing
│   │   ├── mtl_uw.py      # MTL + Uncertainty Weighting
│   │   └── mtl_pcgrad.py  # MTL + PCGrad gradient projection
│   │
│   ├── MTML_baselines/
│   │   ├── si.py          # Subject-Independent baseline
│   │   ├── tlft.py        # Transfer Learning + Fine-Tuning
│   │   ├── mtl_retrain.py # MTL Retrain (from scratch per test user)
│   │   ├── transfer_mtl.py# Transfer MTL (pretrain + fine-tune head)
│   │   ├── pure_meta.py   # Pure Reptile (no task heads)
│   │   ├── reptile_st.py  # Reptile Single-Task episodes
│   │   ├── reptile_mt.py  # Reptile Multi-Task episodes
│   │   └── reptile_mi.py  # Reptile MI-guided episode sampling
│   │
│   ├── collect_results.py # Aggregate all PKL results into a single CSV
│   └── run_all_experiments.py  # Sequential runner for all 13 scripts
│
├── analysis/
│   ├── class_balance.py         # Label distribution check
│   ├── mtl_roc_auc.py           # MTL baseline ROC curves
│   ├── mtl_vs_stl_gains.py      # Per-participant NTG (Negative Transfer Gap)
│   ├── mtml_roc_auc.py          # MTML all-methods ROC comparison
│   └── statistical_analysis.py  # Paired t-tests, rescue effect, variance analysis
│
├── data/                  # CSV and PKL files (not tracked in git)
└── results/               # Output directories (not tracked in git)
```

---

## Datasets

### VREED
- 26 participants (8 excluded for incomplete data from the original 34)
- ECG + EDA signals at 256 Hz, downsampled from 2000 Hz
- 12 immersive 360° VR environments spanning all four quadrants of Russell's circumplex
- Self-reported SAM ratings for valence and arousal after each VE
- 10-second windows, 50% overlap → window size 2560, stride 1280
- Train/test split: 10 videos train, 2 videos test per participant

### DSSN-EQ (Equivital sensor)
- 34 participants (5 excluded for low-quality recordings)
- ECG 1, ECG 2, GSR (EDA) at 256 Hz
- 6 space-themed immersive 180° VR videos per participant
- Window size 2560, stride 1280
- Train/test split: 5 videos train, 1 video test

### DSSN-EM (Empatica EmbracePlus)
- 28 participants
- BVP (64 Hz), EDA (4 Hz upsampled to 64 Hz), Heart Rate (derived from systolic peaks)
- Same 6-video VR protocol as DSSN-EQ
- Window size 640, stride 320
- Train/test split: 5 videos train, 1 video test

All signals are band-pass-filtered, z-score normalised per participant, and segmented into 10-second sliding windows. Binary labels: rating ≥ 5 = High (1), < 5 = Low (0).

---

## Models

### MTL Baselines (known users)

| Script | Description |
|---|---|
| `pstl.py` | Single global model trained on all participants pooled |
| `stl.py` | Independent CNN–LSTM per participant; no parameter sharing |
| `mtl_hps.py` | Shared CNN–LSTM backbone + per-user dense heads |
| `mtl_uw.py` | MTL-HPS with learned homoscedastic uncertainty weighting per task |
| `mtl_pcgrad.py` | MTL-HPS with PCGrad gradient projection for conflict resolution |

### MTML Baselines (unseen users)

| Script | Description |
|---|---|
| `si.py` | Train on all train participants, evaluate directly on test participants |
| `tlft.py` | Pretrain on train participants, fine-tune all parameters per test user |
| `mtl_retrain.py` | Retrain full MTL from scratch for each test participant |
| `transfer_mtl.py` | Pretrain MTL, add new task head, fine-tune backbone + new head |
| `pure_meta.py` | Reptile with a single monolithic model, 1 participant per step |
| `reptile_st.py` | Reptile with backbone + per-participant heads, 1 participant per episode |
| `reptile_mt.py` | Reptile with sequential multi-participant episodes |
| `reptile_mi.py` | Reptile with MI-guided episode sampling (anchor + similar + diverse) |

### Architecture

All models share the same CNN–LSTM backbone:

```
Conv1D(C→128, k=2) → BN → ReLU → MaxPool
Conv1D(128→64, k=1) → BN → ReLU → MaxPool
LSTM(64, hidden=64)
Mean pooling over time
[Task head: Linear(64→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)]
```

Valence and arousal are trained as two independent binary classification tasks.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy
```

Python 3.9+ and PyTorch 2.0+ are recommended. CUDA is supported but not required.

---

## Usage

### Running a single experiment

All scripts accept a `--dataset` argument: `vreed` (default), `dssn_eq`, or `dssn_em`.

```bash
# MTL baselines
python experiments/MTL_baselines/pstl.py --dataset vreed
python experiments/MTL_baselines/stl.py --dataset dssn_eq
python experiments/MTL_baselines/mtl_hps.py --dataset dssn_em
python experiments/MTL_baselines/mtl_uw.py --dataset vreed
python experiments/MTL_baselines/mtl_pcgrad.py --dataset vreed

# MTML baselines
python experiments/MTML_baselines/si.py --dataset vreed
python experiments/MTML_baselines/tlft.py --dataset dssn_eq
python experiments/MTML_baselines/reptile_mt.py --dataset vreed
```

### Running all experiments sequentially

```bash
python experiments/run_all_experiments.py --dataset vreed
```

This runs all 13 scripts in order and prints a pass/fail summary with elapsed times.

### Collecting results

After experiments complete, aggregate all PKL files into a single CSV:

```bash
python experiments/collect_results.py
# Output: results/results_summary.csv
```

### Analysis scripts

```bash
# Check class balance
python analysis/class_balance.py --dataset vreed

# MTL ROC curves
python analysis/mtl_roc_auc.py --dataset vreed

# Per-participant NTG (MTL vs STL accuracy gain)
python analysis/mtl_vs_stl_gains.py --dataset vreed

# MTML all-methods ROC comparison
python analysis/mtml_roc_auc.py --dataset vreed

# Comprehensive statistical analysis (rescue effect, paired t-tests, etc.)
python analysis/statistical_analysis.py --dataset vreed
```

---

## Configuration

All hyperparameters, dataset paths, and train/test splits are centralised in `src/config.py`. The dataset registry (`_DATASET_REGISTRY`) holds all dataset-specific settings; experiment scripts call `get_dataset_config(name)` rather than hard-wiring values.

Key hyperparameters:

| Parameter | Value | Description |
|---|---|---|
| `SEED` | 42 | Global random seed |
| `EPOCHS` | 30 | Training epochs |
| `MTL_SHARED_LR` | 3e-4 | Shared backbone learning rate |
| `MTL_TASK_LR` | 1e-4 | Task-head learning rate |
| `L2_TASK` | 1e-5 | L2 regularisation on task heads |
| `META_STEPS` | 50 | Reptile meta-training steps |
| `META_LR` | 0.01 | Reptile outer-loop learning rate |
| `INNER_STEPS` | 10 | Inner-loop adaptation steps |
| `INNER_LR` | 1e-3 | Inner-loop learning rate |
| `EPISODE_SIZE` | 5 | Participants per multi-task episode |
| `K_PER_CLASS` | None | Balanced k-shot subsampling (None = all windows) |
| `FT_EPOCHS` | 10 | Fine-tuning epochs (TL-FT, Transfer-MTL) |

---

## Reproducibility

Determinism is enforced globally via `set_all_seeds()` in `src/utils.py`:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
```

All scripts call `set_all_seeds(SEED)` before each AR and VA training pass independently. Results are saved as PKL files with mean ± std determinism verification blocks printed at the end of each run.

---

## Output Structure

Each experiment writes results to `results/{PREFIX}_{MTL|MTML}/{PREFIX}_{method}/`:

```
results/
├── VREED_MTL/
│   ├── VREED_pstl_results/
│   │   ├── pstl_results.pkl
│   │   ├── per_participant_results.csv
│   │   ├── VREED_pstl_misclassification_rates.csv
│   │   ├── ar_cm.png  /  va_cm.png
│   │   └── ar_roc.png /  va_roc.png
│   ├── VREED_stl_results/
│   ├── VREED_hps_results/
│   ├── VREED_hps_uw_results/
│   └── VREED_hps_pcgrad_results/
│
├── VREED_MTML/
│   ├── VREED_SI/
│   ├── VREED_TF/
│   ├── VREED_mtl_retrain/
│   ├── VREED_transfer_mtl/
│   ├── VREED_pure_meta/
│   ├── VREED_reptile_st/
│   ├── VREED_reptile_mt/
│   ├── VREED_reptile_mi/
│   └── ROC_Comparisons_All/
│
├── VREED_MTL_vs_STL_Gains.csv
├── VREED_comprehensive_statistical_summary.csv
└── results_summary.csv
```

Each PKL file stores aggregate metrics, per-participant results, confusion matrices, and raw prediction arrays for downstream analysis.

---

## Key Findings

**MTL for known users:** Multitask learning provides the most consistent improvements for valence classification. Task-balancing strategies (MTL-UW, MTL-PCGrad) improve robustness in highly imbalanced settings. The Negative Transfer Gap (NTG) analysis reveals substantial inter-individual heterogeneity: parameter sharing benefits some participants while degrading others.

**Negative transfer:** TAG (Task Affinity Grouping) analysis shows that negative transfer arises from localised gradient conflict between specific incompatible participant pairs rather than uniform degradation. Counterintuitively, moderate gradient conflict can act as an implicit regulariser, producing beneficial transfer for some participants.

**MTML for unseen users:** Multi-task episodic training (Reptile-MT, Reptile-MI) yields stronger meta-initialisations than single-task episodes on datasets with moderate class balance. Valence benefits more from multi-user meta-training than arousal. DSSN-EQ's severe class imbalance makes macro-averaged metrics particularly sensitive to episodic design choices.

**Arousal vs valence:** Arousal is consistently easier to classify due to stronger autonomic correlates (heart rate, electrodermal activity). Valence shows higher sensitivity to personalization and cross-user episodic diversity.

---

## Citation

If you use this code or the DSSN dataset in your research, please cite:

## License

This project is released for academic research purposes. Please contact the authors before using the DSSN dataset or the code in commercial applications. -->

# Few-Shot Personalized Emotion Recognition from Physiological Signals with Multitask Meta-Learning

## Overview

This repository implements a personalized framework for dimensional emotion recognition from physiological signals. Inter-individual variability in physiological responses poses a core challenge for affect recognition: population-level models generalise poorly to individuals, while per-user models require prohibitive amounts of labelled data.

The framework addresses this through two complementary strategies:

- **Multitask Learning (MTL)**: Each user is modelled as a distinct task within a hard parameter sharing (HPS) CNN–LSTM architecture. A shared backbone learns population-level physiological representations; user-specific heads capture individual affective patterns.
- **Multitask Meta-Learning (MTML)**: A Reptile-based meta-learning extension learns an initialisation that enables rapid few-shot adaptation to previously unseen users with minimal calibration data.

Experiments are conducted on three VR-based physiological datasets — **VREED**, **DSSN-EQ**, and **DSSN-EM** — across ECG, EDA, and BVP modalities.

---

## Repository Structure

```
.
├── src/                        # config, data, models, training, utils
├── datasets/dataset_configs/   # dataset-specific loading and preprocessing
├── datasets/preprocessing/     # signal filtering, normalisation, windowing
├── experiments/
│   ├── MTL_baselines/          # pstl, stl, mtl_hps, mtl_uw, mtl_pcgrad
│   ├── MTML_baselines/         # si, tlft, mtl_retrain, transfer_mtl,
│   │                           # pure_meta, reptile_st, reptile_mt, reptile_mi
│   ├── collect_results.py      # aggregate all PKL results → CSV
│   └── run_all_experiments.py  # sequential runner for all 13 scripts
└── analysis/                   # ROC curves, NTG, statistical tests

```

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

All scripts accept `--dataset vreed` (default), `dssn_eq`, or `dssn_em`.

```bash
# Run a single experiment
python experiments/MTL_baselines/mtl_hps.py --dataset vreed
python experiments/MTML_baselines/reptile_mt.py --dataset dssn_eq

# Run all 13 experiments sequentially
python experiments/run_all_experiments.py --dataset vreed

```

All hyperparameters and train/test splits live in `src/config.py`.

---

