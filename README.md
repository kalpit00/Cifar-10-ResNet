# CIFAR-10 ResNet under 5M parameters

NYU Tandon Deep Learning, Fall 2026, Project 1 (Kaggle competition "NYU Tandon Deep Learning F26").
Author: Kalpit Patel (kkp10045).

The task is to train a modified ResNet from scratch on CIFAR-10 with at most 5,000,000 trainable parameters, without pretrained weights or external data. This repository holds every notebook from the project, in the order the techniques were added, with the executed copies, training curves and Kaggle submission files for each run.

## Final model

| | |
|---|---|
| Architecture | Post-activation ResNet: 3×3 stem, 3 residual layers with 64 / 128 / 256 channels and 3 blocks each, 1×1 projection shortcuts when the shape changes, global average pooling, linear classifier |
| Trainable parameters | **4,327,754** |
| Training | SGD (momentum 0.9, weight decay 5e-4), lr 0.1 with 5-epoch warmup and cosine decay to 1e-4, batch size 128, 200 epochs, mixed precision |
| Augmentation | Random crop (padding 4), horizontal flip, TrivialAugment, Cutout 16 |
| Distillation | Learns from the averaged predictions of two models with the same architecture, trained from scratch here (`long_run` and `trivial_augment`), at temperature 4 with alpha 0.5 |
| Inference | Flip test-time augmentation (average of the image and its mirror) |
| Local test accuracy | **97.06%** (40k-image training run, 5,000 held-out labeled images, with flip TTA) |
| Kaggle submission | `submissions/09_distillation/submission_distillation_T4_a0.5_c64-128-256_b3-3-3_cut16_mix0.0_ep200_bs128_all50k_tta.csv` (same recipe trained on all 50,000 images) |
| Kaggle accuracy | **85.9% private**, 88.9% public |

## Repository layout

```
local-notebooks/     Clean notebooks, one per technique, with no outputs. Start here to rerun anything.
colab-notebooks/     The same notebooks after running on Colab, with all outputs (training logs, plots, accuracy, parameter count).
reports/             Training curves (loss and accuracy per epoch) saved by each run.
submissions/         Kaggle submission files written by each run.
experiments.csv      One row per run: parameters, epochs, validation and local test accuracy, minutes, and the full config.
```

`colab-notebooks/`, `reports/` and `submissions/` are split into one folder per notebook, numbered in the order the techniques were added:

| Folder | Notebook in `local-notebooks/` | What it adds |
|---|---|---|
| `01_baseline` | `baseline_colab.ipynb` | Plain ResNet, SGD with constant LR, no augmentation |
| `02_crop_flip` | `crop_flip_colab.ipynb` | Random crop and horizontal flip; 30 epochs |
| `03_optimizer_scheduler` | `optimizer_scheduler_colab.ipynb` | SGD vs Adam vs RMSProp, cosine schedule, weight decay |
| `04_architecture` | `architecture_colab.ipynb` | Pre-activation WRN-22-4 vs post-activation blocks vs 4 narrower layers; warmup, mixed precision |
| `05_strong_augmentation` | `strong_augmentation_colab.ipynb` | Cutout, Mixup, brightness/contrast jitter, flip TTA |
| `06_dropout` | `dropout_colab.ipynb` | Dropout inside residual blocks |
| `07_long_run` | `long_run_colab.ipynb` | 160 epochs, EMA, BatchNorm recalibration, checkpoint/resume, 40k vs all 50k images |
| `08_trivial_augment` | `trivial_augment_colab.ipynb` | TrivialAugment, 200 epochs |
| `09_distillation` | `distillation_colab.ipynb` | Teacher-student distillation from the `long_run` and `trivial_augment` models |

Each notebook is a copy of the one before it with one technique added, so reading them in order shows the progression. Settings that were tried within one notebook, like optimizers or architectures, are switched in its `CONFIG` cell, and several executed copies are numbered `_1`, `_2`, `_3` in `colab-notebooks/`. File names include the settings of each run, for example `cut16_mix0.2_ep160`, and runs on all 50,000 images end in `_all50k`.

## Results

Validation and local test use a fixed stratified split of the 50,000 labeled images: 40,000 train, 5,000 validation, 5,000 local test (seed 0). Runs marked "50k" train on all labeled images, so they have no local scores. The Kaggle test set is split into public and private halves of about 1,000 images each, so differences under about 1 point are noise. Final rankings use the private score. Accuracies from run 9 onward use flip TTA.

| # | Run | Params | Val | Local test | Kaggle public | Kaggle private |
|---|---|---|---|---|---|---|
| 1 | Baseline, 15 epochs | 2,777,674 | 84.50% | 85.80% | 71.5% | 68.9% |
| 2 | + crop and flip, 30 epochs | 2,777,674 | 87.06% | 87.46% | 75.8% | 72.4% |
| 3 | + SGD lr 0.1, cosine, weight decay 5e-4 | 2,777,674 | 92.32% | 92.94% | 81.1% | 78.9% |
| 4 | Adam instead (lr 1e-3) | 2,777,674 | 91.36% | 91.56% | | |
| 5 | RMSProp instead (lr 1e-3) | 2,777,674 | 90.94% | 91.34% | | |
| 6 | WRN-22-4, pre-activation | 4,298,970 | 92.66% | 93.46% | | |
| 7 | Post-activation, 64/128/256, 3 blocks each | 4,327,754 | 93.30% | 93.76% | 84.0% | 80.5% |
| 8 | Pre-activation, 4 narrower layers | 4,804,854 | 92.76% | 92.78% | | |
| 9 | Run 7 + Cutout 16 | 4,327,754 | 94.00% | 94.10% | | |
| 10 | + Mixup 0.2 | 4,327,754 | 94.24% | 94.10% | 83.5% | 80.0% |
| 11 | Run 7 + brightness/contrast jitter | 4,327,754 | 93.56% | 93.60% | | |
| 12 | Run 10 + dropout 0.1 | 4,327,754 | 93.76% | 93.50% | | |
| 13 | Run 10, 160 epochs | 4,327,754 | 96.34% | 96.42% | 86.6% | 83.6% |
| 14 | Run 13 on 50k | 4,327,754 | | | 87.3% | 84.6% |
| 15 | TrivialAugment + Cutout, no Mixup, 200 epochs | 4,327,754 | 96.62% | 96.86% | | |
| 16 | Run 15 on 50k | 4,327,754 | | | 87.6% | 85.0% |
| 17 | Run 15 + distillation | 4,327,754 | 97.02% | 97.06% | | |
| **18** | **Run 17 on 50k (final)** | **4,327,754** | | | **88.9%** | **85.9%** |

Two things were tried and dropped in `long_run`: an exponential moving average of the weights gave no gain, and recalibrating BatchNorm on clean training images lowered accuracy by about 2 points locally and by about 3–4 points on Kaggle (run 14 scored 83.4% public and 81.4% private with it).

The Kaggle scores sit about 9–10 points (public) and 11–12 points (private) below the local test set across runs, but they rise and fall with it, and the two halves rank the submitted runs almost the same way. The Kaggle images measure slightly brighter and higher in contrast than the training images, so they appear to come from a shifted version of CIFAR-10.

## Reproducing the results

Everything runs on Google Colab with a GPU. The notebooks read data from and write all outputs to `MyDrive/dl_proj1/` on Google Drive.

1. **Get the data.** Download the competition data from the Kaggle competition's Data tab. The zip must contain `train_images/`, `test_images/`, `train_labels.csv`, `test.csv` and `sample_submission.csv` at its top level. Name it `data.zip` and put it in `MyDrive/dl_proj1/`.
2. **Open a notebook** from `local-notebooks/` in Colab and pick a GPU runtime (Runtime → Change runtime type). The A100 is the fastest; a T4 also works.
3. **Run all.** The notebook mounts Drive, unzips the data, and caches the decoded images in `cache.npz` on the first run. It prints the parameter count and the final validation and local test accuracy, and saves to `MyDrive/dl_proj1/`:
   - `curves_<run_name>.png`
   - `model_<run_name>.pt`
   - `submission_<run_name>_tta.csv`, or one file per model version in `long_run`
   - one row appended to `experiments.csv`
4. **Change settings** in the `CONFIG` cell, for example `use_all_data = True` to train on all 50,000 images. Then use Run all again. A run's name, and so its output files, include its settings.

### Final submission

Distillation needs the teacher models, so run these notebooks in order. Each run uses Run all; only `use_all_data` changes between the two runs of each notebook.

| Step | Notebook | `use_all_data` | Saves | Time on A100 |
|---|---|---|---|---|
| 1 | `long_run_colab.ipynb` | `False` | 40k teacher 1 and local scores | about 25 min |
| 2 | `long_run_colab.ipynb` | `True` | 50k teacher 1 | about 29 min |
| 3 | `trivial_augment_colab.ipynb` | `False` | 40k teacher 2 and local scores | about 26 min |
| 4 | `trivial_augment_colab.ipynb` | `True` | 50k teacher 2 | about 32 min |
| 5 | `distillation_colab.ipynb` | `False` | 40k student: the reported local test accuracy | about 27 min |
| 6 | `distillation_colab.ipynb` | `True` | 50k student: the Kaggle submission file | about 33 min |

To get the submission file only, steps 2, 4 and 6 are enough, since the 50k student uses the 50k teachers.

The long runs save a checkpoint to Drive every 5 epochs. If Colab disconnects, use Run all with the same `CONFIG` and training resumes from the last checkpoint.

### Trained weights

The trained weights are too large for GitHub (about 540 MB), so they're in a public Google Drive folder: [models](https://drive.google.com/drive/folders/1V6-z-gJElQ7U4cHG94eOEsGwEV6OOlHY?usp=sharing).

- `model_<run_name>.pt`: final weights (a PyTorch `state_dict`) for every run in the results table. `long_run` also has `_ema` versions.
- `checkpoint_<run_name>.pt`: the finished training checkpoints for `long_run`, `trivial_augment` and `distillation`.

**To regenerate a submission without retraining:** copy that run's `checkpoint_<run_name>.pt` into `MyDrive/dl_proj1/` next to `data.zip`, open its notebook with the same `CONFIG`, and use Run all. The notebook finds the finished checkpoint, skips training, and goes straight to scoring and writing the submission file. It also appends another row to `experiments.csv`. For the final submission, use `checkpoint_distillation_T4_a0.5_c64-128-256_b3-3-3_cut16_mix0.0_ep200_bs128_all50k.pt` with `distillation_colab.ipynb` and `use_all_data = True`. Distillation also loads its two teachers, so copy `model_long_run_c64-128-256_b3-3-3_cut16_mix0.2_ep160_bs128_all50k.pt` and `model_trivial_augment_c64-128-256_b3-3-3_cut16_mix0.0_ep200_bs128_all50k.pt` as well.

**To train a single step on its own**, such as only the distillation student, put the teacher `model_*.pt` files in `MyDrive/dl_proj1/` instead of retraining them.

**Expect small differences.** Seeds are fixed (`seed = 0`), but GPU kernels, mixed precision and data-loader workers aren't fully deterministic, so a rerun can differ by a few tenths of a point. Run 13 was resumed after Colab disconnects, and the random state isn't restored on resume, so it can't be repeated exactly.

## References

- K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
- S. Zagoruyko, N. Komodakis. Wide Residual Networks. BMVC 2016.
- I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.
- T. DeVries, G. W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. arXiv:1708.04552, 2017.
- H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz. mixup: Beyond Empirical Risk Minimization. ICLR 2018.
- S. G. Müller, F. Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV 2021.
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.
- Crop and flip augmentation follow [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
