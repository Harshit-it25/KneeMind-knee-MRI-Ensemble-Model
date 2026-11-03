# KneeMind-knee-MRI-Ensemble-Model

Multi-view knee MRI model for the RSNA knee abnormality competition on Kaggle. It reads up to 5 MRI series per study with a ConvNeXt-nano slice encoder and a small transformer, and predicts 12 findings (ACL, MCL, menisci, osteoarthritis, effusion and more). The final submission averages 4 independently seeded members.

**Hidden-set score 0.9116** (v11: 0.908, v7: 0.897) | **Rank 311 of about 6,500 teams (top 5%) on the efficiency track**

## Contents
1. [Team Rocket](#team-rocket)
2. [Results](#results)
3. [Repository files](#repository-files)
4. [Approach in brief](#approach-in-brief)
5. [Model and training settings](#model-and-training-settings)
6. [How to run (Kaggle)](#how-to-run-kaggle)
7. [Limitations](#limitations)
8. [Acknowledgements](#acknowledgements)

## Team Rocket
| Member | Role |
|---|---|
| Harshit Ranbhare | Built v7 to v12. Designed and trained the ML model using ConvNeXt |
| Krishnendu Jana | Team Leader. Built v8 and v9 |
| Ajinkya Supate | Got results using a ResNet model |
| Harshal Upadhye | Built v1 and v2 |
| Amol Subhash Lange | Built v5 and v6 |

## Results
| Run | Measure | Score |
|---|---|---|
| v7 | hidden set | 0.897 |
| v11 | hidden set | 0.908 |
| v12 member s0 | gold-58 AUC, final epoch | 0.881 |
| v12 member s1 | gold-58 AUC, final epoch | 0.899 |
| **v12 ensemble of 4** | **hidden set** | **0.9116** |

- **Efficiency track:** Team Rocket placed 311 of about 6,500 teams (top 5%).
- **Gold-58 AUC** is the macro AUC over the 12 findings on the 58 expert-labelled studies. These studies have a standard error of about 0.02, so they can catch large failures but cannot rank close recipes. The hidden-set score is the real test.
- The ensemble size can be changed to trade AUC against GPU inference time (see [How to run](#how-to-run-kaggle)).

## Repository files
| File | Content |
|---|---|
| `kneemind-v12.ipynb` | Full pipeline: caching, training, inference and submission. Written for Kaggle. |
| `KneeMind_v12_approach.pdf` | 8-page explanation of the approach, with figures and tables. Read this for the details behind this summary. |

Trained weights and the competition data are not part of this repository.

## Approach in brief

```
DICOM series -> canonical volume -> 2.5D encoder -> study transformer -> attention pool -> 12 probabilities
 (5 views)      (336 px, 12 sl.)    (ConvNeXt-nano)    (2 layers, 384-d)   (1 query/finding)   (mean of 4 models)
```

**Data.** 4,407 training studies: 4,349 with labels extracted from radiology reports (weak labels) and 58 with expert labels (gold). The gold studies are never trained on; they are used only for validation and to weigh label sources.

**Preprocessing.** One function builds the input for both the training cache and the hidden-set run, so the two cannot drift apart. Each study gets up to 5 series (sagittal, coronal and axial fluid-sensitive, plus sagittal and coronal non-fluid). Every series is sorted along its true slice normal, resampled to 0.457 mm pixels, centre-cropped to 336 x 336 px (153.6 mm field of view), windowed by percentiles, and reduced to 12 slices over the central 15-85 % of the stack. Series that cannot be decoded are masked instead of being fed as black frames.

**Labels.** Report text is turned into soft labels by combining a multilingual regex labeler with four LLM-extracted public label tables. Each source is weighted per finding by its agreement with the gold studies. A leak guard detects tables that copy the gold labels and stops them from taking over the weights.

**Model.** A shared ConvNeXt-nano encodes every sampled slice as a 2.5D input (previous, current and next cached slice as 3 channels). Tokens get view, slice-position and side-tag information. A 2-layer transformer mixes the tokens of all views, and one learned attention query per finding pools the evidence into a probability. About 19 million parameters per member.

**Training.** Each member trains on all 4,349 weak studies for 24 epochs with AdamW, an EMA of the weights, and a binary cross-entropy loss with positive weights. The last-epoch EMA weights are kept; no epoch is picked on the 58 gold studies. Two members train side by side on two T4 GPUs.

**Inference.** The 2 new v12 members and the 2 v11 members are averaged, together with 2 slice-set passes (test-time augmentation). A study that fails to decode gets 0.5 for every finding, a partial submission is written every 100 studies, and the final file is validated against the sample submission.

## Model and training settings
| Setting | Value |
|---|---|
| Backbone | `convnext_nano.in12k_ft_in1k` (timm) |
| Input | 5 views, 336 x 336 px, 0.457 mm, 12 cached slices per view |
| Windows per study | 20 random (train), 30 per pass (test) |
| Token dimension | 384, 2 transformer layers, 6 heads |
| Epochs | 24 (v11 used 18) |
| Batch | 4 studies x 4 accumulation steps = 16 |
| Learning rate | 1.5e-4 (encoder), 5e-4 (head), weight decay 0.05 |
| EMA decay | 0.997 |
| Loss | BCE, positive weight clip((1-p)/p, 1, 5) |
| Augmentation | affine and intensity on the GPU, no flips |
| Seeds | 142 and 143 (v11 used 42 and 43) |
| Training time | 239 and 263 minutes for the two members |

## How to run (Kaggle)
The notebook expects the Kaggle layout (`/kaggle/input`, `/kaggle/working`).

1. **Train:** `MODE="full"`, GPU T4 x2, Internet ON. Attach the competition data, the two label datasets, the v11 cache output (same cache tag, so no re-caching) and the v11 pack. Check that there are no lines starting with `!!`, that the epochs equal the planned 24, and that the line `gold predictions of:` lists both v11 and v12 files.
2. **Submit:** `MODE="infer"`, GPU, Internet OFF. Attach the competition data, the v12 output and the v11 pack. The log must say `ensemble of 4`.

| Environment variable | Default | Effect |
|---|---|---|
| `KNEE_ENSEMBLE_WITH` | `knee_v11` | Packs whose members join the ensemble. Empty string = own members only (faster inference). |
| `KNEE_MEMBERS` | `2` | Number of members trained. |
| `KNEE_SEED0` | `100` | Member seeds are 42 + SEED0 + index. |
| `KNEE_EPOCHS` | `24` | Training epochs. |
| `KNEE_INPUT_ROOT` | `/kaggle/input` | Where attached data is looked up. |
| `KNEE_WORK` | `/kaggle/working` | Where caches, weights and the submission are written. |

Dependencies: PyTorch, timm 1.0.30, numpy, pandas, scikit-learn, pydicom, psutil, tqdm, plus the DICOM decoders python-gdcm and pylibjpeg for compressed series (the notebook installs these itself when Internet is ON).

## Limitations
- The model is label-bound: the fused weak labels have a macro AUC of 0.891 against the gold studies, so the labels, not the image model, set the ceiling.
- The gold-based numbers are mildly optimistic because the label-fusion weights were fit on the same 58 studies.
- Inference runs 4 models with 2 passes each, so the GPU part of the hidden run is roughly twice that of a 2-member ensemble. Set `KNEE_ENSEMBLE_WITH=""` when GPU cost matters more than the last few thousandths of AUC.
- v7 weights and the v7 cache cannot be reused (different pixels). The v12 and v11 members share preprocessing and can be mixed.

## Acknowledgements
- RSNA and Kaggle for the competition and the data.
- The public report-label datasets by stevenleehans and pilkwang, used as label sources.
- The public repository TranBaDat2607/RSNA-Knee-Abnormality-Detection, whose A/B notes guided the v11 settings.
