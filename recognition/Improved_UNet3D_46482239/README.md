# 3D Improved U-Net for Prostate MRI Segmentation (HipMRI)

<!--
HOW TO USE THIS TEMPLATE
- Replace every [PLACEHOLDER]. Delete the <!-- guidance --> comments before submitting.
- Marking (Section 5.3): working principles (5), usage + comments (3), GitHub markdown formatting (2).
- Fill sections as you go (commit the README progressively; it also helps the commit log).
- Rename the title if you end at Normal tier (e.g. "3D U-Net ...").
-->

**Author:** [Your name], [student ID]
**Course:** COMP3710 / COMP7710 Pattern Recognition, The University of Queensland
**Project:** Segmentation, [3D Improved U-Net (Hard) / 3D U-Net (Normal)] on downsampled HipMRI 3D data
**Folder:** `recognition/[Improved_UNet3D_sSTUDENTID]/`

---

## 1. Overview

### 1.1 Problem
<!-- ~1 paragraph: what is being segmented and why it matters. Prostate radiotherapy planning, millimetre-level
contouring, organs at risk, class imbalance (prostate < 2% of the scan). State the dataset (38 patients, 211 3D volumes)
and the labels (body, bone, bladder, rectum, prostate; confirm exact label values from your data audit). -->

[PLACEHOLDER]

### 1.2 How the algorithm works
<!-- ~1 paragraph: encoder-decoder with skip connections; residual context modules; instance norm + LeakyReLU;
strided-conv downsampling; deep supervision heads; loss function (soft Dice); one-hot output. Cite Isensee et al. [2]. -->

[PLACEHOLDER]

### 1.3 Figure
<!-- Architecture diagram OR an example prediction overlay. Commit the image file (small!) and link it. -->

![Architecture / example prediction](images/[figure_name].png)

---

## 2. Feasibility Review (Section 3 of the task sheet)

<!-- One page. Must address the open engineering dilemma (spatial context vs clinical boundary utility). -->

### 2.1 User need, scope and acceptance criteria
- **Intended user / purpose:** [e.g. radiation oncologist reviewing auto-drafted contours]
- **Acceptance criteria (3-5, testable, include one resource constraint):**
  1. [Test Dice >= 0.7 on the held-out test patients]
  2. [Per-slice Dice reported for base / mid / apex regions]
  3. [3D model compared against 2D U-Net baseline on identical test patients]
  4. [Peak training VRAM <= X GiB / inference time <= Y s per volume]

### 2.2 Model choice and course concepts
[Dataset configuration (downsampled HipMRI 3D), architecture, difficulty band, links to lecture concepts and references.]

### 2.3 Preliminary feasibility evidence
[Data audit results, baseline smoke test, measured VRAM / parameter count.]

### 2.4 Risks, budget and fallback
- **Risks:** [3D memory limits, class imbalance, slow convergence]
- **Compute budget:** [GPU hours on Rangpur]
- **Next experiment:** [...]
- **Fallback:** [Submit 3D U-Net (Normal tier) if the Improved U-Net fails to converge by Day 6 decision point]

---

## 3. Repository Structure

```
recognition/[Improved_UNet3D_sSTUDENTID]/
├── modules.py     # model components (PyTorch only, no NumPy)
├── dataset.py     # loading, patient-level split, preprocessing, augmentation
├── train.py       # training, validation, testing, saving, loss/metric plots
├── predict.py     # inference + visualisation on test cases
├── utils.py       # [optional helpers]
├── images/        # figures used in this README
└── README.md
```

---

## 4. Dependencies and Reproducibility

| Package | Version |
|---|---|
| Python | [x.y.z] |
| PyTorch | [x.y.z] |
| CUDA | [x.y] |
| nibabel | [x.y.z] |
| matplotlib | [x.y.z] |
| [others] | [ ] |

- **Hardware:** [e.g. NVIDIA A100 on UQ Rangpur]
- **Random seeds:** [seed values and where they are set]
- **Data location (not committed):** `/home/groups/comp3710/HipMRI_Study_open/`
- **Environment setup:** [conda / pip commands]

### Usage
```bash
# Train
python train.py --data_dir [PATH] --epochs [N] --batch_size [B]

# Predict / visualise
python predict.py --weights [PATH_TO_WEIGHTS] --data_dir [PATH]
```
<!-- Include the Slurm script you used if helpful (commit the script, not the data or weights). -->

---

## 5. Data and Preprocessing

- **Source:** HipMRI Study (3D volumes in `semantic_MRs`, labels in `semantic_labels_only`).
- **Preprocessing:** [downsampling factor, intensity normalisation (z-score), one-hot encoding, resizing/padding; cite refs].
- **Augmentation:** [3D flips / rotations / elastic, etc.].
- **Train / validation / test split:** [numbers of patients and volumes in each].
- **Split justification:** Patient-level split (all volumes from a patient in exactly one split) to prevent leakage.
  [Describe how this is enforced and verified in `dataset.py`.]

---

## 6. Results

### 6.1 Example inputs and outputs
<!-- Input slice, ground truth, prediction, difference overlay (FP/FN). -->

![Example prediction](images/[example].png)

### 6.2 Training curves
![Loss and Dice curves](images/[curves].png)

---

## 7. Investigation of the Open Research Dilemma
**Spatial context vs. clinical boundary utility:** does 3D context fix base/apex and boundary-leakage errors that a 2D baseline misses, or does it only improve easy central slices at 5-10x the compute?

### 7.1 Quantitative benchmark (identical test split)

| Model | Prostate Dice | Bladder Dice | Rectum Dice | Bone Dice | Body Dice | Mean IoU |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | | | | | | |
| 3D U-Net | | | | | | |
| 3D Improved U-Net | | | | | | |

### 7.2 Dice by slice region (prostate)

| Model | Base | Mid | Apex |
|---|---|---|---|
| 2D U-Net (baseline) | | | |
| 3D U-Net | | | |
| 3D Improved U-Net | | | |

### 7.3 Resource profile

| Model | Parameters (M) | Peak training VRAM (GiB) | Training time | Inference per volume (s) |
|---|---|---|---|---|
| 2D U-Net (baseline) | | | | |
| 3D U-Net | | | | |
| 3D Improved U-Net | | | | |

### 7.4 Failure case autopsies (3-5 cases)
<!-- For each: multi-panel figure (image | ground truth | prediction | FP/FN overlay), what failed, and the likely root trigger. -->

**Case 1: [e.g. prostate apex slice, over-segmentation into rectal wall]**
![Case 1](images/[case1].png)
[Diagnosis]

**Case 2:** [...]

**Case 3:** [...]

### 7.5 Engineering recommendation to the project manager
[Where spatial context delivers measurable gain, where it breaks down, which anatomical cases remain unsuitable for automated drafting, and whether the extra compute is justified.]

---

## 8. Artificial Intelligence Usage Disclosure

- **Tools used:** [e.g. Claude for concept explanation and code scaffolding]
- **Tasks AI assisted with:** [e.g. drafting model-building blocks, debugging shape errors, README template]
- **How I audited and verified AI output:** [e.g. unit-checked tensor shapes, verified Dice against a manual calculation, confirmed no patient overlap between splits, ran a tiny-subset overfit test]

---

## 9. References

1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.
2. F. Isensee et al., "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," 2018.
3. Ö. Çiçek et al., "3D U-Net: Learning dense volumetric segmentation from sparse annotation," MICCAI, 2016.
4. [Any additional sources adapted in code or text. Cite in docstrings too.]
