# 3D Improved U-Net for Prostate Segmentation on HipMRI

**Student:** Joseph Zhang, 46482239\
**Course:** COMP3710 Pattern Analysis, 2026\
**Project:** 3D Improved U-Net for HipMRI Segmentation\
**Github:** `recognition/Improved_UNet3D_46482239/`

> Replace every `<...>` placeholder. Delete these instruction blockquotes before submitting.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Feasibility Review](#2-feasibility-review)
3. [How the Algorithm Works](#3-how-the-algorithm-works)
4. [Data and Preprocessing](#4-data-and-preprocessing)
5. [Setup and Reproducibility](#5-setup-and-reproducibility)
6. [Usage](#6-usage)
7. [Results](#7-results)
8. [Investigation of the Open Research Dilemma](#8-investigation-of-the-open-research-dilemma)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [Artificial Intelligence Usage Disclosure](#10-artificial-intelligence-usage-disclosure)
11. [References](#11-references)

---

## 1. Overview

> About one paragraph: the problem, why it matters clinically, what your model does.

Prostate radiotherapy planning requires millimetre-level delineation of the prostate while sparing nearby organs at risk (bladder, rectum, femoral heads). This project implements a **3D Improved U-Net** [2] that segments the <body outline, bone, bladder, rectum, prostate> from pelvic MRI volumes in the HipMRI study. It is benchmarked against a <2D U-Net> baseline to investigate whether volumetric context removes boundary errors at the prostate base and apex.

![Headline figure](images/headline_figure.png)
*Figure 1: <Caption — e.g. input, ground truth and prediction overlay.>*

---

## 2. Feasibility Review

### 2.1 User Need, Scope and Acceptance Criteria
The model will primarily be targeted towards healthcare professionals, with the main goal being to investigate the 
feasibility of using a 3D automated segmentation model in medical imaging. Specifically, to aid in the identification of 
the prostate in MRI scans. The primary dilemma here is the accuracy trade-off for using 3D scans instead of 2D scans 
since training a 3D segmentation model will up many times more the computation resources.
- **Acceptance criteria (3–5, testable):**
  1. Prostate Dice Score (present) ≥ 0.70 on unseen test set;
  2. Prostate IoU Score (present) ≥ 0.70 on unseen test set;
  3. Peak GPU VRAM ≤ 35 GiB during training;
  4. Inference time ≤ 1.0 s per volume;
  5. No data leaks by ensuring patient-level splits. We can gauge this using ‘assert’ in code.

### 2.2 Model Choice and Course Concepts
The dataset will be split at the patient-level, and we will separate the data into a training set, validation set, and 
test set with the following weights of 0.80, 0.08, 0.12 (or similar). As an initial model, we will take inspiration from
<___67876> and use a 3D Improved U-Net with residual blocks and deep supervision and choose LeakyReLU and InstanceNorm 
units in the residual blocks. Four (encoder & decoder) layers will be implemented, and we will use 2-layer deep 
supervision. Dice-CE loss will be the basic loss function, and Tversky-CE loss can be considered if more suitable. We 
can implement the base U-Net in accordance with the example shown in week 7/8 of COMP3710. We will compare the 
performance to a 2D Improved U-Net with the same residual blocks and deep supervision.

### 2.3 Preliminary Feasibility Evidence
Initial data audit yields the following information shown. Data is compatible, consistent, and can be split at a patient
level.

| Item                         | Description                                                                                                |
|------------------------------|------------------------------------------------------------------------------------------------------------|
| Total Scans                  | 211                                                                                                        |
| Patient Count                | 38                                                                                                         |
| Image Dimensions/shape       | All scans are `(256, 256, 128)` in $H$, $W$, $D$                                                           |
| Patient ID format            | All file names are in the format of `LNNN_WeekX_LFOV/SEMANTIC.nii.gz`<br/>Ex. `B010_Week3_SEMANTIC.nii.gz` |
| 2D Improved U-Net VRAM usage |                                                                                                            |


### 2.4 Risks, Budget and Fallback
| Risk                   | Mitigation                                                                                                   |
|------------------------|--------------------------------------------------------------------------------------------------------------|
| OOM (VRAM)             | Random patch in sizing `__get_item__`; Mixed precision; Small batch size; Load tensor only in `__get_item__` |
| Class imbalance        | Dice Loss + Cross Entropy Loss in training                                                                   |
| Exorbant training time | 2D Improved U-Net is available and working                                                                   |

---

## 3. How the Algorithm Works

> Explain the working principles (this is worth 5 marks). Aim for clarity over length.

### 3.1 U-Net Backbone
Simply put, a U-Net is an autoencoder-like network. But, unlike autoencoders, U-Nets are supervised and feature
skip connections that transfer information between the encoder and the decoder of the same resolution. This skip 
connection is what mitigates the loss of information (preserves spatial detail) as it gets downsampled in the encoders 
and passes through the (bottleneck) latent layer.

The role of the encoder is to extract abstract features of an input. In the middle, the bottleneck holds the most abstract, highest-
channel representation while still retaining spatial structure. The decoder takes the data from this latent 
layer and progressively upsamples it to the original input-resolution while also attempting to produce a per-pixel class 
prediction. At each level, the encoder feature map of matching resolution is concatenated onto the decoder’s feature 
map, which restores the fine spatial detail lost during downsampling. Unlike an autoencoder, which reconstructs its 
input, a U-Net outputs $K$ channels (one per class) via a final 1×1 convolution. Then, a softmax over these channels 
gives us the per-pixel class probabilities, and taking the argmax yields the segmentation mask (akin to a silhouette) of
the every object in the original input.

### 3.2 Improvements over the Standard U-Net
| Component              | Function                                                                                                                                      | Why it helps here                                                                                                                                                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Residual modules       | Adds two 3x3 convolution blocks with a shortcut added to the block's output; acts as a skip connection between each encoder/decoder           | Allows the network to only have to learn the difference before and after each block rather than across the entire network.<br/>Resolves the vanishing gradient problem; better results when training.                               |
| Deep supervision       | Segmentation maps (1×1 conv) are taken from several decoder levels, upsampled to a common size and summed element-wise into the final output. | The loss gradient reaches the earlier decoder layers through a short path, so training converges faster and vanishing gradients are reduced. It also forces intermediate features to be useful for segmentation at multiple scales. |
| leaky ReLU             | Introduces a small slope on the negative side.                                                                                                | Ensures a small gradient still flows back during backpropagation; doesn't kill neurons.                                                                                                                                             |
| Instance normalisation | normalises each channel of each sample independently across its spatial dimensions.                                                           | Allows for smaller batch sizes to save on VRAM usage since its outputs don’t depend on the batch                                                                                                                                    |

### 3.3 3D vs 2D Convolutions
<Inter-slice context vs memory cost. Include a kernel/feature-map size comparison if useful.>

### 3.4 Loss Function
In an image segmentation task, we want the model to make accurate predictions on both at pixel-level and general-level.
For other models where we have similar goals, Cross Entropy Loss is usually the standard loss function for its ability 
to perform pixel classifications with the softmax function. However, for datasets with sever imbalance, CE by itself may 
plunder the model by learning to just predict the dominant class. To mitigate this, we introduce the Dice Similarity
Coefficient (DSC). This is because DSC calculates the overlap between the predicted segmentation and the ground truth, 
and focuses on the intersection over union of the specific classes rather than raw pixel counts. Consequently, this is
what makes it a suitable choice for datasets (like the HipMRI) with class imbalance.
The equations for DSC, Dice Loss, and CE are as follows: 
$$\text{DSC}_c = \frac{2 \sum_{i=1}^{N} y_{i,c} \hat{y}_{i,c} + \epsilon}{\sum_{i=1}^{N} y_{i,c} + \sum_{i=1}^{N} \hat{y}_{i,c} + \epsilon}$$
$$\mathcal{L}_{DSC} = 1 - \frac{1}{C} \sum_{c=1}^{C} \text{DSC}_c$$
$$\mathcal{L}_{CE} = - \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$
Implementation of these loss functions can be found in modules.py

![Architecture diagram](images/architecture.png)
*Figure 2: <Architecture of the 3D Improved U-Net.>*

---

## 4. Data and Preprocessing

### 4.1 Dataset
- **Source:** HipMRI Study (`/home/groups/comp3710/HipMRI_Study_open`) — 38 patients, 211 volumes
- **Labels:** `0: background, 1: body outline, 2: bone, 3: bladder, 4: rectum, 5: prostate`
- **Input shape:** `(128, 256, 256)`
- **Voxel spacing:** `<...>`

### 4.2 Preprocessing
- **Intensity normalisation:** <e.g. per-volume z-score> — <justification and reference>
- **Label encoding:** one-hot with 6 channels (labels)
- **Resampling / downsampling:** 2D scans are cropped to 256x128 when necessary

### 4.3 Train / Validation / Test Split
A total of 211 scans across 38 patients was provided.

| Split      | Proportion | Patients | Volumes |
|------------|------------|----------|---------|
| Train      | 0.80       |        |      |
| Validation | 0.08       |         |       |
| Test       | 0.12       |         |       |

**Justification:** <Split is by patient, so all volumes from one patient are in a single split. State the ratio, seed, and why.>

---

## 5. Setup

### 5.1 Dependencies
| Package    | Version |
|------------|---------|
| Python     | 3.14.5  |
| PyTorch    | 2.14.0  |
| ROCM       | 10.1.0  |
| nibabel    | 5.4.2   |
| numpy      | 2.5.2   |
| matplotlib | 3.11.1  |

### 5.2 Reproducibility
- **Random seed:** Where relevant, a `seed` of `67` was set for all stochastic processes on import for reproducibility.
  - `torch.manual_seed(seed)` 
  - `np.random.seed(seed)`
  - `random.seed(seed)`
  - `torch.cuda.manual_seed(seed)`
- **Hardware:** Nvidia A100 on rangpur
- **Training settings**
  - ***2D Improved U-Net:***
    - `batch_size = 75`
    - `num_epochs = 28`
    - `dropout = 0.23`
    - `learning_rate = 0.001`
    - `optimiser = torch.optim.Adam`
    - `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=num_epochs, eta_min=7e-5)`
  - ***3D Improved U-Net:***
    - `batch_size = 3`
    - `num_epochs = `
    - `dropout = `
    - `learning_rate = `
    - `optimiser = torch.optim.Adam`
    - `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=num_epochs, eta_min=)`

### 5.3 Repository Structure
```
recognition/Improved_UNet3D_46482239/
├── modules.py      # Model definitions and loss functions
├── dataset.py      # Data preprocessing, data loading, patient-level splits
├── train.py        # Training, validation, and saving the model
├── predict.py      # Inference and visualisation
├── utils.py        # Plotting and data Audit
├── audit_out/      # Path for data audit outputs
├── images/         # Figures used in this README and from inference
└── README.md
```

---

## 6. Usage

```bash
# Train
python train.py --model <"3D" or "2D"> --epochs <N>

# Predict and run inference
# Assumes .pth exists under <file>
python predict.py --model <"3D" or "2D">
```

| Argument   | Description                   | Default |
|------------|-------------------------------|---------|
| `--model`  | 2D or 3D Improved U-Net       | None    |
| `--epochs` | Number of epochs to train for | 60      |

---

## 7. Results

### 7.1 Training Curves
![Loss and Dice curves](images/training_curves.png)
*Figure 3: <Training and validation loss and Dice per epoch.>*

### 7.2 Example Predictions
![Predictions](images/predictions.png)
*Figure 4: <Input, ground truth and prediction.>*

### 7.3 Test Set Performance
| Class        | Dice Score (Aggregate) | Dice Score (Present) | Dice Score (Pooled) | IoU (Pooled) | FPR |
|--------------|------------------------|----------------------|---------------------|--------------|-----|
| Background   |                        |                      |                     |              |     |
| Body outline | <...>                  |                      |                     | <...>        |     |
| Bone         | <...>                  |                      |                     | <...>        |     |
| Bladder      | <...>                  |                      |                     | <...>        |     |
| Rectum       | <...>                  |                      |                     | <...>        |     |
| **Prostate** | **<...>**              |                      |                     | **<...>**    |     |

The aggregate dice scores are the dice scores across all slices, including the slices where the class of interest is
absent; it includes the dice scores that predict the empty slices for the said class. On the other hand, the 
'pooled' dice scores consider the aggregate scores across all the slices where the class of interest is present in the 
ground truth mask. Likewise, the 'present' dice score is a mean of the dice scores of the slices where the class of
interest is present in the ground truth masks.\
`FPR` denotes the pixel-wise false positive rates.

---

## 8. Investigation of the Open Research Dilemma

> **Question:** Does multi-scale context or true 3D continuity eliminate hazardous boundary and slice-transition errors compared with a standard 2D baseline, or does it only smooth easy central slices at 5–10× the cost?

### 8.1 Baseline
<Describe the baseline model(s): e.g. 2D U-Net, 2D Improved U-Net. State that all models use the **same test patients and metrics**. If 2D predictions are stacked into 3D for scoring, explain how.>

### 8.2 Quantitative Benchmarking
| Model                 | Prostate Dice Score (present) | Prostate Dice Score | Prostate IoU | FPR    | Dice Score (Aggregate) |
|-----------------------|-------------------------------|---------------------|--------------|--------|------------------------|
| 2D U-Net (baseline)   | <...>                         |                     | <...>        | <...>  | <...>                  |
| 2D Improved U-Net     | 0.762                         | 0.867               | 0.766        | 0.0006 | 0.905                  |
| **3D Improved U-Net** | <...>                         |                     | <...>        | <...>  | <...>                  |

<Brief interpretation. State whether differences are likely meaningful given the test set size, e.g. report the spread across patients.>

### 8.3 Where Does Context Help? (Slice-wise Analysis)
![Dice by slice position](images/dice_by_slice.png)
*Figure 5: Prostate Dice versus slice position (inferior apex → superior base) for each model.*

<Do the models differ at the apex and base compared with the central slices? Discuss empty-slice handling.>

### 8.4 Resource Profiling
| Model             | Parameters | Peak VRAM (GB) | Training time | Inference time / volume |
|-------------------|------------|----------------|---------------|-------------------------|
| 2D U-Net          | <...>      | <...>          | <...>         | <...>                   |
| 2D Improved U-Net | <...>      | <...>          | <...>         | <...>                   |
| 3D Improved U-Net | <...>      | <...>          | <...>         | <...>                   |

### 8.5 Failure Case Autopsy (3–5 cases)

For each case, include the multi-panel figure (image, ground truth, prediction, FP/FN overlay).

#### Case 1: <e.g. prostate apex slice, Patient X>
![Failure case 1](images/failure_1.png)
- **What happened:** <...>
- **Likely trigger:** <e.g. low soft-tissue contrast, partial volume effect>
- **Did the baseline fail too?** <...>

#### Case 2: <...>
<Repeat the same structure for Cases 3–5.>

### 8.6 Recommendation to the Project Manager
- **Does 3D context justify its cost?** <Evidence-based answer.>
- **Where it helps / where it does not:** <...>
- **Cases unsuitable for automated drafting:** <...>
- **Suggested workflow:** <e.g. automate central slices, route base/apex to a human>

---

## 9. Limitations and Future Work
- <Small number of patients, single-centre data, downsampling effects, etc.>
- <Possible extensions.>

---

## 10. Generative AI Disclosure

| Tool              | Used for                             | Verification Processes                                                                                   |
|:------------------|--------------------------------------|----------------------------------------------------------------------------------------------------------|
| Claude Sonnet 5.5 | Explaining theory                    | Cross-reference with course material and other sources                                                   |
| ...               | Subset-sum problem for patient split | Manual checks on split patient lists using pycharm's debug mode and asserts to detect duplicate patients |
| ...               | Debugging OOM                        | Check training graph and inference results to ascertain the plausibility of results                      |
| ...               | Debugging stdout buffering           | `cat log.out` and `cat log.err` to ensure training is happening                                          |
| ...               | Report template                      | Cross-check with task sheet                                                                              |

---

## 11. References
1. <Baseline / dataset reference [1]>
2. F. Isensee et al., "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge," 2017. <Check the exact reference given in the assignment.>
3. <3D U-Net reference [4]>
4. <Any code, library, or online resource adapted — cite it here and in docstrings.>