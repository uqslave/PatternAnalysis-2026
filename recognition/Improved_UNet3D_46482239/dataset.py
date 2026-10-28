import os
import numpy as np
import nibabel as nib
import glob
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter


DATASET_PATH_2D = r"./home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
DATASET_PATH_3D = r"./home/groups/comp3710/HipMRI_Study_open"
if not os.path.exists(DATASET_PATH_2D):
    DATASET_PATH_2D = r"/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    DATASET_PATH_3D = r"/home/groups/comp3710/HipMRI_Study_open"

# Dictionary of tuples of raw, mask pairs for 2D
DATASET_2D_TYPES = {
    "train": (DATASET_PATH_2D + r"/keras_slices_train", DATASET_PATH_2D + r"/keras_slices_seg_train"),
    "test": (DATASET_PATH_2D + r"/keras_slices_test", DATASET_PATH_2D + r"/keras_slices_seg_test"),
    "validate": (DATASET_PATH_2D + r"/keras_slices_validate", DATASET_PATH_2D + r"/keras_slices_seg_validate"),
}

# Only provided raw, mask pairs; need to do train-test-val split ourselves
DATASET_3D_TYPES = {
    "raw": DATASET_PATH_3D + r"/semantic_MRs",
    "mask": DATASET_PATH_3D + r"/semantic_labels_only"
}


class HipMRI3DDataset(torch.utils.data.Dataset):
    """
    HipMRI 3D dataset class.
    Patient-level splits to prevent data leakage.
    """
    def __init__(self, dataset: str, transform=None):
        """
        We must do our own patient split
        :param dataset:
        :param transform:
        """
        patient_ids = SPLIT[dataset]

        self.raw = list()
        self.mask = list()
        for p_id in patient_ids:
            raw_scans = RAW_COMPILED[p_id]
            mask_scans = MASKS_COMPILED[p_id]
            for index in range(len(raw_scans)):
                self.raw.append(raw_scans[index])
                self.mask.append(mask_scans[index])

        self.label_values = (i for i in range(6))   # 0 - Background,
                                                    # 1 - Body outline,
                                                    # 2 - Bone,
                                                    # 3 - Bladder,
                                                    # 4 - Rectum,
                                                    # 5 - Prostate

        self.transform = transform

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index: int):
        # Better to use from_numpy() here since the masks are assigned an integer in sequential order.
        # for Conv3d, shapes per batch should be (B, 1, D, H, W) / (B, D, H, W) or similar:
        raw = torch.from_numpy(self.raw[index]).permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(self.mask[index]).permute(3, 2, 0, 1)
        mask = mask.float()     # Can leave as float32 for now. Consider mixed precision later.

        return raw, mask


class HipMRIDataset(torch.utils.data.Dataset):
    """
    HipMRI dataset class.
    Patient-level splits to prevent data leakage.
    """
    def __init__(self, dataset: str, img_dim: tuple[int, int], transform=None):
        image_path = DATASET_2D_TYPES[dataset][0]
        mask_path = DATASET_2D_TYPES[dataset][1]

        images = sorted(glob.glob(os.path.join(image_path, "*.nii*")))
        masks = sorted(glob.glob(os.path.join(mask_path, "*.nii*")))

        #X = load_data_2D(images, target_shape=(256, 144), normImage=True)
        #Y = load_data_2D(masks, target_shape=(256, 144), categorical=True, num_classes=6, dtype=np.uint8)

        # Sanity checks
        # print(X.shape, Y.shape)
        # first = Y[0][4]
        # mid = Y[29][4]
        # last = Y[59][4]
        # print(0)

        # Need to load train, test, and validation sets for 2D
        # For 3D, we need to split it ourselves
        self.raw = load_data_2D(images, target_shape=img_dim, normImage=True)
        self.mask = load_data_2D(masks, target_shape=img_dim, categorical=True, num_classes=6, dtype=np.uint8)

        self.label_values = (i for i in range(6))   # 0 - Background,
                                                    # 1 - Body outline,
                                                    # 2 - Bone,
                                                    # 3 - Bladder,
                                                    # 4 - Rectum,
                                                    # 5 - Prostate

        # Separate resize for masks — NEAREST is critical, do not blend labels
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_dim, interpolation=transforms.InterpolationMode.NEAREST),
        ]) if transform is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index: int):
        # Better to use from_numpy() here since the masks are assigned an integer in sequential order.
        raw_image = torch.from_numpy(self.raw[index])
        mask_image = torch.from_numpy(self.mask[index])

        raw_image = raw_image.unsqueeze(0)
        mask_image = mask_image.float()     # Can leave as float32 for now. Consider mixed precision later.

        return raw_image, mask_image

# ======================================= Helper methods to load Nifti files ===========================================
def preload3D(image_names: list, mask_names: list):
    raw = load_data_3D(image_names, normImage=True)                      # (N, H, W, D) float32
    mask = load_data_3D(mask_names, categorical=True, dtype=np.uint8)    # (N, H, W, D) integer labels

    # Restructure to dictionary
    raw_images = dict()
    mask_images = dict()

    for index in range(len(image_names)):
        raw_image = raw[index]
        mask_image = mask[index]
        patient_id = image_names[index].split("\\")[1].split("_")[0]

        if patient_id not in raw_images:
            raw_images[patient_id] = list()
            mask_images[patient_id] = list()
        raw_images[patient_id].append(raw_image)
        mask_images[patient_id].append(mask_image)

    return raw_images, mask_images, raw, mask


def fit_to_shape(img, target_shape: tuple[int, int]):
    """
    Pad the 2D image array
    supposedly because the image sizes aren't consistent and 256,144 seems to be the biggest.
    """
    height, width = img.shape
    target_height, target_width = target_shape

    if target_height <= height and target_width <= width:
        dy = height // 2 - target_height // 2
        dx = width // 2 - target_width // 2
        return img[dy : height - dy, dx : width - dx]
    if (target_height <= height and target_width > width) or (target_height >= height and target_width < width):
        return None     # Or change to raise an error

    # Using centre padding
    dx, dy = target_width - width, target_height - height
    sx, sy = dx // 2, dy // 2   # Scaled pad sizes
    cx = cy = 0                 # Correction factors if dx|dy is odd
    if dx % 2 != 0:
        cx = 1
    if dy % 2 != 0:
        cy = 1

    return np.pad(img, ((sy, sy + cy), (sx, sx + cx)))


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res


# load medical image functions
def load_data_3D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, orient=False,
                 early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive
    memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    orient : Apply orientation and resample image? Good for images with large
    slice thickness or anisotropic resolution
    dtype : Type of the data. If dtype = np.uint8, it is assumed that the data
    is labels
    early_stop : Stop loading pre-maturely? Leaves arrays mostly empty, for
    quick loading and testing scripts.
    '''
    affines = []

    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_2D(imageNames, target_shape=(256, 128), normImage=False,
                 categorical=False, num_classes=None, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive
    memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    early_stop : Stop loading pre-maturely, leaves arrays mostly empty, for
    quick loading and testing scripts.

    :param imageNames: List of all .nii images in the target directory
    :param target_shape: Target dimensions of the images once loaded in
    :param normImage: idk
    :param categorical: Whether to split into categories using one-hot encoding
    :param num_classes: Number of classes/channels if we were to split into categories
    :param dtype: Data type
    :param getAffines: idk
    :param early_stop: Early stop
    :return: numpy array of loaded Nifti images
    """

    affines = []

    # get fixed size
    num = len(imageNames)
    if categorical:
        assert num_classes is not None, "pass num_classes for masks"
        images = np.zeros((num, num_classes, *target_shape), dtype=dtype)
    else:
        images = np.zeros((num, *target_shape), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        nifti_image = nib.load(inName)
        in_image = nifti_image.get_fdata(caching='unchanged')
        if in_image.ndim == 3:
            in_image = in_image[:, :, 0]

        in_image = fit_to_shape(in_image, target_shape)

        if categorical:
            # fixed class count: no missing-label shape errors
            one_hot = np.eye(num_classes, dtype=dtype)[in_image.astype(int)]
            images[i] = np.moveaxis(one_hot, -1, 0) # Reorders shape from [N,H,W,K] to [N,K,H,W]
        else:
            in_image = in_image.astype(dtype)
            if normImage:
                in_image = (in_image - in_image.mean()) / (in_image.std() + 1e-8)
            images[i] = in_image

        affines.append(nifti_image.affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

# ======================================================================================================================
def split3D(image_names: list, mask_names: list, seed: int | str = 69) -> tuple:
    assert len(image_names) == len(mask_names) == len(MASKS) == len(RAW), "Inconsistent number of images and masks"

    patient_scans_count = list()  # Useful for data audit
    split_raw = dict()
    split_mask = dict()
    for index in range(len(image_names)):
        raw_split = image_names[index].split("\\")[1].split("_")
        mask_split = mask_names[index].split("\\")[1].split("_")
        patient_id_raw = raw_split[0]
        patient_id_mask = mask_split[0]
        assert patient_id_raw == patient_id_mask, "Raw/mask mismatch in dataset."

        patient_scans_count.append(patient_id_raw)

        split_raw[patient_id_raw] = raw_split[1]
        split_mask[patient_id_mask] = mask_split[1]


    rng = random.Random(seed)
    counts = Counter(patient_scans_count)                 # patient -> number of images
    patients = list(counts)
    rng.shuffle(patients)                         # otherwise the DP always favours the same patients
    total = sum(counts.values())

    splits, remaining = [], patients
    for frac in SPLIT_PROPS[:-1]:                   # last split takes whatever is left
        sizes = [counts[p] for p in remaining]
        idx = set(subset_sum_pick(sizes, round(frac * total)))
        splits.append({remaining[i] for i in idx})
        remaining = [p for i, p in enumerate(remaining) if i not in idx]
    splits.append(set(remaining))

    # sanity check: no patient appears in two splits
    assert sum(len(s) for s in splits) == len(set().union(*splits))

    return tuple(splits)


def subset_sum_pick(sizes, target):
    """Indices into `sizes` whose sum is closest to `target` (0/1 DP)."""
    cap = 2 * target
    reach = [False] * (cap + 1)
    parent = [None] * (cap + 1)
    reach[0] = True
    for i, a in enumerate(sizes):
        for s in range(cap, a - 1, -1):          # backwards => each patient used once
            if not reach[s] and reach[s - a]:
                reach[s] = True
                parent[s] = i
    best = min((s for s in range(cap + 1) if reach[s]), key=lambda s: abs(s - target))
    chosen, s = [], best
    while s > 0:
        chosen.append(parent[s])
        s -= sizes[parent[s]]
    return chosen


LABELS = ("Background", "Body Outline", "Bone", "Bladder", "Rectum", "Prostate")
IMAGE_SIZE = (256, 128)
BATCH_SIZE_2D = {
    "train":    75,
    "validate": 37,
    "test":     42,
}
BATCH_SIZE_3D = {
    "train": 6,
    "validate": 3,
    "test": 4,
}

SPLIT_PROPS = ( # For 3D scans
    0.7,    # Train
    0.2,    # Test
    0.1,    # Validate
)
assert sum(SPLIT_PROPS) == 1.0, "Split ratios not correct!"
"""
images = sorted(glob.glob(os.path.join(DATASET_3D_TYPES["raw"], "*.nii*")))
masks = sorted(glob.glob(os.path.join(DATASET_3D_TYPES["mask"], "*.nii*")))
RAW_COMPILED, MASKS_COMPILED, RAW, MASKS = preload3D(images, masks)
PATIENT_SPLIT = split3D(images, masks)
SPLIT = {
    "train":    PATIENT_SPLIT[0],
    "validate": PATIENT_SPLIT[1],
    "test":     PATIENT_SPLIT[2],
}
"""
def get_dataloader_2d(dataset: str) -> DataLoader:
    mri_dataset = HipMRIDataset(dataset, IMAGE_SIZE)
    shuffle = False
    if dataset == "train" or dataset == "test":
        shuffle = True
    return DataLoader(mri_dataset, batch_size = BATCH_SIZE_2D[dataset], shuffle = shuffle)

def get_dataloader_3d(dataset: str) -> DataLoader:
    mri_dataset = HipMRI3DDataset("train")
    shuffle = False
    if dataset == "train" or dataset == "test":
        shuffle = True
    return DataLoader(mri_dataset, batch_size = BATCH_SIZE_3D[dataset], shuffle=shuffle)

if __name__ == "__main__":
    # Try loading in data

    train_dataset = HipMRI3DDataset("train")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(len(train_dataloader))
    for i, x in enumerate(train_dataloader):
        print(x[0].shape, " | ", x[1].shape)
