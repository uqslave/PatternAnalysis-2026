import os
import numpy as np
import nibabel as nib
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


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
        self.raw = load_data_2D(images, target_shape=(256, 144), normImage=True)
        self.mask = load_data_2D(masks, target_shape=(256, 144), categorical=True, num_classes=6, dtype=np.uint8)

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

    def __getitem__(self, idx):
        # Better to use from_numpy() here since the masks are assigned an integer in sequential order.
        raw_image = torch.from_numpy(self.raw[idx])
        mask_image = torch.from_numpy(self.mask[idx])

        raw_image = raw_image.unsqueeze(0)
        mask_image = mask_image.float()     # Can leave as float32 for now. Consider mixed precision later.

        return raw_image, mask_image

# ======================================= Helper methods to load Nifti files ===========================================

def fit_to_shape(img, target_shape: tuple[int, int]):
    """
    Pad the 2D image array
    supposedly because the image sizes aren't consistent and 256,144 seems to be the biggest.
    """
    height, width = img.shape
    target_height, target_width = target_shape

    if target_height < height or target_width < width:
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

""" # Unused
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res
"""

# load medical image functions
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
LABELS = ("Background", "Body Outline", "Bone", "Bladder", "Rectum", "Prostate")
IMAGE_SIZE = (256, 144)
TRAIN_BATCH_SIZE = 106
TEST_BATCH_SIZE = 40
VALIDATE_BATCH_SIZE = 56

def get_train_loader() -> DataLoader:
    train_dataset = HipMRIDataset("train", IMAGE_SIZE)
    return DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)

def get_test_loader() -> DataLoader:
    test_dataset = HipMRIDataset("test", IMAGE_SIZE)
    return DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, shuffle=True)

def get_validate_loader() -> DataLoader:
    validate_dataset = HipMRIDataset("validate", IMAGE_SIZE)
    return DataLoader(validate_dataset, batch_size = VALIDATE_BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    # Try loading in data

    train_dataset = HipMRIDataset("train", IMAGE_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(len(train_dataloader))
    for i, x in enumerate(train_dataloader):
        print(f"{i}: {x}")
