import math
import random

import kagglehub
import csv
from pathlib import Path

import torchvision.transforms.functional as F
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATASET_HANDLE = "roua91/isic-melanoma-dataset-resized-to-224x224"
DATASET_PATH = r"./isic_melanoma_dataset/"

ISIC_ID, PATIENT_ID, TARGET = "isic_id", "patient_id", "target"

def download_isic() -> None:
    # Download latest version
    path = kagglehub.dataset_download(handle = DATASET_HANDLE, output_dir = DATASET_PATH)
    print("Path to dataset files:", path)

SEED = 0
BENIGN, MALIGNANT = 0, 1
IMAGE_DIMS = (224, 224)
RGB_MEAN, RGB_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
BATCH_SIZES = {
    "train": 125,
    "validate": 80,
    "test": 100,
}
SHUFFLE = {
    "train": True,
    "validate": False,
    "test": False,
}
TRANSFORMS = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ]),
    "validate": transforms.Compose([
        transforms.Resize(IMAGE_DIMS),
        transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD),
    ]),
    "test": transforms.Compose([
        transforms.Resize(IMAGE_DIMS),
        transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD),
    ]),
}
TTV_SPLITS = {
    "train":    0.76,
    "validate": 0.10,
    "test":     0.14,
}
assert math.isclose(sum(TTV_SPLITS.values()), 1.0), "Train/Test/Validate proportion must sum to 1.0"

def get_isic_pairloader(dataset_id: str):
    # Train: fresh random pairs every epoch. Validate/test: fixed pairs so metrics are comparable across epochs.
    pair_seed = None if dataset_id == "train" else SEED
    dataset = PairData(dataset_id, seed = pair_seed)
    return DataLoader(dataset, batch_size = BATCH_SIZES[dataset_id], shuffle = SHUFFLE[dataset_id], num_workers=0)


class ISICMelanomaDataset(Dataset):
    @staticmethod
    def load_isic_csv() -> list[list[str]]:
        """
        Load the isic melanoma dataset csv file and perform patient split.

        isic_id:
            Unique identifier for each image, corresponding to a filename in the dataset.
        target:
            Label indicating the diagnostic result:
                0.0: Benign
                1.0: Malignant
        patient_id:
            Optional identifier for the patient, providing additional metadata for longitudinal studies or patient-level analyses.
            dummy_*;
            HAM_*;
            IP_*;
            Some patients have both HAM_x and dummy_* ID
        :return:
        """
        csv_path = Path(DATASET_PATH + "cleaned_metadata.csv")
        csv_file = open(csv_path, mode='r', newline='')
        reader = csv.reader(csv_file)

        return list(reader)[1:]

    @staticmethod
    def patient_split(isic_data: list[list[str]]) -> dict[str, dict[str, float]]:
        """
        patient_data = {
            patient_id: {
                ISIC ID 1: label 1,\n
                ISIC ID 2: label 2, ...
            }
        }
        :param isic_data: Loaded into a list
        :return: Data with patient-level split
        """
        parent = dict()

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for _, patient_ids, _ in isic_data:
            ids = patient_ids.split(",")
            root = find(ids[0])
            for other in ids[1:]:
                parent[find(other)] = root   # merge all IDs belonging to the same patient

        patient_data = dict()  # one entry per real patient
        for isic_id, patient_ids, target in isic_data:
            patient_key = find(patient_ids.split(",")[0])
            patient_data.setdefault(patient_key, dict())[isic_id] = int(float(target))

        return patient_data

    ISIC_DATA = load_isic_csv()
    PATIENT_SPLIT_DATA = patient_split(ISIC_DATA)
    PATIENT_ORDER = sorted(PATIENT_SPLIT_DATA.items())   # deterministic base order
    random.Random(SEED).shuffle(PATIENT_ORDER)           # shuffled ONCE, identical for every split

    def __init__(self, dataset_id: str, img_dim: tuple[int, int], transform: transforms.Compose = None) -> None:
        super().__init__()
        self._image_path = Path(r"./isic_melanoma_dataset/combined_images")
        self._img_dim = img_dim
        self._transform = transform

        # Patients are shuffled once (seeded) at class level, then cut into contiguous train/validate/test blocks.
        n = len(self.PATIENT_ORDER)
        n_train = int(n * TTV_SPLITS["train"])
        n_val_end = int(n * (TTV_SPLITS["train"] + TTV_SPLITS["validate"]))
        bounds = {"train": (0, n_train), "validate": (n_train, n_val_end), "test": (n_val_end, n)}
        if dataset_id not in bounds:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        self._dataset = self.PATIENT_ORDER[bounds[dataset_id][0]:bounds[dataset_id][1]]
        images = list()
        labels = list()
        for patient_data in self._dataset:
            _, scans = patient_data
            for pair in scans.items():
                images.append(pair[0] + ".jpg")
                labels.append(int(float(pair[1])))
        self._image_file_names = images
        self._labels = labels
        assert len(self._image_file_names) == len(self._labels)

        # umm = {k: v for k, v in self._patient_split_data.items() if len(v.values()) > 1}
        # print(len(umm.keys()))

    def __len__(self) -> int:
        return len(self._image_file_names)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_folder_path = Path(DATASET_PATH + r"combined_images/")
        image_file = self._image_file_names[index]
        label = self._labels[index]
        label = torch.tensor(label)

        image_path = image_folder_path / image_file
        image = Image.open(image_path).convert("RGB")   # 'convert' forces RGB 3 channels
        if self._transform:
            image = self._transform(image)
        else:
            image = F.to_tensor(image)

        return image, label


class PairData(Dataset):
    """
    Class-balanced pair sampling (stores indices only; images are loaded lazily in __getitem__).
      1. anchor class chosen 50/50 (benign / malignant), anchor drawn from that class
      2. partner is the same class with probability POSITIVE_RATIO, otherwise the other class
    Returns ((image1, label1), (image2, label2)).
    seed=None -> fresh random pairs each call; seed=int -> index always maps to the same pair.
    """
    POSITIVE_RATIO = 0.50   # probability that a pair is same-class

    def __init__(self, dataset_id: str, seed: int | None = None) -> None:
        super().__init__()
        self._dataset_id = dataset_id
        self._seed = seed
        self._dataset = ISICMelanomaDataset(dataset_id, IMAGE_DIMS, TRANSFORMS[dataset_id])
        labels = self._dataset._labels
        self._pools = {
            BENIGN: [i for i, l in enumerate(labels) if l == BENIGN],
            MALIGNANT: [i for i, l in enumerate(labels) if l == MALIGNANT],
        }
        if not self._pools[BENIGN] or not self._pools[MALIGNANT]:
            raise ValueError(f"'{dataset_id}' split lacks one class; cannot form negative pairs.")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        rng = random.Random(self._seed + index) if self._seed is not None else random
        anchor_cls = rng.choice((BENIGN, MALIGNANT))
        partner_cls = anchor_cls if rng.random() < self.POSITIVE_RATIO else 1 - anchor_cls
        i = rng.choice(self._pools[anchor_cls])
        j = rng.choice(self._pools[partner_cls])
        if partner_cls == anchor_cls and len(self._pools[anchor_cls]) > 1:
            while j == i:                      # never pair an image with itself
                j = rng.choice(self._pools[partner_cls])
        return self._dataset[i], self._dataset[j]


class DataAudit:
    def __init__(self):
        self._isic_melanoma_dataset = ISICMelanomaDataset.load_isic_csv()
        self._patient_split = ISICMelanomaDataset.patient_split(self._isic_melanoma_dataset)

        self._target_count = self._count_targets()
        self._image_dims, self._downsampled_count = self._check_images()
        self._duplicate_patients = self._check_dupe_patients()

    def __repr__(self) -> str:
        return self._before_split() + self._after_split()

    def _check_images(self) -> tuple[dict[str, int], int]:
        image_dimensions = dict()
        downsampled_count = 0
        for row in self._isic_melanoma_dataset:
            isic_image_id = row[0]
            if "downsampled" in isic_image_id:
                downsampled_count = downsampled_count + 1
            image = Image.open(DATASET_PATH + r"combined_images/" + isic_image_id + ".jpg")
            image_dimensions[image.size] = image_dimensions.get(image.size, 0) + 1
        return image_dimensions, downsampled_count


    def _count_targets(self) -> tuple[int]:
        target_count = [0, 0]
        for row in self._isic_melanoma_dataset:
            _, _, target = row
            target = int(float(target))
            target_count[target] = target_count[target] + 1
        return tuple(target_count)


    def _check_dupe_patients(self) -> dict[str, int]:
        patient_id_records = dict()
        for row in self._isic_melanoma_dataset:
            _, patient_id, _ = row
            patient_id_split = patient_id.split(",")
            patient_id = patient_id_split
            for p_id in patient_id:
                patient_id_records[p_id] = patient_id_records.get(p_id, 0) + 1

        duplicate_patients = {k: v for k, v in patient_id_records.items() if v > 1}
        return duplicate_patients

    def _before_split(self) -> str:
        prologue = 20 * "=" + " Data audit before patient split " + 20 * "=" + "\n"

        target_sum = sum(self._target_count)
        target_ratios = (
            round(100 * self._target_count[0] / target_sum, 4),
            round(100 * self._target_count[1] / target_sum, 4),
        )
        duplicate_sum = dict()
        for k, v in self._duplicate_patients.items():
            duplicate_sum[v] = duplicate_sum.get(v, 0) + 1
        duplicate_sum = dict(sorted(duplicate_sum.items()))
        duplicate_sum = sum(duplicate_sum.values())

        target_summary = (f"Benign count: {self._target_count[0]} ({target_ratios[0]}%)\n"
                          f"Malignant count: {self._target_count[1]} ({target_ratios[1]}%)\n")
        image_dimension_summary = f"Image dimensions count: {str(self._image_dims)}\n"
        image_downsample_summary = f"Downsampled image count: {self._downsampled_count}\n"
        duplicate_patient_summary = f"Duplicate patients: {str(duplicate_sum)}\n"

        return prologue + target_summary + image_dimension_summary + image_downsample_summary + duplicate_patient_summary

    def _after_split(self) -> str:
        prologue = 20 * "=" + " Data audit after patient split " + 20 * "=" + "\n"

        patient_count = len(self._patient_split.keys())
        patient_count_summary = f"Unique patients: {patient_count}\n"

        return prologue + patient_count_summary

if __name__ == "__main__":
    isic_data_audit = DataAudit()
    print(isic_data_audit)