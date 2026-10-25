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

BENIGN, MALIGNANT = 0.0, 1.0
IMAGE_DIMS = (224, 224)
RGB_MEAN, RGB_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
BATCH_SIZES = {
    "train": 250,
    "validate": 150,
    "test": 195,
}
SHUFFLE = {
    "train": False,
    "validate": False,
    "test": False,
}
TRANSFORMS = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ]),
    "validate": transforms.Compose([
        transforms.Resize(IMAGE_DIMS), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN,  # mean R, G, B empirical standard
                             RGB_STD),  # std R, G, B empirical standard
    ]),
}
TTV_SPLITS = {
    "train": 0.78,
    "validate": 0.10,
    "test": 0.12,
}
assert sum(TTV_SPLITS.values()) == 1.0, "Train/Test/Validate proportion must sum to 1.0"

def get_isic_pairloader(dataset_id: str):
    dataset = PairData(dataset_id)
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
        read_patient_id = set()
        patient_data = dict()  # ISIC data with patient split

        for row in isic_data:
            isic_id, patient_ids, target = row
            patient_id_split = patient_ids.split(",")
            patient_ids = patient_id_split

            result = all(item in read_patient_id for item in patient_ids)
            if not result:
                read_patient_id.add(patient_ids[0])
            id_to_use = ""
            for patient_id in patient_ids:
                if patient_id in read_patient_id:
                    id_to_use = patient_id
                    break
            if id_to_use not in patient_data.keys():
                patient_data[id_to_use] = dict()
            patient_data[id_to_use].update({isic_id: int(float(target))})

        return patient_data

    ISIC_DATA = load_isic_csv()
    PATIENT_SPLIT_DATA = patient_split(ISIC_DATA)

    def __init__(self, dataset_id: str, img_dim: tuple[int, int], transform: transforms.Compose = None) -> None:
        super().__init__()
        self._image_path = Path(r"./isic_melanoma_dataset/combined_images")
        self._img_dim = img_dim
        self._transform = transform

        # Train: first x% of patients
        # Validate: next y% of patients
        # Test: last z% of patients
        start_idx = end_idx = -1
        length = len(self.PATIENT_SPLIT_DATA.keys())
        if dataset_id == "train":
            start_idx = 0
            end_idx = int(length * TTV_SPLITS[dataset_id] // 1)
        elif dataset_id == "validate":
            start_idx = int(length * TTV_SPLITS["train"] // 1) + 1
            end_idx = length * int(TTV_SPLITS["train"] + TTV_SPLITS[dataset_id]) // 1
        elif dataset_id == "test":
            start_idx = int(length * (length - TTV_SPLITS[dataset_id]) // 1) + 1
            end_idx = length
        else:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        dataset_whole = list(self.PATIENT_SPLIT_DATA.items())
        random.shuffle(dataset_whole)
        self._dataset = dataset_whole[start_idx:end_idx]
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
    Randomly choose one for the first image the try forcing 50:50 chance for second image.
    """
    FORCED = 0.50   # Chance of getting malignant scan as second image.

    def __init__(self, dataset_id: str) -> None:
        super().__init__()
        self._dataset_id = dataset_id
        self._dataset = self._load_isic_dataset(dataset_id)
        self._benigns, self._malignants = self._extract_scans()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        choice1 = self._dataset[index]  # Dataset is shuffled when loaded.
        if random.random() < self.FORCED:   # Randomly sample from malignant scans
            choice2 = random.choice(tuple(self._malignants))
        else:                               # Randomly sample from self._dataloader again
            random_index = (index + random.randint(1, len(self._dataset) - 1)) % len(self._dataset)
            choice2 = self._dataset[random_index]
        return choice1, choice2

    def _load_isic_dataset(self, dataset_id: str) -> ISICMelanomaDataset:
        transform = TRANSFORMS[dataset_id]
        return ISICMelanomaDataset(dataset_id, IMAGE_DIMS, transform)

    def _extract_scans(self) -> tuple[tuple[tuple[Tensor, Tensor]], tuple[tuple[Tensor, Tensor]]]:
        benign_scans = list()
        malignant_scans = list()
        for image, label in self._dataset:
            if label == MALIGNANT:
                malignant_scans.append((image, label),)
            elif label == BENIGN:
                benign_scans.append((image, label),)
            else:
                raise KeyError(f"Unknown label: {label}; Data labels not processed correctly")
        return tuple(benign_scans), tuple(malignant_scans)


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

