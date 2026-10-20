import kagglehub
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATASET_HANDLE = "roua91/isic-melanoma-dataset-resized-to-224x224"
DATASET_PATH = r"./isic_melanoma_dataset/"

ISIC_ID, PATIENT_ID, TARGET = "isic_id", "patient_id", "target"

def download_isic() -> None:
    # Download latest version
    path = kagglehub.dataset_download(handle = DATASET_HANDLE, output_dir = DATASET_PATH)
    print("Path to dataset files:", path)


train_batch_size = 32
validate_batch_size = 32
test_batch_size = 32
image_dims = (224, 224)

def get_train_dataloader() -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ])
    train_dataset = ISICMelanomaDataset("train", image_dims, transform)
    return DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

def get_validate_dataloader() -> DataLoader:
    validate_dataset = ISICMelanomaDataset("validate", image_dims)
    return DataLoader(validate_dataset, batch_size=validate_batch_size, shuffle=False, num_workers=0)

def get_test_dataloader() -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean R, G, B empirical standard
                             (0.2023, 0.1994, 0.2010)),   # std R, G, B empirical standard
    ])
    test_dataset = ISICMelanomaDataset("test", image_dims, transform)
    return DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)


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
            patient_data[id_to_use].update({isic_id: float(target)})

        return patient_data

    ISIC_DATA = load_isic_csv()
    PATIENT_SPLIT_DATA = patient_split(ISIC_DATA)
    TTV_SPLITS = {
        "train": 0.78,
        "validate": 0.10,
        "test": 0.12,
    }
    assert sum(TTV_SPLITS.values()) == 1.0, "Train/Test/Validate proportion must sum to 1.0"

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
            end_idx = int(length * self.TTV_SPLITS[dataset_id] // 1)
        elif dataset_id == "validate":
            start_idx = int(length * self.TTV_SPLITS["train"] // 1) + 1
            end_idx = length * int(self.TTV_SPLITS["train"] + self.TTV_SPLITS[dataset_id]) // 1
        elif dataset_id == "test":
            start_idx = int(length * (length - self.TTV_SPLITS[dataset_id]) // 1) + 1
            end_idx = length
        else:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        self._dataset = list(self.PATIENT_SPLIT_DATA.items())[start_idx:end_idx]
        images = list()
        labels = list()
        for patient_data in self._dataset:
            _, pairs = patient_data
            for pair in pairs.items():
                images.append(pair[0] + ".jpg")
                labels.append(pair[1])
        self._images = images
        self._labels = labels
        assert len(self._images) == len(self._labels)

        # umm = {k: v for k, v in self._patient_split_data.items() if len(v.values()) > 1}
        # print(len(umm.keys()))


    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        image_folder_path = Path(DATASET_PATH + r"combined_images/")
        image = self._images[idx]
        label = self._labels[idx]

        image_path = image_folder_path / image
        image = Image.open(image_path)
        if self._transform:
            image = self._transform(image)

        return image, label


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

