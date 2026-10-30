import dataset
import glob
import os
import torch
import pandas as pd
from pathlib import Path

SEED = 67
torch.manual_seed(SEED)

output_dir = Path("audit_out/")
output_dir.mkdir(parents=True, exist_ok=True)

class DataAudit:
    def __init__(self):
        self.image_files = sorted(glob.glob(os.path.join(dataset.DATASET_3D_TYPES["raw"], "*.nii*")))
        self.mask_files = sorted(glob.glob(os.path.join(dataset.DATASET_3D_TYPES["mask"], "*.nii*")))

        self.raw_scan_shapes        = list()
        self.mask_scan_shapes       = list()
        self.raw_mask_match         = list()
        self.mask_num_channels      = list()
        self.mask_channels_present  = list()
        self.patient_ids            = list()
        self.raw_file_names         = list()
        self.mask_file_names        = list()
        self.patient_count          = dict()
        self.to_csv = None

        # Data loader are post-split
        #self.train_dataloader = dataset.get_dataloader_3d("train")
        #self.validate_dataloader = dataset.get_dataloader_3d("validate")
        #self.test_dataloader = dataset.get_dataloader_3d("test")

    def __repr__(self) -> str:
        if self.to_csv is None:
            raise "Data audit not yet evaluated!"
        out_str = f"Number of unique patients: {len(self.patient_count.keys())}\n"
        out_str = out_str + str(self.to_csv)
        return out_str

    def __str__(self) -> str:
        return self.__repr__()

    def evaluate(self):
        assert len(self.image_files) == len(self.mask_files), "Data Mismatch!"
        patient_ids = list()
        raw_file_names, mask_file_names = list(), list()
        patient_count = dict()
        for index in range(len(self.image_files)):
            raw_file_name, mask_file_name = self.image_files[index].split("\\")[-1], self.mask_files[index].split("\\")[-1]
            raw_file_names.append(raw_file_name)
            mask_file_names.append(mask_file_name)
            patient_id = raw_file_name.split("_")[0]
            patient_ids.append(patient_id)
            patient_count[patient_id] = patient_count.get(patient_id, 0) + 1

        raw_scan_shapes = list()
        mask_scan_shapes = list()
        raw_mask_match = list()
        mask_num_channels = list()
        mask_channels_present = list()
        for patient_id in patient_count.keys():
            # shape for both tensors:
            #   raw shape   [H, W, D]
            #   mask shape  [H, W, D, K]
            for raw_scan, mask_scan in zip(dataset.RAW_COMPILED[patient_id], dataset.MASKS_COMPILED[patient_id]):
                raw_scan = torch.from_numpy(raw_scan)
                mask_scan = torch.from_numpy(mask_scan)

                raw_scan_shapes.append(raw_scan.shape)
                mask_labels = mask_scan.argmax(dim = 3)  # Also converts to [H, W, D]
                mask_scan_shapes.append(mask_labels.shape)

                mask_num_channels.append(mask_scan.shape[-1])
                labels = torch.unique(mask_labels)
                mask_channels_present.append(labels)

                raw_mask_match.append(raw_scan.shape == mask_labels.shape)


        assert (len(raw_file_names) == len(mask_file_names) ==
                len(raw_scan_shapes) == len(mask_scan_shapes) ==
                len(raw_mask_match) == len(mask_num_channels) == len(mask_channels_present)
            ), "Data Mismatch!"

        self.raw_scan_shapes        = raw_scan_shapes
        self.mask_scan_shapes       = mask_scan_shapes
        self.raw_mask_match         = raw_mask_match
        self.mask_num_channels      = mask_num_channels
        self.mask_channels_present  = mask_channels_present
        self.patient_ids            = patient_ids
        self.raw_file_names         = raw_file_names
        self.mask_file_names        = mask_file_names
        self.patient_count          = patient_count

        # Ceebs changing it
        data = {
            "patient_id":               self.patient_ids,           # 0
            "raw_file_name":            self.raw_file_names,        # 1
            "mask_file_name":           self.mask_file_names,       # 2
            "raw_shape":                self.raw_scan_shapes,       # 3
            "mask_shape":               self.mask_scan_shapes,      # 4
            "shape_match":              self.raw_mask_match,        # 5
            "num_channels":             self.mask_num_channels,     # 6
            "mask_channels_present":    self.mask_channels_present, # 7
        }
        self.to_csv = pd.DataFrame(data)
        self.to_csv.to_csv(output_dir / "data_audit.csv", index=True)

if __name__ == "__main__":
    audit = DataAudit()
    audit.evaluate()
    print(audit)