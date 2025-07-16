import os
import shutil
from pathlib import Path
from typing import List
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json

import sys
sys.path.append('.')
from nnseq2seq.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnseq2seq.paths import nnSeq2Seq_raw, nnSeq2Seq_preprocessed



def make_out_dirs(dataset_id: int, task_name="BraTS"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnSeq2Seq_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_train_folder: Path, src_data_test_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the BraTS dataset to the nnSeq2Seq dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_train_folder).iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_test_folder).iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        file_name = os.path.basename(patient_dir)
        shutil.copy(os.path.join(patient_dir, '{}_t1.nii.gz'.format(file_name)), train_dir / '{}_0000.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_t1ce.nii.gz'.format(file_name)), train_dir / '{}_0001.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_t2.nii.gz'.format(file_name)), train_dir / '{}_0002.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_flair.nii.gz'.format(file_name)), train_dir / '{}_0003.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_seg.nii.gz'.format(file_name)), labels_dir / '{}.nii.gz'.format(file_name))
        num_training_cases += 1

    # Copy test files.
    for patient_dir in patients_test:
        file_name = os.path.basename(patient_dir)
        shutil.copy(os.path.join(patient_dir, '{}_t1.nii.gz'.format(file_name)), test_dir / '{}_0000.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_t1ce.nii.gz'.format(file_name)), test_dir / '{}_0001.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_t2.nii.gz'.format(file_name)), test_dir / '{}_0002.nii.gz'.format(file_name))
        shutil.copy(os.path.join(patient_dir, '{}_flair.nii.gz'.format(file_name)), test_dir / '{}_0003.nii.gz'.format(file_name))
        
    return num_training_cases


def convert_brats(src_data_train_folder: str, src_data_test_folder: str, dataset_id=1, version_id='2021'):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_train_folder), Path(src_data_test_folder), train_dir, labels_dir, test_dir)

    if version_id in ['2020', '2021']:
        num_channels = 4
        generate_dataset_json(
            str(out_dir),
            channel_names={
                0: "T1",
                1: "T1Gd",
                2: "T2",
                3: "Flair",
            },
            labels={
                "background": 0,
                "NCR": 1,
                "ED": 2,
                "unknown": 3,
                "ET": 4,
            },
            file_ending=".nii.gz",
            num_channels=num_channels,
            num_training_cases=num_training_cases,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-itr",
        "--input_train_folder",
        type=str,
        help="The downloaded BraTS dataset train dir.",
    )
    parser.add_argument(
        "-its",
        "--input_test_folder",
        type=str,
        help="The downloaded BraTS dataset test dir.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=1, help="nnSeq2Seq Dataset ID, default: 1"
    )
    parser.add_argument(
        "-v", "--version_id", required=False, type=str, default='2021', help="BraTS Dataset ID, default: 2021"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_brats(args.input_train_folder, args.input_test_folder, args.dataset_id, args.version_id)

    print("Done!")