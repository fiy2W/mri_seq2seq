# Dataset format for Inference
Read the documentation on the overall [data format](./dataset_format.md) first!

The data format for inference must match the one used for the raw data (specifically, the images must be in the same format as in the `imagesTr` folder). The filenames must start with a unique identifier, followed by a 4-digit modality identifier. Here is an example of BraTS2021 dataset:

```
input_folder
├── BraTS2021_00001_0000.nii.gz
├── BraTS2021_00001_0003.nii.gz
├── BraTS2021_00013_0001.nii.gz
├── BraTS2021_00013_0002.nii.gz
├── ...
```

Remember that the file format used for inference (`.nii.gz` in this example) must be the same as that used for training (as specified in `file_ending` in the `dataset.json`)!