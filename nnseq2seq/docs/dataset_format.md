# nnSeq2Seq dataset format
nnSeq2Seq adopts the same dataset structure as nnU-Net. Our improvement is that nnSeq2Seq can **allow incomplete sequences or modalities** in the dataset.

[Quick start](convert_dataset.md) for some public datasets.

## What do training cases look like?
Each training case is associated with an identifier (`CASE_IDENTIFIER`), a unique name for that case. Each training case consists of multiple images and their corresponding segmentation.

Images are multiple sequences or modalities for the same case, and they **MUST** have the same geometry (same shape, spacing, direction, etc.) and must be co-registered.
Image files must therefore follow the following naming convention: `{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}`.
Hereby, `XXXX` is the 4-digit modality/channel identifier (it should be unique for each modality/channel, e.g., "0000" for T1, "0001" for T1Gd, "0002" for T2, "0003" for Flair, …), and `FILE_ENDING` is the file extension used by your image format (.nii.gz, ...). See below for concrete examples. The dataset.json file connects channel names with the channel identifiers in the 'channel_names' key (see below for details).

Segmentation is not necessary for us if only synthesising missing sequence. However, since nnSeq2Seq will oversample the segmentation foreground, this helps the model pay more attention to the lesion area. Segmentations must share the same geometry with their corresponding images. Segmentations are integer maps, with each value representing a semantic class. The background must be 0. If there is no background, do not use the label 0 for something else! Integer values of your semantic classes must be consecutive (1, 2, 3, ...). Of course, not all labels have to be present in each training case. Segmentations are saved as `{CASE_IDENTIFER}.{FILE_ENDING}`. **Let all pixels be 0 in the segmentation if you do not have any annotations.**

**Note: Missing sequences or modalities are allowed in cases.**

## Supported file formats
nnSeq2Seq requires the same file format for images and segmentations!

We have only verified files in the ".nii.gz" format. It is not clear whether other formats can be well-supported.

## Dataset folder structure
Datasets must be located in the `nnSeq2Seq_raw` folder. Each dataset is stored as a separate "Dataset". Datasets are associated with a dataset ID, a three-digit integer, and a dataset name (which you can freely choose). For example, "Dataset001_BraTS" has "BraTS" as the dataset name, and the dataset ID is 1. Datasets are stored in the `nnSeq2Seq_raw` folder like this:
```
nnSeq2Seq_raw/
├── Dataset001_BraTS
├── Dataset002_XXXXX
├── Dataset003_XXXXX
├── ...
```

Within each dataset folder, the following structure is expected:
```
Dataset001_BraTS/
├── dataset.json
├── imagesTr
├── imagesTs  # optional
└── labelsTr
```

- `imagesTr` contains the images belonging to the training cases. nnSeq2Seq will perform pipeline configuration, training with cross-validation, as well as finding postprocessing and the best ensemble using this data.
- `imagesTs` (optional) contains the images that belong to the test cases. nnSeq2Seq does not use them!
- `labelsTr` contains the images with the ground truth segmentation maps for the training cases.
- `dataset.json` contains metadata of the dataset.

An example of the structural results for the BraTS2021 Dataset (script can be found [here](convert_dataset.md)). This dataset has four input channels: T1 (0000), T1Gd (0001), T2 (0002), and Flair (0003). Note that the `imagesTs` folder is optional and does not have to be present.

```
nnSeq2Seq_raw/Dataset001_BraTS/
├── dataset.json
├── imagesTr
│   ├── BraTS2021_00000_0000.nii.gz
│   ├── BraTS2021_00000_0001.nii.gz
│   ├── BraTS2021_00000_0002.nii.gz
│   ├── BraTS2021_00000_0003.nii.gz
│   ├── BraTS2021_00002_0000.nii.gz
│   ├── BraTS2021_00002_0001.nii.gz
│   ├── BraTS2021_00002_0002.nii.gz
│   ├── BraTS2021_00002_0003.nii.gz
│   ├── ...
├── imagesTs
│   ├── BraTS2021_00001_0000.nii.gz
│   ├── BraTS2021_00001_0001.nii.gz
│   ├── BraTS2021_00001_0002.nii.gz
│   ├── BraTS2021_00001_0003.nii.gz
│   ├── BraTS2021_00013_0000.nii.gz
│   ├── BraTS2021_00013_0001.nii.gz
│   ├── BraTS2021_00013_0002.nii.gz
│   ├── BraTS2021_00013_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BraTS2021_00000.nii.gz
    ├── BraTS2021_00002.nii.gz
    ├── ...
```

Here is an example of the missing sequence simulation on the BraTS2021 dataset.
```
nnSeq2Seq_raw/Dataset001_BraTS/
├── dataset.json
├── imagesTr
│   ├── BraTS2021_00000_0000.nii.gz
│   ├── BraTS2021_00000_0001.nii.gz
│   ├── BraTS2021_00002_0002.nii.gz
│   ├── BraTS2021_00002_0003.nii.gz
│   ├── ...
├── imagesTs
│   ├── BraTS2021_00001_0000.nii.gz
│   ├── BraTS2021_00001_0001.nii.gz
│   ├── BraTS2021_00001_0002.nii.gz
│   ├── BraTS2021_00013_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BraTS2021_00000.nii.gz
    ├── BraTS2021_00002.nii.gz
    ├── ...
```

[See also the dataset format for inference!](dataset_format_inference.md)

## dataset.json
The `dataset.json` contains metadata that nnSeq2Seq needs for training.

Here is what the `dataset.json` should look like in the example of the `Dataset001_BraTS`:
```
{
    "channel_names": {
        "0": "T1",
        "1": "T1Gd",
        "2": "T2",
        "3": "Flair"
    },
    "labels": {
        "background": 0,
        "NCR": 1,
        "ED": 2,
        "unknown": 3,
        "ET": 4,
        # "foreground": 5,
    },
    "numChannel": 4,  # Declare the total number of sequences or modalities
    "numTraining": 1251,
    "file_ending": ".nii.gz"
}
```
We added `numChannel` to declare the total number of sequences or modalities.

**Note: Cases with `"foreground"` in the `"label"` are ignored when calculating the segmentation loss during training.** This is primarily used when there are cases of missing segmentation labels in the dataset.

The `channel_names` determine the normalization used by nnSeq2Seq. If a channel is marked as `CT`, then a global normalization based on the intensities in the foreground pixels will be used. If it is something else, a per-channel percentile will be used. See [here](intensity_normalization.md) for more information about custom intensity normalization.