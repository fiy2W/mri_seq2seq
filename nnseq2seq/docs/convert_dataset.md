# Convert Dataset
Example for conversion of some public datasets.

## BraTS
Convert BraTS dataset.
```sh
# run with source code
python nnseq2seq/dataset_conversion/Dataset001_BraTS.py \
    -itr /path/to/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021 \  # training folder
    -its /path/to/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_ValidationData \           # testing/valiation folder
    -d 1 \                                                                        # dataset ID
    -v 2021                                                                       # Year of BraTS
```