# How to run nnSeq2Seq on a new dataset
Given a dataset, nnSeq2Seq fully automatically configures an entire image-to-image translation pipeline that matches its properties. nnSeq2Seq covers the whole pipeline, from preprocessing to model configuration, model training, postprocessing, and ensembling. After running nnSeq2Seq, the trained model(s) can be applied to the test cases for inference.

## Dataset Format
nnSeq2Seq requires the same dataset structure as nnU-Net.
Please read [this](dataset_format.md) for information on how to set up datasets to be compatible with nnSeq2Seq.

We have only verified files in the ".nii.gz" format. It is not clear whether other formats can be well-supported.


## Experiment planning and preprocessing
Given a new dataset, nnSeq2Seq will extract a dataset fingerprint (a set of dataset-specific properties such as image sizes, voxel spacings, intensity information, etc.). This information is used to design network configurations. Each pipeline operates on its own preprocessed version of the dataset.

The easiest way to run fingerprint extraction, experiment planning, and preprocessing is to use:

```sh
# run with source code
python nnseq2seq/experiment_planning/plan_and_preprocess_entrypoints.py -d DATASET_ID

# run with command
nnSeq2Seq_plan_and_preprocess -d DATASET_ID
```

Where `DATASET_ID` is the dataset id, for example, 1 for `Dataset001_BraTS`.

`nnSeq2Seq_plan_and_preprocess` will create a new subfolder in your `nnSeq2Seq_preprocessed` folder named after the dataset. Once the command is completed, there will be a `dataset_fingerprint.json` file and a `nnSeq2SeqPlans.json` file for you to look at (in case you are interested!). Subfolders containing the preprocessed data for your network configurations will also be created.

## Model training
The command to train the model is as follows:
```sh
# run with source code
python nnseq2seq/run/run_training.py \
    -dataset_name_or_id DATASET_ID \  # DATASET_ID can be ID (1) or name (Dataset001_BraTS)
    -configuration CONFIG \           # CONFIG in [2d, 3d]
    -fold FOLD_ID                     # FOLD_ID in [0, 1, 2, 3, 4, all]

# run with command
nnSeq2Seq_train -dataset_name_or_id DATASET_ID -configuration CONFIG -fold FOLD_ID
```

Tips:
- It is recommended to use a `2d` configuration, because the performance of a `3d` configuration needs to be verified.
- It is recommended to train only one fold rather than five folds, because the performance of five folds needs to be verified.

The trained models will be written to the `nnSeq2Seq_results` folder. Each training obtains an automatically generated output folder name:

nnSeq2Seq_results/DatasetXXX_MYNAME/TRAINER_CLASS_NAME__PLANS_NAME__CONFIGURATION/FOLD

For `Dataset001_BraTS`, for example, this looks like this:
```
nnSeq2Seq_results/
├── Dataset001_BraTS
    │── nnSeq2SeqTrainer__nnSeq2SeqPlans__2d
    │    ├── fold_0
    │    ├── fold_1
    │    ├── fold_2
    │    ├── fold_3
    │    ├── fold_4
    │    ├── dataset.json
    │    ├── dataset_fingerprint.json
    │    └── plans.json
    └── nnSeq2SeqTrainer__nnSeq2SeqPlans__3d
         ├── fold_0
         ├── fold_1
         ├── fold_2
         ├── fold_3
         ├── fold_4
         ├── dataset.json
         ├── dataset_fingerprint.json
         └── plans.json
```

## Run inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on and must adhere to the nnU-Net naming scheme for image files (see [dataset format](dataset_format.md) and [inference data format](dataset_format_inference.md)!)

Use the following commands to specify the configuration(s) used for inference manually:
```sh
# run with source code
python nnseq2seq/inference/predict_from_raw_data.py \
    -i INPUT_FOLDER \   # input folder
    -o OUTPUT_FOLDER \  # output folder
    -d DATASET_ID \     # DATASET_ID can be ID (1) or name (Dataset001_BraTS)
    -c CONFIG \         # CONFIG in [2d, 3d]
    -f FOLD_ID \        # FOLD_ID in [0, 1, 2, 3, 4, all]
    -chk CKPT_NAME      # CKPT_NAME in [checkpoint_final.pth, checkpoint_best.pth]. Default is checkpoint_final.pth

# run with command
nnSeq2Seq_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIG -f FOLD_ID -chk CKPT_NAME
```

The results in `OUTPUT_FOLDER` look like this:
```
OUTPUT_FOLDER/
├── CASE_IDENTIFER_1
│   ├── src_{ID1}.nii.gz
│   ├── src_{ID2}.nii.gz
│   ├── pred_src_{ID1}_tgt_{ID1}.nii.gz
│   ├── pred_src_{ID1}_tgt_{ID2}.nii.gz
│   ├── pred_src_{ID2}_tgt_{ID1}.nii.gz
│   ├── pred_src_{ID2}_tgt_{ID2}.nii.gz
│   ├── pred_average_tgt_{ID1}.nii.gz
│   ├── pred_average_tgt_{ID2}.nii.gz
│   ├── pred_md_tgt_{ID1}.nii.gz
│   ├── pred_md_tgt_{ID2}.nii.gz
│   ├── ...
├── CASE_IDENTIFER_2
│   ├── ...
├── CASE_IDENTIFER_3
│   ├── ...
└── ...
```

Each `CASE_IDENTIFER` has a separate folder where the results can be saved. Results include:
- `src_{ID}.nii.gz`: Normalized source image of channel `ID`.
- `pred_src_{ID1}_tgt_{ID2}.nii.gz`: Prediction of translating channel `ID1` to `ID2`.
- `pred_average_tgt_{ID}.nii.gz`: Average of all predictions for channel `ID`.
- `pred_md_tgt_{ID}.nii.gz`: $\mathcal{M}_d$ for channel `ID`.