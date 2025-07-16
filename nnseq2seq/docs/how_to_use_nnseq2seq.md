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

If you already know what configuration you need, you can also specify that with `-c 2d`.

`nnSeq2Seq_plan_and_preprocess` will create a new subfolder in your `nnSeq2Seq_preprocessed` folder named after the dataset. Once the command is completed, there will be a `dataset_fingerprint.json` file and a `nnSeq2SeqPlans.json` file for you to look at (in case you are interested!). Subfolders containing the preprocessed data for your network configurations will also be created.

If you want to use more or less GPU memory, please modify the `batch_size` or `patch_size` in `nnSeq2SeqPlans.json`. Note that, `patch_size` needs to be divisible by 4.

```json
{
    ...
    "configurations": {
        "2d": {
            ...
            "batch_size": 8,
            "patch_size": [
                192,
                160
            ],
            ...
        },
        "3d": {
            ...
            "batch_size": 2,
            "patch_size": [
                96,
                96,
                96
            ],
            ...
        },
    },
    ...
}
```

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
nnSeq2Seq stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `--c` to the training command.

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

In each model training output folder (each of the fold_x folder), the following files will be created:

- debug.json: Contains a summary of blueprint and inferred parameters used for training this model as well as a bunch of additional stuff. Not easy to read, but very useful for debugging ;-)
- checkpoint_best.pth: checkpoint files of the best model identified during training. Not used right now unless you explicitly tell nnSeq2Seq to use it.
- checkpoint_final.pth: checkpoint file of the final model (after training has ended). This is what is used for both validation and inference. If training is interrupted, this file does not exist and is replaced by "checkpoint_latest.pth".
- progress.png: Shows losses, pseudo-PSNR, learning rate, and epoch times throughout the training. At the top is a plot of the training (blue) and validation (red) loss during training. It also shows an approximation of the PSNR (green) and a moving average PSNR (dotted green line). This approximation is the average PSNR score. It needs to be taken with a big (!) grain of salt because it is computed on randomly drawn patches from the validation data at the end of each epoch.
- visualization: Some intermediate results to monitor model training.
  - epoch_X.jpg: synthesis images and segmentation at epoch X
    |  |  |  |  |  |  |  |  |
    |--|--|--|--|--|--|--|--|
    | src_1 | src_2 | ... | pred_1 | pred_2 | ... | mask | pred_mask |
  - deep_X.jpg: synthesis images of deep supervision
    |  |  |  |  |  |  |  |  |
    |--|--|--|--|--|--|--|--|
    | src_1 | src_2 | ... | pred_1 | pred_2 | ... | mask | pred_mask |

## Run inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on and must adhere to the nnU-Net naming scheme for image files (see [dataset format](dataset_format.md) and [inference data format](dataset_format_inference.md)!)

### Run inference with commands
Use the following commands to specify the configuration(s) used for inference manually:
```sh
# run with source code
python nnseq2seq/inference/predict_from_raw_data.py \
    -i INPUT_FOLDER \   # input folder
    -o OUTPUT_FOLDER \  # output folder
    -d DATASET_ID \     # DATASET_ID can be ID (1) or name (Dataset001_BraTS)
    -c CONFIG \         # CONFIG in [2d, 3d]
    -f FOLD_ID \        # FOLD_ID in [0, 1, 2, 3, 4, all]
    -chk CKPT_NAME      # CKPT_NAME in [checkpoint_final.pth, checkpoint_latest.pth, checkpoint_best.pth]. Default is checkpoint_final.pth
    --infer_all         # inference all the results

# run with command
nnSeq2Seq_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIG -f FOLD_ID -chk CKPT_NAME --infer_all
```

Using `--infer_all` to output all the results for each subject. If you only want to output some results, add tags for the corresponding results, e.g., `--infer_input` for normalized input images, `--infer_segment` for segmentation, `--infer_translate` for image-to-image translation.

The results in `OUTPUT_FOLDER` look like this:
```
OUTPUT_FOLDER/
├── synthesis-based_sequence_contribution.csv
├── task-specific_sequence_contribution.csv
├── CASE_IDENTIFER_1
│   ├── normalized_source_images
│   │   ├── norm_src_{ID1}.nii.gz
│   │   ├── norm_src_{ID2}.nii.gz
│   │   ├── ...
│   ├── multi2one_inference
│   │   ├── translate_tgt_{ID1}.nii.gz
│   │   ├── translate_tgt_{ID2}.nii.gz
│   │   ├── segmentation.nii.gz
│   │   ├── ...
├── CASE_IDENTIFER_2
│   ├── ...
├── CASE_IDENTIFER_3
│   ├── ...
└── ...
```

Each `CASE_IDENTIFER` has a separate folder where the results can be saved. Results include:
- `norm_src_{ID}.nii.gz`: Normalized source image of channel `ID`.
- `translate_tgt_{ID}.nii.gz`: Prediction of translating all available channels to channel `ID`.
- `segmentation.nii.gz`: Prediction of segmenting using all available channels.


### Run inference in your script
Here are examples of running inference in your Python code.

#### **Step 1:** Import necessary package
Import `nnSeq2SeqPipeline` from source code.
```Python
import numpy as np
import SimpleITK as sitk
import torch

import sys
sys.path.append('/path/to/mri_seq2seq/nnseq2seq')
from nnseq2seq.api.inference import nnSeq2SeqPipeline
```

#### **Step 2:** Initial nnSeq2SeqPipeline
Set `device`, `initialize_from_trained_model_folder`, `use_folds`, and `checkpoint_name` for `nnSeq2SeqPipeline`.
```Python
pipe = nnSeq2SeqPipeline(device='cuda')
pipe.initialize_from_trained_model_folder(
    model_training_output_dir='/path/to/nnSeq2Seq_results/Dataset00X_XXX/nnSeq2SeqTrainer__nnSeq2SeqPlans__2d',
    use_folds=[0],
    checkpoint_name='checkpoint_best.pth'
)
```

#### **Step 3:** Run inference
- Similar to `Run inference with commands`, we can run inference from a folder and save all the outputs as files. We only need to set `input_folder` and `output_folder`.
```Python
pipe.predict_from_image_files(
    input_folder='/path/to/load/input',
    output_folder='/path/to/save/output')
```

- For more flexible usage, we can run inference by inputting with `numpy.Array` image variables and outputing with `numpy.Array` results.
```Python
# Read multi-sequence MRI images with SimpleITK
s1 = sitk.GetArrayFromImage(sitk.ReadImage('case_id_0000.nii.gz'))
s2 = sitk.GetArrayFromImage(sitk.ReadImage('case_id_0001.nii.gz'))
s3 = sitk.GetArrayFromImage(sitk.ReadImage('case_id_0002.nii.gz'))
s4 = sitk.GetArrayFromImage(sitk.ReadImage('case_id_0003.nii.gz'))

# Combine all images into one variable, if the sequence is missing, remember padding with 0. Please check the size of the images here by yourself. Make sure D, W, H can be divided by 16.
data = np.stack([s1, s2, s3, s4], axis=0)

# Set propoerties of the sample
target_id = 1
properties = {
    'num_channel': 4,
    'available_channel': [0,1,2,3],
}

# Run inference
pred, pred_mask = pipe.predict_from_image_volume(data=data, tgt_seq=target_id, properties=properties)

# Save output
sitk.WriteImage(sitk.GetImageFromArray(pred), 'translate_tgt_1.nii.gz')
sitk.WriteImage(sitk.GetImageFromArray(pred_mask), 'segmentation.nii.gz')
```


- It's also possible to run inference by inputting with `torch.Tensor` image variables and outputing with `torch.Tensor` results.
```Python
# Preprocess data and convert it to torch.Tensor
data = np.array(np.stack([s1, s2, s3, s4], axis=0), dtype=np.half)
data = pipe.data_normalize(data=data, seg=np.ones_like(data),
                            available_sequence=properties['available_channel'])
data = torch.from_numpy(data)

# Get one slice from volume
slice_id = data.shape[1]//2
data_slice = data[:,slice_id].unsqueeze(0).to(device='cuda', dtype=torch.half)  # (1,C,W,H)

# Run inference
pred, pred_mask = pipe.predict_from_image_slice_tensor(data=data_slice, tgt_seq=target_id, properties=properties)

# save as image
import cv2
cv2.imwrite('translate_tgt_1.png', np.uint8(np.clip(pred.cpu().numpy()[0,0], a_min=0, a_max=1)*255))
```