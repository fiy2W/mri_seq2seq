# nnSeq2Seq for MAMA-MIA
Breast cancer remains the most common cancer among women and a leading cause of female mortality. Dynamic contrast-enhanced MRI (DCE-MRI) is a powerful imaging tool for evaluating breast tumours, yet the field lacks a standardized benchmark for analyzing treatment responses and guiding personalized care.

The [MAMA-MIA](https://www.ub.edu/mama-mia/) Challenge bridges this gap by introducing a dual-task framework to advance AI-driven solutions for:

1) Primary Tumour Segmentation in DCE-MRI
2) Prediction of Pathologic Complete Response (pCR) to Neoadjuvant Chemotherapy (NAC)

## Preparation
1) Register in [MAMA-MIA Challenge](https://www.codabench.org/competitions/7425/).
2) Download the data in [MAMA-MIA Dataset Synapse page](https://doi.org/10.7303/syn60868042).
3) Translate dataset into nnSeq2Seq format.
```python
import os
import SimpleITK as sitk
import json


src_root = '/path/to/mamamia-dataset/images'
src_seg_root = '/path/to/mamamia-dataset/segmentations/expert/'
tgt_root = '/path/to/nnSeq2Seq_raw/Dataset021_mamamia2task/imagesTr'
tgt_seg_root = '/path/to/nnSeq2Seq_raw/Dataset021_mamamia2task/labelsTr'

os.makedirs(tgt_root, exist_ok=True)
os.makedirs(tgt_seg_root, exist_ok=True)

for aid in os.listdir(src_root):
    print(aid)
    src_path1 = os.path.join(src_root, '{}_0000.nii.gz'.format(aid))
    src_path2 = os.path.join(src_root, '{}_0001.nii.gz'.format(aid))
    seg_path = os.path.join(src_seg_root, '{}.nii.gz'.format(aid))
    json_path = '/path/to/mamamia-dataset/patient_info_files/{}.json'.format(aid)

    with open(json_path, 'r') as f:
        data_info = json.load(f)
    
    coords = data_info["primary_lesion"]["breast_coordinates"]
    x_min, x_max = coords["x_min"], coords["x_max"]
    y_min, y_max = coords["y_min"], coords["y_max"]
    z_min, z_max = coords["z_min"], coords["z_max"]

    img1 = sitk.ReadImage(src_path1)[z_min:z_max, y_min:y_max, x_min:x_max]
    img2 = sitk.ReadImage(src_path2)[z_min:z_max, y_min:y_max, x_min:x_max]
    seg = sitk.ReadImage(seg_path)[z_min:z_max, y_min:y_max, x_min:x_max]
    sinwas = img2 - img1
    
    sitk.WriteImage(img1, os.path.join(tgt_root, '{}_0000.nii.gz'.format(aid)))
    sitk.WriteImage(img2, os.path.join(tgt_root, '{}_0001.nii.gz'.format(aid)))
    sitk.WriteImage(sinwas, os.path.join(tgt_root, '{}_0002.nii.gz'.format(aid)))
    sitk.WriteImage(seg, os.path.join(tgt_seg_root, '{}.nii.gz'.format(aid)))

    pcr = data_info['primary_lesion']['pcr']
    if pcr==0 or pcr==1:
        if pcr==0:
            pcr = 0.5
            
        seg = seg * pcr
        sitk.WriteImage(seg, os.path.join(tgt_root, '{}_0003.nii.gz'.format(aid)))
```

4) Create ``dataset.json`` file in ``/path/to/nnSeq2Seq_raw/Dataset021_mamamia2task`` folder.
```json
{
    "channel_names": {
        "0": "pre-contrast",
        "1": "post-contrast",
        "2": "sinwas_input",
        "3": "pcr_output"
    },
    "labels": {
        "background": 0,
        "tumor": 1
    },
    "numChannel": 4,
    "missing": "random",
    "numTraining": 1506,
    "file_ending": ".nii.gz"
}
```

## Run nnSeq2Seq
Run following script.
```sh
# make dataset
python nnseq2seq/experiment_planning/plan_and_preprocess_entrypoints.py -d 21 -c 3d

# training
python nnseq2seq/run/run_training.py -dataset_name_or_id 21 -configuration 3d -fold 0
python nnseq2seq/run/run_training.py -dataset_name_or_id 21 -configuration 3d -fold 1
python nnseq2seq/run/run_training.py -dataset_name_or_id 21 -configuration 3d -fold 2
python nnseq2seq/run/run_training.py -dataset_name_or_id 21 -configuration 3d -fold 3
python nnseq2seq/run/run_training.py -dataset_name_or_id 21 -configuration 3d -fold 4
```